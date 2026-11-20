// nanollm — a minimal, from-scratch LLM inference runtime in C++17.
//
// Target model family: Qwen2 / Qwen2.5 dense (GQA attention, NeoX-style RoPE,
// RMSNorm, SwiGLU MLP, tied embeddings in the 0.5B variant).
//
// Everything is implemented here on purpose — tokenizer (GPT-2 byte-level
// BPE), weight loading (mmap'd .nnlm format), the transformer forward pass
// with an explicit KV cache, and sampling. No external dependencies.
//
// Build:  make
// Export: python3 export.py --hf-dir models/Qwen2.5-0.5B-Instruct --out models/qwen05.nnlm
// Run:    ./build/nanollm -m models/qwen05.nnlm -p "hello"
//
// License: MIT

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define NANOLLM_NEON 1
#endif

// ---------------------------------------------------------------------------
// utilities
// ---------------------------------------------------------------------------

static void die(const std::string& msg) {
    std::fprintf(stderr, "nanollm: %s\n", msg.c_str());
    std::exit(1);
}

// Run `fn(begin, end)` over [0, n) split across hardware threads.
// Each range is independent, so results are deterministic w.r.t. thread count.
static void parallel_for(size_t n, const std::function<void(size_t, size_t)>& fn) {
    size_t nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 8;
    if (nthreads > 16) nthreads = 16;
    if (n < 512 || nthreads <= 1) { if (n) fn(0, n); return; }  // spawn overhead not worth it below this
    std::vector<std::thread> pool;
    size_t chunk = (n + nthreads - 1) / nthreads;
    for (size_t t = 0; t < nthreads; t++) {
        pool.emplace_back([&, t] {
            size_t lo = t * chunk;
            size_t hi = std::min(n, lo + chunk);
            if (lo < hi) fn(lo, hi);
        });
    }
    for (auto& th : pool) th.join();
}

// UTF-8 helpers ---------------------------------------------------------------

// Decode one UTF-8 codepoint at s[*i]; advances *i. Returns -1 on invalid byte
// (treated as a single-byte "codepoint" so we never lose data).
static int32_t utf8_decode(const std::string& s, size_t* i) {
    unsigned char c = s[*i];
    if (c < 0x80) { (*i)++; return c; }
    int len; uint32_t cp;
    if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
    else { (*i)++; return -1; }
    for (int k = 1; k < len; k++) {
        if (*i + (size_t)k >= s.size() || (s[*i + k] & 0xC0) != 0x80) { (*i)++; return -1; }
        cp = (cp << 6) | (s[*i + k] & 0x3F);
    }
    *i += len;
    return (int32_t)cp;
}

static void utf8_encode(int32_t cp, std::string* out) {
    if (cp < 0) return;
    uint32_t u = (uint32_t)cp;
    if (u < 0x80) { *out += (char)u; }
    else if (u < 0x800) {
        *out += (char)(0xC0 | (u >> 6)); *out += (char)(0x80 | (u & 0x3F));
    } else if (u < 0x10000) {
        *out += (char)(0xE0 | (u >> 12)); *out += (char)(0x80 | ((u >> 6) & 0x3F));
        *out += (char)(0x80 | (u & 0x3F));
    } else {
        *out += (char)(0xF0 | (u >> 18)); *out += (char)(0x80 | ((u >> 12) & 0x3F));
        *out += (char)(0x80 | ((u >> 6) & 0x3F)); *out += (char)(0x80 | (u & 0x3F));
    }
}

// ---------------------------------------------------------------------------
// tokenizer: GPT-2 byte-level BPE (Qwen vocab)
// ---------------------------------------------------------------------------

// GPT-2 "bytes to unicode" mapping: printable bytes map to themselves, the
// rest are assigned consecutive codepoints starting at 256, so every byte
// sequence round-trips through printable text.
struct ByteUni {
    std::array<int32_t, 256> b2u;
    std::array<int32_t, 512> u2b;  // index by codepoint; -1 = not a mapped byte
    ByteUni() {
        u2b.fill(-1);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                             (b >= 0xAE && b <= 0xFF);
            int32_t u = printable ? b : 256 + n++;
            b2u[b] = u;
            u2b[u < 512 ? u : 511] = b;  // max mapped cp is 256+67 < 512
        }
    }
    std::string bytes_to_uni(const char* p, size_t n) const {
        std::string out;
        out.reserve(n * 2);
        for (size_t i = 0; i < n; i++) utf8_encode(b2u[(unsigned char)p[i]], &out);
        return out;
    }
    std::string uni_to_bytes(const std::string& uni) const {
        std::string out;
        size_t i = 0;
        while (i < uni.size()) {
            int32_t cp = utf8_decode(uni, &i);
            if (cp >= 0 && cp < 512 && u2b[cp] >= 0) out += (char)u2b[cp];
        }
        return out;
    }
};
static const ByteUni kByteUni;

static std::string hex_decode(const std::string& hex) {
    std::string out;
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        int hi = nib(hex[i]), lo = nib(hex[i + 1]);
        if (hi < 0 || lo < 0) die("bad hex in tokenizer file");
        out += (char)((hi << 4) | lo);
    }
    return out;
}

struct Tokenizer {
    std::unordered_map<std::string, int> vocab;          // byte-uni token -> id
    std::vector<std::string> id2tok;                     // id -> byte-uni token
    std::map<std::pair<std::string, std::string>, int> merge_rank;
    std::vector<std::pair<std::string, std::string>> specials;  // text -> handled at encode
    int im_start = -1, im_end = -1, eos = -1;

    void load(const std::string& tsv_path, const std::string& merges_path);

    // Encode raw text (specials split out first) to ids.
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

  private:
    std::vector<std::string> pretokenize(const std::string& text) const;
    std::vector<int> bpe(const std::string& uni_word) const;
};

void Tokenizer::load(const std::string& tsv_path, const std::string& merges_path) {
    std::ifstream f(tsv_path);
    if (!f) die("cannot open " + tsv_path);
    std::string line;
    std::getline(f, line);
    size_t count = (size_t)std::stoull(line);
    id2tok.clear();
    id2tok.reserve(count);
    int max_id = -1;
    while (std::getline(f, line)) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        int id = std::stoi(line.substr(0, tab));
        std::string tok = hex_decode(line.substr(tab + 1));
        vocab[tok] = id;
        if (id >= (int)id2tok.size()) id2tok.resize(id + 1);
        id2tok[id] = tok;
        max_id = std::max(max_id, id);
    }
    if ((size_t)max_id + 1 != count) die("tokenizer count mismatch");

    std::ifstream m(merges_path);
    if (!m) die("cannot open " + merges_path);
    std::getline(m, line);
    int rank = 0;
    while (std::getline(m, line)) {
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        merge_rank[{hex_decode(line.substr(0, sp)), hex_decode(line.substr(sp + 1))}] = rank++;
    }

    auto lookup = [&](const char* s) { auto it = vocab.find(s); return it == vocab.end() ? -1 : it->second; };
    im_start = lookup("<|im_start|>");
    im_end = lookup("<|im_end|>");
    eos = lookup("<|endoftext|>");
}

// Approximate Qwen's tiktoken pre-tokenizer regex:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} |
//    ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// ASCII is exact; for codepoints >= 0x80 we classify via common Unicode
// blocks (Han/Kana/Hangul/etc. count as letters, CJK/fullwidth punctuation
// as symbols). Documented limitation; good enough for mixed CJK/English.
static bool uni_is_letter(int32_t cp) {
    if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return true;
    if (cp < 0x80) return false;
    // treat as symbol blocks (punctuation / fullwidth forms)
    if ((cp >= 0x2000 && cp <= 0x206F) || (cp >= 0x3000 && cp <= 0x303F) ||
        (cp >= 0xFF01 && cp <= 0xFF60)) return false;
    return true;  // Han, Kana, Hangul, accented Latin, ...
}
static bool uni_is_digit(int32_t cp) { return cp >= '0' && cp <= '9'; }
static bool uni_is_space(int32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v';
}
static bool lower_ascii(int32_t cp) { return (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp; }

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
    // decode to codepoints, remembering byte spans
    std::vector<int32_t> cps;
    std::vector<std::pair<size_t, size_t>> spans;  // byte offset/len per codepoint
    size_t i = 0;
    while (i < text.size()) {
        size_t start = i;
        int32_t cp = utf8_decode(text, &i);
        cps.push_back(cp);
        spans.push_back({start, i - start});
    }
    size_t n = cps.size();
    std::vector<std::string> out;
    size_t p = 0;
    auto emit = [&](size_t lo, size_t hi) {
        std::string bytes;
        for (size_t k = lo; k < hi; k++) bytes.append(text, spans[k].first, spans[k].second);
        if (!bytes.empty()) out.push_back(bytes);
    };
    while (p < n) {
        int32_t cp = cps[p];
        // contractions: 's 't 're 've 'm 'll 'd (case-insensitive)
        if (cp == '\'' && p + 1 < n) {
            static const char* forms3[] = {"re", "ve", "ll"};
            int32_t a = lower_ascii(cps[p + 1]);
            int32_t b = p + 2 < n ? lower_ascii(cps[p + 2]) : -1;
            bool two = false;
            for (auto* s : forms3)
                if (a == s[0] && b == s[1]) two = true;
            if (two) { emit(p, p + 3); p += 3; continue; }
            if (a == 's' || a == 't' || a == 'm' || a == 'd') { emit(p, p + 2); p += 2; continue; }
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+
        {
            size_t j = p;
            if (cps[j] != '\r' && cps[j] != '\n' && !uni_is_letter(cps[j]) && !uni_is_digit(cps[j]) &&
                j + 1 < n)
                j++;
            if (j < n && uni_is_letter(cps[j])) {
                size_t k = j;
                while (k < n && uni_is_letter(cps[k])) k++;
                emit(p, k);
                p = k;
                continue;
            }
        }
        // \p{N} (single digit)
        if (uni_is_digit(cp)) { emit(p, p + 1); p++; continue; }
        //  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t j = p;
            if (cps[j] == ' ' && j + 1 < n) j++;
            if (j < n && !uni_is_space(cps[j]) && !uni_is_letter(cps[j]) && !uni_is_digit(cps[j])) {
                size_t k = j;
                while (k < n && !uni_is_space(cps[k]) && !uni_is_letter(cps[k]) && !uni_is_digit(cps[k])) k++;
                while (k < n && (cps[k] == '\r' || cps[k] == '\n')) k++;
                emit(p, k);
                p = k;
                continue;
            }
        }
        // whitespace runs. Mirror tiktoken: a space run followed by a
        // non-space keeps its last space for the next branch (" word").
        if (uni_is_space(cp)) {
            size_t k = p;
            while (k < n && uni_is_space(cps[k])) k++;
            bool has_newline = false;
            for (size_t q = p; q < k; q++)
                if (cps[q] == '\n' || cps[q] == '\r') has_newline = true;
            if (!has_newline && k < n && k - p > 1) k--;  // leave trailing space
            emit(p, k);
            p = k;
            continue;
        }
        emit(p, p + 1);  // fallback: single codepoint
        p++;
    }
    return out;
}

std::vector<int> Tokenizer::bpe(const std::string& uni_word) const {
    // split into single byte-uni characters
    std::vector<std::string> sym;
    size_t i = 0;
    while (i < uni_word.size()) {
        size_t start = i;
        utf8_decode(uni_word, &i);
        sym.push_back(uni_word.substr(start, i - start));
    }
    while (sym.size() > 1) {
        int best_rank = 1 << 30;
        size_t best_i = 0;
        for (size_t k = 0; k + 1 < sym.size(); k++) {
            auto it = merge_rank.find({sym[k], sym[k + 1]});
            if (it != merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = k;
            }
        }
        if (best_rank == (1 << 30)) break;
        sym[best_i] += sym[best_i + 1];
        sym.erase(sym.begin() + best_i + 1);
    }
    std::vector<int> ids;
    ids.reserve(sym.size());
    for (auto& s : sym) {
        auto it = vocab.find(s);
        if (it != vocab.end()) ids.push_back(it->second);
    }
    return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    const std::string specials_[] = {"<|im_start|>", "<|im_end|>", "<|endoftext|>"};
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best = std::string::npos;
        size_t best_len = 0;
        for (auto& s : specials_) {
            size_t at = text.find(s, pos);
            if (at != std::string::npos && (best == std::string::npos || at < best)) {
                best = at;
                best_len = s.size();
            }
        }
        size_t chunk_end = (best == std::string::npos) ? text.size() : best;
        for (auto& pre : pretokenize(text.substr(pos, chunk_end - pos))) {
            std::string uni = kByteUni.bytes_to_uni(pre.data(), pre.size());
            for (int id : bpe(uni)) ids.push_back(id);
        }
        if (best != std::string::npos) {
            auto it = vocab.find(text.substr(best, best_len));
            if (it != vocab.end()) ids.push_back(it->second);
            pos = best + best_len;
        } else {
            pos = chunk_end;
        }
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string uni;
    for (int id : ids)
        if (id >= 0 && id < (int)id2tok.size()) uni += id2tok[id];
    return kByteUni.uni_to_bytes(uni);
}

// ---------------------------------------------------------------------------
// model: mmap'd .nnlm format
// ---------------------------------------------------------------------------

struct Config {
    int n_layers, hidden, n_heads, n_kv_heads, head_dim, ffn, vocab;
    float rope_theta, rms_eps;
    int max_seq, tie_embeddings;
};

// Little-endian cursor over the mmap'd file.
struct Cursor {
    const uint8_t* p;
    template <typename T>
    T get() {
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str(size_t n) {
        std::string s((const char*)p, n);
        p += n;
        return s;
    }
};

// mmap'd .nnlm model: fixed header + per-tensor directory of absolute offsets.
// Per-tensor dtype: 0 = f32, 1 = int8 (weights only, with a "<name>.s" f32
// per-output-channel scale tensor — symmetric per-channel quantization).
struct Model {
    void* map = nullptr;
    size_t map_len = 0;
    Config cfg{};
    struct TView {
        std::vector<uint64_t> dims;
        uint64_t off;  // absolute offset in file
        uint32_t dtype = 0;
    };
    std::unordered_map<std::string, TView> dir;
    std::unordered_map<std::string, const float*> cache;

    ~Model() {
        if (map) munmap(map, map_len);
    }
    const float* W(const std::string& name) {
        auto it = cache.find(name);
        if (it != cache.end()) return it->second;
        auto d = dir.find(name);
        if (d == dir.end()) die("missing tensor: " + name);
        if (d->second.dtype != 0) die("tensor " + name + " is int8; use W8()");
        const float* ptr = (const float*)((const uint8_t*)map + d->second.off);
        cache[name] = ptr;
        return ptr;
    }
    const int8_t* W8(const std::string& name) {
        auto d = dir.find(name);
        if (d == dir.end()) die("missing tensor: " + name);
        if (d->second.dtype != 1) die("tensor " + name + " is not int8");
        return (const int8_t*)((const uint8_t*)map + d->second.off);
    }
    const float* Wscale(const std::string& name) { return W(name + ".s"); }
    bool is_int8(const std::string& name) const {
        auto d = dir.find(name);
        return d != dir.end() && d->second.dtype == 1;
    }
    bool has(const std::string& name) { return dir.count(name) > 0; }
    void load(const std::string& path);
};

void Model::load(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) die("cannot open model " + path);
    struct stat st;
    if (fstat(fd, &st) != 0) die("fstat failed");
    map_len = (size_t)st.st_size;
    map = mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) die("mmap failed");

    Cursor c{(const uint8_t*)map};
    if (c.str(8) != "nanollm1") die("bad magic (not a .nnlm file)");
    if (c.get<uint32_t>() != 1) die("unsupported .nnlm version");
    cfg.n_layers = c.get<int32_t>();
    cfg.hidden = c.get<int32_t>();
    cfg.n_heads = c.get<int32_t>();
    cfg.n_kv_heads = c.get<int32_t>();
    cfg.head_dim = c.get<int32_t>();
    cfg.ffn = c.get<int32_t>();
    cfg.vocab = c.get<int32_t>();
    cfg.rope_theta = c.get<float>();
    cfg.rms_eps = c.get<float>();
    cfg.max_seq = c.get<int32_t>();
    cfg.tie_embeddings = c.get<int32_t>();
    if (c.get<uint32_t>() != 0) die("unsupported dtype (v1 = f32 only)");
    c.get<uint64_t>();  // data_offset (kept for format compat; offsets below are absolute)
    uint32_t n_tensors = c.get<uint32_t>();
    for (uint32_t i = 0; i < n_tensors; i++) {
        std::string name = c.str(c.get<uint16_t>());
        TView tv;
        uint32_t dtype = c.get<uint32_t>();
        uint32_t ndims = c.get<uint32_t>();
        uint64_t total = 1;
        for (uint32_t d = 0; d < ndims; d++) {
            uint64_t dim = c.get<uint64_t>();
            tv.dims.push_back(dim);
            total *= dim;
        }
        uint64_t nbytes = c.get<uint64_t>();
        tv.off = c.get<uint64_t>();  // absolute file offset of tensor data
        tv.dtype = dtype;
        size_t expect = (dtype == 0) ? total * 4 : total;
        if ((dtype != 0 && dtype != 1) || nbytes != expect)
            die("tensor " + name + ": bad dtype/size");
        dir[name] = tv;
    }
}

// ---------------------------------------------------------------------------
// transformer forward pass with explicit KV cache
// ---------------------------------------------------------------------------

struct KVCache {
    // layout: [layer][pos][kv_head][head_dim]
    std::vector<float> k, v;
    int max_seq, n_kv_heads, head_dim;
    int len = 0;  // tokens currently cached
    void alloc(const Config& c) {
        max_seq = c.max_seq;
        n_kv_heads = c.n_kv_heads;
        head_dim = c.head_dim;
        size_t n = (size_t)c.n_layers * max_seq * n_kv_heads * head_dim;
        k.resize(n);
        v.resize(n);
        len = 0;
    }
    float* K(int layer, int pos) {
        return &k[(((size_t)layer * max_seq) + pos) * n_kv_heads * head_dim];
    }
    float* V(int layer, int pos) {
        return &v[(((size_t)layer * max_seq) + pos) * n_kv_heads * head_dim];
    }
};

// out (M,N) = A (M,K) @ W (N,K)^T (+ bias)
// Decode (M<=2) parallelizes over output columns instead of rows: with a
// single activation row there is no row parallelism to exploit.
static void matmul(float* out, const float* A, const float* W, const float* bias, int M, int K,
                   int N) {
    if (M <= 2) {
        parallel_for((size_t)N, [&](size_t lo, size_t hi) {
            for (size_t n = lo; n < hi; n++) {
                const float* w = W + n * (size_t)K;
                for (int m = 0; m < M; m++) {
                    const float* a = A + m * (size_t)K;
                    float acc = 0.f;
                    for (int k = 0; k < K; k++) acc += a[k] * w[k];
                    out[m * (size_t)N + n] = bias ? acc + bias[n] : acc;
                }
            }
        });
        return;
    }
    parallel_for((size_t)M, [&](size_t lo, size_t hi) {
        for (size_t m = lo; m < hi; m++) {
            const float* a = A + m * (size_t)K;
            float* o = out + m * (size_t)N;
            for (int n = 0; n < N; n++) {
                const float* w = W + (size_t)n * K;
                float acc = 0.f;
                for (int k = 0; k < K; k++) acc += a[k] * w[k];
                o[n] = bias ? acc + bias[n] : acc;
            }
        }
    });
}

// INT8 weight-only variant: out[m,n] = scale[n] * Σ a[k]·w8[n][k] (+ bias).
// Weights stay int8 in cache-friendly memory; dequantization folds into the
// matmul epilogue, halving weight bytes read per token versus F32.
//
// The inner dot product has an ARM NEON path: 16 int8 weights are widened to
// four float32x4 lanes and fused-multiplied with the activation vector. On
// the scalar fallback the int8→float conversion defeats autovectorization,
// which makes INT8 no faster than F32 despite half the memory traffic.
#if defined(NANOLLM_NEON)
static inline float dot_w8(const float* a, const int8_t* w, int K) {
    float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
    float32x4_t acc2 = vdupq_n_f32(0.f), acc3 = vdupq_n_f32(0.f);
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        int8x16_t q = vld1q_s8(w + k);
        int16x8_t lo = vmovl_s8(vget_low_s8(q));
        int16x8_t hi = vmovl_high_s8(q);
        acc0 = vfmaq_f32(acc0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
                         vld1q_f32(a + k + 0));
        acc1 = vfmaq_f32(acc1, vcvtq_f32_s32(vmovl_high_s16(lo)), vld1q_f32(a + k + 4));
        acc2 = vfmaq_f32(acc2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
                         vld1q_f32(a + k + 8));
        acc3 = vfmaq_f32(acc3, vcvtq_f32_s32(vmovl_high_s16(hi)), vld1q_f32(a + k + 12));
    }
    float32x4_t s01 = vaddq_f32(acc0, acc1);
    float32x4_t s23 = vaddq_f32(acc2, acc3);
    float32x4_t s = vaddq_f32(s01, s23);
    float acc = vaddvq_f32(s);
    for (; k < K; k++) acc += a[k] * (float)w[k];
    return acc;
}
#else
static inline float dot_w8(const float* a, const int8_t* w, int K) {
    float acc = 0.f;
    for (int k = 0; k < K; k++) acc += a[k] * (float)w[k];
    return acc;
}
#endif

static void matmul_w8(float* out, const float* A, const int8_t* W8t, const float* scale,
                      const float* bias, int M, int K, int N) {
    if (M <= 2) {
        parallel_for((size_t)N, [&](size_t lo, size_t hi) {
            for (size_t n = lo; n < hi; n++) {
                const int8_t* w = W8t + n * (size_t)K;
                for (int m = 0; m < M; m++) {
                    const float* a = A + m * (size_t)K;
                    out[m * (size_t)N + n] = scale[n] * dot_w8(a, w, K) + (bias ? bias[n] : 0.f);
                }
            }
        });
        return;
    }
    parallel_for((size_t)M, [&](size_t lo, size_t hi) {
        for (size_t m = lo; m < hi; m++) {
            const float* a = A + m * (size_t)K;
            float* o = out + m * (size_t)N;
            for (int n = 0; n < N; n++) {
                const int8_t* w = W8t + (size_t)n * K;
                o[n] = scale[n] * dot_w8(a, w, K) + (bias ? bias[n] : 0.f);
            }
        }
    });
}

static void rmsnorm(float* out, const float* x, const float* w, int rows, int H, float eps) {
    parallel_for((size_t)rows, [&](size_t lo, size_t hi) {
        for (size_t r = lo; r < hi; r++) {
            const float* xr = x + r * H;
            float* orow = out + r * H;
            float ss = 0.f;
            for (int i = 0; i < H; i++) ss += xr[i] * xr[i];
            float inv = 1.0f / std::sqrt(ss / H + eps);
            for (int i = 0; i < H; i++) orow[i] = xr[i] * inv * w[i];
        }
    });
}

static void rmsnorm_inplace(float* x, const float* w, int H, float eps) {
    float ss = 0.f;
    for (int i = 0; i < H; i++) ss += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ss / H + eps);
    for (int i = 0; i < H; i++) x[i] = x[i] * inv * w[i];
}

// NeoX-style RoPE (HF rotate_half convention).
// buf layout: [token][head][head_dim]; positions absolute.
static void rope(float* buf, int base_pos, int T, int n_heads, int head_dim, float theta) {
    int half = head_dim / 2;
    for (int t = 0; t < T; t++) {
        float* row = buf + (size_t)t * n_heads * head_dim;
        for (int h = 0; h < n_heads; h++) {
            float* qh = row + (size_t)h * head_dim;
            for (int i = 0; i < half; i++) {
                float freq = std::pow(theta, -2.0f * i / head_dim);
                float ang = (base_pos + t) * freq;
                float cs = std::cos(ang), sn = std::sin(ang);
                float x1 = qh[i], x2 = qh[i + half];
                qh[i] = x1 * cs - x2 * sn;
                qh[i + half] = x2 * cs + x1 * sn;
            }
        }
    }
}

struct Runtime {
    Model model;
    KVCache kv;
    Tokenizer tok;

    // scratch buffers (sized on load)
    std::vector<float> x, xb, q, k, v, attn, projout, g, u, d, logits;
    std::vector<float> scores;

    void load(const std::string& model_path) {
        model.load(model_path);
        std::string base = model_path;
        auto dot = base.rfind('.');
        if (dot != std::string::npos) base = base.substr(0, dot);
        tok.load(base + ".tok.tsv", base + ".merges");
        kv.alloc(model.cfg);
        const Config& c = model.cfg;
        size_t T = 1;  // grows with prefill
        x.resize(T * c.hidden);
        xb.resize(T * c.hidden);
        q.resize(T * c.n_heads * c.head_dim);
        k.resize(T * c.n_kv_heads * c.head_dim);
        v.resize(T * c.n_kv_heads * c.head_dim);
        attn.resize(T * c.hidden);
        logits.resize(c.vocab);
        scores.resize(c.max_seq);
    }

    void ensure_T(size_t T) {
        const Config& c = model.cfg;
        size_t need = T * (size_t)c.hidden;
        if (x.size() < need) {
            x.assign(need, 0.f);
            xb.assign(need, 0.f);
            q.assign(T * c.n_heads * c.head_dim, 0.f);
            k.assign(T * c.n_kv_heads * c.head_dim, 0.f);
            v.assign(T * c.n_kv_heads * c.head_dim, 0.f);
            attn.assign(need, 0.f);
            projout.assign(need, 0.f);
        }
    }

    // Dispatch a projection to the F32 or INT8-weight path based on how the
    // tensor is stored in the model file.
    void proj(float* out, const float* in, const std::string& wname, const float* bias, int M,
              int K, int N) {
        if (model.is_int8(wname))
            matmul_w8(out, in, model.W8(wname), model.Wscale(wname), bias, M, K, N);
        else
            matmul(out, in, model.W(wname), bias, M, K, N);
    }

    // Forward a batch of tokens (prefill: T>1, decode: T=1). Appends to KV
    // cache. Returns logits for the last position; if all_logits is given it
    // receives (T × vocab) logits for every position (used by --ppl).
    const std::vector<float>& forward(const std::vector<int>& tokens,
                                      std::vector<float>* all_logits = nullptr) {
        const Config& c = model.cfg;
        int T = (int)tokens.size();
        if (kv.len + T > c.max_seq) die("context overflow (max_seq)");
        ensure_T(T);

        const float* embed = model.W("embed");
        for (int t = 0; t < T; t++) {
            if (tokens[t] < 0 || tokens[t] >= c.vocab) die("token id out of range");
            std::memcpy(&x[(size_t)t * c.hidden], embed + (size_t)tokens[t] * c.hidden,
                        c.hidden * sizeof(float));
        }

        int head_dim = c.head_dim;
        int nq = c.n_heads, nkv = c.n_kv_heads, ratio = nq / nkv;
        float scale = 1.0f / std::sqrt((float)head_dim);

        for (int l = 0; l < c.n_layers; l++) {
            std::string lp = "l" + std::to_string(l) + ".";
            // ---- attention block ----
            rmsnorm(xb.data(), x.data(), model.W(lp + "ln1.w"), T, c.hidden, c.rms_eps);
            proj(q.data(), xb.data(), lp + "wq.w",
                 model.has(lp + "wq.b") ? model.W(lp + "wq.b") : nullptr, T, c.hidden,
                 nq * head_dim);
            proj(k.data(), xb.data(), lp + "wk.w",
                 model.has(lp + "wk.b") ? model.W(lp + "wk.b") : nullptr, T, c.hidden,
                 nkv * head_dim);
            proj(v.data(), xb.data(), lp + "wv.w",
                 model.has(lp + "wv.b") ? model.W(lp + "wv.b") : nullptr, T, c.hidden,
                 nkv * head_dim);
            rope(q.data(), kv.len, T, nq, head_dim, c.rope_theta);
            rope(k.data(), kv.len, T, nkv, head_dim, c.rope_theta);
            for (int t = 0; t < T; t++) {
                std::memcpy(kv.K(l, kv.len + t), &k[(size_t)t * nkv * head_dim],
                            (size_t)nkv * head_dim * sizeof(float));
                std::memcpy(kv.V(l, kv.len + t), &v[(size_t)t * nkv * head_dim],
                            (size_t)nkv * head_dim * sizeof(float));
            }
            // attention: causal during prefill, full history during decode
            for (int t = 0; t < T; t++) {
                for (int h = 0; h < nq; h++) {
                    int kvh = h / ratio;
                    const float* qh = &q[((size_t)t * nq + h) * head_dim];
                    int upto = kv.len + t;  // attend to [0, upto]
                    float maxs = -1e30f;
                    for (int j = 0; j <= upto; j++) {
                        const float* kj = kv.K(l, j) + (size_t)kvh * head_dim;
                        float s = 0.f;
                        for (int i = 0; i < head_dim; i++) s += qh[i] * kj[i];
                        s *= scale;
                        scores[j] = s;
                        if (s > maxs) maxs = s;
                    }
                    float sum = 0.f;
                    for (int j = 0; j <= upto; j++) {
                        scores[j] = std::exp(scores[j] - maxs);
                        sum += scores[j];
                    }
                    float inv = 1.0f / sum;
                    float* oh = &attn[((size_t)t * nq + h) * head_dim];
                    std::fill(oh, oh + head_dim, 0.f);
                    for (int j = 0; j <= upto; j++) {
                        const float* vj = kv.V(l, j) + (size_t)kvh * head_dim;
                        float p = scores[j] * inv;
                        for (int i = 0; i < head_dim; i++) oh[i] += p * vj[i];
                    }
                }
            }
            proj(projout.data(), attn.data(), lp + "wo.w", nullptr, T, c.hidden, c.hidden);
            for (size_t i = 0; i < (size_t)T * c.hidden; i++) x[i] += projout[i];

            // ---- MLP block (SwiGLU) ----
            rmsnorm(xb.data(), x.data(), model.W(lp + "ln2.w"), T, c.hidden, c.rms_eps);
            g.resize((size_t)T * c.ffn);
            u.resize((size_t)T * c.ffn);
            d.resize((size_t)T * c.ffn);
            proj(g.data(), xb.data(), lp + "wg.w", nullptr, T, c.hidden, c.ffn);
            proj(u.data(), xb.data(), lp + "wu.w", nullptr, T, c.hidden, c.ffn);
            for (size_t i = 0; i < (size_t)T * c.ffn; i++) {
                float s = g[i];
                g[i] = s / (1.0f + std::exp(-s)) * u[i];  // silu(gate) * up
            }
            proj(d.data(), g.data(), lp + "wd.w", nullptr, T, c.ffn, c.hidden);
            for (size_t i = 0; i < (size_t)T * c.hidden; i++) x[i] += d[i];
        }

        kv.len += T;

        // final norm + lm_head. Normally only the last position's logits are
        // needed; --ppl requests every position (T × vocab).
        const std::string head_name = model.has("lm_head") ? "lm_head" : "embed";
        std::vector<float> last(c.hidden);
        if (all_logits) {
            all_logits->assign((size_t)T * c.vocab, 0.f);
            // one batched lm_head over all T rows: reads weights once per chunk
            rmsnorm(xb.data(), x.data(), model.W("norm.w"), T, c.hidden, c.rms_eps);
            proj(all_logits->data(), xb.data(), head_name, nullptr, T, c.hidden, c.vocab);
            std::memcpy(logits.data(), &(*all_logits)[(size_t)(T - 1) * c.vocab],
                        c.vocab * sizeof(float));
            return logits;
        }
        std::memcpy(last.data(), &x[(size_t)(T - 1) * c.hidden], c.hidden * sizeof(float));
        rmsnorm_inplace(last.data(), model.W("norm.w"), c.hidden, c.rms_eps);
        proj(logits.data(), last.data(), head_name, nullptr, 1, c.hidden, c.vocab);
        return logits;
    }

    // ---- sampling ----
    int sample(float temp, float top_p, std::mt19937& rng) const {
        const std::vector<float>& lg = logits;
        if (temp <= 0.f) {
            int best = 0;
            for (int i = 1; i < (int)lg.size(); i++)
                if (lg[i] > lg[best]) best = i;
            return best;
        }
        std::vector<float> probs(lg.size());
        float mx = -1e30f;
        for (size_t i = 0; i < lg.size(); i++) mx = std::max(mx, lg[i] / temp);
        float sum = 0.f;
        for (size_t i = 0; i < lg.size(); i++) {
            probs[i] = std::exp(lg[i] / temp - mx);
            sum += probs[i];
        }
        for (auto& p : probs) p /= sum;
        std::vector<int> idx(probs.size());
        for (size_t i = 0; i < idx.size(); i++) idx[i] = (int)i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] > probs[b]; });
        float cum = 0.f;
        size_t keep = 0;
        for (; keep < idx.size(); keep++) {
            cum += probs[idx[keep]];
            if (cum >= top_p) { keep++; break; }
        }
        float r = std::uniform_real_distribution<float>(0.f, 1.f)(rng);
        float acc = 0.f;
        for (size_t i = 0; i < keep; i++) {
            acc += probs[idx[i]];
            if (r <= acc) return idx[i];
        }
        return idx[0];
    }
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static std::string chat_template_user(const std::string& user) {
    std::ostringstream ss;
    ss << "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
       << "<|im_start|>user\n" << user << "<|im_end|>\n<|im_start|>assistant\n";
    return ss.str();
}

int main(int argc, char** argv) {
    std::string model_path, prompt, forward_ids, ppl_file;
    float temp = 0.7f, top_p = 0.9f;
    int max_new = 256, seed = 42;
    bool raw = false, bench = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) die("missing value for " + a);
            return argv[++i];
        };
        if (a == "-m") model_path = next();
        else if (a == "-p") prompt = next();
        else if (a == "-n") max_new = std::stoi(next());
        else if (a == "--temp") temp = std::stof(next());
        else if (a == "--top-p") top_p = std::stof(next());
        else if (a == "--seed") seed = std::stoi(next());
        else if (a == "--raw") raw = true;
        else if (a == "--bench") bench = true;
        else if (a == "--forward") forward_ids = next();
        else if (a == "--ppl") ppl_file = next();
        else die("unknown flag " + a);
    }
    if (model_path.empty()) die("usage: nanollm -m model.nnlm [-p prompt] [--raw] [--bench] [--forward ids]");

    Runtime rt;
    rt.load(model_path);
    const Config& c = rt.model.cfg;
    std::mt19937 rng(seed);
    std::fprintf(stderr, "[nanollm] %dL H=%d heads=%d/%d hd=%d ffn=%d vocab=%d max_seq=%d\n",
                 c.n_layers, c.hidden, c.n_heads, c.n_kv_heads, c.head_dim, c.ffn, c.vocab,
                 c.max_seq);

    // --forward: prefill explicit token ids, dump last-position logits for
    // golden testing against a HuggingFace reference.
    if (!forward_ids.empty()) {
        std::vector<int> ids;
        std::stringstream ss(forward_ids);
        std::string item;
        while (std::getline(ss, item, ',')) ids.push_back(std::stoi(item));
        rt.forward(ids);
        for (float v : rt.logits) std::printf("%.6f\n", v);
        return 0;
    }

    // --ppl: perplexity of a token sequence (comma-separated ids in a file).
    // Chunks of 64 tokens share one running KV cache; every position's
    // next-token log-probability contributes. ppl = exp(-Σ nll / count).
    if (!ppl_file.empty()) {
        std::ifstream pf(ppl_file);
        if (!pf) die("cannot open " + ppl_file);
        std::vector<int> ids;
        std::string item;
        while (std::getline(pf, item, ',')) ids.push_back(std::stoi(item));
        if (ids.size() < 2) die("need at least 2 tokens for ppl");
        const int V = c.vocab;
        double nll = 0.0;
        size_t count = 0;
        std::vector<float> all;
        const int CHUNK = 64;
        for (size_t s = 0; s < ids.size(); s += CHUNK) {
            size_t len = std::min((size_t)CHUNK, ids.size() - s);
            std::vector<int> chunk(ids.begin() + s, ids.begin() + s + len);
            rt.forward(chunk, &all);
            for (size_t t = 0; t < len; t++) {
                size_t next_idx = s + t + 1;
                if (next_idx >= ids.size()) break;
                const float* row = &all[t * (size_t)V];
                float mx = -1e30f;
                for (int i = 0; i < V; i++) mx = std::max(mx, row[i]);
                double lse = 0.0;
                for (int i = 0; i < V; i++) lse += std::exp(row[i] - mx);
                lse = std::log(lse) + mx;
                nll += lse - row[ids[next_idx]];
                count++;
            }
        }
        std::printf("tokens,%zu,nll_mean,%.6f,ppl,%.4f\n", count, nll / (double)count,
                    std::exp(nll / (double)count));
        return 0;
    }

    // --bench: synthetic tokens, report prefill and decode throughput.
    if (bench) {
        std::vector<int> ids;
        for (int i = 0; i < 128; i++) ids.push_back(1000 + (i * 7) % 40000);
        auto t0 = std::chrono::steady_clock::now();
        rt.forward(ids);
        auto t1 = std::chrono::steady_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        int n_decode = 32;
        auto t2 = std::chrono::steady_clock::now();
        int next = rt.sample(0.0f, 1.0f, rng);
        for (int i = 0; i < n_decode; i++) {
            rt.forward({next});
            next = rt.sample(0.0f, 1.0f, rng);
        }
        auto t3 = std::chrono::steady_clock::now();
        double decode_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        std::printf("bench,prefill_tokens,%.3f,prefill_ms,%.1f,decode_tokens,%d,decode_ms,%.1f,"
                    "decode_tokens_per_s,%.2f\n",
                    prefill_ms / 128.0, prefill_ms, n_decode, decode_ms,
                    n_decode / (decode_ms / 1000.0));
        return 0;
    }

    auto generate = [&](const std::vector<int>& prompt_ids) {
        std::vector<int> gen;
        if (!prompt_ids.empty()) rt.forward(prompt_ids);
        int next = rt.sample(temp, top_p, rng);
        for (int i = 0; i < max_new; i++) {
            if (next == rt.tok.im_end || next == rt.tok.eos) break;
            gen.push_back(next);
            std::fputs(rt.tok.decode({next}).c_str(), stdout);
            std::fflush(stdout);
            rt.forward({next});
            next = rt.sample(temp, top_p, rng);
        }
        std::fputs("\n", stdout);
        return gen;
    };

    if (!prompt.empty()) {
        std::string text = raw ? prompt : chat_template_user(prompt);
        generate(rt.tok.encode(text));
        return 0;
    }

    // interactive multi-turn chat; the KV cache persists across turns
    std::fprintf(stderr, "[nanollm] interactive chat — empty line to quit\n");
    std::string line;
    while (true) {
        std::fprintf(stderr, "user> ");
        if (!std::getline(std::cin, line) || line.empty()) break;
        std::vector<int> ids = rt.tok.encode("<|im_start|>user\n" + line + "<|im_end|>\n<|im_start|>assistant\n");
        if (rt.kv.len == 0) {
            std::vector<int> sys = rt.tok.encode(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n");
            ids.insert(ids.begin(), sys.begin(), sys.end());
        }
        std::fprintf(stderr, "assistant> ");
        generate(ids);
        // cache the turn terminator so the next prefill continues cleanly
        rt.forward(rt.tok.encode("<|im_end|>\n"));
    }
    return 0;
}
