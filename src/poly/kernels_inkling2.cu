// src/poly/kernels_inkling2.cu - Inkling 2 (inkling2_loop / inkling2) CUDA kernels.
//
// Reference: mainline llama.cpp src/models/inkling2.cpp and src/models/inkling2-loop.cpp.
#include "strata/poly/kernels.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

constexpr int WARP = 32;
constexpr int CONV_ROWS = 16;
constexpr int SPLIT_KEYS = 512;
constexpr int MAX_SPLITS = 64;
constexpr size_t kAttendResults = size_t(8) << 20;

cudaStream_t cs(void* s) { return (cudaStream_t) s; }
int grid(size_t n, int b) { return (int) ((n + (size_t) b - 1) / (size_t) b); }

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

void blas_ck(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": cuBLAS error " + std::to_string((int) st));
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o, WARP));
    return x;
}

__device__ __forceinline__ float block_sum(float x, float* part) {
    x = warp_sum(x);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float s = 0.0f;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) s += part[k];
    __syncthreads();
    return s;
}

__device__ __forceinline__ float block_max(float x, float* part) {
    x = warp_max(x);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float m = part[0];
    for (int k = 1; k < int(blockDim.x) / WARP; ++k) m = fmaxf(m, part[k]);
    __syncthreads();
    return m;
}

__device__ __forceinline__ float sigmoidf_d(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float2 row_scales(const ExpertScales& s, int row) {
    if (!s.gate && !s.up) return make_float2(1.0f, 1.0f);
    const int e = s.ids ? s.ids[row] : s.e[row];
    return make_float2(s.gate ? s.gate[e] : 1.0f, s.up ? s.up[e] : 1.0f);
}

// ---- 1. SiTU activation ----

__global__ void situ_rows_kernel(const float* __restrict__ gate_up, float* __restrict__ out, int n, float beta,
                                 float linear_beta, ExpertScales s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = gate_up + size_t(blockIdx.y) * 2 * n;
    const float2 sc = row_scales(s, blockIdx.y);
    const float g = row[i] * sc.x;
    const float u = row[n + i] * sc.y;
    const float act = beta * tanhf(g / beta) * sigmoidf_d(g);
    const float upv = linear_beta > 0.0f ? linear_beta * tanhf(u / linear_beta) : u;
    out[size_t(blockIdx.y) * n + i] = act * upv;
}

// ---- 2. Block Attention Residual stream mixing ----

__global__ void inkling_mix_kernel(const float* __restrict__ r, int streams, MixSlots slots, int n_slots,
                                   const float* __restrict__ query, int n_queries, float* __restrict__ out,
                                   int n_embd, float eps) {
    __shared__ float part[8];
    __shared__ float inv_rms[6];
    __shared__ float probs[4][6];

    const int t = blockIdx.x;
    const float* tok_r = r + size_t(t) * streams * n_embd;
    const float score_scale = rsqrtf((float) n_embd);

    float logits[4][6];
    for (int q = 0; q < n_queries; ++q)
        for (int k = 0; k < n_slots; ++k) logits[q][k] = 0.0f;

    for (int k = 0; k < n_slots; ++k) {
        const float* src = tok_r + size_t(slots.s[k]) * n_embd;
        float sq = 0.0f;
        float dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
            const float v = src[i];
            sq += v * v;
            for (int q = 0; q < n_queries; ++q) dot[q] += v * query[size_t(q) * n_embd + i];
        }
        const float mean_sq = block_sum(sq, part) / (float) n_embd;
        const float inv = rsqrtf(mean_sq + eps);
        if (threadIdx.x == 0) inv_rms[k] = inv;
        for (int q = 0; q < n_queries; ++q) {
            const float d = block_sum(dot[q], part);
            if (threadIdx.x == 0) logits[q][k] = d * inv * score_scale;
        }
    }
    if (threadIdx.x == 0) {
        for (int q = 0; q < n_queries; ++q) {
            float mx = logits[q][0];
            for (int k = 1; k < n_slots; ++k) mx = fmaxf(mx, logits[q][k]);
            float sum = 0.0f;
            float exps[6];
            for (int k = 0; k < n_slots; ++k) {
                exps[k] = expf(logits[q][k] - mx);
                sum += exps[k];
            }
            const float inv_s = 1.0f / sum;
            for (int k = 0; k < n_slots; ++k) probs[q][k] = exps[k] * inv_s;
        }
    }
    __syncthreads();

    const int rows = gridDim.x;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int k = 0; k < n_slots; ++k) {
            const float nv = tok_r[size_t(slots.s[k]) * n_embd + i] * inv_rms[k];
            for (int q = 0; q < n_queries; ++q) acc[q] += probs[q][k] * nv;
        }
        for (int q = 0; q < n_queries; ++q) out[(size_t(q) * rows + t) * n_embd + i] = acc[q];
    }
}

// ---- 3. Cross-pass loop controllers ----

__global__ void inkling_control_prep_kernel(const float* __restrict__ cur, const float* __restrict__ msg,
                                            float* __restrict__ out_concat, int n_embd, float eps) {
    __shared__ float part[8];
    const int t = blockIdx.x;
    const float* c_row = cur + size_t(t) * n_embd;
    const float* m_row = msg + size_t(t) * n_embd;
    float* dst = out_concat + size_t(t) * 2 * n_embd;

    float sq_c = 0.0f, sq_m = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        const float cv = c_row[i], mv = m_row[i];
        sq_c += cv * cv;
        sq_m += mv * mv;
    }
    const float inv_c = rsqrtf(block_sum(sq_c, part) / (float) n_embd + eps);
    const float inv_m = rsqrtf(block_sum(sq_m, part) / (float) n_embd + eps);
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        dst[i] = c_row[i] * inv_c;
        dst[n_embd + i] = m_row[i] * inv_m;
    }
}

__global__ void inkling_control_apply_kernel(const float* __restrict__ cur, const float* __restrict__ msg,
                                             const float* __restrict__ gain_raw, const float* __restrict__ forget_raw,
                                             float* __restrict__ out, int n_embd, int gf_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    const size_t idx = size_t(blockIdx.y) * n_embd + i;
    const size_t gidx = size_t(blockIdx.y) * gf_stride + i;
    const float c = cur[idx], m = msg[idx];
    out[idx] = c * (1.0f - tanhf(forget_raw[gidx])) + m * tanhf(gain_raw[gidx]);
}

// ---- 4. Causal 4-tap short convolution (static + optional dynamic per-head reversed-lag taps) ----

__device__ __forceinline__ float* snap_of(float* snap, size_t stride, unsigned snaps, int row) {
    return row < 32 && (snaps >> row & 1u) ? snap + size_t(__popc(snaps & ((1u << row) - 1u))) * stride : nullptr;
}

__global__ void inkling_conv4_walk_kernel(const float* __restrict__ x, float* __restrict__ state,
                                          const float* __restrict__ w, const float* __restrict__ dyn, int head_dim,
                                          int n_heads, float* __restrict__ out, int channels, int rows,
                                          float* __restrict__ snap, size_t stride, unsigned snaps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float* wc = w + 4 * size_t(c);
    float* st = state + size_t(c) * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    const int h = dyn ? c / head_dim : 0;
    const float scale = dyn ? (1.0f / 3.0f) : rsqrtf(5.0f);
    for (int row = 0; row < rows; ++row) {
        const size_t at = size_t(row) * channels + c;
        const float v = x[at];
        float y = s0 * wc[0] + s1 * wc[1] + s2 * wc[2] + v * wc[3];
        if (dyn) {
            const float* d = dyn + (size_t(row) * n_heads + h) * 4;
            y += d[0] * v + d[1] * s2 + d[2] * s1 + d[3] * s0;
        }
        s0 = s1;
        s1 = s2;
        s2 = v;
        out[at] = (v + y) * scale;
        if (float* m = snap_of(snap, stride, snaps, row)) {
            m += size_t(c) * 3;
            m[0] = s0;
            m[1] = s1;
            m[2] = s2;
        }
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
}

__global__ void inkling_conv4_rows_kernel(const float* __restrict__ x, const float* __restrict__ state,
                                          const float* __restrict__ w, const float* __restrict__ dyn, int head_dim,
                                          int n_heads, float* __restrict__ out, int channels) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int row = blockIdx.y;
    const float* wc = w + 4 * size_t(c);
    const float* st = state + size_t(c) * 3;
    auto in = [&](int r) { return r >= 0 ? x[size_t(r) * channels + c] : st[3 + r]; };
    const float s0 = in(row - 3), s1 = in(row - 2), s2 = in(row - 1), v = x[size_t(row) * channels + c];
    float y = s0 * wc[0] + s1 * wc[1] + s2 * wc[2] + v * wc[3];
    if (dyn) {
        const float* d = dyn + (size_t(row) * n_heads + c / head_dim) * 4;
        y += d[0] * v + d[1] * s2 + d[2] * s1 + d[3] * s0;
    }
    out[size_t(row) * channels + c] = (v + y) * (dyn ? (1.0f / 3.0f) : rsqrtf(5.0f));
}

__global__ void inkling_conv4_advance_kernel(const float* __restrict__ x, float* __restrict__ state, int channels,
                                             int rows) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float* st = state + size_t(c) * 3;
    st[0] = x[size_t(rows - 3) * channels + c];
    st[1] = x[size_t(rows - 2) * channels + c];
    st[2] = x[size_t(rows - 1) * channels + c];
}

// ---- 5. L2 norm clip + optional global log-scale tau(pos) ----

__global__ void inkling_l2_clip_kernel(const float* __restrict__ x, const float* __restrict__ scale,
                                       float* __restrict__ out, int dim, int heads, float threshold, long long pos,
                                       bool is_global, float log_floor, float log_alpha, const long long* at) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const size_t base = (size_t(t) * heads + h) * dim;
    float sq = 0.0f;
    for (int d = threadIdx.x; d < dim; d += WARP) {
        float v = x[base + d];
        if (scale) v *= scale[size_t(h) * dim + d];
        sq += v * v;
    }
    const float norm = sqrtf(warp_sum(sq)) + 1e-6f;
    float factor = fminf(1.0f, fmaxf(0.0f, threshold / norm));
    if (is_global) {
        const long long p = at ? at[2 * t] : pos + t;
        const float ratio = float(p + 1) / log_floor;
        if (ratio > 1.0f) factor *= (1.0f + log_alpha * logf(ratio));
    }
    for (int d = threadIdx.x; d < dim; d += WARP) {
        float v = x[base + d];
        if (scale) v *= scale[size_t(h) * dim + d];
        out[base + d] = v * factor;
    }
}

// ---- 6. Per-query relative-position bias table ----

__global__ void inkling_rel_table_kernel(const float* __restrict__ r_cur, const float* __restrict__ rel_bias,
                                         const float* __restrict__ sinks, float* __restrict__ table, int n_rel,
                                         int n_extent, int n_head) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const float* rq = r_cur + (size_t(t) * n_head + h) * n_rel;
    const float sink = sinks[size_t(t) * n_head + h];
    float* dst = table + (size_t(t) * n_head + h) * n_extent;

    float rreg[16];
    if (n_rel <= 16) {
        for (int k = 0; k < n_rel; ++k) rreg[k] = rq[k];
    }
    for (int d = threadIdx.x; d < n_extent; d += blockDim.x) {
        const float* rb = rel_bias + size_t(d) * n_rel;
        float dot = 0.0f;
        if (n_rel <= 16) {
            for (int k = 0; k < n_rel; ++k) dot += rreg[k] * rb[k];
        } else {
            for (int k = 0; k < n_rel; ++k) dot += rq[k] * rb[k];
        }
        dst[d] = dot - sink;
    }
}

// ---- 7. Inkling 2 attention (decode split-K + batch cuBLAS prefill) ----

__device__ __forceinline__ int attn_rows_span(long long p, int window, long long* lo) {
    *lo = window > 0 && p >= window ? p - window + 1 : 0;
    return int(p - *lo) + 1;
}

__device__ __forceinline__ float read_kv_val(const uint16_t* cache, Kvq8 q8, Kvq4 q4, size_t elem_idx,
                                             size_t scale_idx) {
    if (q8.codes) return float(q8.codes[elem_idx]) * __half2float(__ushort_as_half(q8.scales[scale_idx]));
    if (q4.codes) {
        const uint8_t packed = q4.codes[(scale_idx << 4) + (elem_idx & 15)];
        const int q = (elem_idx & 16) ? (packed >> 4) : (packed & 0x0f);
        return float(q - 8) * __half2float(__ushort_as_half(q4.scales[scale_idx]));
    }
    return __half2float(__ushort_as_half(cache[elem_idx]));
}

__device__ __forceinline__ void quant_q4_block(const float* xr, uint8_t* cr, uint16_t* sr) {
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(xr[j]));
    const half dh = __float2half(amax / 7.0f);
    const float df = __half2float(dh);
    const float id = df > 0.0f ? 1.0f / df : 0.0f;
    *sr = __half_as_ushort(dh);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float lo_f = fminf(fmaxf(rintf(xr[j] * id), -8.0f), 7.0f);
        const float hi_f = fminf(fmaxf(rintf(xr[j + 16] * id), -8.0f), 7.0f);
        const unsigned lo = unsigned(int(lo_f) + 8) & 0x0fu;
        const unsigned hi = unsigned(int(hi_f) + 8) & 0x0fu;
        cr[j] = uint8_t(lo | (hi << 4));
    }
}

__global__ void store_q4_kernel(const float* __restrict__ x, uint8_t* __restrict__ codes,
                                uint16_t* __restrict__ scales, int n, int rows, int nb) {
    const int gb = blockIdx.x * blockDim.x + threadIdx.x;
    if (gb >= nb * rows) return;
    const int row = gb / nb, blk = gb % nb;
    quant_q4_block(x + (size_t) row * n + blk * 32, codes + (size_t) row * (n >> 1) + blk * 16,
                   scales + (size_t) row * nb + blk);
}

__global__ void store_q4_at_kernel(const float* __restrict__ x, uint8_t* __restrict__ codes,
                                   uint16_t* __restrict__ scales, int n, int rows, int nb,
                                   const long long* __restrict__ at, int ring) {
    const int gb = blockIdx.x * blockDim.x + threadIdx.x;
    if (gb >= nb * rows) return;
    const int trow = gb / nb, blk = gb % nb;
    const long long row = ring > 0 ? (*at + trow) % ring : *at + trow;
    quant_q4_block(x + (size_t) trow * n + blk * 32, codes + (size_t) row * (n >> 1) + blk * 16,
                   scales + (size_t) row * nb + blk);
}

template <bool WHOLE = false>
__global__ void inkling_attend_split_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kcache,
                                            const uint16_t* __restrict__ vcache, const float* __restrict__ table,
                                            const float* __restrict__ sinks, int n_extent, long long pos,
                                            float* __restrict__ partials, int n_splits, int dim, int n_head,
                                            int kv_heads, float scale, int window, int ring,
                                            const long long* __restrict__ at, Kvq8 kq8, Kvq8 vq8,
                                            Kvq4 kq4, Kvq4 vq4) {
    const int s = blockIdx.x;
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    const long long q_pos = at ? at[2 * t] : pos + t;
    const long long p0 = at ? at[0] : pos;

    long long lo;
    const int total = WHOLE ? [&]() {
        lo = window > 0 && q_pos >= window ? q_pos - window + 1 : 0;
        return int((p0 + gridDim.z - 1) - lo) + 1;
    }() : attn_rows_span(q_pos, window, &lo);
    const int per_split = max(SPLIT_KEYS, (total + n_splits - 1) / n_splits);
    const int k_begin = s * per_split;
    const int pstride = dim + 2;
    float* dst = partials + (((size_t(t) * n_head + h) * n_splits + s) * pstride);

    if (k_begin >= total) {
        if (threadIdx.x == 0) {
            dst[dim] = 0.0f;
            dst[dim + 1] = 0.0f;
        }
        return;
    }
    const int k_end = min(total, k_begin + per_split);
    const int per = n_head / kv_heads;
    const int kv_h = h / per;
    const int kv_width = kv_heads * dim;
    const int kv_swidth = kv_heads * (dim >> 5);
    const float* q_ptr = q + (size_t(t) * n_head + h) * dim;
    const float* tbl = WHOLE ? nullptr : table + (size_t(t) * n_head + h) * n_extent;
    const float neg_sink = WHOLE ? 0.0f : -sinks[size_t(t) * n_head + h];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float q_reg[4];
    float o_reg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int d = u * 32 + lane;
        q_reg[u] = d < dim ? q_ptr[d] * scale : 0.0f;
    }

    float m_w = -INFINITY;
    float l_w = 0.0f;
    for (int off = k_begin + warp; off < k_end; off += 4) {
        const long long j = lo + off;
        const size_t row = size_t(ring > 0 ? j % ring : j);
        const size_t k_off = row * kv_width + size_t(kv_h) * dim;
        const size_t s_off = row * kv_swidth + size_t(kv_h) * (dim >> 5);

        float dot = 0.0f;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int d = u * 32 + lane;
            if (d < dim) {
                const float kv = read_kv_val(kcache, kq8, kq4, k_off + d, s_off + (d >> 5));
                dot += q_reg[u] * kv;
            }
        }
        dot = warp_sum(dot);
        const long long dist = q_pos - j;
        const float bias = WHOLE ? 0.0f : (dist < n_extent ? tbl[dist] : neg_sink);
        const float score = dot + bias;

        const float m_new = fmaxf(m_w, score);
        const float alpha = l_w > 0.0f ? expf(m_w - m_new) : 0.0f;
        const float w_j = expf(score - m_new);
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int d = u * 32 + lane;
            if (d < dim) {
                const float vv = read_kv_val(vcache, vq8, vq4, k_off + d, s_off + (d >> 5));
                o_reg[u] = o_reg[u] * alpha + w_j * vv;
            }
        }
        l_w = l_w * alpha + w_j;
        m_w = m_new;
    }

    __shared__ float sm_m[4];
    __shared__ float sm_l[4];
    __shared__ float sm_o[4][128];
    if (lane == 0) {
        sm_m[warp] = m_w;
        sm_l[warp] = l_w;
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int d = u * 32 + lane;
        if (d < dim) sm_o[warp][d] = o_reg[u];
    }
    __syncthreads();

    if (threadIdx.x < dim) {
        const int d = threadIdx.x;
        float mb = -INFINITY, lb = 0.0f, ob = 0.0f;
        for (int w = 0; w < 4; ++w) {
            if (sm_l[w] > 0.0f) {
                const float mn = fmaxf(mb, sm_m[w]);
                const float a = lb > 0.0f ? expf(mb - mn) : 0.0f;
                const float b = expf(sm_m[w] - mn);
                lb = lb * a + sm_l[w] * b;
                ob = ob * a + sm_o[w][d] * b;
                mb = mn;
            }
        }
        dst[d] = ob;
        if (d == 0) {
            dst[dim] = mb;
            dst[dim + 1] = lb;
        }
    }
}

template <int PER, bool WHOLE = false>
__global__ void inkling_attend_split_gqa_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kcache,
                                                const uint16_t* __restrict__ vcache, const float* __restrict__ table,
                                                const float* __restrict__ sinks, int n_extent, long long pos,
                                                float* __restrict__ partials, int n_splits, int dim, int n_head,
                                                int kv_heads, float scale, int window, int ring,
                                                const long long* __restrict__ at, Kvq8 kq8, Kvq8 vq8,
                                                Kvq4 kq4, Kvq4 vq4) {
    const int s = blockIdx.x;
    const int kv_h = blockIdx.y;
    const int t = blockIdx.z;
    const int h_base = kv_h * PER;
    const long long q_pos = at ? at[2 * t] : pos + t;
    const long long p0 = at ? at[0] : pos;

    long long lo;
    const int total = WHOLE ? [&]() {
        lo = window > 0 && q_pos >= window ? q_pos - window + 1 : 0;
        return int((p0 + gridDim.z - 1) - lo) + 1;
    }() : attn_rows_span(q_pos, window, &lo);
    const int per_split = max(SPLIT_KEYS, (total + n_splits - 1) / n_splits);
    const int k_begin = s * per_split;
    const int pstride = dim + 2;

    if (k_begin >= total) {
        if (threadIdx.x == 0) {
            for (int p = 0; p < PER; ++p) {
                float* dst = partials + (((size_t(t) * n_head + (h_base + p)) * n_splits + s) * pstride);
                dst[dim] = 0.0f;
                dst[dim + 1] = 0.0f;
            }
        }
        return;
    }

    const int k_end = min(total, k_begin + per_split);
    const int kv_width = kv_heads * dim;
    const int kv_swidth = kv_heads * (dim >> 5);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    float q_reg[PER][4];
    float o_reg[PER][4];
    float m_w[PER];
    float l_w[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p) {
        const float* q_ptr = q + (size_t(t) * n_head + (h_base + p)) * dim;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int d = u * 32 + lane;
            q_reg[p][u] = q_ptr[d] * scale;
            o_reg[p][u] = 0.0f;
        }
        m_w[p] = -INFINITY;
        l_w[p] = 0.0f;
    }

    for (int off = k_begin + warp; off < k_end; off += 4) {
        const long long j = lo + off;
        const size_t row = size_t(ring > 0 ? j % ring : j);
        const size_t k_off = row * kv_width + size_t(kv_h) * dim;
        const size_t s_off = row * kv_swidth + size_t(kv_h) * (dim >> 5);

        float dot[PER];
#pragma unroll
        for (int p = 0; p < PER; ++p) dot[p] = 0.0f;

#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int d = u * 32 + lane;
            const float kv = read_kv_val(kcache, kq8, kq4, k_off + d, s_off + u);
#pragma unroll
            for (int p = 0; p < PER; ++p) dot[p] += q_reg[p][u] * kv;
        }

        const long long dist = q_pos - j;
        float w_j[PER];
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int h = h_base + p;
            const float dp = warp_sum(dot[p]);
            const float bias = WHOLE ? 0.0f
                                     : (dist < n_extent ? table[(size_t(t) * n_head + h) * n_extent + dist]
                                                        : -sinks[size_t(t) * n_head + h]);
            const float score = dp + bias;
            const float m_new = fmaxf(m_w[p], score);
            const float alpha = l_w[p] > 0.0f ? expf(m_w[p] - m_new) : 0.0f;
            const float wv = expf(score - m_new);
            w_j[p] = wv;
#pragma unroll
            for (int u = 0; u < 4; ++u) o_reg[p][u] *= alpha;
            l_w[p] = l_w[p] * alpha + wv;
            m_w[p] = m_new;
        }

#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int d = u * 32 + lane;
            const float vv = read_kv_val(vcache, vq8, vq4, k_off + d, s_off + u);
#pragma unroll
            for (int p = 0; p < PER; ++p) o_reg[p][u] += w_j[p] * vv;
        }
    }

    __shared__ float sm_m[4];
    __shared__ float sm_l[4];
    __shared__ float sm_o[4][128];
    for (int p = 0; p < PER; ++p) {
        if (lane == 0) {
            sm_m[warp] = m_w[p];
            sm_l[warp] = l_w[p];
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            sm_o[warp][u * 32 + lane] = o_reg[p][u];
        }
        __syncthreads();

        if (threadIdx.x < dim) {
            const int d = threadIdx.x;
            float mb = -INFINITY, lb = 0.0f, ob = 0.0f;
            for (int w = 0; w < 4; ++w) {
                if (sm_l[w] > 0.0f) {
                    const float mn = fmaxf(mb, sm_m[w]);
                    const float a = lb > 0.0f ? expf(mb - mn) : 0.0f;
                    const float b = expf(sm_m[w] - mn);
                    lb = lb * a + sm_l[w] * b;
                    ob = ob * a + sm_o[w][d] * b;
                    mb = mn;
                }
            }
            float* dst = partials + (((size_t(t) * n_head + (h_base + p)) * n_splits + s) * pstride);
            dst[d] = ob;
            if (d == 0) {
                dst[dim] = mb;
                dst[dim + 1] = lb;
            }
        }
        if (p + 1 < PER) __syncthreads();
    }
}

template <bool ZERO_SINK = true, bool WHOLE = false>
__global__ void inkling_attend_join_kernel(const float* __restrict__ partials, float* __restrict__ out, int n_splits,
                                           int dim, int n_head, long long pos, int window,
                                           const long long* __restrict__ at) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= dim) return;

    const long long q_pos = at ? at[2 * t] : pos + t;
    const long long p0 = at ? at[0] : pos;
    long long lo;
    const int total = WHOLE ? [&]() {
        lo = window > 0 && q_pos >= window ? q_pos - window + 1 : 0;
        return int((p0 + gridDim.y - 1) - lo) + 1;
    }() : attn_rows_span(q_pos, window, &lo);
    const int per_split = max(SPLIT_KEYS, (total + n_splits - 1) / n_splits);
    const int active = min(n_splits, (total + per_split - 1) / per_split);
    const int pstride = dim + 2;
    const float* base = partials + (size_t(t) * n_head + h) * n_splits * pstride;

    // ZERO_SINK=true: virtual zero sink (m = 0.0f, l = 1.0f, o = 0.0f). ZERO_SINK=false: standard softmax.
    float m = ZERO_SINK ? 0.0f : -INFINITY;
    float l = ZERO_SINK ? 1.0f : 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < active; ++s) {
        const float* p = base + size_t(s) * pstride;
        const float ls = p[dim + 1];
        if (ls <= 0.0f) continue;
        const float ms = p[dim];
        const float m_new = fmaxf(m, ms);
        const float alpha = l > 0.0f ? expf(m - m_new) : 0.0f;
        const float beta = expf(ms - m_new);
        l = l * alpha + ls * beta;
        acc = acc * alpha + p[d] * beta;
        m = m_new;
    }
    out[(size_t(t) * n_head + h) * dim + d] = l > 0.0f ? acc / l : 0.0f;
}

__global__ void dequant_q8_kernel(const int8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                                  half* __restrict__ out, int width, long long row) {
    const int i4 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i4 >= width) return;
    const size_t r = size_t(row + blockIdx.y), at = r * width;
    const float d = __half2float(__ushort_as_half(scales[r * (width >> 5) + (i4 >> 5)]));
    const char4 c = *reinterpret_cast<const char4*>(codes + at + i4);
    half* dst = out + size_t(blockIdx.y) * width + i4;
    *reinterpret_cast<half2*>(dst) = __floats2half2_rn(float(c.x) * d, float(c.y) * d);
    *reinterpret_cast<half2*>(dst + 2) = __floats2half2_rn(float(c.z) * d, float(c.w) * d);
}

__global__ void dequant_q4_kernel(const uint8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
                                  half* __restrict__ out, int width, long long row) {
    const int b2 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int half_w = width >> 1;
    if (b2 >= half_w) return;
    const size_t r = size_t(row + blockIdx.y);
    const int blk = b2 >> 4;
    const int j = b2 & 15;
    const uint16_t packed2 = *reinterpret_cast<const uint16_t*>(codes + r * half_w + b2);
    const int p0 = packed2 & 0xff;
    const int p1 = packed2 >> 8;
    const float d = __half2float(__ushort_as_half(scales[r * (width >> 5) + blk]));
    const size_t base = size_t(blockIdx.y) * width + size_t(blk) * 32 + j;
    half* dst = out + base;
    *reinterpret_cast<half2*>(dst) = __floats2half2_rn(float((p0 & 0x0f) - 8) * d, float((p1 & 0x0f) - 8) * d);
    *reinterpret_cast<half2*>(dst + 16) = __floats2half2_rn(float((p0 >> 4) - 8) * d, float((p1 >> 4) - 8) * d);
}

__global__ void attn_group_kernel(const float* __restrict__ src, half* __restrict__ dst, int dim, int n_head,
                                  int kv_heads, int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const int per = n_head / kv_heads;
    const int h = blockIdx.y % n_head, tok = blockIdx.y / n_head;
    const size_t g = (size_t(h / per) * tokens + tok) * per + h % per;
    dst[g * dim + i] = __float2half(src[size_t(blockIdx.y) * dim + i]);
}

__global__ void attn_ungroup_kernel(const float* __restrict__ src, float* __restrict__ dst, int dim, int n_head,
                                    int kv_heads, int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const int per = n_head / kv_heads;
    const int h = blockIdx.y % n_head, tok = blockIdx.y / n_head;
    const size_t g = (size_t(h / per) * tokens + tok) * per + h % per;
    dst[size_t(blockIdx.y) * dim + i] = src[g * dim + i];
}

__global__ void inkling_softmax_block_kernel(const float* __restrict__ score, half* __restrict__ weight, int n_keys,
                                             long long row0, long long pos, int first, int window, int tokens,
                                             int per, int n_head, const float* __restrict__ table,
                                             const float* __restrict__ sinks, int n_extent, float* top, float* sum,
                                             float* shrink, bool fresh) {
    __shared__ float part[8];
    const int b = blockIdx.x;
    const int p = b % per;
    const int tok_in_chunk = (b / per) % tokens;
    const int kv_g = b / (tokens * per);
    const int h = kv_g * per + p;
    const int tok = first + tok_in_chunk;
    const long long q_pos = pos + tok;

    long long lo;
    const int n = attn_rows_span(q_pos, window, &lo);
    const long long end = lo + n;
    const int j0 = lo > row0 ? int(lo - row0) : 0;
    const int j1 = end - row0 < n_keys ? int(end - row0) : n_keys;

    const float* row = score + size_t(b) * n_keys;
    half* w = weight + size_t(b) * n_keys;
    const float* tbl = table + (size_t(tok) * n_head + h) * n_extent;
    const float neg_sink = -sinks[size_t(tok) * n_head + h];

    float mx = -INFINITY;
    for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
        const long long dist = q_pos - (row0 + j);
        const float bias = dist < n_extent ? tbl[dist] : neg_sink;
        mx = fmaxf(mx, row[j] + bias);
    }
    mx = block_max(mx, part);
    // Start online softmax with the virtual zero sink (m = 0, l = 1) on fresh == true.
    const float before = fresh ? 0.0f : top[b];
    const float now = fmaxf(before, mx);
    const float f = expf(before - now);
    float s = 0.0f;
    for (int j = threadIdx.x; j < n_keys; j += blockDim.x) {
        float val = 0.0f;
        if (j >= j0 && j < j1) {
            const long long dist = q_pos - (row0 + j);
            const float bias = dist < n_extent ? tbl[dist] : neg_sink;
            val = expf(row[j] + bias - now);
        }
        w[j] = __float2half(val);
        s += __half2float(w[j]);
    }
    s = block_sum(s, part);
    if (threadIdx.x != 0) return;
    top[b] = now;
    sum[b] = (fresh ? 1.0f : sum[b]) * f + s;
    shrink[b] = f;
}

__global__ void attn_scale_kernel(float* out, const float* __restrict__ factor, int dim, bool divide) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const float f = factor[blockIdx.y];
    if (!divide && f == 1.0f) return;
    out[size_t(blockIdx.y) * dim + i] *= divide ? 1.0f / f : f;
}

// ---- 8. Stream helpers ----

__global__ void inkling_copy_stream_kernel(float* r, int streams, int dst_stream, int src_stream, int n_embd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float* tok = r + size_t(blockIdx.y) * streams * n_embd;
    tok[size_t(dst_stream) * n_embd + i] = tok[size_t(src_stream) * n_embd + i];
}

__global__ void inkling_write_stream_kernel(const float* __restrict__ x, float* r, int streams, int dst_stream,
                                            int n_embd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    r[(size_t(blockIdx.y) * streams + dst_stream) * n_embd + i] = x[size_t(blockIdx.y) * n_embd + i];
}

__global__ void inkling_read_stream_kernel(const float* __restrict__ r, int streams, int src_stream,
                                           float* __restrict__ out, int n_embd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    out[size_t(blockIdx.y) * n_embd + i] = r[(size_t(blockIdx.y) * streams + src_stream) * n_embd + i];
}

__global__ void inkling_add_to_stream_kernel(const float* __restrict__ a, const float* __restrict__ b, float* r,
                                             int streams, int dst_stream, int n_embd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    const size_t src_i = size_t(blockIdx.y) * n_embd + i;
    const size_t dst_i = (size_t(blockIdx.y) * streams + dst_stream) * n_embd + i;
    r[dst_i] = a ? (a[src_i] + b[src_i]) : (r[dst_i] + b[src_i]);
}

struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81Block) == 36);

__global__ void inkling_swiglu_gate_up_q8_1_kernel(const float* __restrict__ gate_up, Q81Block* __restrict__ out_q8_1,
                                                   int ff_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ff_dim) return;
    const size_t t = blockIdx.y;
    const float* row = gate_up + t * 2 * size_t(ff_dim);
    const float gi = row[i];
    const float ui = row[ff_dim + i];
    const float xi = __fmul_rn(__fdividef(gi, __fadd_rn(1.0f, __expf(-gi))), ui);
    const float amax = warp_max(fabsf(xi));
    const float sum = warp_sum(xi);
    const float d = strata::kernels::q8_1_finite(__fdividef(amax, 127.0f));
    int8_t q = 0;
    if (amax != 0.0f) {
        const float qf = roundf(__fdividef(xi, d));
        q = (int8_t) (qf > 127.0f ? 127.0f : qf < -127.0f ? -127.0f : qf);
    }
    const int nb = ff_dim >> 5;
    Q81Block& dst = out_q8_1[t * nb + (i >> 5)];
    dst.qs[i & 31] = q;
    if ((i & 31) == 0) dst.ds = strata::kernels::q8_1_ds(d, sum);
}

__global__ void inkling_dspark_extract_taps_kernel(const float* __restrict__ r, int n_streams, int first_tap_stream,
                                                   float* __restrict__ taps012, int n_embd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    const size_t row = blockIdx.y;
    const float* src = r + (row * n_streams + first_tap_stream) * n_embd + i;
    float* dst = taps012 + row * 3 * n_embd + i;
    dst[0] = src[0];
    dst[n_embd] = src[n_embd];
    dst[2 * size_t(n_embd)] = src[2 * size_t(n_embd)];
}

__global__ void inkling_dspark_mix_norm_kernel(const float* __restrict__ taps012, const float* __restrict__ tap3,
                                               float a0, float a1, float a2, float a3, const float* __restrict__ w_norm,
                                               float* __restrict__ out, int n_embd, float eps) {
    __shared__ float part[8];
    const size_t row = blockIdx.x;
    const float* t0 = taps012 + row * 3 * n_embd;
    const float* t1 = t0 + n_embd;
    const float* t2 = t1 + n_embd;
    const float* t3 = tap3 + row * n_embd;
    float* y = out + row * n_embd;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        const float m = a0 * t0[i] + a1 * t1[i] + a2 * t2[i] + a3 * t3[i];
        ss += m * m;
    }
    ss = block_sum(ss, part);
    const float inv = rsqrtf(ss / float(n_embd) + eps);
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        const float m = a0 * t0[i] + a1 * t1[i] + a2 * t2[i] + a3 * t3[i];
        y[i] = m * inv * (w_norm ? w_norm[i] : 1.0f);
    }
}

__global__ void inkling_dspark_prep_rnn_step_kernel(const float* __restrict__ state,
                                                  const uint16_t* __restrict__ w1_bf16, int32_t anchor_tok,
                                                  const Likeliest* __restrict__ picks, int step, int anchor_slot,
                                                  int rank, int w1_vocab, float* __restrict__ rnn_in_sp) {
    const int i = threadIdx.x;
    if (i >= rank) return;
    rnn_in_sp[i] = state[i];
    const int32_t tok = (step == 0) ? (anchor_tok >= 0 ? anchor_tok : picks[anchor_slot].token) : picks[step - 1].token;
    if (tok >= 0 && tok < w1_vocab) {
        const uint32_t bits = (uint32_t) w1_bf16[size_t(tok) * rank + i] << 16;
        rnn_in_sp[rank + i] = __uint_as_float(bits);
    } else {
        rnn_in_sp[rank + i] = 0.0f;
    }
}

__global__ void inkling_dspark_rnn_gate_kernel(const float* __restrict__ gates_sp, const float* __restrict__ gates_h,
                                               const float* __restrict__ gates_f, const float* __restrict__ bias,
                                               float* __restrict__ state, float* __restrict__ latent, int rank) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rank) return;
    const float h0 = gates_h ? gates_h[i] : 0.0f;
    const float h1 = gates_h ? gates_h[rank + i] : 0.0f;
    const float h2 = gates_h ? gates_h[2 * rank + i] : 0.0f;
    const float f0 = gates_f ? gates_f[i] : 0.0f;
    const float f1 = gates_f ? gates_f[rank + i] : 0.0f;
    const float f2 = gates_f ? gates_f[2 * rank + i] : 0.0f;
    const float g0 = gates_sp[i] + h0 + f0 + (bias ? bias[i] : 0.0f);
    const float g1 = gates_sp[rank + i] + h1 + f1 + (bias ? bias[rank + i] : 0.0f);
    const float g2 = gates_sp[2 * rank + i] + h2 + f2 + (bias ? bias[2 * rank + i] : 0.0f);
    const float keep = 1.0f / (1.0f + expf(-g0));
    const float upd = tanhf(g1);
    const float lat = tanhf(g2);
    state[i] = keep * state[i] + (1.0f - keep) * upd;
    latent[i] = lat;
}

__global__ void inkling_dspark_add_softcap_kernel(const float* __restrict__ base, const float* __restrict__ delta,
                                                  const float* __restrict__ bias, float* __restrict__ out, int n,
                                                  float cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = base[i] + (delta ? delta[i] : 0.0f) + (bias ? bias[i] : 0.0f);
    out[i] = cap > 0.0f ? cap * tanhf(v / cap) : v;
}

__global__ void inkling_dspark_pack_q8_0_block_kernel(const float* __restrict__ cols32, uint8_t* __restrict__ dst_q8,
                                                      int n_vocab, int block_idx, int n_blocks) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vocab) return;
    float vals[32];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const float x = cols32[size_t(j) * n_vocab + v];
        vals[j] = x;
        amax = fmaxf(amax, fabsf(x));
    }
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    uint8_t* blk = dst_q8 + (size_t(v) * n_blocks + block_idx) * 34;
    const uint16_t dh = __half_as_ushort(__float2half(d));
    blk[0] = uint8_t(dh & 0xff);
    blk[1] = uint8_t((dh >> 8) & 0xff);
    int8_t* qs = (int8_t*) (blk + 2);
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int q = (int) roundf(vals[j] * id);
        qs[j] = (int8_t) (q < -127 ? -127 : q > 127 ? 127 : q);
    }
}

__global__ void inkling_dspark_pack_q4_0_block_kernel(const float* __restrict__ cols32, uint8_t* __restrict__ dst_q4,
                                                      int n_vocab, int block_idx, int n_blocks) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vocab) return;
    float vals[32];
    float amax = 0.0f, vmax = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const float x = cols32[size_t(j) * n_vocab + v];
        vals[j] = x;
        const float ax = fabsf(x);
        if (ax > amax || (ax == amax && x > vmax)) {
            amax = ax;
            vmax = x;
        }
    }
    const float d = vmax / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    uint8_t* blk = dst_q4 + (size_t(v) * n_blocks + block_idx) * 18;
    const uint16_t dh = __half_as_ushort(__float2half(d));
    blk[0] = uint8_t(dh & 0xff);
    blk[1] = uint8_t((dh >> 8) & 0xff);
    uint8_t* qs = blk + 2;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int q0 = __float2int_rz(vals[j] * id + 8.5f);
        int q1 = __float2int_rz(vals[j + 16] * id + 8.5f);
        q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
        q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
        qs[j] = uint8_t(q0 | (q1 << 4));
    }
}

__global__ void inkling_dspark_split_qkv_kernel(const float* __restrict__ qkv, float* __restrict__ qraw,
                                                float* __restrict__ kraw, float* __restrict__ v,
                                                int q_dim, int kv_dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = q_dim + 2 * kv_dim;
    if (i >= stride) return;
    const size_t t = blockIdx.y;
    const float x = qkv[t * stride + i];
    if (i < q_dim) qraw[t * q_dim + i] = x;
    else if (i < q_dim + kv_dim) kraw[t * kv_dim + (i - q_dim)] = x;
    else v[t * kv_dim + (i - q_dim - kv_dim)] = x;
}

}  // namespace

void situ_rows(const float* gate_up, float* out, int n, int rows, float beta, float linear_beta, void* stream,
               const ExpertScales* scales) {
    if (rows <= 0) return;
    const ExpertScales s = scales ? *scales : ExpertScales{};
    situ_rows_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(gate_up, out, n, beta, linear_beta, s);
    ck("situ_rows");
}

void inkling_mix(const float* r, int streams, MixSlots slots, int n_slots, const float* query, int n_queries,
                 float* out, int n_embd, int rows, float eps, void* stream) {
    if (rows <= 0) return;
    if (n_slots < 1 || n_slots > 6 || n_queries < 1 || n_queries > 4)
        throw std::invalid_argument("inkling_mix: n_slots must be in 1..6 and n_queries in 1..4");
    inkling_mix_kernel<<<rows, 256, 0, cs(stream)>>>(r, streams, slots, n_slots, query, n_queries, out, n_embd, eps);
    ck("inkling_mix");
}

void inkling_control_prep(const float* cur, const float* msg, float* out_concat, int n_embd, int rows, float eps,
                          void* stream) {
    if (rows <= 0) return;
    inkling_control_prep_kernel<<<rows, 256, 0, cs(stream)>>>(cur, msg, out_concat, n_embd, eps);
    ck("inkling_control_prep");
}

void inkling_control_apply(const float* cur, const float* msg, const float* gain_raw, const float* forget_raw,
                           float* out, int n_embd, int rows, void* stream, int gf_stride) {
    if (rows <= 0) return;
    const int stride = gf_stride > 0 ? gf_stride : n_embd;
    inkling_control_apply_kernel<<<dim3(grid(n_embd, 256), rows), 256, 0, cs(stream)>>>(cur, msg, gain_raw,
                                                                                        forget_raw, out, n_embd,
                                                                                        stride);
    ck("inkling_control_apply");
}

void inkling_conv4(const float* x, float* state, const float* w, const float* dyn, int head_dim, float* out,
                   int channels, int rows, void* stream, float* snap, size_t stride, unsigned snaps) {
    if (rows <= 0) return;
    const int n_heads = dyn ? channels / head_dim : 0;
    if (rows >= CONV_ROWS && !snap) {
        inkling_conv4_rows_kernel<<<dim3(grid(channels, 256), rows), 256, 0, cs(stream)>>>(
            x, state, w, dyn, head_dim, n_heads, out, channels);
        inkling_conv4_advance_kernel<<<grid(channels, 256), 256, 0, cs(stream)>>>(x, state, channels, rows);
    } else {
        inkling_conv4_walk_kernel<<<grid(channels, 256), 256, 0, cs(stream)>>>(
            x, state, w, dyn, head_dim, n_heads, out, channels, rows, snap, stride, snaps);
    }
    ck("inkling_conv4");
}

void inkling_l2_clip(const float* x, const float* scale, float* out, int dim, int heads, float threshold,
                     int64_t pos, bool is_global, float log_floor, float log_alpha, int rows,
                     const long long* at, void* stream) {
    if (rows <= 0) return;
    if (dim > 128) throw std::invalid_argument("inkling_l2_clip: head dim must be <= 128");
    inkling_l2_clip_kernel<<<dim3(heads, rows), WARP, 0, cs(stream)>>>(
        x, scale, out, dim, heads, threshold, (long long) pos, is_global, log_floor, log_alpha, at);
    ck("inkling_l2_clip");
}

void inkling_rel_table(const float* r_cur, const float* rel_bias, const float* sinks, float* table, int n_rel,
                       int n_extent, int n_head, int rows, void* stream) {
    if (rows <= 0) return;
    inkling_rel_table_kernel<<<dim3(n_head, rows), 256, 0, cs(stream)>>>(r_cur, rel_bias, sinks, table, n_rel, n_extent,
                                                                        n_head);
    ck("inkling_rel_table");
}

void store_q4(const float* x, uint8_t* codes, uint16_t* scales, int n, int64_t pos, int rows, void* stream) {
    if (rows <= 0) return;
    if (n <= 0 || n % 32 != 0) throw std::invalid_argument("store_q4: width must be a multiple of 32");
    const int nb = n / 32;
    store_q4_kernel<<<grid((size_t) nb * rows, 256), 256, 0, cs(stream)>>>(
        x, codes + (size_t) pos * (n >> 1), scales + (size_t) pos * nb, n, rows, nb);
    ck("store_q4");
}

void store_q4_at(const float* x, uint8_t* codes, uint16_t* scales, int n, const long long* at, int rows, int ring,
                 void* stream) {
    if (rows <= 0) return;
    if (n <= 0 || n % 32 != 0) throw std::invalid_argument("store_q4_at: width must be a multiple of 32");
    const int nb = n / 32;
    store_q4_at_kernel<<<grid((size_t) nb * rows, 256), 256, 0, cs(stream)>>>(x, codes, scales, n, rows, nb, at, ring);
    ck("store_q4_at");
}

void inkling_attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, const float* table,
                    const float* sinks, int n_extent, int64_t pos, int rows, float* score, int chunk, float* out,
                    int dim, int n_head, int kv_heads, float scale, void* stream, int window, int ring,
                    const long long* at, int room, Kvq8 kq8, Kvq8 vq8, Kvq4 kq4, Kvq4 vq4) {
    (void) chunk;
    if (rows <= 0) return;
    if (dim > 128) throw std::invalid_argument("inkling_attend: head dim must be <= 128");
    const long long max_pos = at ? (long long) room : (long long) pos + rows - 1;
    const int span = window > 0 ? std::min<int>((int) (max_pos + 1), window) : (int) (max_pos + 1);
    const int n_splits = std::clamp((span + SPLIT_KEYS - 1) / SPLIT_KEYS, 1, MAX_SPLITS);
    if (dim == 128 && n_head == kv_heads * 3) {
        inkling_attend_split_gqa_kernel<3><<<dim3(n_splits, kv_heads, rows), 128, 0, cs(stream)>>>(
            q, kcache, vcache, table, sinks, n_extent, (long long) pos, score, n_splits, dim, n_head, kv_heads, scale,
            window, ring, at, kq8, vq8, kq4, vq4);
    } else {
        inkling_attend_split_kernel<<<dim3(n_splits, n_head, rows), 128, 0, cs(stream)>>>(
            q, kcache, vcache, table, sinks, n_extent, (long long) pos, score, n_splits, dim, n_head, kv_heads, scale,
            window, ring, at, kq8, vq8, kq4, vq4);
    }
    inkling_attend_join_kernel<<<dim3(n_head, rows), dim, 0, cs(stream)>>>(score, out, n_splits, dim, n_head,
                                                                            (long long) pos, window, at);
    ck("inkling_attend");
}

void inkling_dspark_attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows,
                           float* score, float* out, int dim, int n_head, int kv_heads, float scale, void* stream,
                           int window, int ring, const long long* at, int room, Kvq8 kq8, Kvq8 vq8,
                           Kvq4 kq4, Kvq4 vq4) {
    if (rows <= 0) return;
    if (dim > 128) throw std::invalid_argument("inkling_dspark_attend: head dim must be <= 128");
    const long long max_pos = at ? (long long) room : (long long) pos + rows - 1;
    const int span = window > 0 ? std::min<int>((int) (max_pos + 1), window + rows - 1) : (int) (max_pos + 1);
    const int n_splits = std::clamp((span + SPLIT_KEYS - 1) / SPLIT_KEYS, 1, MAX_SPLITS);
    if (dim == 128 && n_head == kv_heads * 3) {
        inkling_attend_split_gqa_kernel<3, true><<<dim3(n_splits, kv_heads, rows), 128, 0, cs(stream)>>>(
            q, kcache, vcache, nullptr, nullptr, 0, (long long) pos, score, n_splits, dim, n_head, kv_heads, scale,
            window, ring, at, kq8, vq8, kq4, vq4);
    } else {
        inkling_attend_split_kernel<true><<<dim3(n_splits, n_head, rows), 128, 0, cs(stream)>>>(
            q, kcache, vcache, nullptr, nullptr, 0, (long long) pos, score, n_splits, dim, n_head, kv_heads, scale,
            window, ring, at, kq8, vq8, kq4, vq4);
    }
    inkling_attend_join_kernel<false, true><<<dim3(n_head, rows), dim, 0, cs(stream)>>>(
        score, out, n_splits, dim, n_head, (long long) pos, window, at);
    ck("inkling_dspark_attend");
}

void inkling_attend_batch(void* blas, const float* q, const uint16_t* kcache, const uint16_t* vcache,
                          const float* table, const float* sinks, int n_extent, int64_t pos, int rows, float* score,
                          size_t score_floats, float* work, float* out, int dim, int n_head, int kv_heads,
                          float scale, void* stream, int window, int ring, Kvq8 kq8, Kvq8 vq8,
                          uint16_t* kwork, uint16_t* vwork, size_t work_halves, Kvq4 kq4, Kvq4 vq4) {
    if (rows <= 0) return;
    const cublasHandle_t h = (cublasHandle_t) blas;
    const float one = 1.0f, zero = 0.0f;
    const int per = n_head / kv_heads, width = kv_heads * dim;
    const bool kq = kq8.codes != nullptr, vq = vq8.codes != nullptr;
    const bool k4 = kq4.codes != nullptr, v4 = vq4.codes != nullptr;
    const int work_rows = width > 0 ? (int) std::min<size_t>(work_halves / (size_t) width, (size_t) INT_MAX) : 0;
    if ((kq || k4) && (kwork == nullptr || work_rows < 1))
        throw std::invalid_argument("inkling_attend_batch: quantized key cache needs workspace");
    if ((vq || v4) && (vwork == nullptr || work_rows < 1))
        throw std::invalid_argument("inkling_attend_batch: quantized value cache needs workspace");
    const size_t cells = score_floats / 3 * 2;
    half* weight = (half*) (score + cells);
    for (int first = 0, c = 0; first < rows; first += c) {
        c = (int) std::min({size_t(rows - first), size_t(65535 / n_head), cells / n_head,
                            std::max<size_t>(16, kAttendResults / (size_t(n_head) * dim * sizeof(float)))});
        const long long p0 = pos + first;
        const long long lo0 = window > 0 && p0 >= window ? p0 - window + 1 : 0;
        const int n_keys = int((p0 + c - 1) - lo0) + 1;
        const size_t heads = size_t(c) * n_head;
        half* qg = (half*) work;
        float* og = work + (heads * dim + 1) / 2;
        float* top = og + heads * dim;
        float* sum = top + heads;
        float* shrink = sum + heads;
        attn_group_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(
            q + size_t(first) * n_head * dim, qg, dim, n_head, kv_heads, c);
        const long long cols = (long long) c * per;
        const int fit = (int) (cells / heads);
        const int block = fit > 16 ? fit & ~15 : fit;
        for (int k0 = 0, m = 0; k0 < n_keys; k0 += m) {
            const long long row = ring > 0 ? (lo0 + k0) % ring : lo0 + k0;
            m = (int) std::min<long long>(std::min(block, n_keys - k0), ring > 0 ? ring - row : n_keys);
            if ((kq || vq || k4 || v4) && m > work_rows) m = work_rows;
            if (kq)
                dequant_q8_kernel<<<dim3(grid(width >> 2, 256), m), 256, 0, cs(stream)>>>(kq8.codes, kq8.scales,
                                                                                          (half*) kwork, width, row);
            else if (k4)
                dequant_q4_kernel<<<dim3(grid(width >> 2, 256), m), 256, 0, cs(stream)>>>(kq4.codes, kq4.scales,
                                                                                          (half*) kwork, width, row);
            if (vq)
                dequant_q8_kernel<<<dim3(grid(width >> 2, 256), m), 256, 0, cs(stream)>>>(vq8.codes, vq8.scales,
                                                                                          (half*) vwork, width, row);
            else if (v4)
                dequant_q4_kernel<<<dim3(grid(width >> 2, 256), m), 256, 0, cs(stream)>>>(vq4.codes, vq4.scales,
                                                                                          (half*) vwork, width, row);
            const half* krow = (kq || k4) ? (const half*) kwork : (const half*) kcache + size_t(row) * width;
            const half* vrow = (vq || v4) ? (const half*) vwork : (const half*) vcache + size_t(row) * width;
            blas_ck(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, (int) cols, dim, &scale,
                                               krow, CUDA_R_16F, width, dim, qg, CUDA_R_16F, dim, cols * dim, &zero,
                                               score, CUDA_R_32F, m, cols * m, kv_heads, CUBLAS_COMPUTE_32F,
                                               CUBLAS_GEMM_DEFAULT),
                    "inkling attention scores");
            inkling_softmax_block_kernel<<<c * n_head, 128, 0, cs(stream)>>>(
                score, weight, m, lo0 + k0, (long long) pos, first, window, c, per, n_head, table, sinks, n_extent,
                top, sum, shrink, k0 == 0);
            if (k0 > 0)
                attn_scale_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(og, shrink, dim, false);
            blas_ck(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, dim, (int) cols, m, &one,
                                               vrow, CUDA_R_16F, width, dim, weight, CUDA_R_16F, m, cols * m,
                                               k0 > 0 ? &one : &zero, og, CUDA_R_32F, dim, cols * dim, kv_heads,
                                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                    "inkling attention values");
        }
        attn_scale_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(og, sum, dim, true);
        attn_ungroup_kernel<<<dim3(grid(dim, 256), c * n_head), 256, 0, cs(stream)>>>(
            og, out + size_t(first) * n_head * dim, dim, n_head, kv_heads, c);
    }
    ck("inkling_attend_batch");
}

void inkling_copy_stream(float* r, int streams, int dst_stream, int src_stream, int n_embd, int rows, void* stream) {
    if (rows <= 0) return;
    inkling_copy_stream_kernel<<<dim3(grid(n_embd, 256), rows), 256, 0, cs(stream)>>>(r, streams, dst_stream,
                                                                                      src_stream, n_embd);
    ck("inkling_copy_stream");
}

void inkling_write_stream(const float* x, float* r, int streams, int dst_stream, int n_embd, int rows, void* stream) {
    if (rows <= 0) return;
    inkling_write_stream_kernel<<<dim3(grid(n_embd, 256), rows), 256, 0, cs(stream)>>>(x, r, streams, dst_stream,
                                                                                       n_embd);
    ck("inkling_write_stream");
}

void inkling_read_stream(const float* r, int streams, int src_stream, float* out, int n_embd, int rows, void* stream) {
    if (rows <= 0) return;
    inkling_read_stream_kernel<<<dim3(grid(n_embd, 256), rows), 256, 0, cs(stream)>>>(r, streams, src_stream, out,
                                                                                      n_embd);
    ck("inkling_read_stream");
}

void inkling_add_to_stream(const float* a, const float* b, float* r, int streams, int dst_stream, int n_embd, int rows,
                           void* stream) {
    if (rows <= 0) return;
    inkling_add_to_stream_kernel<<<dim3(grid(n_embd, 256), rows), 256, 0, cs(stream)>>>(a, b, r, streams, dst_stream,
                                                                                        n_embd);
    ck("inkling_add_to_stream");
}

void inkling_dequant_kv_span(int64_t lo, int count, int width, int ring, Kvq8 kq8, Kvq8 vq8, Kvq4 kq4, Kvq4 vq4,
                             uint16_t* kwork, uint16_t* vwork, void* stream) {
    if (count <= 0 || width <= 0) return;
    const int w4 = grid(width >> 2, 256);
    for (int done = 0; done < count;) {
        const long long row = ring > 0 ? (lo + done) % ring : lo + done;
        const int m = ring > 0 ? (int) std::min<long long>(count - done, ring - row) : count - done;
        half* ko = (half*) kwork + size_t(done) * width;
        half* vo = (half*) vwork + size_t(done) * width;
        if (kq8.codes)
            dequant_q8_kernel<<<dim3(w4, m), 256, 0, cs(stream)>>>(kq8.codes, kq8.scales, ko, width, row);
        else if (kq4.codes)
            dequant_q4_kernel<<<dim3(w4, m), 256, 0, cs(stream)>>>(kq4.codes, kq4.scales, ko, width, row);
        if (vq8.codes)
            dequant_q8_kernel<<<dim3(w4, m), 256, 0, cs(stream)>>>(vq8.codes, vq8.scales, vo, width, row);
        else if (vq4.codes)
            dequant_q4_kernel<<<dim3(w4, m), 256, 0, cs(stream)>>>(vq4.codes, vq4.scales, vo, width, row);
        done += m;
    }
    ck("inkling_dequant_kv_span");
}

void inkling_swiglu_gate_up_q8_1(const float* gate_up, void* out_q8_1, int ff_dim, int rows, void* stream) {
    if (rows <= 0) return;
    if (ff_dim <= 0 || ff_dim % 32 != 0)
        throw std::invalid_argument("inkling_swiglu_gate_up_q8_1: ff_dim must be a positive multiple of 32");
    inkling_swiglu_gate_up_q8_1_kernel<<<dim3(grid(ff_dim, 256), rows), 256, 0, cs(stream)>>>(
        gate_up, (Q81Block*) out_q8_1, ff_dim);
    ck("inkling_swiglu_gate_up_q8_1");
}

void inkling_dspark_extract_taps(const float* r, int n_streams, int first_tap_stream, float* taps012, int n_embd,
                                 int rows, void* stream) {
    if (rows <= 0) return;
    inkling_dspark_extract_taps_kernel<<<dim3(grid(n_embd, 256), rows), 256, 0, cs(stream)>>>(
        r, n_streams, first_tap_stream, taps012, n_embd);
    ck("inkling_dspark_extract_taps");
}

void inkling_dspark_mix_norm(const float* taps012, const float* tap3, float a0, float a1, float a2, float a3,
                             const float* w_norm, float* out, int n_embd, int rows, float eps, void* stream) {
    if (rows <= 0) return;
    inkling_dspark_mix_norm_kernel<<<rows, 256, 0, cs(stream)>>>(taps012, tap3, a0, a1, a2, a3, w_norm, out, n_embd, eps);
    ck("inkling_dspark_mix_norm");
}

void inkling_dspark_prep_rnn_step(const float* state, const uint16_t* w1_bf16, int32_t anchor_tok,
                                  const Likeliest* picks, int step, int anchor_slot, int rank, int w1_vocab,
                                  float* rnn_in_sp, void* stream) {
    if (rank <= 0) return;
    if (rank > 1024 || rank % 32 != 0)
        throw std::invalid_argument("inkling_dspark_prep_rnn_step: rank must be a multiple of 32 <= 1024");
    inkling_dspark_prep_rnn_step_kernel<<<1, rank, 0, cs(stream)>>>(
        state, w1_bf16, anchor_tok, picks, step, anchor_slot, rank, w1_vocab, rnn_in_sp);
    ck("inkling_dspark_prep_rnn_step");
}

void inkling_dspark_rnn_gate(const float* gates_sp, const float* gates_h, const float* gates_f, const float* bias,
                             float* state, float* latent, int rank, void* stream) {
    if (rank <= 0) return;
    inkling_dspark_rnn_gate_kernel<<<grid(rank, 256), 256, 0, cs(stream)>>>(
        gates_sp, gates_h, gates_f, bias, state, latent, rank);
    ck("inkling_dspark_rnn_gate");
}

void inkling_dspark_add_softcap(const float* base, const float* delta, const float* bias, float* out, int n, float cap,
                                void* stream) {
    if (n <= 0) return;
    inkling_dspark_add_softcap_kernel<<<grid(n, 256), 256, 0, cs(stream)>>>(base, delta, bias, out, n, cap);
    ck("inkling_dspark_add_softcap");
}

void inkling_dspark_pack_q8_0_block(const float* cols32, uint8_t* dst_q8, int n_vocab, int block_idx, int n_blocks,
                                    void* stream) {
    if (n_vocab <= 0) return;
    inkling_dspark_pack_q8_0_block_kernel<<<grid(n_vocab, 256), 256, 0, cs(stream)>>>(
        cols32, dst_q8, n_vocab, block_idx, n_blocks);
    ck("inkling_dspark_pack_q8_0_block");
}

void inkling_dspark_pack_q4_0_block(const float* cols32, uint8_t* dst_q4, int n_vocab, int block_idx, int n_blocks,
                                    void* stream) {
    if (n_vocab <= 0) return;
    inkling_dspark_pack_q4_0_block_kernel<<<grid(n_vocab, 256), 256, 0, cs(stream)>>>(
        cols32, dst_q4, n_vocab, block_idx, n_blocks);
    ck("inkling_dspark_pack_q4_0_block");
}

void inkling_dspark_split_qkv(const float* qkv, float* qraw, float* kraw, float* v, int q_dim, int kv_dim, int rows,
                              void* stream) {
    if (rows <= 0) return;
    const int stride = q_dim + 2 * kv_dim;
    inkling_dspark_split_qkv_kernel<<<dim3(grid(stride, 256), rows), 256, 0, cs(stream)>>>(
        qkv, qraw, kraw, v, q_dim, kv_dim);
    ck("inkling_dspark_split_qkv");
}

}  // namespace strata::poly
