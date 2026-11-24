// src/poly/kernels_inkling2.cu - Inkling 2 (inkling2_loop / inkling2) CUDA kernels.
//
// Reference: mainline llama.cpp src/models/inkling2.cpp and src/models/inkling2-loop.cpp.
#include "strata/poly/kernels.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace strata::poly {
namespace {

constexpr int WARP = 32;
constexpr int CONV_ROWS = 16;
constexpr int SPLIT_KEYS = 128;
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

struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81Block) == 36);

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

__global__ void situ_rows_q8_1_kernel(const float* __restrict__ gate_up, Q81Block* __restrict__ out_q8_1, int n,
                                      float beta, float linear_beta, ExpertScales s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t t = blockIdx.y;
    const float* row = gate_up + t * 2 * size_t(n);
    const float2 sc = row_scales(s, (int) t);
    const float g = row[i] * sc.x;
    const float u = row[n + i] * sc.y;
    const float act = beta * tanhf(g / beta) * sigmoidf_d(g);
    const float upv = linear_beta > 0.0f ? linear_beta * tanhf(u / linear_beta) : u;
    const float xi = act * upv;
    const float amax = warp_max(fabsf(xi));
    const float bsum = warp_sum(xi);
    const float d = strata::kernels::q8_1_finite(amax / 127.0f);
    const int8_t q = strata::kernels::q8_1_quant(xi, d, amax);
    Q81Block& dst = out_q8_1[t * (n >> 5) + (i >> 5)];
    dst.qs[i & 31] = q;
    if ((i & 31) == 0) dst.ds = strata::kernels::q8_1_ds(d, bsum);
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

constexpr int kMixThreads = 768;
constexpr int kMixHeld = 3;

template <int NS, int NQ>
__global__ void __launch_bounds__(kMixThreads) inkling_mix_held_kernel(
    const float* __restrict__ r, int streams, MixSlots slots,
    const float* __restrict__ query, float* __restrict__ out,
    int n_embd, int rows, float eps) {

    constexpr int kSums = 1 + NQ;
    __shared__ float part[NS * kSums][kMixThreads / WARP];
    __shared__ float whole[NS * kSums];
    __shared__ float inv_rms[NS];
    __shared__ float probs[NQ][NS];

    const int t = blockIdx.x;
    const size_t row_offset = size_t(t) * streams * n_embd;
    const float* tok_r = r + row_offset;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    float x[kMixHeld][NS];
    float sum[NS][kSums];

#pragma unroll
    for (int s = 0; s < NS; ++s) {
#pragma unroll
        for (int k = 0; k < kSums; ++k) sum[s][k] = 0.0f;
    }

#pragma unroll
    for (int h = 0; h < kMixHeld; ++h) {
        const int i = tid + h * kMixThreads;
        if (i < n_embd) {
            float q[NQ];
#pragma unroll
            for (int j = 0; j < NQ; ++j) q[j] = query[size_t(j) * n_embd + i];
#pragma unroll
            for (int s = 0; s < NS; ++s) {
                const float v = tok_r[size_t(slots.s[s]) * n_embd + i];
                x[h][s] = v;
                sum[s][0] += v * v;
#pragma unroll
                for (int j = 0; j < NQ; ++j) sum[s][1 + j] += v * q[j];
            }
        }
    }

#pragma unroll
    for (int s = 0; s < NS; ++s) {
#pragma unroll
        for (int k = 0; k < kSums; ++k) {
            const float w = warp_sum(sum[s][k]);
            if (lane == 0) part[s * kSums + k][warp] = w;
        }
    }
    __syncthreads();

    if (tid < NS * kSums) {
        float val = 0.0f;
#pragma unroll
        for (int w = 0; w < kMixThreads / WARP; ++w) val += part[tid][w];
        whole[tid] = val;
    }
    __syncthreads();

    if (tid == 0) {
        const float score_scale = rsqrtf((float) n_embd);
        for (int s = 0; s < NS; ++s) {
            const float mean_sq = whole[s * kSums] / (float) n_embd;
            inv_rms[s] = rsqrtf(mean_sq + eps);
        }
        for (int q = 0; q < NQ; ++q) {
            float logits[NS];
            float mx = -1e30f;
            for (int s = 0; s < NS; ++s) {
                const float dot = whole[s * kSums + 1 + q];
                logits[s] = dot * inv_rms[s] * score_scale;
                mx = fmaxf(mx, logits[s]);
            }
            float exps[NS];
            float sum_exp = 0.0f;
            for (int s = 0; s < NS; ++s) {
                exps[s] = expf(logits[s] - mx);
                sum_exp += exps[s];
            }
            const float inv_s = 1.0f / sum_exp;
            for (int s = 0; s < NS; ++s) {
                probs[q][s] = exps[s] * inv_s;
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int h = 0; h < kMixHeld; ++h) {
        const int i = tid + h * kMixThreads;
        if (i < n_embd) {
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
                float acc = 0.0f;
#pragma unroll
                for (int s = 0; s < NS; ++s) {
                    acc += probs[q][s] * (x[h][s] * inv_rms[s]);
                }
                out[(size_t(q) * rows + t) * n_embd + i] = acc;
            }
        }
    }
}

template <int NS, int NQ>
struct MixHeldLauncher {
    static void launch(dim3 grid, cudaStream_t s, const float* r, int streams, MixSlots slots,
                       const float* query, float* out, int n_embd, int rows, float eps) {
        inkling_mix_held_kernel<NS, NQ><<<grid, kMixThreads, 0, s>>>(r, streams, slots, query, out, n_embd, rows, eps);
    }
};

template <template <int, int> class K, class... Args>
void launch_mix_held(int n_slots, int n_queries, dim3 grid, cudaStream_t s, Args... args) {
    const auto go = [&](auto ns_c) {
        constexpr int NS = decltype(ns_c)::value;
        switch (n_queries) {
            case 1: K<NS, 1>::launch(grid, s, args...); break;
            case 2: K<NS, 2>::launch(grid, s, args...); break;
            case 3: K<NS, 3>::launch(grid, s, args...); break;
            default: K<NS, 4>::launch(grid, s, args...); break;
        }
    };
    switch (n_slots) {
        case 1: go(std::integral_constant<int, 1>{}); break;
        case 2: go(std::integral_constant<int, 2>{}); break;
        case 3: go(std::integral_constant<int, 3>{}); break;
        case 4: go(std::integral_constant<int, 4>{}); break;
        case 5: go(std::integral_constant<int, 5>{}); break;
        default: go(std::integral_constant<int, 6>{}); break;
    }
}

template <int NS>
__global__ void __launch_bounds__(kMixThreads) inkling_mix_norm_q8_kernel(
    const float* __restrict__ r, int streams, MixSlots slots, const float* __restrict__ query,
    const float* __restrict__ norm_w, float* __restrict__ xn_out, Q81Block* __restrict__ xq_out,
    int n_embd, float mix_eps, float rms_eps) {

    constexpr int kSums = 2;
    __shared__ float part[NS * kSums][kMixThreads / WARP];
    __shared__ float whole[NS * kSums];
    __shared__ float inv_rms[NS];
    __shared__ float probs[NS];
    __shared__ float part_out[kMixThreads / WARP];

    const int t = blockIdx.x;
    const float* tok_r = r + size_t(t) * streams * n_embd;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    float x[kMixHeld][NS];
    float sum[NS][kSums];

#pragma unroll
    for (int s = 0; s < NS; ++s) {
        sum[s][0] = 0.0f;
        sum[s][1] = 0.0f;
    }

#pragma unroll
    for (int h = 0; h < kMixHeld; ++h) {
        const int i = tid + h * kMixThreads;
        if (i < n_embd) {
            const float qv = query[i];
#pragma unroll
            for (int s = 0; s < NS; ++s) {
                const float v = tok_r[size_t(slots.s[s]) * n_embd + i];
                x[h][s] = v;
                sum[s][0] += v * v;
                sum[s][1] += v * qv;
            }
        }
    }

#pragma unroll
    for (int s = 0; s < NS; ++s) {
#pragma unroll
        for (int k = 0; k < kSums; ++k) {
            const float w = warp_sum(sum[s][k]);
            if (lane == 0) part[s * kSums + k][warp] = w;
        }
    }
    __syncthreads();

    if (tid < NS * kSums) {
        float val = 0.0f;
#pragma unroll
        for (int w = 0; w < kMixThreads / WARP; ++w) val += part[tid][w];
        whole[tid] = val;
    }
    __syncthreads();

    if (tid == 0) {
        const float score_scale = rsqrtf((float) n_embd);
        for (int s = 0; s < NS; ++s) {
            inv_rms[s] = rsqrtf(whole[s * kSums] / (float) n_embd + mix_eps);
        }
        float logits[NS];
        float mx = -1e30f;
        for (int s = 0; s < NS; ++s) {
            logits[s] = whole[s * kSums + 1] * inv_rms[s] * score_scale;
            mx = fmaxf(mx, logits[s]);
        }
        float exps[NS];
        float sum_exp = 0.0f;
        for (int s = 0; s < NS; ++s) {
            exps[s] = expf(logits[s] - mx);
            sum_exp += exps[s];
        }
        const float inv_s = 1.0f / sum_exp;
        for (int s = 0; s < NS; ++s) probs[s] = exps[s] * inv_s;
    }
    __syncthreads();

    float m[kMixHeld];
    float sq_m = 0.0f;
#pragma unroll
    for (int h = 0; h < kMixHeld; ++h) {
        const int i = tid + h * kMixThreads;
        float acc = 0.0f;
        if (i < n_embd) {
#pragma unroll
            for (int s = 0; s < NS; ++s) acc += probs[s] * (x[h][s] * inv_rms[s]);
            sq_m += acc * acc;
        }
        m[h] = acc;
    }

    const float w_sq = warp_sum(sq_m);
    if (lane == 0) part_out[warp] = w_sq;
    __syncthreads();

    float total_sq = 0.0f;
#pragma unroll
    for (int w = 0; w < kMixThreads / WARP; ++w) total_sq += part_out[w];
    const float inv_norm = rsqrtf(total_sq / (float) n_embd + rms_eps);

    const int nb = n_embd >> 5;
    Q81Block* qr = xq_out ? xq_out + size_t(t) * nb : nullptr;
#pragma unroll
    for (int h = 0; h < kMixHeld; ++h) {
        const int i = tid + h * kMixThreads;
        if (i < n_embd) {
            const float xi = (m[h] * inv_norm) * norm_w[i];
            xn_out[size_t(t) * n_embd + i] = xi;
            if (qr) {
                const float amax = warp_max(fabsf(xi));
                const float bsum = warp_sum(xi);
                const float d = strata::kernels::q8_1_finite(amax / 127.0f);
                const int8_t q = strata::kernels::q8_1_quant(xi, d, amax);
                Q81Block& dst = qr[i >> 5];
                dst.qs[lane] = q;
                if (lane == 0) dst.ds = strata::kernels::q8_1_ds(d, bsum);
            }
        }
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

constexpr int kCatThreads = 256;
constexpr int kDownThreads = 512;
constexpr int kCtrlSet = 3;

struct CtrlDown {
    const float* cur[kCtrlSet];
    const float* msg[kCtrlSet];
    const float* w[kCtrlSet];
    float* y[kCtrlSet];
};

__device__ __forceinline__ void cat_squares(const float* __restrict__ cur, const float* __restrict__ msg, int n_embd,
                                            int tid, float& a, float& b) {
    a = 0.0f;
    b = 0.0f;
    for (int i = tid; i < n_embd; i += kCatThreads) {
        a += cur[i] * cur[i];
        b += msg[i] * msg[i];
    }
}

__device__ __forceinline__ float cat_scale(float sum, int n_embd) {
    return rsqrtf(sum / (float) n_embd + 1e-6f);
}

__global__ void __launch_bounds__(kDownThreads) ink_ctrl_down_kernel(CtrlDown d, int n_embd, int rank) {
    __shared__ float pa[kCatThreads / WARP], pb[kCatThreads / WARP], part[32];
    const int j = blockIdx.x / rank, o = blockIdx.x % rank, tid = threadIdx.x;
    const size_t row = size_t(blockIdx.y) * n_embd;
    const float* __restrict__ cur = d.cur[j] + row;
    const float* __restrict__ msg = d.msg[j] + row;
    if (tid < kCatThreads) {
        float a, b;
        cat_squares(cur, msg, n_embd, tid, a, b);
        a = warp_sum(a);
        b = warp_sum(b);
        if ((tid & 31) == 0) pa[tid >> 5] = a, pb[tid >> 5] = b;
    }
    __syncthreads();
    float sa = 0.0f, sb = 0.0f;
    for (int k = 0; k < kCatThreads / WARP; ++k) sa += pa[k];
    for (int k = 0; k < kCatThreads / WARP; ++k) sb += pb[k];
    const float ia = cat_scale(sa, n_embd), ib = cat_scale(sb, n_embd);
    const float* __restrict__ wr = d.w[j] + size_t(o) * 2 * n_embd;
    const int n_in = 2 * n_embd;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int i = tid; i < n_in; i += 2 * kDownThreads) {
        const float x0 = i < n_embd ? cur[i] * ia : msg[i - n_embd] * ib;
        acc0 += wr[i] * x0;
        const int i1 = i + kDownThreads;
        if (i1 < n_in) {
            const float x1 = i1 < n_embd ? cur[i1] * ia : msg[i1 - n_embd] * ib;
            acc1 += wr[i1] * x1;
        }
    }
    acc0 = warp_sum(acc0);
    acc1 = warp_sum(acc1);
    if ((tid & 31) == 0) part[tid >> 5] = acc0, part[(tid >> 5) + kDownThreads / WARP] = acc1;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int k = 0; k < 2 * kDownThreads / WARP; ++k) sum += part[k];
        d.y[j][size_t(blockIdx.y) * rank + o] = sum;
    }
}

// STRATA_INK_FAST_CTRL=1 (opt-in, other bits): ink_ctrl_down_kernel in one pass over cur and msg.  Half the threads
// of a block, and cur and msg are read once a block instead of twice.
constexpr int kFastDownThreads = 256;
__global__ void __launch_bounds__(kFastDownThreads) ink_ctrl_down_fast_kernel(CtrlDown d, int n_embd, int rank) {
    __shared__ float part[kFastDownThreads / WARP][4];
    const int j = blockIdx.x / rank, o = blockIdx.x % rank, tid = threadIdx.x;
    const size_t row = size_t(blockIdx.y) * n_embd;
    const float* __restrict__ cur = d.cur[j] + row;
    const float* __restrict__ msg = d.msg[j] + row;
    const float* __restrict__ wl = d.w[j] + size_t(o) * 2 * n_embd;
    const float* __restrict__ wr = wl + n_embd;
    float dl = 0.0f, dr = 0.0f, cc = 0.0f, mm = 0.0f;
    for (int i = tid; i < n_embd; i += kFastDownThreads) {
        const float c = cur[i], m = msg[i];
        dl += wl[i] * c;
        dr += wr[i] * m;
        cc += c * c;
        mm += m * m;
    }
    dl = warp_sum(dl), dr = warp_sum(dr), cc = warp_sum(cc), mm = warp_sum(mm);
    if ((tid & 31) == 0) part[tid >> 5][0] = dl, part[tid >> 5][1] = dr, part[tid >> 5][2] = cc, part[tid >> 5][3] = mm;
    __syncthreads();
    if (tid == 0) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int w = 0; w < kFastDownThreads / WARP; ++w)
#pragma unroll
            for (int k = 0; k < 4; ++k) v[k] += part[w][k];
        d.y[j][size_t(blockIdx.y) * rank + o] = v[0] * cat_scale(v[2], n_embd) + v[1] * cat_scale(v[3], n_embd);
    }
}

int g_fast_ctrl = -1;
bool fast_ctrl_on() {
#if defined(__HIPCC__)
    return false;
#else
    if (g_fast_ctrl < 0) {
        const char* v = std::getenv("STRATA_INK_FAST_CTRL");
        g_fast_ctrl = v != nullptr && v[0] == '1';
    }
    return g_fast_ctrl == 1;
#endif
}

struct CtrlGate {
    const float* cur[kCtrlSet];
    const float* msg[kCtrlSet];
    const float* ch[kCtrlSet];
    const float* gf[kCtrlSet];
    float* out[kCtrlSet];
};

constexpr int kCtrlGateWarps = 8;
__global__ void __launch_bounds__(kCtrlGateWarps * WARP) ink_control3_gate_kernel(CtrlGate d, int count, int n_embd,
                                                                                 int rank) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.x * kCtrlGateWarps + warp;
    if (i >= n_embd) return;
    const int row = blockIdx.y;
    const size_t idx = size_t(row) * n_embd + i;

    float c_val[kCtrlSet];
    float m_val[kCtrlSet];
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < kCtrlSet; ++j) {
            if (j < count) {
                c_val[j] = d.cur[j][idx];
                m_val[j] = d.msg[j][idx];
            }
        }
    }

    float out_val[kCtrlSet];
    for (int j = 0; j < count; ++j) {
        const float* x = d.ch[j] + size_t(row) * rank;
        const float* wg = d.gf[j] + size_t(i) * rank;
        const float* wf = d.gf[j] + size_t(n_embd + i) * rank;
        float g = 0.0f;
        float f = 0.0f;
        for (int k = lane; k < rank; k += WARP) {
            const float xv = x[k];
            g += wg[k] * xv;
            f += wf[k] * xv;
        }
        g = warp_sum(g);
        f = warp_sum(f);
        if (lane == 0) {
            out_val[j] = c_val[j] * (1.0f - tanhf(f)) + m_val[j] * tanhf(g);
        }
    }

    if (lane == 0) {
        for (int j = 0; j < count; ++j) {
            d.out[j][idx] = out_val[j];
        }
    }
}

// ---- 4. Causal 4-tap short convolution (static + optional dynamic per-head reversed-lag taps) ----

__device__ __forceinline__ float* snap_of(float* snap, size_t stride, unsigned snaps, int row) {
    return row < 32 && (snaps >> row & 1u) ? snap + size_t(__popc(snaps & ((1u << row) - 1u))) * stride : nullptr;
}

__global__ void inkling_conv4_stream_kernel(const float* __restrict__ x, float* __restrict__ state,
                                            const float* __restrict__ w, float* __restrict__ r, int n_streams,
                                            int n_embd, int rows, int pre_copy_dst, bool replace_s0, int post_op,
                                            float* __restrict__ snap, size_t stride, unsigned snaps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_embd) return;
    const float* wc = w + 4 * size_t(c);
    float* st = state + size_t(c) * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    const float scale = rsqrtf(5.0f);
    for (int row = 0; row < rows; ++row) {
        const float v = x[size_t(row) * n_embd + c];
        const float y = (v + (s0 * wc[0] + s1 * wc[1] + s2 * wc[2] + v * wc[3])) * scale;
        s0 = s1;
        s1 = s2;
        s2 = v;
        if (float* m = snap_of(snap, stride, snaps, row)) {
            m += size_t(c) * 3;
            m[0] = s0;
            m[1] = s1;
            m[2] = s2;
        }
        float* r_row = r + size_t(row) * n_streams * n_embd + c;
        const float s0_old = r_row[0];
        if (pre_copy_dst >= 0) r_row[size_t(pre_copy_dst) * n_embd] = s0_old;
        const float s0_new = replace_s0 ? y : (s0_old + y);
        if (post_op == 0) {
            r_row[0] = s0_new;
        } else if (post_op == 1) {
            r_row[0] = s0_new;
            r_row[2 * size_t(n_embd)] = s0_new;
        } else if (post_op == 2) {
            r_row[4 * size_t(n_embd)] = r_row[3 * size_t(n_embd)];
            r_row[5 * size_t(n_embd)] = s0_new;
            r_row[0] = r_row[2 * size_t(n_embd)];
        }
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
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

__global__ void ink_conv_few_kernel(InkConv j0, InkConv j1, size_t stride, unsigned snaps, int channels, int head_dim,
                                    int rows) {
    const InkConv& j = blockIdx.y ? j1 : j0;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const float* wc = j.w + 4 * size_t(c);
    float* st = j.state + size_t(c) * 3;
    float s0 = st[0], s1 = st[1], s2 = st[2];
    const int n_heads = j.dyn ? channels / head_dim : 0;
    const int h = j.dyn ? c / head_dim : 0;
    const float scale = j.norm > 0.0f ? j.norm : (j.dyn ? (1.0f / 3.0f) : rsqrtf(5.0f));
    const int x_stride = j.x_stride > 0 ? j.x_stride : channels;

    for (int row = 0; row < rows; ++row) {
        const size_t at = size_t(row) * channels + c;
        const float v = j.x[size_t(row) * x_stride + c];
        float y = s0 * wc[0] + s1 * wc[1] + s2 * wc[2] + v * wc[3];
        if (j.dyn) {
            const float* d = j.dyn + (size_t(row) * n_heads + h) * 4;
            y += d[0] * v + d[1] * s2 + d[2] * s1 + d[3] * s0;
        }
        s0 = s1;
        s1 = s2;
        s2 = v;
        const float out_val = (v + y) * scale;
        j.out[at] = out_val;
        if (j.acc) {
            if (j.mode == 0) {
                j.acc[at] += out_val * j.a;
            } else if (j.mode == 1) {
                if (j.keep) j.keep[at] = j.acc[at];
                j.acc[at] = out_val;
            } else if (j.mode == 2) {
                j.acc[at] = out_val;
            }
        }
        if (j.snap) {
            if (float* m = snap_of(j.snap, stride, snaps, row)) {
                float* dst = m + size_t(c) * 3;
                dst[0] = s0;
                dst[1] = s1;
                dst[2] = s2;
            }
        }
    }
    st[0] = s0;
    st[1] = s1;
    st[2] = s2;
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

__device__ __forceinline__ float block128_sum(float v, float* sm) {
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = v;
    __syncthreads();
    return sm[0] + sm[1] + sm[2] + sm[3];
}

__device__ __forceinline__ void store_kv128(float v, uint16_t* fp16, int8_t* q8, uint8_t* q4, uint16_t* scales,
                                            size_t dst_row, int kv_width, int kv_nb, int h, int tid, int warp,
                                            int lane) {
    if (q8) {
        const float amax = warp_max(fabsf(v));
        const float d = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        if (lane == 0) scales[dst_row * kv_nb + h * 4 + warp] = __half_as_ushort(__float2half(d));
        float q = rintf(v * id);
        q = fminf(fmaxf(q, -127.0f), 127.0f);
        q8[dst_row * kv_width + h * 128 + tid] = (int8_t) q;
    } else if (q4) {
        const float amax = warp_max(fabsf(v));
        const half dh = __float2half(amax / 7.0f);
        const float df = __half2float(dh);
        const float id = df > 0.0f ? 1.0f / df : 0.0f;
        if (lane == 0) scales[dst_row * kv_nb + h * 4 + warp] = __half_as_ushort(dh);
        const float qf = fminf(fmaxf(rintf(v * id), -8.0f), 7.0f);
        const unsigned qi = unsigned(int(qf) + 8) & 0x0fu;
        const unsigned qhi = __shfl_down_sync(0xffffffffu, qi, 16);
        if (lane < 16) {
            q4[dst_row * (kv_width >> 1) + h * 64 + warp * 16 + lane] = uint8_t(qi | (qhi << 4));
        }
    } else {
        fp16[dst_row * kv_width + h * 128 + tid] = __half_as_ushort(__float2half(v));
    }
}

__global__ void inkling_clips_store_kernel(InkClipsStore c) {
    __shared__ float sm[4];
    const int b = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if (b < c.n_head) {
        const int h = b;
        const size_t base = (size_t(t) * c.n_head + h) * 128;
        float v = c.q_in[base + tid];
        if (c.q_scale) v *= c.q_scale[size_t(h) * 128 + tid];
        const float norm = sqrtf(block128_sum(v * v, sm)) + 1e-6f;
        float factor = fminf(1.0f, fmaxf(0.0f, c.qk_clip / norm));
        if (c.is_global) {
            const long long p = c.at ? c.at[2 * t] : c.pos + t;
            const float ratio = float(p + 1) / c.log_floor;
            if (ratio > 1.0f) factor *= (1.0f + c.log_alpha * logf(ratio));
        }
        c.q_out[base + tid] = v * factor;
    } else if (b < c.n_head + c.n_kv) {
        const int h = b - c.n_head;
        const int kv_width = c.n_kv * 128;
        const int kv_nb = c.n_kv * 4;
        const size_t base = size_t(t) * kv_width + size_t(h) * 128;
        const float v = c.k_in[base + tid];
        const float norm = sqrtf(block128_sum(v * v, sm)) + 1e-6f;
        const float factor = fminf(1.0f, fmaxf(0.0f, c.qk_clip / norm));
        const long long p_row = c.at ? (*c.at + t) : (c.pos + t);
        const size_t dst_row = size_t(c.ring > 0 ? p_row % c.ring : p_row);
        store_kv128(v * factor, c.k_fp16, c.k_q8, c.k_q4, c.k_scales, dst_row, kv_width, kv_nb, h, tid, warp, lane);
    } else if (b < c.n_head + 2 * c.n_kv) {
        const int h = b - c.n_head - c.n_kv;
        const int kv_width = c.n_kv * 128;
        const int kv_nb = c.n_kv * 4;
        const size_t base = size_t(t) * kv_width + size_t(h) * 128;
        const float v = c.v_in[base + tid];
        const float inv = rsqrtf(block128_sum(v * v, sm) / 128.0f + c.rms_eps);
        const long long p_row = c.at ? (*c.at + t) : (c.pos + t);
        const size_t dst_row = size_t(c.ring > 0 ? p_row % c.ring : p_row);
        store_kv128(v * inv, c.v_fp16, c.v_q8, c.v_q4, c.v_scales, dst_row, kv_width, kv_nb, h, tid, warp, lane);
    } else if (warp == 0) {
        const int h = b - c.n_head - 2 * c.n_kv;
        const size_t base = (size_t(t) * c.n_head + h) * c.n_rel;
        float v = 0.0f;
        if (lane < c.n_rel) {
            v = c.r_in[base + lane];
            if (c.r_scale) v *= c.r_scale[size_t(h) * c.n_rel + lane];
        }
        const float norm = sqrtf(warp_sum(v * v)) + 1e-6f;
        float factor = fminf(1.0f, fmaxf(0.0f, c.rel_clip / norm));
        if (c.is_global) {
            const long long p = c.at ? c.at[2 * t] : c.pos + t;
            const float ratio = float(p + 1) / c.log_floor;
            if (ratio > 1.0f) factor *= (1.0f + c.log_alpha * logf(ratio));
        }
        if (lane < c.n_rel) c.r_out[base + lane] = v * factor;
    }
}

// ---- 6. Per-query relative-position bias table ----

template <bool TermMajor>
__global__ void fixed_relative_bias(const float* __restrict__ query, const float* __restrict__ bias,
                                    const float* __restrict__ sinks, float* __restrict__ out,
                                    int n_extent, int n_head) {
    const size_t row = size_t(blockIdx.y) * n_head + blockIdx.x;
    const float* r = query + row * 16;
    float coef[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) coef[k] = r[k];
    for (int d = threadIdx.x; d < n_extent; d += blockDim.x) {
        float dot = 0.0f;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const size_t at = TermMajor ? size_t(k) * n_extent + d : size_t(d) * 16 + k;
            dot += coef[k] * bias[at];
        }
        out[row * n_extent + d] = dot - sinks[row];
    }
}

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

template <int PER, bool WHOLE = false, bool DIRECT_REL = false>
__global__ void inkling_attend_split_gqa_kernel(const float* __restrict__ q, const uint16_t* __restrict__ kcache,
                                                const uint16_t* __restrict__ vcache, const float* __restrict__ table,
                                                const float* __restrict__ sinks, int n_extent, long long pos,
                                                float* __restrict__ partials, int n_splits, int dim, int n_head,
                                                int kv_heads, float scale, int window, int ring,
                                                const long long* __restrict__ at, Kvq8 kq8, Kvq8 vq8,
                                                Kvq4 kq4, Kvq4 vq4, const float* __restrict__ r_cur,
                                                const float* __restrict__ rel_bias) {
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
    float r_lane[PER];
    float neg_sink[PER];
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
        if constexpr (DIRECT_REL) {
            const int h = h_base + p;
            r_lane[p] = (lane < 16) ? r_cur[(size_t(t) * n_head + h) * 16 + lane] : 0.0f;
            neg_sink[p] = -sinks[size_t(t) * n_head + h];
        }
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
        if constexpr (DIRECT_REL) {
            const float rb = (lane < 16 && dist < n_extent) ? __ldg(rel_bias + size_t(dist) * 16 + lane) : 0.0f;
#pragma unroll
            for (int p = 0; p < PER; ++p) dot[p] += r_lane[p] * rb;
        }
        float w_j[PER];
#pragma unroll
        for (int p = 0; p < PER; ++p) {
            const int h = h_base + p;
            const float dp = warp_sum(dot[p]);
            const float bias = WHOLE ? 0.0f
                                     : (DIRECT_REL ? neg_sink[p]
                                                   : (dist < n_extent ? table[(size_t(t) * n_head + h) * n_extent + dist]
                                                                      : -sinks[size_t(t) * n_head + h]));
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
                                           const long long* __restrict__ at, const float* __restrict__ gate,
                                           Q81Block* __restrict__ out_q8) {
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
    const size_t out_idx = (size_t(t) * n_head + h) * dim + d;
    float v = l > 0.0f ? acc / l : 0.0f;
    if (gate) v *= sigmoidf_d(gate[out_idx]);
    if (out) out[out_idx] = v;
    if (out_q8) {
        const int lane = d & 31;
        const float amax = warp_max(fabsf(v));
        const float bsum = warp_sum(v);
        const float sc = strata::kernels::q8_1_finite(amax / 127.0f);
        const int8_t q = strata::kernels::q8_1_quant(v, sc, amax);
        Q81Block& dst = out_q8[(size_t(t) * n_head + h) * (dim >> 5) + (d >> 5)];
        dst.qs[lane] = q;
        if (lane == 0) dst.ds = strata::kernels::q8_1_ds(sc, bsum);
    }
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

void situ_rows_q8_1(const float* gate_up, void* out_q8_1, int n, int rows, float beta, float linear_beta, void* stream,
                    const ExpertScales* scales) {
    if (rows <= 0) return;
    if (n <= 0 || (n & 31) != 0)
        throw std::invalid_argument("situ_rows_q8_1: n must be a positive multiple of 32");
    const ExpertScales s = scales ? *scales : ExpertScales{};
    situ_rows_q8_1_kernel<<<dim3(grid(n, 256), rows), 256, 0, cs(stream)>>>(
        gate_up, (Q81Block*) out_q8_1, n, beta, linear_beta, s);
    ck("situ_rows_q8_1");
}

void inkling_mix(const float* r, int streams, MixSlots slots, int n_slots, const float* query, int n_queries,
                 float* out, int n_embd, int rows, float eps, void* stream) {
    if (rows <= 0) return;
    if (n_slots < 1 || n_slots > 6 || n_queries < 1 || n_queries > 4)
        throw std::invalid_argument("inkling_mix: n_slots must be in 1..6 and n_queries in 1..4");

    static const bool s_mix_held = []{
        const char* e = std::getenv("STRATA_MIX_HELD");
        if (e) return std::strcmp(e, "1") == 0 || std::strcmp(e, "true") == 0 || std::strcmp(e, "on") == 0;
        const char* no = std::getenv("STRATA_NO_MIX_HELD");
        if (no && (std::strcmp(no, "1") == 0 || std::strcmp(no, "true") == 0)) return false;
        return true;
    }();

    if (s_mix_held && n_embd <= kMixHeld * kMixThreads) {
        launch_mix_held<MixHeldLauncher>(n_slots, n_queries, dim3(rows), cs(stream), r, streams, slots, query, out,
                                         n_embd, rows, eps);
    } else {
        inkling_mix_kernel<<<rows, 256, 0, cs(stream)>>>(r, streams, slots, n_slots, query, n_queries, out, n_embd, eps);
    }
    ck("inkling_mix");
}

bool inkling_mix_norm_q8(const float* r, int streams, MixSlots slots, int n_slots, const float* query,
                         const float* norm_w, float* xn_out, void* xq_out, int n_embd, int rows, float mix_eps,
                         float rms_eps, void* stream) {
    if (rows <= 0) return true;
    if (n_slots < 1 || n_slots > 6 || n_embd <= 0 || n_embd > kMixHeld * kMixThreads || (n_embd & 31) != 0)
        return false;
    Q81Block* xq = (Q81Block*) xq_out;
    switch (n_slots) {
        case 1: inkling_mix_norm_q8_kernel<1><<<rows, kMixThreads, 0, cs(stream)>>>(
            r, streams, slots, query, norm_w, xn_out, xq, n_embd, mix_eps, rms_eps); break;
        case 2: inkling_mix_norm_q8_kernel<2><<<rows, kMixThreads, 0, cs(stream)>>>(
            r, streams, slots, query, norm_w, xn_out, xq, n_embd, mix_eps, rms_eps); break;
        case 3: inkling_mix_norm_q8_kernel<3><<<rows, kMixThreads, 0, cs(stream)>>>(
            r, streams, slots, query, norm_w, xn_out, xq, n_embd, mix_eps, rms_eps); break;
        case 4: inkling_mix_norm_q8_kernel<4><<<rows, kMixThreads, 0, cs(stream)>>>(
            r, streams, slots, query, norm_w, xn_out, xq, n_embd, mix_eps, rms_eps); break;
        case 5: inkling_mix_norm_q8_kernel<5><<<rows, kMixThreads, 0, cs(stream)>>>(
            r, streams, slots, query, norm_w, xn_out, xq, n_embd, mix_eps, rms_eps); break;
        default: inkling_mix_norm_q8_kernel<6><<<rows, kMixThreads, 0, cs(stream)>>>(
            r, streams, slots, query, norm_w, xn_out, xq, n_embd, mix_eps, rms_eps); break;
    }
    ck("inkling_mix_norm_q8");
    return true;
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

void ink_set_fast_ctrl(bool on) { g_fast_ctrl = on ? 1 : 0; }

bool ink_ctrl_down(const float* const* cur, const float* const* msg, const float* const* down, float* const* ch,
                   int count, int n_embd, int rank, int rows, void* stream) {
    if (count < 1 || count > kCtrlSet) throw std::invalid_argument("ink_ctrl_down: 1 to 3 controllers");
    if (2 * n_embd < 1024) return false;
    if (rows <= 0) return true;
    if (rows >= 16) return false;
    CtrlDown d{};
    for (int j = 0; j < count; ++j) {
        d.cur[j] = cur[j];
        d.msg[j] = msg[j];
        d.w[j] = down[j];
        d.y[j] = ch[j];
    }
    if (fast_ctrl_on()) {
        ink_ctrl_down_fast_kernel<<<dim3(count * rank, rows), kFastDownThreads, 0, cs(stream)>>>(d, n_embd, rank);
        ck("ink_ctrl_down_fast");
        return true;
    }
    ink_ctrl_down_kernel<<<dim3(count * rank, rows), kDownThreads, 0, cs(stream)>>>(d, n_embd, rank);
    ck("ink_ctrl_down");
    return true;
}

bool ink_control3_gate(const float* const* cur, const float* const* msg, const float* const* ch, const float* const* gf,
                       float* const* out, int count, int n_embd, int rank, int rows, void* stream) {
    if (count < 1 || count > kCtrlSet) throw std::invalid_argument("ink_control3_gate: 1 to 3 controllers");
    if (rows <= 0) return true;
    if (rows >= CONV_ROWS || rank <= 0) return false;
    CtrlGate d{};
    for (int j = 0; j < count; ++j) {
        d.cur[j] = cur[j];
        d.msg[j] = msg[j];
        d.ch[j] = ch[j];
        d.gf[j] = gf[j];
        d.out[j] = out[j];
    }
    ink_control3_gate_kernel<<<dim3(grid(n_embd, kCtrlGateWarps), rows), kCtrlGateWarps * WARP, 0, cs(stream)>>>(
        d, count, n_embd, rank);
    ck("ink_control3_gate");
    return true;
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

void inkling_conv4_stream(const float* x, float* state, const float* w, float* r, int n_streams, int n_embd, int rows,
                          int pre_copy_dst, bool replace_s0, int post_op, void* stream, float* snap, size_t stride,
                          unsigned snaps) {
    if (rows <= 0) return;
    if (rows >= CONV_ROWS) throw std::invalid_argument("inkling_conv4_stream: requires rows < 16");
    inkling_conv4_stream_kernel<<<grid(n_embd, 256), 256, 0, cs(stream)>>>(
        x, state, w, r, n_streams, n_embd, rows, pre_copy_dst, replace_s0, post_op, snap, stride, snaps);
    ck("inkling_conv4_stream");
}

__global__ void inkling_unpack_q_fused_kernel(const float* __restrict__ src, float* __restrict__ qraw,
                                              float* __restrict__ gate, float* __restrict__ r_raw,
                                              float* __restrict__ sinks, int q_dim, int rel_dim, int n_head,
                                              int rel_off, int sink_off, int stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= stride) return;
    const int t = blockIdx.y;
    const float v = src[size_t(t) * stride + i];
    if (i < q_dim) {
        qraw[size_t(t) * q_dim + i] = v;
    } else if (i < 2 * q_dim) {
        gate[size_t(t) * q_dim + (i - q_dim)] = v;
    } else if (r_raw && i >= rel_off && i < rel_off + rel_dim) {
        r_raw[size_t(t) * rel_dim + (i - rel_off)] = v;
    } else if (sinks && i >= sink_off && i < sink_off + n_head) {
        sinks[size_t(t) * n_head + (i - sink_off)] = v;
    }
}

void inkling_unpack_q_fused(const float* src, float* qraw, float* gate, float* r_raw, float* sinks, int q_dim,
                            int rel_dim, int n_head, int rel_off, int sink_off, int stride, int rows, void* stream) {
    if (rows <= 0 || stride <= 0) return;
    inkling_unpack_q_fused_kernel<<<dim3(grid(stride, 256), rows), 256, 0, cs(stream)>>>(
        src, qraw, gate, r_raw, sinks, q_dim, rel_dim, n_head, rel_off, sink_off, stride);
    ck("inkling_unpack_q_fused");
}

void ink_conv_few(const InkConv& j0, const InkConv& j1, int channels, int head_dim, int rows, void* stream,
                  float* snap, size_t stride, unsigned snaps) {
    if (rows <= 0) return;
    const int num_jobs = j1.x ? 2 : 1;
    InkConv a0 = j0, a1 = j1;
    if (snap && snaps) {
        if (!a0.snap) a0.snap = snap;
        if (num_jobs == 2 && !a1.snap) a1.snap = snap;
    }
    ink_conv_few_kernel<<<dim3(grid(channels, 256), num_jobs), 256, 0, cs(stream)>>>(
        a0, a1, stride, snaps, channels, head_dim, rows);
    ck("ink_conv_few");
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

void inkling_clips_store(const InkClipsStore& c, void* stream) {
    if (c.rows <= 0) return;
    if (c.dim != 128 || c.n_rel <= 0 || c.n_rel > 32)
        throw std::invalid_argument("inkling_clips_store: requires dim=128 and 1 <= n_rel <= 32");
    const int n_blocks = 2 * c.n_head + 2 * c.n_kv;
    inkling_clips_store_kernel<<<dim3(n_blocks, c.rows), 128, 0, cs(stream)>>>(c);
    ck("inkling_clips_store");
}

void inkling_rel_table(const float* r_cur, const float* rel_bias, const float* sinks, float* table, int n_rel,
                       int n_extent, int n_head, int rows, void* stream, bool term_major) {
    if (rows <= 0) return;
    if (n_rel == 16) {
        if (term_major) {
            fixed_relative_bias<true><<<dim3(n_head, rows), 256, 0, cs(stream)>>>(
                r_cur, rel_bias, sinks, table, n_extent, n_head);
        } else {
            fixed_relative_bias<false><<<dim3(n_head, rows), 256, 0, cs(stream)>>>(
                r_cur, rel_bias, sinks, table, n_extent, n_head);
        }
    } else {
        inkling_rel_table_kernel<<<dim3(n_head, rows), 256, 0, cs(stream)>>>(r_cur, rel_bias, sinks, table, n_rel, n_extent,
                                                                            n_head);
    }
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
                    const long long* at, int room, Kvq8 kq8, Kvq8 vq8, Kvq4 kq4, Kvq4 vq4,
                    const float* r_cur, const float* rel_bias, int n_rel, const float* gate, void* out_q8) {
    (void) chunk;
    if (rows <= 0) return;
    if (dim > 128) throw std::invalid_argument("inkling_attend: head dim must be <= 128");
    if (out_q8 && (dim & 31) != 0) throw std::invalid_argument("inkling_attend: out_q8 requires dim % 32 == 0");
    const long long max_pos = at ? (long long) room : (long long) pos + rows - 1;
    const int span = window > 0 ? std::min<int>((int) (max_pos + 1), window) : (int) (max_pos + 1);
    const int n_splits = std::clamp((span + SPLIT_KEYS - 1) / SPLIT_KEYS, 1, MAX_SPLITS);
    if (!table) {
        if (!r_cur || !rel_bias || dim != 128 || n_head != kv_heads * 3 || n_rel != 16)
            throw std::invalid_argument("inkling_attend: direct relative bias requires dim=128, n_head=3*kv_heads, n_rel=16");
        inkling_attend_split_gqa_kernel<3, false, true><<<dim3(n_splits, kv_heads, rows), 128, 0, cs(stream)>>>(
            q, kcache, vcache, nullptr, sinks, n_extent, (long long) pos, score, n_splits, dim, n_head, kv_heads, scale,
            window, ring, at, kq8, vq8, kq4, vq4, r_cur, rel_bias);
    } else if (dim == 128 && n_head == kv_heads * 3) {
        inkling_attend_split_gqa_kernel<3, false, false><<<dim3(n_splits, kv_heads, rows), 128, 0, cs(stream)>>>(
            q, kcache, vcache, table, sinks, n_extent, (long long) pos, score, n_splits, dim, n_head, kv_heads, scale,
            window, ring, at, kq8, vq8, kq4, vq4, nullptr, nullptr);
    } else {
        inkling_attend_split_kernel<<<dim3(n_splits, n_head, rows), 128, 0, cs(stream)>>>(
            q, kcache, vcache, table, sinks, n_extent, (long long) pos, score, n_splits, dim, n_head, kv_heads, scale,
            window, ring, at, kq8, vq8, kq4, vq4);
    }
    inkling_attend_join_kernel<<<dim3(n_head, rows), dim, 0, cs(stream)>>>(score, out, n_splits, dim, n_head,
                                                                            (long long) pos, window, at, gate,
                                                                            (Q81Block*) out_q8);
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
        inkling_attend_split_gqa_kernel<3, true, false><<<dim3(n_splits, kv_heads, rows), 128, 0, cs(stream)>>>(
            q, kcache, vcache, nullptr, nullptr, 0, (long long) pos, score, n_splits, dim, n_head, kv_heads, scale,
            window, ring, at, kq8, vq8, kq4, vq4, nullptr, nullptr);
    } else {
        inkling_attend_split_kernel<true><<<dim3(n_splits, n_head, rows), 128, 0, cs(stream)>>>(
            q, kcache, vcache, nullptr, nullptr, 0, (long long) pos, score, n_splits, dim, n_head, kv_heads, scale,
            window, ring, at, kq8, vq8, kq4, vq4);
    }
    inkling_attend_join_kernel<false, true><<<dim3(n_head, rows), dim, 0, cs(stream)>>>(
        score, out, n_splits, dim, n_head, (long long) pos, window, at, nullptr, nullptr);
    ck("inkling_dspark_attend");
}

// The tensor cores' m16n8k16 step, ldmatrix and the copies into shared memory, as kernels.cu has them for its
// attn_flash kernels: from Ampere on (ink_attend_batch() asks the kernel's PTX version).
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800)
#define INK_FLASH 0
#else
#define INK_FLASH 1
#endif
#if INK_FLASH
__device__ __forceinline__ void ink_mma(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void ink_ldsm4(uint32_t* r, const __half* p, bool trans) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    if (trans)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
    else
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ uint32_t ink_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// 16 bytes from global to shared memory, or zeros where `whole` is false, a thread's copies a group until
// ink_cp_wait() (cp.async)
__device__ __forceinline__ void ink_cp16(__half* dst, const __half* src, bool whole) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"((unsigned) __cvta_generic_to_shared(dst)), "l"(src), "r"(whole ? 16 : 0));
}

__device__ __forceinline__ void ink_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int ahead> __device__ __forceinline__ void ink_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(ahead));
}
#endif

constexpr int kMaxPer = 4;
constexpr int kFlashKeys = 64, kFlashRow = 128 + 8, kFlashRel = 16;
constexpr size_t kFlashKv = size_t(4) * kFlashKeys * kFlashRow * sizeof(__half);
constexpr size_t kFlashKvQ = size_t(2) * kFlashKeys * kFlashRow * sizeof(__half);

constexpr size_t flash_smem(int g, bool quant, int tokens) {
    return (quant ? kFlashKvQ : kFlashKv) + size_t(tokens) * g * kFlashRel * sizeof(float);
}

template <int G, bool QUANT = false, int TOKENS = 64>
__global__ void __launch_bounds__(TOKENS / 16 * G * 32, 1) ink_flash_kernel(InkAttn a) {
    static_assert(TOKENS % 16 == 0, "a warp takes 16 tokens");
#if INK_FLASH
    constexpr int DIM = 128, KEYS = kFlashKeys, ROW = kFlashRow, NR = kFlashRel, THREADS = TOKENS / 16 * G * 32;
    constexpr int BUFS = QUANT ? 1 : 2;
    extern __shared__ __align__(16) __half ink_flash_sm[];
    __half(*ks)[KEYS][ROW] = reinterpret_cast<__half(*)[KEYS][ROW]>(ink_flash_sm);
    __half(*vs)[KEYS][ROW] = reinterpret_cast<__half(*)[KEYS][ROW]>(ink_flash_sm + BUFS * KEYS * ROW);
    float* rs = reinterpret_cast<float*>(ink_flash_sm + 2 * BUFS * KEYS * ROW);   // [token of the block][head of G][NR]
    const int g = blockIdx.y, t0 = blockIdx.x * TOKENS, lane = threadIdx.x % 32, w = threadIdx.x / 32;
    const int gid = lane / 4, tig = lane % 4, hg = w % G, sub = w / G, head = g * G + hg;
    const int tok[2] = {t0 + 16 * sub + gid, t0 + 16 * sub + gid + 8};
    for (int i = threadIdx.x; i < TOKENS * G * NR; i += THREADS) {
        const int t = t0 + i / (G * NR);
        rs[i] = t < a.rows ? a.r[(size_t(t) * a.n_head + g * G + i / NR % G) * NR + i % NR] : 0.0f;
    }
    uint32_t qa[8][4];   // a block of 16 values a step
#pragma unroll
    for (int k = 0; k < 8; ++k)
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                float2 x = make_float2(0.0f, 0.0f);
                if (tok[i] < a.rows)
                    x = *reinterpret_cast<const float2*>(a.q + (size_t(tok[i]) * a.n_head + head) * DIM + 16 * k +
                                                         8 * hi + 2 * tig);
                qa[k][i + 2 * hi] = ink_h2(x.x, x.y);
            }
    float sink[2];
    for (int i = 0; i < 2; ++i) sink[i] = tok[i] < a.rows ? a.sink[size_t(tok[i]) * a.n_head + head] : 0.0f;
    float o[16][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};
    for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    // the block's tokens attend positions [lo, hi): from the first one's window to the last one
    const long long hi = a.pos + min(t0 + TOKENS, a.rows);
    const long long lo = a.window > 0 ? max(0LL, a.pos + t0 - a.window + 1) : 0;
    const __half* kc = reinterpret_cast<const __half*>(a.keys);
    const __half* vc = reinterpret_cast<const __half*>(a.vals);
    const auto fetch = [&](int buf, long long k0) {
        for (int i = threadIdx.x; i < KEYS * DIM / 8; i += THREADS) {
            const int r = i / (DIM / 8), c = 8 * (i % (DIM / 8));
            const long long key = k0 + r;
            const bool whole = key >= lo && key < hi;   // the others are zeros, which no weight reads
            const size_t at = whole ? (size_t(a.ring > 0 ? key % a.ring : key) * a.n_kv + g) * DIM + c : 0;
            ink_cp16(&ks[buf][r][c], kc + at, whole);
            ink_cp16(&vs[buf][r][c], vc + at, whole);
        }
        ink_cp_commit();
    };
    const auto fetch_quant = [&](long long k0) {
        for (int i = threadIdx.x; i < KEYS * (DIM / 8); i += THREADS) {
            const int r = i / (DIM / 8), s = i % (DIM / 8);
            const int blk = s >> 2, off_in_blk = (s & 3) * 4;
            const long long key = k0 + r;
            const bool whole = key >= lo && key < hi;
            const size_t row_idx = whole ? size_t(a.ring > 0 ? key % a.ring : key) : 0;
            const size_t head_row = row_idx * a.n_kv + g;
            const size_t scale_idx = head_row * (DIM >> 5) + blk;
            const auto load_q = [&](__half* dst_row, const __half* fp16_cache, Kvq8 q8, Kvq4 q4) {
                if (!whole) {
                    *reinterpret_cast<uint4*>(dst_row + s * 8) = make_uint4(0, 0, 0, 0);
                    return;
                }
                if (q8.codes) {
                    const float sc = __half2float(__ushort_as_half(q8.scales[scale_idx]));
                    const char4 c0 = *reinterpret_cast<const char4*>(q8.codes + head_row * DIM + s * 8);
                    const char4 c1 = *reinterpret_cast<const char4*>(q8.codes + head_row * DIM + s * 8 + 4);
                    half2* h2 = reinterpret_cast<half2*>(dst_row + s * 8);
                    h2[0] = __floats2half2_rn(float(c0.x) * sc, float(c0.y) * sc);
                    h2[1] = __floats2half2_rn(float(c0.z) * sc, float(c0.w) * sc);
                    h2[2] = __floats2half2_rn(float(c1.x) * sc, float(c1.y) * sc);
                    h2[3] = __floats2half2_rn(float(c1.z) * sc, float(c1.w) * sc);
                    return;
                }
                if (q4.codes) {
                    const float sc = __half2float(__ushort_as_half(q4.scales[scale_idx]));
                    const uint32_t p4 = *reinterpret_cast<const uint32_t*>(q4.codes + head_row * (DIM >> 1) + blk * 16 + off_in_blk);
                    const int b0 = int(p4 & 0xffu);
                    const int b1 = int((p4 >> 8) & 0xffu);
                    const int b2 = int((p4 >> 16) & 0xffu);
                    const int b3 = int(p4 >> 24);
                    half2* lo_h2 = reinterpret_cast<half2*>(dst_row + blk * 32 + off_in_blk);
                    half2* hi_h2 = reinterpret_cast<half2*>(dst_row + blk * 32 + off_in_blk + 16);
                    lo_h2[0] = __floats2half2_rn(float((b0 & 0x0f) - 8) * sc, float((b1 & 0x0f) - 8) * sc);
                    lo_h2[1] = __floats2half2_rn(float((b2 & 0x0f) - 8) * sc, float((b3 & 0x0f) - 8) * sc);
                    hi_h2[0] = __floats2half2_rn(float((b0 >> 4) - 8) * sc, float((b1 >> 4) - 8) * sc);
                    hi_h2[1] = __floats2half2_rn(float((b2 >> 4) - 8) * sc, float((b3 >> 4) - 8) * sc);
                    return;
                }
                *reinterpret_cast<uint4*>(dst_row + s * 8) =
                    *reinterpret_cast<const uint4*>(fp16_cache + head_row * DIM + s * 8);
            };
            load_q(ks[0][r], kc, a.kq8, a.kq4);
            load_q(vs[0][r], vc, a.vq8, a.vq4);
        }
    };
    const long long first = lo / KEYS * KEYS;
    int buf = 0;
    if constexpr (!QUANT) fetch(0, first);
    for (long long k0 = first; k0 < hi; k0 += KEYS, buf = QUANT ? 0 : (buf ^ 1)) {
        __syncthreads();   // the buffer the last block was worked on in is free (and rs is written)
        if constexpr (QUANT) {
            fetch_quant(k0);
            __syncthreads();
        } else {
            if (k0 + KEYS < hi) {
                fetch(buf ^ 1, k0 + KEYS);
                ink_cp_wait<1>();
            } else {
                ink_cp_wait<0>();
            }
            __syncthreads();
        }
        // the warp's rows' positions own0..own1: skip tiles this warp cannot attend, and elide edge masks inside full tiles
        const int w0 = t0 + 16 * sub, w1 = min(w0 + 15, a.rows - 1);
        const long long own0 = a.pos + w0, own1 = a.pos + w1;
        if (w0 >= a.rows || k0 > own1 || (a.window > 0 && k0 + KEYS - 1 <= own0 - a.window)) continue;
        const bool edge = w0 + 15 >= a.rows || k0 + KEYS - 1 > own0 || (a.window > 0 && k0 <= own1 - a.window);
        float s[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
            for (int k = 0; k < 8; k += 2) {
                uint32_t b[4];   // keys 8 n.., values 16 k.. and 16 (k + 1)..
                ink_ldsm4(b, &ks[buf][8 * n + lane % 8][16 * k + 8 * (lane / 8)], false);
                ink_mma(s[n], qa[k], b);
                ink_mma(s[n], qa[k + 1], b + 2);
            }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const long long own = a.pos + tok[i];   // the row's token attends positions (own - window, own]
            float rr[NR];
            const float* rp = rs + ((16 * sub + gid + 8 * i) * G + hg) * NR;
#pragma unroll
            for (int q = 0; q < NR; ++q) rr[q] = rp[q];
            float mx = -INFINITY;
#pragma unroll
            for (int n = 0; n < 8; ++n)
#pragma unroll
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    const long long key = k0 + 8 * n + 2 * tig + j % 2;
                    float v = -INFINITY;
                    if (!edge || (tok[i] < a.rows && key <= own && (a.window <= 0 || key > own - a.window))) {
                        const long long d = own - key;
                        float add = 0.0f;
                        if (d < a.extent) {
                            const float4* col = reinterpret_cast<const float4*>(a.rel_bias + size_t(d) * NR);
#pragma unroll
                            for (int q = 0; q < NR / 4; ++q) {
                                const float4 c = __ldg(col + q);
                                add += c.x * rr[4 * q];
                                add += c.y * rr[4 * q + 1];
                                add += c.z * rr[4 * q + 2];
                                add += c.w * rr[4 * q + 3];
                            }
                        }
                        v = s[n][j] * a.scale;
                        v += add - sink[i];
                    }
                    s[n][j] = v;
                    mx = fmaxf(mx, v);
                }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float now = fmaxf(m[i], mx), base = now == -INFINITY ? 0.0f : now;
            const float f = expf(m[i] - base);
            m[i] = now;
            l[i] *= f;
            if (__any_sync(0xffffffffu, f != 1.0f))
#pragma unroll
                for (int n = 0; n < 16; ++n) {
                    o[n][2 * i] *= f;
                    o[n][2 * i + 1] *= f;
                }
#pragma unroll
            for (int n = 0; n < 8; ++n)
#pragma unroll
                for (int j = 2 * i; j < 2 * i + 2; ++j) {
                    s[n][j] = __half2float(__float2half(expf(s[n][j] - base)));   // the weight the values get
                    l[i] += s[n][j];
                }
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {   // keys 16 k..: the weights as an A fragment
            const uint32_t pa[4] = {ink_h2(s[2 * k][0], s[2 * k][1]), ink_h2(s[2 * k][2], s[2 * k][3]),
                                    ink_h2(s[2 * k + 1][0], s[2 * k + 1][1]), ink_h2(s[2 * k + 1][2], s[2 * k + 1][3])};
#pragma unroll
            for (int n = 0; n < 16; n += 2) {
                uint32_t b[4];   // keys 16 k.., values 8 n.. and 8 (n + 1)..
                ink_ldsm4(b, &vs[buf][16 * k + lane % 8 + 8 * (lane / 8 % 2)][8 * n + 8 * (lane / 16)], true);
                ink_mma(o[n], pa, b);
                ink_mma(o[n + 1], pa, b + 2);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffffu, l[i], 2);
        if (tok[i] >= a.rows) continue;
        const float inv = 1.0f / (l[i] + expf(0.0f - m[i]));   // the term of score 0, against the row's top
        const size_t at = (size_t(tok[i]) * a.n_head + head) * DIM + 2 * tig;
#pragma unroll
        for (int n = 0; n < 16; ++n)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                float v = o[n][2 * i + e] * inv;
                if (a.gate) v *= 1.0f / (1.0f + expf(-a.gate[at + 8 * n + e]));   // ink_mul_sigmoid_kernel's
                a.out[at + 8 * n + e] = v;
            }
    }
#else
    (void) a;   // ink_attend_batch() launches it only where the PTX has the steps
#endif
}

#if !defined(__HIPCC__)
namespace {

constexpr int kFlashSizes[] = {64, 48, 32};

int flash_size_index(int tokens) { return tokens == 64 ? 0 : tokens == 48 ? 1 : 2; }

template <int G, bool Q, int T> void* flash_fn() { return reinterpret_cast<void*>(ink_flash_kernel<G, Q, T>); }

void* flash_kernel(int g, bool quant, int tokens) {
    static void* const fns[4][2][3] = {
        {{flash_fn<1, false, 64>(), flash_fn<1, false, 48>(), flash_fn<1, false, 32>()},
         {flash_fn<1, true, 64>(), flash_fn<1, true, 48>(), flash_fn<1, true, 32>()}},
        {{flash_fn<2, false, 64>(), flash_fn<2, false, 48>(), flash_fn<2, false, 32>()},
         {flash_fn<2, true, 64>(), flash_fn<2, true, 48>(), flash_fn<2, true, 32>()}},
        {{flash_fn<3, false, 64>(), flash_fn<3, false, 48>(), flash_fn<3, false, 32>()},
         {flash_fn<3, true, 64>(), flash_fn<3, true, 48>(), flash_fn<3, true, 32>()}},
        {{flash_fn<4, false, 64>(), flash_fn<4, false, 48>(), flash_fn<4, false, 32>()},
         {flash_fn<4, true, 64>(), flash_fn<4, true, 48>(), flash_fn<4, true, 32>()}},
    };
    return fns[g - 1][quant ? 1 : 0][flash_size_index(tokens)];
}

int flash_resident(int g, bool quant, int tokens) {
    constexpr int kDevs = 16;
    static std::atomic<int> cache[kDevs][4][2][3] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) dev = 0;
    std::atomic<int>* c = dev >= 0 && dev < kDevs ? &cache[dev][g - 1][quant ? 1 : 0][flash_size_index(tokens)] : nullptr;
    if (c && c->load(std::memory_order_relaxed) > 0) return c->load(std::memory_order_relaxed) - 1;
    void* fn = flash_kernel(g, quant, tokens);
    const size_t smem = flash_smem(g, quant, tokens);
    const cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("ink_attend_batch: shared memory: ") + cudaGetErrorString(e));
    int n = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, tokens / 16 * g * 32, smem) != cudaSuccess) n = 0;
    if (c) c->store(1 + n, std::memory_order_relaxed);
    return n;
}

int flash_sms() {
    int dev = 0, n = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    return cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess ? n : 0;
}

int flash_pick(const InkAttn& a, int per, bool quant) {
    static const int fixed = [] {
        const char* v = std::getenv("STRATA_INK_FLASH_TOKENS");
        const int t = v ? std::atoi(v) : 0;
        return t == 64 || t == 48 || t == 32 ? t : 0;
    }();
    if (fixed) return fixed;
    constexpr double kFlashCost[] = {1.0, 0.9, 0.65}, kFlashGain = 0.05;
    const int sms = flash_sms();
    if (sms < 1 || a.rows <= 0) return 64;
    double best = 0.0, main_cost = 0.0;
    int pick = 64;
    for (int i = 0; i < 3; ++i) {
        const int t = kFlashSizes[i], res = flash_resident(per, quant, t);
        if (res < 1) continue;
        const long long blocks = (long long) grid((size_t) a.rows, t) * a.n_kv, slots = (long long) sms * res;
        const double cost = (double) ((blocks + slots - 1) / slots) * res * kFlashCost[i];
        if (t == 64) {
            main_cost = best = cost;
        } else if (main_cost > 0.0 && cost < best && cost <= main_cost * (1.0 - kFlashGain)) {
            best = cost;
            pick = t;
        }
    }
    return pick;
}

}  // namespace
#endif

bool ink_attend_batch(const InkAttn& a, void* stream) {
#if defined(__HIPCC__)
    (void) a, (void) stream;
    return false;   // the kernel is NVIDIA's tensor cores' (INK_FLASH)
#else
    if (a.rows <= 0) return true;
    static const bool off = [] {   // STRATA_INK_FLASH=0: attend_batch()'s path
        const char* v = std::getenv("STRATA_INK_FLASH");
        return v && v[0] == '0';
    }();
    const int per = a.n_kv > 0 && a.n_head % a.n_kv == 0 ? a.n_head / a.n_kv : 0;
    if (off) return false;
    if (a.dim != 128 || a.n_rel != kFlashRel || per < 1 || per > kMaxPer || a.at || a.q8 || a.kv8) return false;
    static const int ptx = [] {
        cudaFuncAttributes f{};
        return cudaFuncGetAttributes(&f, ink_flash_kernel<1, false>) == cudaSuccess ? f.ptxVersion : 0;
    }();
    if (ptx < 80) return false;
    const bool has_quant = a.kq8.codes || a.vq8.codes || a.kq4.codes || a.vq4.codes;
    const int tokens = flash_pick(a, per, has_quant);
    flash_resident(per, has_quant, tokens);
    InkAttn arg = a;
    void* args[] = {&arg};
    const cudaError_t e = cudaLaunchKernel(flash_kernel(per, has_quant, tokens),
                                           dim3(grid((size_t) a.rows, tokens), a.n_kv),
                                           dim3(tokens / 16 * per * 32), args,
                                           flash_smem(per, has_quant, tokens), cs(stream));
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("ink_attend_batch: launch: ") + cudaGetErrorString(e));
    ck("ink_flash");
    return true;
#endif
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
