// src/poly/inkling2_test.cu - Inkling 2 CUDA kernels & CPU SiTU parity against float64 reference math.
#include "cpu_experts.hpp"
#include "strata/poly/kernels.hpp"
#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

std::mt19937 rng(42);

std::vector<float> random_vec(size_t n, float scale = 1.f) {
    std::normal_distribution<float> nd(0.f, scale);
    std::vector<float> v(n);
    for (float& x : v) x = nd(rng);
    return v;
}

template <class T>
T* dev(const std::vector<T>& v) {
    T* p = nullptr;
    cudaMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(T));
    if (!v.empty()) cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return p;
}

template <class T>
std::vector<T> host(const T* p, size_t n) {
    std::vector<T> v(n);
    if (n > 0) cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

int report(const char* what, const std::vector<float>& got, const std::vector<float>& want, double tol) {
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
        worst = std::max(worst, std::abs((double) got[i] - (double) want[i]) / (1.0 + std::abs((double) want[i])));
    const bool ok = worst <= tol;
    std::printf("%-42s %8zu vals  err=%.3g  %s\n", what, got.size(), worst, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int test_situ_gpu_and_cpu() {
    int bad = 0;
    const int rows = 5, n_ff = 64, n_embd = 32;
    const float beta = 4.0f, lbeta = 25.0f;

    auto gu = random_vec((size_t) rows * 2 * n_ff, 3.0f);
    auto sg = random_vec(8, 0.5f);
    auto su = random_vec(8, 0.5f);
    for (auto& v : sg) v = std::abs(v) + 0.1f;
    for (auto& v : su) v = std::abs(v) + 0.1f;
    std::vector<int32_t> ids = {2, 5, 0, 7, 3};

    float* d_gu = dev(gu);
    float* d_out = dev(std::vector<float>((size_t) rows * n_ff, 0.f));
    float* d_sg = dev(sg);
    float* d_su = dev(su);
    int32_t* d_ids = dev(ids);
    ExpertScales es{d_sg, d_su, d_ids};

    situ_rows(d_gu, d_out, n_ff, rows, beta, lbeta, nullptr, &es);
    auto got = host(d_out, (size_t) rows * n_ff);
    std::vector<float> want((size_t) rows * n_ff);
    for (int r = 0; r < rows; ++r) {
        const int e = ids[(size_t) r];
        for (int i = 0; i < n_ff; ++i) {
            const double g = (double) gu[(size_t) r * 2 * n_ff + i] * sg[(size_t) e];
            const double u = (double) gu[(size_t) r * 2 * n_ff + n_ff + i] * su[(size_t) e];
            const double sig = 1.0 / (1.0 + std::exp(-g));
            const double act = beta * std::tanh(g / beta) * sig;
            const double upv = lbeta * std::tanh(u / lbeta);
            want[(size_t) r * n_ff + i] = (float) (act * upv);
        }
    }
    bad += report("situ_rows (with ExpertScales)", got, want, 1e-5);

    // CPU SiTU check with F32 weights (two distinct experts, each with its own weight buffers and scale)
    auto x = random_vec((size_t) 2 * n_embd, 1.f);
    auto w_g0 = random_vec((size_t) n_ff * n_embd, 0.3f);
    auto w_u0 = random_vec((size_t) n_ff * n_embd, 0.3f);
    auto w_d0 = random_vec((size_t) n_embd * n_ff, 0.3f);
    auto w_g1 = random_vec((size_t) n_ff * n_embd, 0.3f);
    auto w_u1 = random_vec((size_t) n_ff * n_embd, 0.3f);
    auto w_d1 = random_vec((size_t) n_embd * n_ff, 0.3f);
    std::vector<CpuExpert> exps = {
        {(const uint8_t*) w_g0.data(), (const uint8_t*) w_u0.data(), (const uint8_t*) w_d0.data(), 1.5f, 0, 1.2f, 0.8f},
        {(const uint8_t*) w_g1.data(), (const uint8_t*) w_u1.data(), (const uint8_t*) w_d1.data(), 0.7f, 1, 0.9f, 1.1f},
    };
    Workers workers(2);
    std::vector<float> cpu_out((size_t) 2 * n_embd, 0.f);
    cpu_experts(workers, exps.data(), (int) exps.size(), 2, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32,
                n_embd, n_ff, x.data(), INFINITY, false, cpu_out.data(), nullptr, nullptr, nullptr, beta, lbeta);
    std::vector<float> want_cpu((size_t) 2 * n_embd, 0.f);
    for (int row = 0; row < 2; ++row) {
        const auto& ex = exps[(size_t) row];
        const auto& wg = row == 0 ? w_g0 : w_g1;
        const auto& wu = row == 0 ? w_u0 : w_u1;
        const auto& wd = row == 0 ? w_d0 : w_d1;
        std::vector<double> hid(n_ff);
        for (int j = 0; j < n_ff; ++j) {
            double dg = 0.0, du = 0.0;
            for (int k = 0; k < n_embd; ++k) {
                dg += (double) wg[(size_t) j * n_embd + k] * x[(size_t) row * n_embd + k];
                du += (double) wu[(size_t) j * n_embd + k] * x[(size_t) row * n_embd + k];
            }
            dg *= ex.gate_scale;
            du *= ex.up_scale;
            const double sig = 1.0 / (1.0 + std::exp(-dg));
            hid[(size_t) j] = (beta * std::tanh(dg / beta) * sig) * (lbeta * std::tanh(du / lbeta));
        }
        for (int r = 0; r < n_embd; ++r) {
            double acc = 0.0;
            for (int j = 0; j < n_ff; ++j) acc += (double) wd[(size_t) r * n_ff + j] * hid[(size_t) j];
            want_cpu[(size_t) row * n_embd + r] = (float) (ex.weight * acc);
        }
    }
    bad += report("cpu_experts SiTU", cpu_out, want_cpu, 1e-5);

    cudaFree(d_gu);
    cudaFree(d_out);
    cudaFree(d_sg);
    cudaFree(d_su);
    cudaFree(d_ids);
    return bad;
}

int test_mix_and_control() {
    int bad = 0;
    const int streams = 6, n_embd = 256, rows = 4, n_slots = 4, n_queries = 3;
    const float eps = 1e-6f;
    auto r = random_vec((size_t) rows * streams * n_embd, 1.f);
    auto q = random_vec((size_t) n_queries * n_embd, 0.5f);
    MixSlots slots{{0, 1, 2, 3, 0, 0}};

    float* d_r = dev(r);
    float* d_q = dev(q);
    float* d_out = dev(std::vector<float>((size_t) n_queries * rows * n_embd, 0.f));
    inkling_mix(d_r, streams, slots, n_slots, d_q, n_queries, d_out, n_embd, rows, eps, nullptr);
    auto got = host(d_out, (size_t) n_queries * rows * n_embd);

    std::vector<float> want((size_t) n_queries * rows * n_embd, 0.f);
    for (int t = 0; t < rows; ++t) {
        std::vector<std::vector<double>> normed(n_slots, std::vector<double>(n_embd));
        for (int k = 0; k < n_slots; ++k) {
            const float* src = r.data() + ((size_t) t * streams + slots.s[k]) * n_embd;
            double sq = 0.0;
            for (int i = 0; i < n_embd; ++i) sq += (double) src[i] * src[i];
            const double inv = 1.0 / std::sqrt(sq / n_embd + eps);
            for (int i = 0; i < n_embd; ++i) normed[(size_t) k][(size_t) i] = src[i] * inv;
        }
        for (int qi = 0; qi < n_queries; ++qi) {
            std::vector<double> sc(n_slots);
            double mx = -1e30;
            for (int k = 0; k < n_slots; ++k) {
                double dot = 0.0;
                for (int i = 0; i < n_embd; ++i)
                    dot += normed[(size_t) k][(size_t) i] * q[(size_t) qi * n_embd + i];
                sc[(size_t) k] = dot / std::sqrt((double) n_embd);
                mx = std::max(mx, sc[(size_t) k]);
            }
            double sum = 0.0;
            for (int k = 0; k < n_slots; ++k) { sc[(size_t) k] = std::exp(sc[(size_t) k] - mx); sum += sc[(size_t) k]; }
            for (int i = 0; i < n_embd; ++i) {
                double acc = 0.0;
                for (int k = 0; k < n_slots; ++k) acc += (sc[(size_t) k] / sum) * normed[(size_t) k][(size_t) i];
                want[((size_t) qi * rows + t) * n_embd + i] = (float) acc;
            }
        }
    }
    bad += report("inkling_mix (4 slots, 3 queries)", got, want, 1e-5);

    auto cur = random_vec((size_t) rows * n_embd, 1.f);
    auto msg = random_vec((size_t) rows * n_embd, 1.f);
    auto gain = random_vec((size_t) rows * n_embd, 0.7f);
    auto forg = random_vec((size_t) rows * n_embd, 0.7f);
    float *d_c = dev(cur), *d_m = dev(msg), *d_cat = dev(std::vector<float>((size_t) rows * 2 * n_embd, 0.f));
    float *d_g = dev(gain), *d_f = dev(forg), *d_co = dev(std::vector<float>((size_t) rows * n_embd, 0.f));

    inkling_control_prep(d_c, d_m, d_cat, n_embd, rows, eps, nullptr);
    inkling_control_apply(d_c, d_m, d_g, d_f, d_co, n_embd, rows, nullptr);
    auto got_cat = host(d_cat, (size_t) rows * 2 * n_embd);
    auto got_co = host(d_co, (size_t) rows * n_embd);
    std::vector<float> want_cat((size_t) rows * 2 * n_embd), want_co((size_t) rows * n_embd);
    for (int t = 0; t < rows; ++t) {
        double sqc = 0.0, sqm = 0.0;
        for (int i = 0; i < n_embd; ++i) {
            sqc += (double) cur[(size_t) t * n_embd + i] * cur[(size_t) t * n_embd + i];
            sqm += (double) msg[(size_t) t * n_embd + i] * msg[(size_t) t * n_embd + i];
        }
        const double ic = 1.0 / std::sqrt(sqc / n_embd + eps), im = 1.0 / std::sqrt(sqm / n_embd + eps);
        for (int i = 0; i < n_embd; ++i) {
            const size_t idx = (size_t) t * n_embd + i;
            want_cat[(size_t) t * 2 * n_embd + i] = (float) (cur[idx] * ic);
            want_cat[(size_t) t * 2 * n_embd + n_embd + i] = (float) (msg[idx] * im);
            want_co[idx] = (float) (cur[idx] * (1.0 - std::tanh((double) forg[idx])) +
                                    msg[idx] * std::tanh((double) gain[idx]));
        }
    }
    bad += report("inkling_control_prep", got_cat, want_cat, 1e-5);
    bad += report("inkling_control_apply", got_co, want_co, 1e-5);

    cudaFree(d_r); cudaFree(d_q); cudaFree(d_out);
    cudaFree(d_c); cudaFree(d_m); cudaFree(d_cat); cudaFree(d_g); cudaFree(d_f); cudaFree(d_co);
    return bad;
}

int test_conv4() {
    int bad = 0;
    const int head_dim = 64, n_heads = 4, channels = head_dim * n_heads, rows = 20;
    auto st0 = random_vec((size_t) 3 * channels, 0.5f);
    auto w = random_vec((size_t) 4 * channels, 0.3f);
    auto x = random_vec((size_t) rows * channels, 1.f);
    auto dyn = random_vec((size_t) rows * n_heads * 4, 0.2f);

    auto ref_conv = [&](bool use_dyn) {
        std::vector<float> st = st0, out((size_t) rows * channels);
        const double scale = 1.0 / std::sqrt(use_dyn ? 9.0 : 5.0);
        for (int t = 0; t < rows; ++t) {
            for (int c = 0; c < channels; ++c) {
                const float* wc = w.data() + 4 * c;
                double s0 = st[3 * c + 0], s1 = st[3 * c + 1], s2 = st[3 * c + 2], v = x[(size_t) t * channels + c];
                double y = s0 * wc[0] + s1 * wc[1] + s2 * wc[2] + v * wc[3];
                if (use_dyn) {
                    const float* d = dyn.data() + ((size_t) t * n_heads + c / head_dim) * 4;
                    y += d[0] * v + d[1] * s2 + d[2] * s1 + d[3] * s0;
                }
                st[3 * c + 0] = (float) s1; st[3 * c + 1] = (float) s2; st[3 * c + 2] = (float) v;
                out[(size_t) t * channels + c] = (float) ((v + y) * scale);
            }
        }
        return std::pair{out, st};
    };

    // 1. Dynamic batch path (rows = 20 >= 16)
    float *d_x = dev(x), *d_w = dev(w), *d_dyn = dev(dyn), *d_st = dev(st0);
    float *d_out = dev(std::vector<float>((size_t) rows * channels, 0.f));
    inkling_conv4(d_x, d_st, d_w, d_dyn, head_dim, d_out, channels, rows, nullptr);
    auto [want_dyn, want_st_dyn] = ref_conv(true);
    bad += report("inkling_conv4 dynamic batch out", host(d_out, x.size()), want_dyn, 1e-5);
    bad += report("inkling_conv4 dynamic batch state", host(d_st, st0.size()), want_st_dyn, 1e-5);

    // 2. Static walk + snapshots (rows = 6, snap after tok 1 & tok 4)
    const int wrows = 6;
    const unsigned snaps = (1u << 1) | (1u << 4);
    cudaMemcpy(d_st, st0.data(), st0.size() * sizeof(float), cudaMemcpyHostToDevice);
    float* d_snap = dev(std::vector<float>(2 * st0.size(), 0.f));
    inkling_conv4(d_x, d_st, d_w, nullptr, 0, d_out, channels, wrows, nullptr, d_snap, st0.size(), snaps);
    auto [want_st, _unused] = ref_conv(false);
    want_st.resize((size_t) wrows * channels);
    bad += report("inkling_conv4 static walk out", host(d_out, (size_t) wrows * channels), want_st, 1e-5);

    // Check snapshot after tok 1 matches running 2 rows alone
    float* d_st2 = dev(st0);
    inkling_conv4(d_x, d_st2, d_w, nullptr, 0, d_out, channels, 2, nullptr);
    bad += report("inkling_conv4 snap rollback tok 1", host(d_snap, st0.size()), host(d_st2, st0.size()), 1e-6);

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_dyn); cudaFree(d_st); cudaFree(d_out); cudaFree(d_snap); cudaFree(d_st2);
    return bad;
}

int test_l2_clip() {
    int bad = 0;
    const int dim = 128, heads = 24, rows = 4, floor_p = 10;
    const float threshold = 22.6274166f, alpha = 0.1f;
    const int64_t pos = 18;
    auto x = random_vec((size_t) rows * heads * dim, 3.0f);
    auto scale = random_vec((size_t) heads * dim, 1.0f);
    long long at_h[8] = {pos, pos, pos + 1, pos + 1, pos + 2, pos + 2, pos + 3, pos + 3};

    float *d_x = dev(x), *d_sc = dev(scale), *d_out = dev(std::vector<float>(x.size(), 0.f));
    long long* d_at = dev(std::vector<long long>(at_h, at_h + 8));

    inkling_l2_clip(d_x, d_sc, d_out, dim, heads, threshold, pos, true, (float) floor_p, alpha, rows, d_at, nullptr);
    auto got = host(d_out, x.size());

    std::vector<float> want(x.size());
    for (int t = 0; t < rows; ++t) {
        const double ratio = (double) (pos + t + 1) / floor_p;
        const double tau = ratio > 1.0 ? 1.0 + alpha * std::log(ratio) : 1.0;
        for (int h = 0; h < heads; ++h) {
            const size_t base = ((size_t) t * heads + h) * dim;
            double sq = 0.0;
            for (int d = 0; d < dim; ++d) {
                const double v = (double) x[base + d] * scale[(size_t) h * dim + d];
                sq += v * v;
            }
            const double norm = std::sqrt(sq) + 1e-6;
            const double factor = std::min(1.0, std::max(0.0, threshold / norm));
            for (int d = 0; d < dim; ++d)
                want[base + d] = (float) ((double) x[base + d] * scale[(size_t) h * dim + d] * factor * tau);
        }
    }
    bad += report("inkling_l2_clip (global + scale + at)", got, want, 1e-5);
    cudaFree(d_x); cudaFree(d_sc); cudaFree(d_out); cudaFree(d_at);
    return bad;
}

int test_rel_and_attend() {
    int bad = 0;
    const int dim = 128, n_rel = 16, n_extent = 1024;
    const float scale = 1.0f / std::sqrt((float) dim);

    enum class Mode { kFp16, kQ8, kQ4 };
    auto run_case = [&](const char* label, int n_head, int kv_heads, int total_keys, int window, int ring, int q_rows,
                        Mode mode, bool use_at, bool via_gpu_store) {
        const int width = kv_heads * dim;
        const int cache_rows = ring > 0 ? ring : total_keys + q_rows;
        auto q = random_vec((size_t) q_rows * n_head * dim, 0.5f);
        auto r_cur = random_vec((size_t) q_rows * n_head * n_rel, 0.5f);
        auto rel_bias = random_vec((size_t) n_extent * n_rel, 0.25f);
        auto sinks = random_vec((size_t) q_rows * n_head, 0.5f);

        auto k_f32 = random_vec((size_t) total_keys * width, 0.5f);
        auto v_f32 = random_vec((size_t) total_keys * width, 0.5f);
        auto q_new_k = random_vec((size_t) q_rows * width, 0.5f);
        auto q_new_v = random_vec((size_t) q_rows * width, 0.5f);

        std::vector<uint16_t> kf16((size_t) cache_rows * width, 0), vf16((size_t) cache_rows * width, 0);
        std::vector<int8_t> kc((size_t) cache_rows * width, 0), vc((size_t) cache_rows * width, 0);
        std::vector<uint8_t> k4((size_t) cache_rows * (width / 2), 0), v4((size_t) cache_rows * (width / 2), 0);
        std::vector<uint16_t> ks((size_t) cache_rows * (width / 32), 0), vs((size_t) cache_rows * (width / 32), 0);
        std::vector<uint16_t> ks4((size_t) cache_rows * (width / 32), 0), vs4((size_t) cache_rows * (width / 32), 0);

        auto write_row = [&](int64_t j, const float* kp, const float* vp) {
            const int r = ring > 0 ? (int) (j % ring) : (int) j;
            for (int c = 0; c < width; ++c) {
                kf16[(size_t) r * width + c] = __half_as_ushort(__float2half(kp[c]));
                vf16[(size_t) r * width + c] = __half_as_ushort(__float2half(vp[c]));
            }
            for (int b = 0; b < width / 32; ++b) {
                auto q8block = [&](const float* src, int8_t* dstc, uint16_t* dsts) {
                    float amax = 0.f;
                    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::abs(src[b * 32 + i]));
                    const float d = amax / 127.f;
                    const uint16_t dh = __half_as_ushort(__float2half(d));
                    const float df = __half2float(__ushort_as_half(dh));
                    dsts[(size_t) r * (width / 32) + b] = dh;
                    for (int i = 0; i < 32; ++i)
                        dstc[(size_t) r * width + b * 32 + i] = df > 0.f ? (int8_t) std::llround(src[b * 32 + i] / df) : 0;
                };
                auto q4block = [&](const float* src, uint8_t* dstc, uint16_t* dsts) {
                    float amax = 0.f;
                    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::abs(src[b * 32 + i]));
                    const uint16_t dh = __half_as_ushort(__float2half(amax / 7.f));
                    const float df = __half2float(__ushort_as_half(dh));
                    dsts[(size_t) r * (width / 32) + b] = dh;
                    for (int i = 0; i < 16; ++i) {
                        auto qnib = [&](int idx) -> unsigned {
                            if (df <= 0.f) return 8u;
                            long long qv = std::llround(src[b * 32 + idx] / df);
                            qv = std::clamp<long long>(qv, -8, 7);
                            return (unsigned) (qv + 8);
                        };
                        dstc[(size_t) r * (width / 2) + b * 16 + i] = (uint8_t) (qnib(i) | (qnib(i + 16) << 4));
                    }
                };
                q8block(kp, kc.data(), ks.data());
                q8block(vp, vc.data(), vs.data());
                q4block(kp, k4.data(), ks4.data());
                q4block(vp, v4.data(), vs4.data());
            }
        };
        if (!via_gpu_store) {
            for (int64_t j = 0; j < total_keys; ++j)
                write_row(j, k_f32.data() + (size_t) j * width, v_f32.data() + (size_t) j * width);
            for (int t = 0; t < q_rows; ++t)
                write_row(total_keys + t, q_new_k.data() + (size_t) t * width, q_new_v.data() + (size_t) t * width);
        }

        float* d_q = dev(q);
        float* d_rcur = dev(r_cur);
        float* d_rb = dev(rel_bias);
        float* d_snk = dev(sinks);
        float* d_tbl = dev(std::vector<float>((size_t) q_rows * n_head * n_extent, 0.f));
        inkling_rel_table(d_rcur, d_rb, d_snk, d_tbl, n_rel, n_extent, n_head, q_rows, nullptr);

        // Check rel_table vs float64
        std::vector<float> want_tbl((size_t) q_rows * n_head * n_extent);
        for (int t = 0; t < q_rows; ++t)
            for (int h = 0; h < n_head; ++h)
                for (int d = 0; d < n_extent; ++d) {
                    double dot = 0.0;
                    for (int k = 0; k < n_rel; ++k)
                        dot += (double) r_cur[((size_t) t * n_head + h) * n_rel + k] * rel_bias[(size_t) d * n_rel + k];
                    want_tbl[((size_t) t * n_head + h) * n_extent + d] = (float) (dot - sinks[(size_t) t * n_head + h]);
                }
        if (total_keys == 1500 && mode == Mode::kFp16 && n_head == 24)
            bad += report("inkling_rel_table", host(d_tbl, want_tbl.size()), want_tbl, 1e-5);

        uint16_t *d_kf16 = dev(kf16), *d_vf16 = dev(vf16);
        int8_t *d_kc = dev(kc), *d_vc = dev(vc);
        uint8_t *d_k4 = dev(k4), *d_v4 = dev(v4);
        uint16_t *d_ks = dev(ks), *d_vs = dev(vs);
        uint16_t *d_ks4 = dev(ks4), *d_vs4 = dev(vs4);

        if (via_gpu_store) {
            // Exercise store_q4 + store_q4_at on GPU and read packed codes/scales back for exact FP64 reference
            float* d_kf = dev(k_f32);
            float* d_vf = dev(v_f32);
            float* d_qnk = dev(q_new_k);
            float* d_qnv = dev(q_new_v);
            if (ring > 0) {
                for (int64_t j = 0; j < total_keys; j += ring) {
                    const int chunk_j = (int) std::min<int64_t>(ring, total_keys - j);
                    const int at = (int) (j % ring), head = std::min(chunk_j, ring - at);
                    store_q4(d_kf + (size_t) j * width, d_k4, d_ks4, width, at, head, nullptr);
                    store_q4(d_vf + (size_t) j * width, d_v4, d_vs4, width, at, head, nullptr);
                    if (head < chunk_j) {
                        store_q4(d_kf + (size_t) (j + head) * width, d_k4, d_ks4, width, 0, chunk_j - head, nullptr);
                        store_q4(d_vf + (size_t) (j + head) * width, d_v4, d_vs4, width, 0, chunk_j - head, nullptr);
                    }
                }
            } else {
                store_q4(d_kf, d_k4, d_ks4, width, 0, total_keys, nullptr);
                store_q4(d_vf, d_v4, d_vs4, width, 0, total_keys, nullptr);
            }
            if (use_at) {
                std::vector<long long> sat{total_keys, total_keys};
                long long* d_sat = dev(sat);
                store_q4_at(d_qnk, d_k4, d_ks4, width, d_sat, q_rows, ring, nullptr);
                store_q4_at(d_qnv, d_v4, d_vs4, width, d_sat, q_rows, ring, nullptr);
                cudaFree(d_sat);
            } else {
                const int at = ring > 0 ? (int) (total_keys % ring) : total_keys;
                const int head = ring > 0 ? std::min(q_rows, ring - at) : q_rows;
                store_q4(d_qnk, d_k4, d_ks4, width, at, head, nullptr);
                store_q4(d_qnv, d_v4, d_vs4, width, at, head, nullptr);
                if (head < q_rows) {
                    store_q4(d_qnk + (size_t) head * width, d_k4, d_ks4, width, 0, q_rows - head, nullptr);
                    store_q4(d_qnv + (size_t) head * width, d_v4, d_vs4, width, 0, q_rows - head, nullptr);
                }
            }
            k4 = host(d_k4, k4.size());
            v4 = host(d_v4, v4.size());
            ks4 = host(d_ks4, ks4.size());
            vs4 = host(d_vs4, vs4.size());
            cudaFree(d_kf); cudaFree(d_vf); cudaFree(d_qnk); cudaFree(d_qnv);
        }

        const bool q8 = mode == Mode::kQ8;
        const bool q4 = mode == Mode::kQ4;
        const bool quant = q8 || q4;
        Kvq8 kq = q8 ? Kvq8{d_kc, d_ks} : Kvq8{};
        Kvq8 vq = q8 ? Kvq8{d_vc, d_vs} : Kvq8{};
        Kvq4 kq4 = q4 ? Kvq4{d_k4, d_ks4} : Kvq4{};
        Kvq4 vq4 = q4 ? Kvq4{d_v4, d_vs4} : Kvq4{};

        const int n_splits = std::clamp((total_keys + q_rows + 511) / 512 + 1, 1, 64);
        const size_t score_n = std::max((size_t) q_rows * n_head * n_splits * (dim + 2),
                                        (size_t) q_rows * n_head * 2048);
        float* d_score = dev(std::vector<float>(score_n, 0.f));
        float* d_out_dec = dev(std::vector<float>(q.size(), 0.f));
        float* d_out_bat = dev(std::vector<float>(q.size(), 0.f));
        float* d_work = dev(std::vector<float>((size_t) q_rows * n_head * (dim + dim / 2 + 3) + 1, 0.f));
        const size_t wh = (size_t) 2048 * width;
        uint16_t* d_kw = quant ? dev(std::vector<uint16_t>(wh, 0)) : nullptr;
        uint16_t* d_vw = quant ? dev(std::vector<uint16_t>(wh, 0)) : nullptr;

        std::vector<long long> at_v((size_t) q_rows * 2);
        for (int t = 0; t < q_rows; ++t) at_v[2 * t] = at_v[2 * t + 1] = total_keys + t;
        long long* d_at = use_at ? dev(at_v) : nullptr;

        inkling_attend(d_q, quant ? nullptr : d_kf16, quant ? nullptr : d_vf16, d_tbl, d_snk, n_extent, total_keys,
                       q_rows, d_score, q_rows, d_out_dec, dim, n_head, kv_heads, scale, nullptr, window, ring, d_at,
                       use_at ? total_keys + q_rows + 16 : 0, kq, vq, kq4, vq4);

        void* blas = blas_create(nullptr);
        inkling_attend_batch(blas, d_q, quant ? nullptr : d_kf16, quant ? nullptr : d_vf16, d_tbl, d_snk, n_extent,
                             total_keys, q_rows, d_score, score_n, d_work, d_out_bat, dim, n_head, kv_heads, scale,
                             nullptr, window, ring, kq, vq, d_kw, d_vw, quant ? wh : 0, kq4, vq4);
        blas_destroy(blas);

        auto read_host_val = [&](const std::vector<uint16_t>& f16, const std::vector<int8_t>& c8,
                                 const std::vector<uint8_t>& c4, const std::vector<uint16_t>& sc, size_t r,
                                 size_t col) -> double {
            const size_t ei = r * width + col;
            if (q8) return (double) c8[ei] * (double) __half2float(__ushort_as_half(sc[r * (width / 32) + col / 32]));
            if (q4) {
                const size_t blk = col / 32;
                const size_t pos32 = col & 31;
                const uint8_t packed = c4[r * (width / 2) + blk * 16 + (pos32 & 15)];
                const int qv = (pos32 & 16) ? (packed >> 4) : (packed & 0x0f);
                return (double) (qv - 8) * (double) __half2float(__ushort_as_half(sc[r * (width / 32) + blk]));
            }
            return (double) __half2float(__ushort_as_half(f16[ei]));
        };

        // Float64 reference reading the exact stored FP16, Q8_0, or Q4_0 cache values
        std::vector<float> want_att(q.size(), 0.f);
        const int per = n_head / kv_heads;
        for (int t = 0; t < q_rows; ++t) {
            const int64_t qp = total_keys + t;
            const int64_t lo = window > 0 && qp >= window ? qp - window + 1 : 0;
            for (int h = 0; h < n_head; ++h) {
                const int kvh = h / per;
                std::vector<double> scores((size_t) (qp - lo + 1));
                double mx = 0.0;   // virtual zero sink!
                for (int64_t j = lo; j <= qp; ++j) {
                    const size_t r = (size_t) (ring > 0 ? j % ring : j);
                    double dot = 0.0;
                    for (int d = 0; d < dim; ++d) {
                        const double kv = read_host_val(kf16, kc, k4, q4 ? ks4 : ks, r, (size_t) kvh * dim + d);
                        dot += (double) q[((size_t) t * n_head + h) * dim + d] * kv;
                    }
                    const int64_t dist = qp - j;
                    const double bias = dist < n_extent
                        ? (double) want_tbl[((size_t) t * n_head + h) * n_extent + dist]
                        : -(double) sinks[(size_t) t * n_head + h];
                    const double sc = dot * scale + bias;
                    scores[(size_t) (j - lo)] = sc;
                    mx = std::max(mx, sc);
                }
                double lsum = std::exp(0.0 - mx);
                std::vector<double> acc(dim, 0.0);
                for (int64_t j = lo; j <= qp; ++j) {
                    const size_t r = (size_t) (ring > 0 ? j % ring : j);
                    const double wj = std::exp(scores[(size_t) (j - lo)] - mx);
                    lsum += wj;
                    for (int d = 0; d < dim; ++d) {
                        const double vv = read_host_val(vf16, vc, v4, q4 ? vs4 : vs, r, (size_t) kvh * dim + d);
                        acc[(size_t) d] += wj * vv;
                    }
                }
                for (int d = 0; d < dim; ++d)
                    want_att[((size_t) t * n_head + h) * dim + d] = (float) (acc[(size_t) d] / lsum);
            }
        }

        auto got_dec = host(d_out_dec, q.size());
        auto got_bat = host(d_out_bat, q.size());
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%s attend vs fp64", label);
        bad += report(buf, got_dec, want_att, 1e-5);
        std::snprintf(buf, sizeof(buf), "%s attend_batch vs attend", label);
        bad += report(buf, got_bat, got_dec, 5e-3);

        cudaFree(d_q); cudaFree(d_rcur); cudaFree(d_rb); cudaFree(d_snk); cudaFree(d_tbl);
        cudaFree(d_kf16); cudaFree(d_vf16); cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_k4); cudaFree(d_v4);
        cudaFree(d_ks); cudaFree(d_vs); cudaFree(d_ks4); cudaFree(d_vs4);
        cudaFree(d_score); cudaFree(d_out_dec); cudaFree(d_out_bat); cudaFree(d_work);
        if (d_kw) cudaFree(d_kw);
        if (d_vw) cudaFree(d_vw);
        if (d_at) cudaFree(d_at);
    };

    run_case("SWA ring FP16 (GQA3)", 24, 8, 1500, 1024, 1100, 32, Mode::kFp16, false, false);
    run_case("SWA ring Q8_0 (GQA3)", 24, 8, 1500, 1024, 1100, 32, Mode::kQ8, false, false);
    run_case("SWA ring Q4_0 GPU-store", 24, 8, 1500, 1024, 1100, 32, Mode::kQ4, false, true);
    run_case("Global >1024 FP16 + at (GQA3)", 24, 8, 1200, 0, 0, 16, Mode::kFp16, true, false);
    run_case("Global >1024 Q8_0 (GQA3)", 24, 8, 1200, 0, 0, 16, Mode::kQ8, false, false);
    run_case("Global >1024 Q4_0 GPU-store + at", 24, 8, 1200, 0, 0, 16, Mode::kQ4, true, true);
    run_case("Global >32K bounded-splits Q4_0", 24, 8, 35000, 0, 0, 2, Mode::kQ4, true, true);
    run_case("Global >1024 generic fallback (GQA2)", 16, 8, 1200, 0, 0, 8, Mode::kQ4, false, true);
    return bad;
}

int test_dspark_kernels() {
    int bad = 0;
    const int rows = 7, n_embd = 288, n_streams = 9;
    const float eps = 1e-6f;

    // 1. extract_taps + mix_norm
    auto r_host = random_vec((size_t) rows * n_streams * n_embd, 1.5f);
    auto tap3_host = random_vec((size_t) rows * n_embd, 1.5f);
    auto w_norm = random_vec(n_embd, 0.8f);
    const float a0 = 0.41f, a1 = 0.19f, a2 = 0.27f, a3 = 0.13f;

    float* d_r = dev(r_host);
    float* d_taps = dev(std::vector<float>((size_t) rows * 3 * n_embd, 0.f));
    float* d_tap3 = dev(tap3_host);
    float* d_wn = dev(w_norm);
    float* d_mix = dev(std::vector<float>((size_t) rows * n_embd, 0.f));

    inkling_dspark_extract_taps(d_r, n_streams, 6, d_taps, n_embd, rows, nullptr);
    inkling_dspark_mix_norm(d_taps, d_tap3, a0, a1, a2, a3, d_wn, d_mix, n_embd, rows, eps, nullptr);
    cudaDeviceSynchronize();

    std::vector<float> want_taps((size_t) rows * 3 * n_embd);
    std::vector<float> want_mix((size_t) rows * n_embd);
    for (int r = 0; r < rows; ++r) {
        std::vector<double> m(n_embd);
        double ss = 0.0;
        for (int t = 0; t < 3; ++t)
            for (int i = 0; i < n_embd; ++i)
                want_taps[((size_t) r * 3 + t) * n_embd + i] =
                    r_host[((size_t) r * n_streams + (6 + t)) * n_embd + i];
        for (int i = 0; i < n_embd; ++i) {
            m[i] = (double) a0 * want_taps[(size_t) r * 3 * n_embd + i] +
                   (double) a1 * want_taps[((size_t) r * 3 + 1) * n_embd + i] +
                   (double) a2 * want_taps[((size_t) r * 3 + 2) * n_embd + i] +
                   (double) a3 * tap3_host[(size_t) r * n_embd + i];
            ss += m[i] * m[i];
        }
        const double inv = 1.0 / std::sqrt(ss / n_embd + eps);
        for (int i = 0; i < n_embd; ++i)
            want_mix[(size_t) r * n_embd + i] = (float) (m[i] * inv * (double) w_norm[i]);
    }
    bad += report("DSpark extract_taps", host(d_taps, want_taps.size()), want_taps, 1e-6);
    bad += report("DSpark mix_norm vs FP64", host(d_mix, want_mix.size()), want_mix, 1e-5);

    cudaFree(d_r); cudaFree(d_taps); cudaFree(d_tap3); cudaFree(d_wn); cudaFree(d_mix);

    // 2. rnn_gate with gates_sp + gates_h + gates_f (two sequential updates) + prep_rnn_step
    const int rank = 256, w1_voc = 12;
    auto g1 = random_vec(3 * rank, 1.2f), g2 = random_vec(3 * rank, 1.2f),
         gh = random_vec(3 * rank, 0.8f), gf = random_vec(3 * rank, 0.6f), rb = random_vec(3 * rank, 0.3f);
    std::vector<float> st_init(rank, 0.f);
    float* d_g = dev(g1);
    float* d_gh = dev(gh);
    float* d_gf = dev(gf);
    float* d_rb = dev(rb);
    float* d_st = dev(st_init);
    float* d_lat = dev(std::vector<float>(rank, 0.f));

    auto ref_gate = [&](const std::vector<float>& g, std::vector<double>& s_ref, std::vector<float>& l_ref) {
        for (int i = 0; i < rank; ++i) {
            const double k = 1.0 / (1.0 + std::exp(-((double) g[i] + gh[i] + gf[i] + rb[i])));
            const double u = std::tanh((double) g[rank + i] + gh[rank + i] + gf[rank + i] + rb[rank + i]);
            const double l = std::tanh((double) g[2 * rank + i] + gh[2 * rank + i] + gf[2 * rank + i] + rb[2 * rank + i]);
            s_ref[i] = k * s_ref[i] + (1.0 - k) * u;
            l_ref[i] = (float) l;
        }
    };
    std::vector<double> st_ref(rank, 0.0);
    std::vector<float> lat_ref(rank);
    inkling_dspark_rnn_gate(d_g, d_gh, d_gf, d_rb, d_st, d_lat, rank, nullptr);
    ref_gate(g1, st_ref, lat_ref);
    cudaMemcpy(d_g, g2.data(), g2.size() * sizeof(float), cudaMemcpyHostToDevice);
    inkling_dspark_rnn_gate(d_g, d_gh, d_gf, d_rb, d_st, d_lat, rank, nullptr);
    ref_gate(g2, st_ref, lat_ref);
    cudaDeviceSynchronize();
    std::vector<float> st_ref_f(rank);
    for (int i = 0; i < rank; ++i) st_ref_f[i] = (float) st_ref[i];
    bad += report("DSpark rnn_gate state vs FP64", host(d_st, rank), st_ref_f, 1e-5);
    bad += report("DSpark rnn_gate latent vs FP64", host(d_lat, rank), lat_ref, 1e-5);

    // 2b. prep_rnn_step: reads anchor (-1 -> picks[anchor_slot]) on step 0 and picks[step - 1] on step > 0 + BF16 w1
    auto w1_f = random_vec((size_t) w1_voc * rank, 1.4f);
    std::vector<ggml_bf16_t> w1_bf16((size_t) w1_voc * rank);
    ggml_fp32_to_bf16_row(w1_f.data(), w1_bf16.data(), (int64_t) w1_f.size());
    std::vector<float> w1_deq((size_t) w1_voc * rank);
    ggml_get_type_traits(GGML_TYPE_BF16)->to_float(w1_bf16.data(), w1_deq.data(), (int64_t) w1_f.size());
    std::vector<uint16_t> w1_u16((size_t) w1_voc * rank);
    std::memcpy(w1_u16.data(), w1_bf16.data(), w1_u16.size() * sizeof(uint16_t));
    uint16_t* d_w1 = dev(w1_u16);
    std::vector<Likeliest> h_picks = {{7, 0.5f}, {3, 0.4f}, {4, 1.0f}};
    Likeliest* d_picks = dev(h_picks);
    float* d_in_sp = dev(std::vector<float>(2 * rank, 0.f));
    inkling_dspark_prep_rnn_step(d_st, d_w1, -1, d_picks, 0, 2, rank, w1_voc, d_in_sp, nullptr);
    cudaDeviceSynchronize();
    std::vector<float> want_sp0(2 * rank);
    for (int i = 0; i < rank; ++i) {
        want_sp0[i] = st_ref_f[i];
        want_sp0[rank + i] = w1_deq[(size_t) 4 * rank + i];
    }
    bad += report("DSpark prep_rnn_step(step=0,slot)", host(d_in_sp, 2 * rank), want_sp0, 1e-6);
    inkling_dspark_prep_rnn_step(d_st, d_w1, -1, d_picks, 1, 2, rank, w1_voc, d_in_sp, nullptr);
    cudaDeviceSynchronize();
    std::vector<float> want_sp1(2 * rank);
    for (int i = 0; i < rank; ++i) {
        want_sp1[i] = st_ref_f[i];
        want_sp1[rank + i] = w1_deq[(size_t) 7 * rank + i];
    }
    bad += report("DSpark prep_rnn_step(step=1)", host(d_in_sp, 2 * rank), want_sp1, 1e-6);
    cudaFree(d_w1); cudaFree(d_picks); cudaFree(d_in_sp);
    cudaFree(d_g); cudaFree(d_gh); cudaFree(d_gf); cudaFree(d_rb); cudaFree(d_st); cudaFree(d_lat);

    // 3. add_softcap
    const int n_voc = 4099;
    const float cap = 30.0f;
    auto base = random_vec(n_voc, 12.f), delta = random_vec(n_voc, 6.f), fbias = random_vec(n_voc, 2.f);
    float* d_base = dev(base);
    float* d_delta = dev(delta);
    float* d_fb = dev(fbias);
    float* d_out = dev(std::vector<float>(n_voc, 0.f));
    inkling_dspark_add_softcap(d_base, d_delta, d_fb, d_out, n_voc, cap, nullptr);
    cudaDeviceSynchronize();
    std::vector<float> want_cap(n_voc);
    for (int i = 0; i < n_voc; ++i) {
        const double v = (double) base[i] + delta[i] + fbias[i];
        want_cap[i] = (float) (cap * std::tanh(v / cap));
    }
    bad += report("DSpark add_softcap vs FP64", host(d_out, n_voc), want_cap, 1e-5);
    cudaFree(d_base); cudaFree(d_delta); cudaFree(d_fb); cudaFree(d_out);

    // 4. pack_q8_0_block dequant verification via ggml
    const int pv = 137, n_blocks = 2;
    auto col0 = random_vec(32 * pv, 2.5f), col1 = random_vec(32 * pv, 1.5f);
    float* d_cols = dev(col0);
    uint8_t* d_q8 = nullptr;
    cudaMalloc(&d_q8, (size_t) pv * n_blocks * 34);
    cudaMemset(d_q8, 0, (size_t) pv * n_blocks * 34);
    inkling_dspark_pack_q8_0_block(d_cols, d_q8, pv, 0, n_blocks, nullptr);
    cudaMemcpy(d_cols, col1.data(), col1.size() * sizeof(float), cudaMemcpyHostToDevice);
    inkling_dspark_pack_q8_0_block(d_cols, d_q8, pv, 1, n_blocks, nullptr);
    cudaDeviceSynchronize();

    std::vector<uint8_t> h_q8((size_t) pv * n_blocks * 34);
    cudaMemcpy(h_q8.data(), d_q8, h_q8.size(), cudaMemcpyDeviceToHost);
    const auto* tt = ggml_get_type_traits(GGML_TYPE_Q8_0);
    std::vector<float> deq((size_t) pv * 64);
    tt->to_float(h_q8.data(), deq.data(), (int64_t) pv * 64);
    double max_q_err = 0.0;
    for (int v = 0; v < pv; ++v) {
        float amax0 = 0.f, amax1 = 0.f;
        for (int j = 0; j < 32; ++j) {
            amax0 = std::max(amax0, std::abs(col0[(size_t) j * pv + v]));
            amax1 = std::max(amax1, std::abs(col1[(size_t) j * pv + v]));
        }
        const double bound0 = (amax0 / 127.0) * 0.52 + 1e-3 * amax0 + 1e-5;
        const double bound1 = (amax1 / 127.0) * 0.52 + 1e-3 * amax1 + 1e-5;
        for (int j = 0; j < 32; ++j) {
            const double e0 = std::abs((double) deq[(size_t) v * 64 + j] - (double) col0[(size_t) j * pv + v]);
            const double e1 = std::abs((double) deq[(size_t) v * 64 + 32 + j] - (double) col1[(size_t) j * pv + v]);
            if (e0 > bound0) ++bad;
            if (e1 > bound1) ++bad;
            max_q_err = std::max({max_q_err, e0, e1});
        }
    }
    std::printf("%-42s %8d vals  err=%.3g  %s\n", "DSpark pack_q8_0_block vs ggml_to_float",
                pv * 64, max_q_err, bad == 0 ? "PASS" : "FAIL");

    // 5. pack_q4_0_block exact byte parity vs ggml_quantize_chunk(GGML_TYPE_Q4_0)
    uint8_t* d_q4 = nullptr;
    cudaMalloc(&d_q4, (size_t) pv * n_blocks * 18);
    cudaMemset(d_q4, 0, (size_t) pv * n_blocks * 18);
    cudaMemcpy(d_cols, col0.data(), col0.size() * sizeof(float), cudaMemcpyHostToDevice);
    inkling_dspark_pack_q4_0_block(d_cols, d_q4, pv, 0, n_blocks, nullptr);
    cudaMemcpy(d_cols, col1.data(), col1.size() * sizeof(float), cudaMemcpyHostToDevice);
    inkling_dspark_pack_q4_0_block(d_cols, d_q4, pv, 1, n_blocks, nullptr);
    cudaDeviceSynchronize();

    std::vector<uint8_t> h_q4((size_t) pv * n_blocks * 18);
    cudaMemcpy(h_q4.data(), d_q4, h_q4.size(), cudaMemcpyDeviceToHost);
    std::vector<float> rows_f((size_t) pv * 64);
    for (int v = 0; v < pv; ++v) {
        for (int j = 0; j < 32; ++j) {
            rows_f[(size_t) v * 64 + j] = col0[(size_t) j * pv + v];
            rows_f[(size_t) v * 64 + 32 + j] = col1[(size_t) j * pv + v];
        }
    }
    std::vector<uint8_t> ggml_q4((size_t) pv * n_blocks * 18);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, rows_f.data(), ggml_q4.data(), 0, pv, 64, nullptr);
    const auto* tt4 = ggml_get_type_traits(GGML_TYPE_Q4_0);
    std::vector<float> deq4((size_t) pv * 64), deq_ref((size_t) pv * 64);
    tt4->to_float(h_q4.data(), deq4.data(), (int64_t) pv * 64);
    tt4->to_float(ggml_q4.data(), deq_ref.data(), (int64_t) pv * 64);
    int bad4 = 0;
    double max_q4_err = 0.0;
    for (size_t i = 0; i < deq4.size(); ++i) {
        const double e = std::abs((double) deq4[i] - (double) deq_ref[i]);
        max_q4_err = std::max(max_q4_err, e);
    }
    // Check dequantized diff against ggml_quantize_chunk (allows at most 1 Q4 step from fp16 rounding on boundary ties)
    for (int b_i = 0; b_i < pv * n_blocks; ++b_i) {
        uint16_t dh = (uint16_t) ggml_q4[(size_t) b_i * 18] | ((uint16_t) ggml_q4[(size_t) b_i * 18 + 1] << 8);
        const double step = std::abs((double) __half2float(__ushort_as_half(dh)));
        for (int j = 0; j < 32; ++j) {
            const size_t idx = (size_t) b_i * 32 + j;
            if (std::abs((double) deq4[idx] - (double) deq_ref[idx]) > step * 1.05 + 1e-4) ++bad4;
        }
    }
    bad += bad4;
    std::printf("%-42s %8d vals  err=%.3g  %s\n", "DSpark pack_q4_0_block vs ggml_quantize",
                pv * 64, max_q4_err, bad4 == 0 ? "PASS" : "FAIL");
    cudaFree(d_cols); cudaFree(d_q8); cudaFree(d_q4);

    // 6. split_qkv
    const int sq_rows = 8, sq_qd = 64, sq_kvd = 32;
    auto qkv_h = random_vec((size_t) sq_rows * (sq_qd + 2 * sq_kvd), 1.f);
    float *d_qkv = dev(qkv_h), *d_sq = dev(std::vector<float>((size_t) sq_rows * sq_qd, 0.f)),
          *d_sk = dev(std::vector<float>((size_t) sq_rows * sq_kvd, 0.f)),
          *d_sv = dev(std::vector<float>((size_t) sq_rows * sq_kvd, 0.f));
    inkling_dspark_split_qkv(d_qkv, d_sq, d_sk, d_sv, sq_qd, sq_kvd, sq_rows, nullptr);
    cudaDeviceSynchronize();
    std::vector<float> want_sq((size_t) sq_rows * sq_qd), want_sk((size_t) sq_rows * sq_kvd), want_sv((size_t) sq_rows * sq_kvd);
    for (int t = 0; t < sq_rows; ++t) {
        const size_t base = (size_t) t * (sq_qd + 2 * sq_kvd);
        std::copy(qkv_h.data() + base, qkv_h.data() + base + sq_qd, want_sq.data() + (size_t) t * sq_qd);
        std::copy(qkv_h.data() + base + sq_qd, qkv_h.data() + base + sq_qd + sq_kvd, want_sk.data() + (size_t) t * sq_kvd);
        std::copy(qkv_h.data() + base + sq_qd + sq_kvd, qkv_h.data() + base + sq_qd + 2 * sq_kvd, want_sv.data() + (size_t) t * sq_kvd);
    }
    bad += report("DSpark split_qkv (Q)", host(d_sq, want_sq.size()), want_sq, 0.0);
    bad += report("DSpark split_qkv (K)", host(d_sk, want_sk.size()), want_sk, 0.0);
    bad += report("DSpark split_qkv (V)", host(d_sv, want_sv.size()), want_sv, 0.0);
    cudaFree(d_qkv); cudaFree(d_sq); cudaFree(d_sk); cudaFree(d_sv);

    // 7. inkling_dspark_attend vs FP64 bidirectional block window reference (FP16, Q8_0, Q4_0)
    auto run_dspark_attend = [&](const char* label, int nh, int nkv, int64_t pos, int blk_rows, int win, int ring, int kv_mode) {
        const int dim = 128;
        const int64_t total_kv = pos + blk_rows;
        const int cache_rows = ring > 0 ? ring : (int) total_kv;
        const int width = nkv * dim, nb = width / 32;
        const float scale = 1.0f / std::sqrt((float) dim);

        auto k_all = random_vec((size_t) total_kv * width, 0.6f);
        auto v_all = random_vec((size_t) total_kv * width, 0.6f);
        auto q_h = random_vec((size_t) blk_rows * nh * dim, 0.6f);

        std::vector<uint16_t> k16((size_t) cache_rows * width, 0), v16((size_t) cache_rows * width, 0);
        std::vector<int8_t> k8c((size_t) cache_rows * width, 0), v8c((size_t) cache_rows * width, 0);
        std::vector<uint8_t> k4c((size_t) cache_rows * (width / 2), 0), v4c((size_t) cache_rows * (width / 2), 0);
        std::vector<uint16_t> ks((size_t) cache_rows * nb, 0), vs((size_t) cache_rows * nb, 0);
        std::vector<float> k_ref((size_t) total_kv * width), v_ref((size_t) total_kv * width);

        for (int64_t p = 0; p < total_kv; ++p) {
            const int64_t r = ring > 0 ? (p % ring) : p;
            if (kv_mode == 0) {
                for (int i = 0; i < width; ++i) {
                    __half hk = __float2half(k_all[(size_t) p * width + i]);
                    __half hv = __float2half(v_all[(size_t) p * width + i]);
                    k16[(size_t) r * width + i] = __half_as_ushort(hk);
                    v16[(size_t) r * width + i] = __half_as_ushort(hv);
                    k_ref[(size_t) p * width + i] = __half2float(hk);
                    v_ref[(size_t) p * width + i] = __half2float(hv);
                }
            } else if (kv_mode == 1) {
                for (int b = 0; b < nb; ++b) {
                    float ak = 0.f, av = 0.f;
                    for (int j = 0; j < 32; ++j) {
                        ak = std::max(ak, std::abs(k_all[(size_t) p * width + b * 32 + j]));
                        av = std::max(av, std::abs(v_all[(size_t) p * width + b * 32 + j]));
                    }
                    __half dkh = __float2half(ak / 127.0f), dvh = __float2half(av / 127.0f);
                    float dkf = __half2float(dkh), dvf = __half2float(dvh);
                    ks[(size_t) r * nb + b] = __half_as_ushort(dkh);
                    vs[(size_t) r * nb + b] = __half_as_ushort(dvh);
                    for (int j = 0; j < 32; ++j) {
                        int qk = dkf > 0 ? std::clamp((int) std::llrint(k_all[(size_t) p * width + b * 32 + j] / dkf), -127, 127) : 0;
                        int qv = dvf > 0 ? std::clamp((int) std::llrint(v_all[(size_t) p * width + b * 32 + j] / dvf), -127, 127) : 0;
                        k8c[(size_t) r * width + b * 32 + j] = (int8_t) qk;
                        v8c[(size_t) r * width + b * 32 + j] = (int8_t) qv;
                        k_ref[(size_t) p * width + b * 32 + j] = qk * dkf;
                        v_ref[(size_t) p * width + b * 32 + j] = qv * dvf;
                    }
                }
            } else if (kv_mode == 2) {
                for (int b = 0; b < nb; ++b) {
                    float ak = 0.f, av = 0.f;
                    for (int j = 0; j < 32; ++j) {
                        ak = std::max(ak, std::abs(k_all[(size_t) p * width + b * 32 + j]));
                        av = std::max(av, std::abs(v_all[(size_t) p * width + b * 32 + j]));
                    }
                    __half dkh = __float2half(ak / 7.0f), dvh = __float2half(av / 7.0f);
                    float dkf = __half2float(dkh), dvf = __half2float(dvh);
                    ks[(size_t) r * nb + b] = __half_as_ushort(dkh);
                    vs[(size_t) r * nb + b] = __half_as_ushort(dvh);
                    for (int j = 0; j < 16; ++j) {
                        int k0 = dkf > 0 ? std::clamp((int) std::llrint(k_all[(size_t) p * width + b * 32 + j] / dkf), -8, 7) + 8 : 8;
                        int k1 = dkf > 0 ? std::clamp((int) std::llrint(k_all[(size_t) p * width + b * 32 + 16 + j] / dkf), -8, 7) + 8 : 8;
                        int v0 = dvf > 0 ? std::clamp((int) std::llrint(v_all[(size_t) p * width + b * 32 + j] / dvf), -8, 7) + 8 : 8;
                        int v1 = dvf > 0 ? std::clamp((int) std::llrint(v_all[(size_t) p * width + b * 32 + 16 + j] / dvf), -8, 7) + 8 : 8;
                        k4c[(size_t) r * (width / 2) + b * 16 + j] = (uint8_t) (k0 | (k1 << 4));
                        v4c[(size_t) r * (width / 2) + b * 16 + j] = (uint8_t) (v0 | (v1 << 4));
                        k_ref[(size_t) p * width + b * 32 + j] = (k0 - 8) * dkf;
                        k_ref[(size_t) p * width + b * 32 + 16 + j] = (k1 - 8) * dkf;
                        v_ref[(size_t) p * width + b * 32 + j] = (v0 - 8) * dvf;
                        v_ref[(size_t) p * width + b * 32 + 16 + j] = (v1 - 8) * dvf;
                    }
                }
            } else {
                for (int b = 0; b < nb; ++b) {
                    float ak = 0.f, av = 0.f;
                    for (int j = 0; j < 32; ++j) {
                        ak = std::max(ak, std::abs(k_all[(size_t) p * width + b * 32 + j]));
                        av = std::max(av, std::abs(v_all[(size_t) p * width + b * 32 + j]));
                    }
                    __half dkh = __float2half(ak / 127.0f), dvh = __float2half(av / 7.0f);
                    float dkf = __half2float(dkh), dvf = __half2float(dvh);
                    ks[(size_t) r * nb + b] = __half_as_ushort(dkh);
                    vs[(size_t) r * nb + b] = __half_as_ushort(dvh);
                    for (int j = 0; j < 32; ++j) {
                        int qk = dkf > 0 ? std::clamp((int) std::llrint(k_all[(size_t) p * width + b * 32 + j] / dkf), -127, 127) : 0;
                        k8c[(size_t) r * width + b * 32 + j] = (int8_t) qk;
                        k_ref[(size_t) p * width + b * 32 + j] = qk * dkf;
                    }
                    for (int j = 0; j < 16; ++j) {
                        int v0 = dvf > 0 ? std::clamp((int) std::llrint(v_all[(size_t) p * width + b * 32 + j] / dvf), -8, 7) + 8 : 8;
                        int v1 = dvf > 0 ? std::clamp((int) std::llrint(v_all[(size_t) p * width + b * 32 + 16 + j] / dvf), -8, 7) + 8 : 8;
                        v4c[(size_t) r * (width / 2) + b * 16 + j] = (uint8_t) (v0 | (v1 << 4));
                        v_ref[(size_t) p * width + b * 32 + j] = (v0 - 8) * dvf;
                        v_ref[(size_t) p * width + b * 32 + 16 + j] = (v1 - 8) * dvf;
                    }
                }
            }
        }

        float *d_q = dev(q_h), *d_sc = dev(std::vector<float>((size_t) blk_rows * nh * 64 * (dim + 2), 0.f)),
              *d_o = dev(std::vector<float>((size_t) blk_rows * nh * dim, 0.f));
        uint16_t *d_k16 = kv_mode == 0 ? dev(k16) : nullptr, *d_v16 = kv_mode == 0 ? dev(v16) : nullptr;
        int8_t *d_k8 = (kv_mode == 1 || kv_mode == 3) ? dev(k8c) : nullptr, *d_v8 = kv_mode == 1 ? dev(v8c) : nullptr;
        uint8_t *d_k4 = kv_mode == 2 ? dev(k4c) : nullptr, *d_v4 = (kv_mode == 2 || kv_mode == 3) ? dev(v4c) : nullptr;
        uint16_t *d_ks = kv_mode > 0 ? dev(ks) : nullptr, *d_vs = kv_mode > 0 ? dev(vs) : nullptr;
        Kvq8 kq8{d_k8, d_ks}, vq8{d_v8, d_vs};
        Kvq4 kq4{d_k4, d_ks}, vq4{d_v4, d_vs};

        inkling_dspark_attend(d_q, d_k16, d_v16, pos, blk_rows, d_sc, d_o, dim, nh, nkv, scale, nullptr,
                              win, ring, nullptr, 0, kq8, vq8, kq4, vq4);
        cudaDeviceSynchronize();

        std::vector<float> want_o((size_t) blk_rows * nh * dim);
        const int per = nh / nkv;
        for (int t = 0; t < blk_rows; ++t) {
            const int64_t q_pos = pos + t;
            const int64_t lo = (win > 0 && q_pos >= win) ? (q_pos - win + 1) : 0;
            const int64_t end = pos + blk_rows - 1;
            for (int h = 0; h < nh; ++h) {
                const int kvh = h / per;
                std::vector<double> scores((size_t) (end - lo + 1));
                double mx = -INFINITY;
                for (int64_t j = lo; j <= end; ++j) {
                    double dot = 0.0;
                    for (int d = 0; d < dim; ++d)
                        dot += (double) q_h[((size_t) t * nh + h) * dim + d] * k_ref[(size_t) j * width + kvh * dim + d];
                    const double s = dot * scale;
                    scores[(size_t) (j - lo)] = s;
                    mx = std::max(mx, s);
                }
                double sum = 0.0;
                std::vector<double> acc(dim, 0.0);
                for (size_t idx = 0; idx < scores.size(); ++idx) {
                    const double w = std::exp(scores[idx] - mx);
                    sum += w;
                    for (int d = 0; d < dim; ++d)
                        acc[d] += w * v_ref[(size_t) (lo + idx) * width + kvh * dim + d];
                }
                for (int d = 0; d < dim; ++d)
                    want_o[((size_t) t * nh + h) * dim + d] = (float) (acc[d] / sum);
            }
        }
        bad += report(label, host(d_o, want_o.size()), want_o, 1e-4);
        cudaFree(d_q); cudaFree(d_sc); cudaFree(d_o);
        cudaFree(d_k16); cudaFree(d_v16); cudaFree(d_k8); cudaFree(d_v8);
        cudaFree(d_k4); cudaFree(d_v4); cudaFree(d_ks); cudaFree(d_vs);
    };
    run_dspark_attend("DSpark attend local SWA ring FP16", 24, 8, 28, 8, 17, 24, 0);
    run_dspark_attend("DSpark attend global split-K FP16", 24, 8, 1100, 8, 0, 0, 0);
    run_dspark_attend("DSpark attend global split-K Q8_0", 24, 8, 1100, 8, 0, 0, 1);
    run_dspark_attend("DSpark attend global split-K Q4_0", 24, 8, 1100, 8, 0, 0, 2);
    run_dspark_attend("DSpark attend global split-K K8V4", 24, 8, 1100, 8, 0, 0, 3);
    return bad;
}

}  // namespace

int main() {
    int bad = 0;
    bad += test_situ_gpu_and_cpu();
    bad += test_mix_and_control();
    bad += test_conv4();
    bad += test_l2_clip();
    bad += test_rel_and_attend();
    bad += test_dspark_kernels();
    std::printf("\n%s\n", bad ? "FAILED" : "ALL PASSED");
    return bad ? 1 : 0;
}
