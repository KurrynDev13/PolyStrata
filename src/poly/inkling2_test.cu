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
    const int dim = 128, n_head = 24, kv_heads = 8, n_rel = 16, n_extent = 1024;
    const float scale = 1.0f / std::sqrt((float) dim);

    auto run_case = [&](const char* label, int total_keys, int window, int ring, int q_rows, bool q8, bool use_at) {
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
        std::vector<uint16_t> ks((size_t) cache_rows * (width / 32), 0), vs((size_t) cache_rows * (width / 32), 0);

        auto write_row = [&](int64_t j, const float* kp, const float* vp) {
            const int r = ring > 0 ? (int) (j % ring) : (int) j;
            for (int c = 0; c < width; ++c) {
                kf16[(size_t) r * width + c] = __half_as_ushort(__float2half(kp[c]));
                vf16[(size_t) r * width + c] = __half_as_ushort(__float2half(vp[c]));
            }
            for (int b = 0; b < width / 32; ++b) {
                auto qblock = [&](const float* src, int8_t* dstc, uint16_t* dsts) {
                    float amax = 0.f;
                    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::abs(src[b * 32 + i]));
                    const float d = amax / 127.f;
                    const uint16_t dh = __half_as_ushort(__float2half(d));
                    const float df = __half2float(__ushort_as_half(dh));
                    dsts[(size_t) r * (width / 32) + b] = dh;
                    for (int i = 0; i < 32; ++i)
                        dstc[(size_t) r * width + b * 32 + i] = df > 0.f ? (int8_t) std::llround(src[b * 32 + i] / df) : 0;
                };
                qblock(kp, kc.data(), ks.data());
                qblock(vp, vc.data(), vs.data());
            }
        };
        for (int64_t j = 0; j < total_keys; ++j)
            write_row(j, k_f32.data() + (size_t) j * width, v_f32.data() + (size_t) j * width);
        for (int t = 0; t < q_rows; ++t)
            write_row(total_keys + t, q_new_k.data() + (size_t) t * width, q_new_v.data() + (size_t) t * width);

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
        if (total_keys == 1500 && !q8) bad += report("inkling_rel_table", host(d_tbl, want_tbl.size()), want_tbl, 1e-5);

        uint16_t *d_kf16 = dev(kf16), *d_vf16 = dev(vf16);
        int8_t *d_kc = dev(kc), *d_vc = dev(vc);
        uint16_t *d_ks = dev(ks), *d_vs = dev(vs);
        Kvq8 kq = q8 ? Kvq8{d_kc, d_ks} : Kvq8{};
        Kvq8 vq = q8 ? Kvq8{d_vc, d_vs} : Kvq8{};

        const int n_splits = (total_keys + q_rows + 511) / 512 + 1;
        const size_t score_n = std::max((size_t) q_rows * n_head * n_splits * (dim + 2),
                                        (size_t) q_rows * n_head * 2048);
        float* d_score = dev(std::vector<float>(score_n, 0.f));
        float* d_out_dec = dev(std::vector<float>(q.size(), 0.f));
        float* d_out_bat = dev(std::vector<float>(q.size(), 0.f));
        float* d_work = dev(std::vector<float>((size_t) q_rows * n_head * (dim + dim / 2 + 3) + 1, 0.f));
        const size_t wh = (size_t) 2048 * width;
        uint16_t* d_kw = q8 ? dev(std::vector<uint16_t>(wh, 0)) : nullptr;
        uint16_t* d_vw = q8 ? dev(std::vector<uint16_t>(wh, 0)) : nullptr;

        std::vector<long long> at_v((size_t) q_rows * 2);
        for (int t = 0; t < q_rows; ++t) at_v[2 * t] = at_v[2 * t + 1] = total_keys + t;
        long long* d_at = use_at ? dev(at_v) : nullptr;

        inkling_attend(d_q, q8 ? nullptr : d_kf16, q8 ? nullptr : d_vf16, d_tbl, d_snk, n_extent, total_keys, q_rows,
                       d_score, q_rows, d_out_dec, dim, n_head, kv_heads, scale, nullptr, window, ring, d_at,
                       use_at ? total_keys + q_rows + 16 : 0, kq, vq);

        void* blas = blas_create(nullptr);
        inkling_attend_batch(blas, d_q, q8 ? nullptr : d_kf16, q8 ? nullptr : d_vf16, d_tbl, d_snk, n_extent,
                             total_keys, q_rows, d_score, score_n, d_work, d_out_bat, dim, n_head, kv_heads, scale,
                             nullptr, window, ring, kq, vq, d_kw, d_vw, q8 ? wh : 0);
        blas_destroy(blas);

        // Float64 reference reading the exact stored FP16 or Q8_0 cache values
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
                        const size_t ei = r * width + (size_t) kvh * dim + d;
                        const double kv = q8
                            ? (double) kc[ei] * (double) __half2float(__ushort_as_half(ks[r * (width / 32) + ((size_t) kvh * dim + d) / 32]))
                            : (double) __half2float(__ushort_as_half(kf16[ei]));
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
                        const size_t ei = r * width + (size_t) kvh * dim + d;
                        const double vv = q8
                            ? (double) vc[ei] * (double) __half2float(__ushort_as_half(vs[r * (width / 32) + ((size_t) kvh * dim + d) / 32]))
                            : (double) __half2float(__ushort_as_half(vf16[ei]));
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
        cudaFree(d_kf16); cudaFree(d_vf16); cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_ks); cudaFree(d_vs);
        cudaFree(d_score); cudaFree(d_out_dec); cudaFree(d_out_bat); cudaFree(d_work);
        if (d_kw) cudaFree(d_kw);
        if (d_vw) cudaFree(d_vw);
        if (d_at) cudaFree(d_at);
    };

    run_case("SWA ring FP16", 1500, 1024, 1100, 32, false, false);
    run_case("SWA ring Q8_0", 1500, 1024, 1100, 32, true, false);
    run_case("Global >1024 FP16 + at", 1200, 0, 0, 16, false, true);
    run_case("Global >1024 Q8_0", 1200, 0, 0, 16, true, false);
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
    std::printf("\n%s\n", bad ? "FAILED" : "ALL PASSED");
    return bad ? 1 : 0;
}
