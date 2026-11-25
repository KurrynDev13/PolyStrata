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

int test_ctrl_down() {
    int bad = 0;
    const int n_embd = 512, rank = 64, rows = 2, count = 2;
    auto c0 = random_vec((size_t) rows * n_embd, 1.f);
    auto m0 = random_vec((size_t) rows * n_embd, 1.f);
    auto w0 = random_vec((size_t) rank * 2 * n_embd, 0.2f);
    auto c1 = random_vec((size_t) rows * n_embd, 1.f);
    auto m1 = random_vec((size_t) rows * n_embd, 1.f);
    auto w1 = random_vec((size_t) rank * 2 * n_embd, 0.2f);

    float *d_c0 = dev(c0), *d_m0 = dev(m0), *d_w0 = dev(w0), *d_y0 = dev(std::vector<float>(rows * rank, 0.f));
    float *d_c1 = dev(c1), *d_m1 = dev(m1), *d_w1 = dev(w1), *d_y1 = dev(std::vector<float>(rows * rank, 0.f));

    const float* cur[2] = {d_c0, d_c1};
    const float* msg[2] = {d_m0, d_m1};
    const float* down[2] = {d_w0, d_w1};
    float* ch[2] = {d_y0, d_y1};

    bool ok = ink_ctrl_down(cur, msg, down, ch, count, n_embd, rank, rows, nullptr);
    if (!ok) {
        printf("ink_ctrl_down returned false!\n");
        return 1;
    }

    auto got0 = host(d_y0, (size_t) rows * rank);
    auto got1 = host(d_y1, (size_t) rows * rank);

    auto ref_down = [&](const std::vector<float>& c, const std::vector<float>& m, const std::vector<float>& w) {
        std::vector<float> want((size_t) rows * rank);
        for (int t = 0; t < rows; ++t) {
            double sq_c = 0.0, sq_m = 0.0;
            for (int i = 0; i < n_embd; ++i) {
                sq_c += (double) c[(size_t) t * n_embd + i] * c[(size_t) t * n_embd + i];
                sq_m += (double) m[(size_t) t * n_embd + i] * m[(size_t) t * n_embd + i];
            }
            float ic = (float) (1.0 / std::sqrt(sq_c / n_embd + 1e-6));
            float im = (float) (1.0 / std::sqrt(sq_m / n_embd + 1e-6));
            for (int o = 0; o < rank; ++o) {
                double acc = 0.0;
                for (int i = 0; i < n_embd; ++i) {
                    acc += (double) w[(size_t) o * 2 * n_embd + i] * (c[(size_t) t * n_embd + i] * ic);
                    acc += (double) w[(size_t) o * 2 * n_embd + n_embd + i] * (m[(size_t) t * n_embd + i] * im);
                }
                want[(size_t) t * rank + o] = (float) acc;
            }
        }
        return want;
    };

    bad += report("ink_ctrl_down (controller 0)", got0, ref_down(c0, m0, w0), 1e-5);
    bad += report("ink_ctrl_down (controller 1)", got1, ref_down(c1, m1, w1), 1e-5);

    cudaFree(d_c0); cudaFree(d_m0); cudaFree(d_w0); cudaFree(d_y0);
    cudaFree(d_c1); cudaFree(d_m1); cudaFree(d_w1); cudaFree(d_y1);
    return bad;
}

int test_conv4() {
    int bad = 0;
    const int head_dim = 64, n_heads = 4, channels = head_dim * n_heads, rows = 20;
    auto st0 = random_vec((size_t) 3 * channels, 0.5f);
    auto w = random_vec((size_t) 4 * channels, 0.3f);
    auto x = random_vec((size_t) rows * channels, 1.f);
    auto dyn = random_vec((size_t) rows * n_heads * 4, 0.2f);

    auto ref_conv = [&](bool use_dyn, int num_rows = 0) {
        const int nr = num_rows > 0 ? num_rows : rows;
        std::vector<float> st = st0, out((size_t) nr * channels);
        const double scale = 1.0 / std::sqrt(use_dyn ? 9.0 : 5.0);
        for (int t = 0; t < nr; ++t) {
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

    // 3. Fused dual-job ink_conv_few (e.g. K and V)
    const int few_rows = 4;
    float* d_st_few0 = dev(st0);
    float* d_st_few1 = dev(st0);
    float* d_out_few0 = dev(std::vector<float>((size_t) few_rows * channels, 0.f));
    float* d_out_few1 = dev(std::vector<float>((size_t) few_rows * channels, 0.f));

    InkConv j0, j1;
    j0.x = d_x; j0.state = d_st_few0; j0.w = d_w; j0.dyn = nullptr; j0.out = d_out_few0;
    j1.x = d_x; j1.state = d_st_few1; j1.w = d_w; j1.dyn = d_dyn; j1.out = d_out_few1;

    ink_conv_few(j0, j1, channels, head_dim, few_rows, nullptr);

    auto [want_few0, want_st_few0] = ref_conv(false, few_rows);
    auto [want_few1, want_st_few1] = ref_conv(true, few_rows);

    bad += report("ink_conv_few dual job 0 (static out)", host(d_out_few0, (size_t) few_rows * channels), want_few0, 1e-5);
    bad += report("ink_conv_few dual job 1 (dynamic out)", host(d_out_few1, (size_t) few_rows * channels), want_few1, 1e-5);
    bad += report("ink_conv_few dual job 0 (state)", host(d_st_few0, st0.size()), want_st_few0, 1e-6);
    bad += report("ink_conv_few dual job 1 (state)", host(d_st_few1, st0.size()), want_st_few1, 1e-6);

    // 4. Strided dual-job ink_conv_few (x_stride = 2 * channels, rows = 4)
    std::vector<float> x_strided((size_t) few_rows * 2 * channels);
    for (int t = 0; t < few_rows; ++t) {
        std::copy_n(x.data() + (size_t) t * channels, channels, x_strided.data() + (size_t) t * 2 * channels);
        std::copy_n(x.data() + (size_t) t * channels, channels, x_strided.data() + (size_t) t * 2 * channels + channels);
    }
    float* d_xs = dev(x_strided);
    cudaMemcpy(d_st_few0, st0.data(), st0.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_st_few1, st0.data(), st0.size() * sizeof(float), cudaMemcpyHostToDevice);
    j0.x = d_xs; j0.x_stride = 2 * channels;
    j1.x = d_xs + channels; j1.x_stride = 2 * channels; j1.dyn = nullptr;
    ink_conv_few(j0, j1, channels, head_dim, few_rows, nullptr);
    bad += report("ink_conv_few strided job 0", host(d_out_few0, (size_t) few_rows * channels), want_few0, 1e-5);
    bad += report("ink_conv_few strided job 1", host(d_out_few1, (size_t) few_rows * channels), want_few0, 1e-5);
    bad += report("ink_conv_few strided state 0", host(d_st_few0, st0.size()), want_st_few0, 1e-6);
    bad += report("ink_conv_few strided state 1", host(d_st_few1, st0.size()), want_st_few0, 1e-6);
    cudaFree(d_xs);

    cudaFree(d_st_few0); cudaFree(d_st_few1); cudaFree(d_out_few0); cudaFree(d_out_few1);
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
        if (total_keys == 1500 && mode == Mode::kFp16 && n_head == 24) {
            bad += report("inkling_rel_table", host(d_tbl, want_tbl.size()), want_tbl, 1e-5);
            if (n_rel == 16) {
                std::vector<float> transposed_bias(rel_bias.size());
                for (int d = 0; d < n_extent; ++d) {
                    for (int k = 0; k < n_rel; ++k) {
                        transposed_bias[(size_t) k * (size_t) n_extent + d] = rel_bias[(size_t) d * (size_t) n_rel + k];
                    }
                }
                float* d_rb_tr = dev(transposed_bias);
                float* d_tbl_tr = dev(std::vector<float>((size_t) q_rows * n_head * n_extent, 0.f));
                inkling_rel_table(d_rcur, d_rb_tr, d_snk, d_tbl_tr, n_rel, n_extent, n_head, q_rows, nullptr, true);
                bad += report("inkling_rel_table (transposed term-major)", host(d_tbl_tr, want_tbl.size()), want_tbl, 1e-5);
                std::vector<float> got_orig = host(d_tbl, want_tbl.size());
                std::vector<float> got_tr = host(d_tbl_tr, want_tbl.size());
                if (std::memcmp(got_orig.data(), got_tr.data(), got_orig.size() * sizeof(float)) != 0) {
                    std::fprintf(stderr, "FAIL: inkling_rel_table bitwise divergence between regular and transposed\n");
                    bad++;
                }
            }
        }

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

        const int n_splits = std::clamp((total_keys + q_rows + 127) / 128 + 1, 1, 64);
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

        float* d_out_flash = nullptr;
        if (!use_at && dim == 128 && n_rel == 16) {
            d_out_flash = dev(std::vector<float>(q.size(), 0.f));
            InkAttn ia;
            ia.q = d_q;
            ia.r = d_rcur;
            ia.sink = d_snk;
            ia.keys = quant ? nullptr : d_kf16;
            ia.vals = quant ? nullptr : d_vf16;
            ia.rel_bias = d_rb;
            ia.out = d_out_flash;
            ia.pos = total_keys;
            ia.rows = q_rows;
            ia.n_head = n_head;
            ia.n_kv = kv_heads;
            ia.dim = dim;
            ia.n_rel = n_rel;
            ia.extent = n_extent;
            ia.window = window;
            ia.ring = ring;
            ia.scale = scale;
            ia.kq8 = kq;
            ia.vq8 = vq;
            ia.kq4 = kq4;
            ia.vq4 = vq4;
            if (ink_attend_batch(ia, nullptr)) {
                auto got_flash = host(d_out_flash, q.size());
                std::snprintf(buf, sizeof(buf), "%s flash vs attend", label);
                bad += report(buf, got_flash, got_dec, 5e-3);
            }
        }

        cudaFree(d_q); cudaFree(d_rcur); cudaFree(d_rb); cudaFree(d_snk); cudaFree(d_tbl);
        cudaFree(d_kf16); cudaFree(d_vf16); cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_k4); cudaFree(d_v4);
        cudaFree(d_ks); cudaFree(d_vs); cudaFree(d_ks4); cudaFree(d_vs4);
        cudaFree(d_score); cudaFree(d_out_dec); cudaFree(d_out_bat); cudaFree(d_work);
        if (d_out_flash) cudaFree(d_out_flash);
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

int test_fused_and_predequant() {
    int bad = 0;

    // 1. In-place inkling_add_to_stream(nullptr, ...)
    {
        const int rows = 5, streams = 6, dst_s = 2, n_embd = 64;
        auto r0 = random_vec((size_t) rows * streams * n_embd);
        auto b = random_vec((size_t) rows * n_embd);
        std::vector<float> want_r = r0;
        for (int t = 0; t < rows; ++t)
            for (int i = 0; i < n_embd; ++i)
                want_r[((size_t) t * streams + dst_s) * n_embd + i] += b[(size_t) t * n_embd + i];
        float *d_r = dev(r0), *d_b = dev(b);
        inkling_add_to_stream(nullptr, d_b, d_r, streams, dst_s, n_embd, rows, nullptr);
        cudaDeviceSynchronize();
        bad += report("inkling_add_to_stream in-place", host(d_r, want_r.size()), want_r, 1e-6);
        cudaFree(d_r); cudaFree(d_b);
    }

    // 2. inkling_swiglu_gate_up_q8_1 vs FP64 Q8_1 reference
    {
        const int rows = 4, ff_dim = 128, nb = ff_dim / 32;
        auto gu = random_vec((size_t) rows * 2 * ff_dim, 2.0f);
        const size_t block_bytes = 36;
        uint8_t* d_q8 = nullptr;
        cudaMalloc(&d_q8, (size_t) rows * nb * block_bytes);
        float* d_gu = dev(gu);
        inkling_swiglu_gate_up_q8_1(d_gu, d_q8, ff_dim, rows, nullptr);
        cudaDeviceSynchronize();
        std::vector<uint8_t> got_bytes((size_t) rows * nb * block_bytes);
        cudaMemcpy(got_bytes.data(), d_q8, got_bytes.size(), cudaMemcpyDeviceToHost);

        std::vector<float> got_deq((size_t) rows * ff_dim), want_deq((size_t) rows * ff_dim);
        std::vector<float> got_sum((size_t) rows * nb), want_sum((size_t) rows * nb);
        for (int t = 0; t < rows; ++t) {
            const float* r = gu.data() + (size_t) t * 2 * ff_dim;
            for (int b = 0; b < nb; ++b) {
                double prod[32], amax = 0.0, s = 0.0;
                for (int j = 0; j < 32; ++j) {
                    const double g = r[b * 32 + j], u = r[ff_dim + b * 32 + j];
                    prod[j] = (g / (1.0 + std::exp(-g))) * u;
                    amax = std::max(amax, std::abs(prod[j]));
                    s += prod[j];
                }
                const float d_raw = (float) (amax / 127.0);
                const float d_ref = __half2float(__float2half(d_raw));
                want_sum[(size_t) t * nb + b] = __half2float(__float2half((float) s));
                const uint8_t* blk = got_bytes.data() + ((size_t) t * nb + b) * block_bytes;
                uint16_t ds_h[2];
                std::memcpy(ds_h, blk, 4);
                const float d_got = __half2float(__ushort_as_half(ds_h[0]));
                got_sum[(size_t) t * nb + b] = __half2float(__ushort_as_half(ds_h[1]));
                for (int j = 0; j < 32; ++j) {
                    const int q_ref = d_raw > 0.f ? std::clamp((int) std::llround(prod[j] / d_raw), -127, 127) : 0;
                    want_deq[((size_t) t * ff_dim) + b * 32 + j] = q_ref * d_ref;
                    got_deq[((size_t) t * ff_dim) + b * 32 + j] = ((int8_t) blk[4 + j]) * d_got;
                }
            }
        }
        bad += report("inkling_swiglu_gate_up_q8_1 deq", got_deq, want_deq, 1e-4);
        bad += report("inkling_swiglu_gate_up_q8_1 sum", got_sum, want_sum, 5e-3);
        cudaFree(d_q8); cudaFree(d_gu);
    }

    // 4. inkling_dequant_kv_span + shifted-pos inkling_attend_batch (Q8_0, Q4_0, K8V4 on ring SWA)
    {
        const int nh = 24, nkv = 8, dim = 128, win = 16, ring = 24, n_extent = 16;
        const int64_t pos = 20;
        const int q_rows = 8;
        const int width = nkv * dim, nb = width / 32;
        const float scale = 1.0f / std::sqrt((float) dim);

        auto k_all = random_vec((size_t) (pos + q_rows) * width, 0.5f);
        auto v_all = random_vec((size_t) (pos + q_rows) * width, 0.5f);
        auto q_h = random_vec((size_t) q_rows * nh * dim, 0.5f);
        auto r_h = random_vec((size_t) q_rows * nh * 16, 0.3f);
        auto rb_h = random_vec((size_t) n_extent * 16, 0.25f);
        auto snk = random_vec((size_t) q_rows * nh, 0.3f);
        std::vector<float> tbl((size_t) q_rows * nh * n_extent);
        for (int t = 0; t < q_rows; ++t)
            for (int h = 0; h < nh; ++h)
                for (int d = 0; d < n_extent; ++d) {
                    double dot = 0.0;
                    for (int k = 0; k < 16; ++k)
                        dot += (double) r_h[((size_t) t * nh + h) * 16 + k] * rb_h[(size_t) d * 16 + k];
                    tbl[((size_t) t * nh + h) * n_extent + d] = (float) (dot - snk[(size_t) t * nh + h]);
                }

        std::vector<int8_t> k8c((size_t) ring * width, 0), v8c((size_t) ring * width, 0);
        std::vector<uint8_t> k4c((size_t) ring * (width / 2), 0), v4c((size_t) ring * (width / 2), 0);
        std::vector<uint16_t> ks8((size_t) ring * nb, 0), vs8((size_t) ring * nb, 0);
        std::vector<uint16_t> ks4((size_t) ring * nb, 0), vs4((size_t) ring * nb, 0);
        for (int64_t p = 0; p < pos + q_rows; ++p) {
            const int64_t r = p % ring;
            for (int b = 0; b < nb; ++b) {
                float ak = 0.f, av = 0.f;
                for (int j = 0; j < 32; ++j) {
                    ak = std::max(ak, std::abs(k_all[(size_t) p * width + b * 32 + j]));
                    av = std::max(av, std::abs(v_all[(size_t) p * width + b * 32 + j]));
                }
                __half dkh8 = __float2half(ak / 127.0f), dvh8 = __float2half(av / 127.0f);
                __half dkh4 = __float2half(ak / 7.0f), dvh4 = __float2half(av / 7.0f);
                ks8[(size_t) r * nb + b] = __half_as_ushort(dkh8);
                vs8[(size_t) r * nb + b] = __half_as_ushort(dvh8);
                ks4[(size_t) r * nb + b] = __half_as_ushort(dkh4);
                vs4[(size_t) r * nb + b] = __half_as_ushort(dvh4);
                float fk8 = __half2float(dkh8), fv8 = __half2float(dvh8);
                float fk4 = __half2float(dkh4), fv4 = __half2float(dvh4);
                for (int j = 0; j < 32; ++j) {
                    k8c[(size_t) r * width + b * 32 + j] =
                        fk8 > 0 ? (int8_t) std::clamp((int) std::llround(k_all[(size_t) p * width + b * 32 + j] / fk8), -127, 127) : 0;
                    v8c[(size_t) r * width + b * 32 + j] =
                        fv8 > 0 ? (int8_t) std::clamp((int) std::llround(v_all[(size_t) p * width + b * 32 + j] / fv8), -127, 127) : 0;
                }
                for (int j = 0; j < 16; ++j) {
                    int k0 = fk4 > 0 ? std::clamp((int) std::llround(k_all[(size_t) p * width + b * 32 + j] / fk4), -8, 7) + 8 : 8;
                    int k1 = fk4 > 0 ? std::clamp((int) std::llround(k_all[(size_t) p * width + b * 32 + 16 + j] / fk4), -8, 7) + 8 : 8;
                    int v0 = fv4 > 0 ? std::clamp((int) std::llround(v_all[(size_t) p * width + b * 32 + j] / fv4), -8, 7) + 8 : 8;
                    int v1 = fv4 > 0 ? std::clamp((int) std::llround(v_all[(size_t) p * width + b * 32 + 16 + j] / fv4), -8, 7) + 8 : 8;
                    k4c[(size_t) r * (width / 2) + b * 16 + j] = (uint8_t) (k0 | (k1 << 4));
                    v4c[(size_t) r * (width / 2) + b * 16 + j] = (uint8_t) (v0 | (v1 << 4));
                }
            }
        }
        float *d_q = dev(q_h), *d_r = dev(r_h), *d_rb = dev(rb_h), *d_tbl = dev(tbl), *d_snk = dev(snk);
        int8_t *d_k8 = dev(k8c), *d_v8 = dev(v8c);
        uint8_t *d_k4 = dev(k4c), *d_v4 = dev(v4c);
        uint16_t *d_ks8 = dev(ks8), *d_vs8 = dev(vs8), *d_ks4 = dev(ks4), *d_vs4 = dev(vs4);
        const size_t score_n = std::max((size_t) q_rows * nh * 64 * (dim + 2), (size_t) q_rows * nh * 2048);
        float *d_sc = dev(std::vector<float>(score_n, 0.f)),
              *d_wrk = dev(std::vector<float>((size_t) q_rows * nh * (dim + dim / 2 + 3) + 1, 0.f)),
              *d_dec = dev(std::vector<float>(q_h.size(), 0.f)),
              *d_bat = dev(std::vector<float>(q_h.size(), 0.f)),
              *d_fls = dev(std::vector<float>(q_h.size(), 0.f));
        const int64_t lo_all = pos >= win ? (pos - win + 1) : 0;
        const int count_all = (int) ((pos + q_rows) - lo_all);
        uint16_t *d_kw = dev(std::vector<uint16_t>((size_t) count_all * width, 0)),
                 *d_vw = dev(std::vector<uint16_t>((size_t) count_all * width, 0));
        void* blas = blas_create(nullptr);
        auto run_pre = [&](const char* label, Kvq8 kq8, Kvq8 vq8, Kvq4 kq4, Kvq4 vq4) {
            inkling_attend(d_q, nullptr, nullptr, d_tbl, d_snk, n_extent, pos, q_rows, d_sc, q_rows, d_dec,
                           dim, nh, nkv, scale, nullptr, win, ring, nullptr, 0, kq8, vq8, kq4, vq4);
            inkling_dequant_kv_span(lo_all, count_all, width, ring, kq8, vq8, kq4, vq4, d_kw, d_vw, nullptr);
            inkling_attend_batch(blas, d_q, d_kw, d_vw, d_tbl, d_snk, n_extent, pos - lo_all, q_rows,
                                 d_sc, score_n, d_wrk, d_bat, dim, nh, nkv, scale, nullptr, win, 0);
            cudaDeviceSynchronize();
            bad += report(label, host(d_bat, q_h.size()), host(d_dec, q_h.size()), 5e-3);
            InkAttn ia;
            ia.q = d_q; ia.r = d_r; ia.sink = d_snk; ia.rel_bias = d_rb; ia.out = d_fls;
            ia.pos = pos; ia.rows = q_rows; ia.n_head = nh; ia.n_kv = nkv; ia.dim = dim;
            ia.n_rel = 16; ia.extent = n_extent; ia.window = win; ia.ring = ring; ia.scale = scale;
            ia.kq8 = kq8; ia.vq8 = vq8; ia.kq4 = kq4; ia.vq4 = vq4;
            if (ink_attend_batch(ia, nullptr)) {
                cudaDeviceSynchronize();
                char flabel[96];
                std::snprintf(flabel, sizeof(flabel), "%s flash", label);
                bad += report(flabel, host(d_fls, q_h.size()), host(d_dec, q_h.size()), 5e-3);
            }
        };
        run_pre("Pre-dequant ring SWA Q8_0", {d_k8, d_ks8}, {d_v8, d_vs8}, {}, {});
        run_pre("Pre-dequant ring SWA Q4_0", {}, {}, {d_k4, d_ks4}, {d_v4, d_vs4});
        run_pre("Pre-dequant ring SWA K8V4", {d_k8, d_ks8}, {}, {}, {d_v4, d_vs4});
        blas_destroy(blas);
        cudaFree(d_q); cudaFree(d_r); cudaFree(d_rb); cudaFree(d_tbl); cudaFree(d_snk);
        cudaFree(d_k8); cudaFree(d_v8); cudaFree(d_k4); cudaFree(d_v4);
        cudaFree(d_ks8); cudaFree(d_vs8); cudaFree(d_ks4); cudaFree(d_vs4);
        cudaFree(d_sc); cudaFree(d_wrk); cudaFree(d_dec); cudaFree(d_bat); cudaFree(d_fls);
        cudaFree(d_kw); cudaFree(d_vw);
    }

    // 5. inkling_unpack_q_fused (rows = 4)
    {
        const int rows = 4, q_dim = 128, rel_dim = 64, n_head = 8;
        const int rel_off = 2 * q_dim, sink_off = rel_off + rel_dim, stride = sink_off + n_head;
        auto src = random_vec((size_t) rows * stride, 1.0f);
        float *d_src = dev(src),
              *d_qraw = dev(std::vector<float>((size_t) rows * q_dim, 0.f)),
              *d_gate = dev(std::vector<float>((size_t) rows * q_dim, 0.f)),
              *d_r = dev(std::vector<float>((size_t) rows * rel_dim, 0.f)),
              *d_s = dev(std::vector<float>((size_t) rows * n_head, 0.f));
        inkling_unpack_q_fused(d_src, d_qraw, d_gate, d_r, d_s, q_dim, rel_dim, n_head, rel_off, sink_off, stride, rows,
                               nullptr);
        cudaDeviceSynchronize();
        std::vector<float> w_q((size_t) rows * q_dim), w_g((size_t) rows * q_dim),
            w_r((size_t) rows * rel_dim), w_s((size_t) rows * n_head);
        for (int t = 0; t < rows; ++t) {
            const float* row = src.data() + (size_t) t * stride;
            std::copy_n(row, q_dim, w_q.data() + (size_t) t * q_dim);
            std::copy_n(row + q_dim, q_dim, w_g.data() + (size_t) t * q_dim);
            std::copy_n(row + rel_off, rel_dim, w_r.data() + (size_t) t * rel_dim);
            std::copy_n(row + sink_off, n_head, w_s.data() + (size_t) t * n_head);
        }
        bad += report("inkling_unpack_q_fused qraw", host(d_qraw, w_q.size()), w_q, 1e-6);
        bad += report("inkling_unpack_q_fused gate", host(d_gate, w_g.size()), w_g, 1e-6);
        bad += report("inkling_unpack_q_fused r_raw", host(d_r, w_r.size()), w_r, 1e-6);
        bad += report("inkling_unpack_q_fused sinks", host(d_s, w_s.size()), w_s, 1e-6);
        cudaFree(d_src); cudaFree(d_qraw); cudaFree(d_gate); cudaFree(d_r); cudaFree(d_s);
    }

    // 6. ink_ctrl_down fast one-pass kernel vs default (count = 3, rows = 4)
    {
        const int n_embd = 1024, rank = 64, rows = 4;
        std::vector<std::vector<float>> cv(3), mv(3), wv(3);
        float *dc[3] = {}, *dm[3] = {}, *dw[3] = {}, *dy[3] = {};
        const float *cc[3] = {}, *cm[3] = {}, *cw[3] = {};
        for (int j = 0; j < 3; ++j) {
            cv[j] = random_vec((size_t) rows * n_embd, 1.0f);
            mv[j] = random_vec((size_t) rows * n_embd, 1.0f);
            wv[j] = random_vec((size_t) rank * 2 * n_embd, 0.25f);
            cc[j] = dc[j] = dev(cv[j]);
            cm[j] = dm[j] = dev(mv[j]);
            cw[j] = dw[j] = dev(wv[j]);
            dy[j] = dev(std::vector<float>((size_t) rows * rank, 0.f));
        }
        ink_set_fast_ctrl(false);
        if (!ink_ctrl_down(cc, cm, cw, dy, 3, n_embd, rank, rows, nullptr)) ++bad;
        cudaDeviceSynchronize();
        std::vector<std::vector<float>> ref(3);
        for (int j = 0; j < 3; ++j) ref[j] = host(dy[j], (size_t) rows * rank);
        ink_set_fast_ctrl(true);
        if (!ink_ctrl_down(cc, cm, cw, dy, 3, n_embd, rank, rows, nullptr)) ++bad;
        cudaDeviceSynchronize();
        for (int j = 0; j < 3; ++j) {
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl), "ink_ctrl_down_fast[%d]", j);
            bad += report(lbl, host(dy[j], (size_t) rows * rank), ref[j], 1e-5);
            cudaFree(dc[j]); cudaFree(dm[j]); cudaFree(dw[j]); cudaFree(dy[j]);
        }
        ink_set_fast_ctrl(false);
    }

    // 7. Multi-token ExpertBell (bell_ring n_tokens=4, skip_from=4 + bell_add + zero-miss short-circuit)
    {
        const int n_tokens = 4, n_used = 8, skip_from = 4, n_embd = 64, n_exp = 16;
        ExpertBell* h_bell = nullptr;
        if (cudaHostAlloc(&h_bell, sizeof(ExpertBell), cudaHostAllocMapped) != cudaSuccess) ++bad;
        std::memset(h_bell, 0, sizeof(ExpertBell));
        std::vector<void*> slots(n_exp, (void*) 0x1000);
        // token 0: k=0..7 all hit except k=5 (skipped >= skip_from)
        // token 1: k=1 -> expert 3 miss (miss #0), k=3 -> expert 7 miss (miss #1)
        // token 2: k=0 -> expert 1 miss (miss #2), k=6 -> expert 3 (skipped >= skip_from)
        // token 3: all hit
        slots[3] = nullptr;
        slots[7] = nullptr;
        slots[1] = nullptr;
        void** d_slots = nullptr;
        cudaMalloc(&d_slots, (size_t) n_exp * sizeof(void*));
        cudaMemcpy(d_slots, slots.data(), (size_t) n_exp * sizeof(void*), cudaMemcpyHostToDevice);
        std::vector<int32_t> h_ids(n_tokens * n_used, 0);
        std::vector<float> h_wts(n_tokens * n_used, 0.1f);
        h_ids[0 * n_used + 5] = 3;
        h_ids[1 * n_used + 1] = 3; h_wts[1 * n_used + 1] = 0.25f;
        h_ids[1 * n_used + 3] = 7; h_wts[1 * n_used + 3] = 0.50f;
        h_ids[2 * n_used + 0] = 1; h_wts[2 * n_used + 0] = 0.75f;
        h_ids[2 * n_used + 6] = 3;
        int32_t* d_ids = dev(h_ids);
        float* d_wts = dev(h_wts);
        auto x_h = random_vec((size_t) n_tokens * n_embd, 1.0f);
        auto acc_h = random_vec((size_t) n_tokens * n_embd, 1.0f);
        auto hy_h = random_vec((size_t) n_tokens * n_embd, 1.0f);
        float *d_x = dev(x_h), *d_acc = dev(acc_h), *d_hx = nullptr, *d_hy = nullptr;
        cudaHostAlloc(&d_hx, (size_t) n_tokens * n_embd * sizeof(float), cudaHostAllocMapped);
        cudaHostAlloc(&d_hy, (size_t) n_tokens * n_embd * sizeof(float), cudaHostAllocMapped);
        std::memcpy(d_hy, hy_h.data(), hy_h.size() * sizeof(float));
        uint32_t* d_seq = nullptr;
        cudaMalloc(&d_seq, 2 * sizeof(uint32_t));
        cudaMemset(d_seq, 0, 2 * sizeof(uint32_t));

        // First test zero-miss token-only call (token 3 alone)
        bell_ring(d_slots, d_ids + 3 * n_used, d_wts + 3 * n_used, 1, n_used, skip_from, 5, d_x, d_hx, n_embd,
                  h_bell, d_seq, nullptr);
        uint32_t seq_h[2] = {};
        cudaMemcpy(seq_h, d_seq, sizeof(seq_h), cudaMemcpyDeviceToHost);
        if (seq_h[0] != 0 || seq_h[1] != 0) {
            std::printf("FAIL bell_ring zero-miss seq=(%u,%u)\n", seq_h[0], seq_h[1]);
            ++bad;
        }

        // Now test n_tokens = 4
        bell_ring(d_slots, d_ids, d_wts, n_tokens, n_used, skip_from, 11, d_x, d_hx, n_embd, h_bell, d_seq, nullptr);
        cudaDeviceSynchronize();
        const bool ok_meta = h_bell->ring == 1 && h_bell->layer == 11 && h_bell->tok_n == 4 && h_bell->n == 3 &&
                             h_bell->e[0] == 3 && h_bell->tok[0] == 1 && std::abs(h_bell->w[0] - 0.25f) < 1e-6f &&
                             h_bell->e[1] == 7 && h_bell->tok[1] == 1 && std::abs(h_bell->w[1] - 0.50f) < 1e-6f &&
                             h_bell->e[2] == 1 && h_bell->tok[2] == 2 && std::abs(h_bell->w[2] - 0.75f) < 1e-6f;
        if (!ok_meta) {
            std::printf("FAIL bell_ring multi-token metadata\n");
            ++bad;
        }
        std::vector<float> got_hx(d_hx, d_hx + (size_t) n_tokens * n_embd);
        bad += report("bell_ring multi-token hx", got_hx, x_h, 1e-6);

        h_bell->ack = h_bell->ring;
        bell_add(h_bell, d_seq, d_hy, d_acc, n_tokens * n_embd, nullptr);
        cudaDeviceSynchronize();
        std::vector<float> want_acc(acc_h.size());
        for (size_t i = 0; i < acc_h.size(); ++i) want_acc[i] = acc_h[i] + hy_h[i];
        bad += report("bell_add multi-token acc", host(d_acc, want_acc.size()), want_acc, 1e-6);

        cudaFree(d_slots); cudaFree(d_ids); cudaFree(d_wts);
        cudaFree(d_x); cudaFree(d_acc); cudaFree(d_seq);
        cudaFreeHost(d_hx); cudaFreeHost(d_hy); cudaFreeHost(h_bell);
    }
    return bad;
}

int test_decode_fusions() {
    int bad = 0;

    // F1. Direct r_cur + rel_bias + fused gate in inkling_attend()
    {
        auto run_f1 = [&](const char* label, int rows, int pos, int window, int ring, int n_extent, int kv_mode) {
            const int n_head = 24, kv_heads = 8, dim = 128, n_rel = 16;
            const int width = kv_heads * dim;
            const int total_rows = ring > 0 ? ring : (pos + rows);
            const float scale = 1.0f / std::sqrt((float) dim);
            auto q = random_vec((size_t) rows * n_head * dim, 0.7f);
            auto r_h = random_vec((size_t) rows * n_head * n_rel, 0.5f);
            auto rb_h = random_vec((size_t) n_extent * n_rel, 0.3f);
            auto snk = random_vec((size_t) rows * n_head, 0.3f);
            auto gate = random_vec((size_t) rows * n_head * dim, 1.2f);
            auto k_raw = random_vec((size_t) total_rows * width, 0.6f);
            auto v_raw = random_vec((size_t) total_rows * width, 0.6f);

            float *d_q = dev(q), *d_r = dev(r_h), *d_rb = dev(rb_h), *d_snk = dev(snk), *d_g = dev(gate);
            float *d_kr = dev(k_raw), *d_vr = dev(v_raw);
            uint16_t *d_k16 = nullptr, *d_v16 = nullptr;
            int8_t *d_kc = nullptr, *d_vc = nullptr;
            uint8_t *d_v4 = nullptr;
            uint16_t *d_ks = nullptr, *d_vs = nullptr;
            if (kv_mode == 0) {
                cudaMalloc(&d_k16, (size_t) total_rows * width * sizeof(uint16_t));
                cudaMalloc(&d_v16, (size_t) total_rows * width * sizeof(uint16_t));
                store_f16(d_kr, d_k16, width, 0, total_rows, nullptr);
                store_f16(d_vr, d_v16, width, 0, total_rows, nullptr);
            } else if (kv_mode == 1) {
                cudaMalloc(&d_kc, (size_t) total_rows * width);
                cudaMalloc(&d_vc, (size_t) total_rows * width);
                cudaMalloc(&d_ks, (size_t) total_rows * (width / 32) * sizeof(uint16_t));
                cudaMalloc(&d_vs, (size_t) total_rows * (width / 32) * sizeof(uint16_t));
                store_q8(d_kr, d_kc, d_ks, width, 0, total_rows, nullptr);
                store_q8(d_vr, d_vc, d_vs, width, 0, total_rows, nullptr);
            } else {
                cudaMalloc(&d_kc, (size_t) total_rows * width);
                cudaMalloc(&d_v4, (size_t) total_rows * (width / 2));
                cudaMalloc(&d_ks, (size_t) total_rows * (width / 32) * sizeof(uint16_t));
                cudaMalloc(&d_vs, (size_t) total_rows * (width / 32) * sizeof(uint16_t));
                store_q8(d_kr, d_kc, d_ks, width, 0, total_rows, nullptr);
                store_q4(d_vr, d_v4, d_vs, width, 0, total_rows, nullptr);
            }
            Kvq8 kq8 = (kv_mode == 1 || kv_mode == 2) ? Kvq8{d_kc, d_ks} : Kvq8{};
            Kvq8 vq8 = (kv_mode == 1) ? Kvq8{d_vc, d_vs} : Kvq8{};
            Kvq4 vq4 = (kv_mode == 2) ? Kvq4{d_v4, d_vs} : Kvq4{};

            const size_t score_n = (size_t) rows * n_head * 64 * (dim + 2);
            float *d_tbl = dev(std::vector<float>((size_t) rows * n_head * n_extent, 0.f));
            float *d_sc = dev(std::vector<float>(score_n, 0.f));
            float *d_ref = dev(std::vector<float>(q.size(), 0.f));
            float *d_got = dev(std::vector<float>(q.size(), 0.f));
            std::vector<long long> at_v((size_t) rows * 2);
            for (int t = 0; t < rows; ++t) at_v[2 * t] = at_v[2 * t + 1] = pos + t;
            long long* d_at = ring > 0 ? dev(at_v) : nullptr;
            const int rm = d_at ? (window > 0 ? std::min(window + rows - 1, pos + rows) : pos + rows) : 0;

            inkling_rel_table(d_r, d_rb, d_snk, d_tbl, n_rel, n_extent, n_head, rows, nullptr, false);
            inkling_attend(d_q, d_k16, d_v16, d_tbl, d_snk, n_extent, pos, rows, d_sc, rows, d_ref, dim, n_head,
                           kv_heads, scale, nullptr, window, ring, d_at, rm, kq8, vq8, {}, vq4);
            mul_sigmoid(d_ref, d_g, n_head * dim, rows, n_head * dim, false, nullptr);

            inkling_attend(d_q, d_k16, d_v16, nullptr, d_snk, n_extent, pos, rows, d_sc, rows, d_got, dim, n_head,
                           kv_heads, scale, nullptr, window, ring, d_at, rm, kq8, vq8, {}, vq4, d_r, d_rb, n_rel,
                           d_g);
            cudaDeviceSynchronize();
            bad += report(label, host(d_got, q.size()), host(d_ref, q.size()), 1e-4);

            cudaFree(d_q); cudaFree(d_r); cudaFree(d_rb); cudaFree(d_snk); cudaFree(d_g);
            cudaFree(d_kr); cudaFree(d_vr);
            cudaFree(d_k16); cudaFree(d_v16); cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_v4);
            cudaFree(d_ks); cudaFree(d_vs); cudaFree(d_tbl); cudaFree(d_sc);
            cudaFree(d_ref); cudaFree(d_got); cudaFree(d_at);
        };
        run_f1("F1 attend direct+gate SWA FP16 r=1", 1, 700, 1024, 1024, 1024, 0);
        run_f1("F1 attend direct+gate SWA K8V4 r=4", 4, 1020, 1024, 1024, 1024, 2);
        run_f1("F1 attend direct+gate Global Q8_0 r=4", 4, 900, 0, 0, 512, 1);
    }

    // F2. inkling_clips_store against inkling_l2_clip + rms_norm + store_*
    {
        auto run_f2 = [&](const char* label, int rows, int pos, int ring, bool is_global, bool use_at, int mode) {
            const int n_head = 24, n_kv = 8, dim = 128, n_rel = 16;
            const int kv_width = n_kv * dim, kv_nb = kv_width / 32;
            const int cap = ring > 0 ? ring : (pos + rows);
            const float qk_clip = 2.0f, rel_clip = 1.5f, eps = 1e-6f;
            const float log_floor = 64.0f, log_alpha = 0.1f;

            auto q_in = random_vec((size_t) rows * n_head * dim, 1.5f);
            auto q_sc = random_vec((size_t) n_head * dim, 0.8f);
            auto k_in = random_vec((size_t) rows * kv_width, 1.5f);
            auto v_in = random_vec((size_t) rows * kv_width, 1.5f);
            auto r_in = random_vec((size_t) rows * n_head * n_rel, 1.2f);
            auto r_sc = random_vec((size_t) n_head * n_rel, 0.8f);

            float *d_qin = dev(q_in), *d_qsc = dev(q_sc), *d_kin = dev(k_in), *d_vin = dev(v_in);
            float *d_rin = dev(r_in), *d_rsc = dev(r_sc);
            float *d_qref = dev(std::vector<float>(q_in.size(), 0.f));
            float *d_kref = dev(std::vector<float>(k_in.size(), 0.f));
            float *d_vref = dev(std::vector<float>(v_in.size(), 0.f));
            float *d_rref = dev(std::vector<float>(r_in.size(), 0.f));
            float *d_qgot = dev(std::vector<float>(q_in.size(), 0.f));
            float *d_rgot = dev(std::vector<float>(r_in.size(), 0.f));

            std::vector<long long> at_v((size_t) rows * 2);
            for (int t = 0; t < rows; ++t) at_v[2 * t] = at_v[2 * t + 1] = pos + t;
            long long* d_at = use_at ? dev(at_v) : nullptr;

            // Ref outputs
            inkling_l2_clip(d_qin, d_qsc, d_qref, dim, n_head, qk_clip, pos, is_global, log_floor, log_alpha, rows,
                            d_at, nullptr);
            inkling_l2_clip(d_kin, nullptr, d_kref, dim, n_kv, qk_clip, pos, false, log_floor, 0.f, rows, nullptr,
                            nullptr);
            rms_norm(d_vin, nullptr, d_vref, dim, rows * n_kv, eps, nullptr);
            inkling_l2_clip(d_rin, d_rsc, d_rref, n_rel, n_head, rel_clip, pos, is_global, log_floor, log_alpha, rows,
                            d_at, nullptr);

            uint16_t *d_k16_r = nullptr, *d_v16_r = nullptr, *d_k16_g = nullptr, *d_v16_g = nullptr;
            int8_t *d_k8_r = nullptr, *d_v8_r = nullptr, *d_k8_g = nullptr, *d_v8_g = nullptr;
            uint8_t *d_k4_r = nullptr, *d_v4_r = nullptr, *d_k4_g = nullptr, *d_v4_g = nullptr;
            uint16_t *d_ks_r = nullptr, *d_vs_r = nullptr, *d_ks_g = nullptr, *d_vs_g = nullptr;

            InkClipsStore cs{};
            cs.q_in = d_qin; cs.q_scale = d_qsc; cs.q_out = d_qgot;
            cs.k_in = d_kin; cs.v_in = d_vin;
            cs.r_in = d_rin; cs.r_scale = d_rsc; cs.r_out = d_rgot;
            cs.n_head = n_head; cs.n_kv = n_kv; cs.dim = dim; cs.n_rel = n_rel;
            cs.rows = rows; cs.ring = ring; cs.pos = pos; cs.at = d_at;
            cs.qk_clip = qk_clip; cs.rel_clip = rel_clip; cs.rms_eps = eps;
            cs.is_global = is_global; cs.log_floor = log_floor; cs.log_alpha = log_alpha;

            auto store_side = [&](const float* x, bool val, uint16_t* f16, int8_t* c8, uint8_t* c4, uint16_t* sc) {
                const bool u4 = (mode == 2) || (val && mode == 3);
                const bool u8 = (mode == 1) || (!val && mode == 3);
                if (use_at) {
                    if (u4) store_q4_at(x, c4, sc, kv_width, d_at, rows, ring, nullptr);
                    else if (u8) store_q8_at(x, c8, sc, kv_width, d_at, rows, ring, nullptr);
                    else store_f16_at(x, f16, kv_width, d_at, rows, ring, nullptr);
                } else if (ring > 0) {
                    const int at = pos % ring, head = std::min(rows, ring - at);
                    if (u4) {
                        store_q4(x, c4, sc, kv_width, at, head, nullptr);
                        if (head < rows) store_q4(x + (size_t) head * kv_width, c4, sc, kv_width, 0, rows - head, nullptr);
                    } else if (u8) {
                        store_q8(x, c8, sc, kv_width, at, head, nullptr);
                        if (head < rows) store_q8(x + (size_t) head * kv_width, c8, sc, kv_width, 0, rows - head, nullptr);
                    } else {
                        store_f16(x, f16, kv_width, at, head, nullptr);
                        if (head < rows) store_f16(x + (size_t) head * kv_width, f16, kv_width, 0, rows - head, nullptr);
                    }
                } else {
                    if (u4) store_q4(x, c4, sc, kv_width, pos, rows, nullptr);
                    else if (u8) store_q8(x, c8, sc, kv_width, pos, rows, nullptr);
                    else store_f16(x, f16, kv_width, pos, rows, nullptr);
                }
            };

            if (mode == 0) {
                d_k16_r = dev(std::vector<uint16_t>((size_t) cap * kv_width, 0));
                d_v16_r = dev(std::vector<uint16_t>((size_t) cap * kv_width, 0));
                d_k16_g = dev(std::vector<uint16_t>((size_t) cap * kv_width, 0));
                d_v16_g = dev(std::vector<uint16_t>((size_t) cap * kv_width, 0));
                cs.k_fp16 = d_k16_g; cs.v_fp16 = d_v16_g;
            } else {
                d_ks_r = dev(std::vector<uint16_t>((size_t) cap * kv_nb, 0));
                d_vs_r = dev(std::vector<uint16_t>((size_t) cap * kv_nb, 0));
                d_ks_g = dev(std::vector<uint16_t>((size_t) cap * kv_nb, 0));
                d_vs_g = dev(std::vector<uint16_t>((size_t) cap * kv_nb, 0));
                if (mode == 1 || mode == 3) {
                    d_k8_r = dev(std::vector<int8_t>((size_t) cap * kv_width, 0));
                    d_k8_g = dev(std::vector<int8_t>((size_t) cap * kv_width, 0));
                    cs.k_q8 = d_k8_g;
                } else {
                    d_k4_r = dev(std::vector<uint8_t>((size_t) cap * (kv_width / 2), 0));
                    d_k4_g = dev(std::vector<uint8_t>((size_t) cap * (kv_width / 2), 0));
                    cs.k_q4 = d_k4_g;
                }
                cs.k_scales = d_ks_g;
                if (mode == 1) {
                    d_v8_r = dev(std::vector<int8_t>((size_t) cap * kv_width, 0));
                    d_v8_g = dev(std::vector<int8_t>((size_t) cap * kv_width, 0));
                    cs.v_q8 = d_v8_g;
                } else {
                    d_v4_r = dev(std::vector<uint8_t>((size_t) cap * (kv_width / 2), 0));
                    d_v4_g = dev(std::vector<uint8_t>((size_t) cap * (kv_width / 2), 0));
                    cs.v_q4 = d_v4_g;
                }
                cs.v_scales = d_vs_g;
            }
            store_side(d_kref, false, d_k16_r, d_k8_r, d_k4_r, d_ks_r);
            store_side(d_vref, true, d_v16_r, d_v8_r, d_v4_r, d_vs_r);
            inkling_clips_store(cs, nullptr);
            cudaDeviceSynchronize();

            bad += report(label, host(d_qgot, q_in.size()), host(d_qref, q_in.size()), 1e-5);
            bad += report(label, host(d_rgot, r_in.size()), host(d_rref, r_in.size()), 1e-5);
            auto check_bytes = [&](const char* what, const void* g_ptr, const void* r_ptr, size_t nbytes) {
                std::vector<uint8_t> hg(nbytes), hr(nbytes);
                cudaMemcpy(hg.data(), g_ptr, nbytes, cudaMemcpyDeviceToHost);
                cudaMemcpy(hr.data(), r_ptr, nbytes, cudaMemcpyDeviceToHost);
                if (hg != hr) {
                    int diff = 0;
                    for (size_t i = 0; i < nbytes; ++i) diff += (hg[i] != hr[i]);
                    std::printf("%-42s MISMATCH %d/%zu bytes  FAIL\n", what, diff, nbytes);
                    ++bad;
                }
            };
            if (mode == 0) {
                check_bytes(label, d_k16_g, d_k16_r, (size_t) cap * kv_width * 2);
                check_bytes(label, d_v16_g, d_v16_r, (size_t) cap * kv_width * 2);
            } else {
                check_bytes(label, d_ks_g, d_ks_r, (size_t) cap * kv_nb * 2);
                check_bytes(label, d_vs_g, d_vs_r, (size_t) cap * kv_nb * 2);
                if (d_k8_g) check_bytes(label, d_k8_g, d_k8_r, (size_t) cap * kv_width);
                if (d_v8_g) check_bytes(label, d_v8_g, d_v8_r, (size_t) cap * kv_width);
                if (d_k4_g) check_bytes(label, d_k4_g, d_k4_r, (size_t) cap * (kv_width / 2));
                if (d_v4_g) check_bytes(label, d_v4_g, d_v4_r, (size_t) cap * (kv_width / 2));
            }
            cudaFree(d_qin); cudaFree(d_qsc); cudaFree(d_kin); cudaFree(d_vin); cudaFree(d_rin); cudaFree(d_rsc);
            cudaFree(d_qref); cudaFree(d_kref); cudaFree(d_vref); cudaFree(d_rref); cudaFree(d_qgot); cudaFree(d_rgot);
            cudaFree(d_at); cudaFree(d_k16_r); cudaFree(d_v16_r); cudaFree(d_k16_g); cudaFree(d_v16_g);
            cudaFree(d_k8_r); cudaFree(d_v8_r); cudaFree(d_k8_g); cudaFree(d_v8_g);
            cudaFree(d_k4_r); cudaFree(d_v4_r); cudaFree(d_k4_g); cudaFree(d_v4_g);
            cudaFree(d_ks_r); cudaFree(d_vs_r); cudaFree(d_ks_g); cudaFree(d_vs_g);
        };
        run_f2("F2 clips_store SWA wrap FP16 r=4", 4, 62, 64, false, false, 0);
        run_f2("F2 clips_store Global Q8_0 at r=4", 4, 150, 0, true, true, 1);
        run_f2("F2 clips_store SWA Q4_0 at r=3", 3, 62, 64, false, true, 2);
        run_f2("F2 clips_store Global K8V4 r=4", 4, 200, 0, true, false, 3);
    }

    // F3. ink_control3_gate (count=3 SWA aliasing + count=1)
    {
        for (int rows : {1, 4}) {
            const int n_embd = 512, rank = 64, count = 3;
            auto in0_h = random_vec((size_t) rows * n_embd, 0.7f);
            std::vector<float> msg_h[3], ch_h[3], gf_h[3], ref_h[3];
            float* dm[3], *dch[3], *dgf[3], *dout[3];
            for (int j = 0; j < count; ++j) {
                msg_h[j] = random_vec((size_t) rows * n_embd, 0.7f);
                ch_h[j] = random_vec((size_t) rows * rank, 0.5f);
                gf_h[j] = random_vec((size_t) 2 * n_embd * rank, 0.25f);
                dm[j] = dev(msg_h[j]);
                dch[j] = dev(ch_h[j]);
                dgf[j] = dev(gf_h[j]);
                ref_h[j].resize((size_t) rows * n_embd);
                for (int r = 0; r < rows; ++r) {
                    for (int i = 0; i < n_embd; ++i) {
                        double g = 0.0, f = 0.0;
                        for (int k = 0; k < rank; ++k) {
                            const double xv = ch_h[j][(size_t) r * rank + k];
                            g += (double) gf_h[j][(size_t) i * rank + k] * xv;
                            f += (double) gf_h[j][(size_t) (n_embd + i) * rank + k] * xv;
                        }
                        const size_t idx = (size_t) r * n_embd + i;
                        ref_h[j][idx] = (float) ((double) in0_h[idx] * (1.0 - std::tanh(f)) +
                                                 (double) msg_h[j][idx] * std::tanh(g));
                    }
                }
            }
            float* din0 = dev(in0_h);
            float* d1 = dev(std::vector<float>((size_t) rows * n_embd, 0.f));
            float* d2 = dev(std::vector<float>((size_t) rows * n_embd, 0.f));
            // SWA aliasing: cur={in0, in0, in0}, out={d1, d2, in0}
            const float* c3[3] = {din0, din0, din0};
            const float* m3[3] = {dm[0], dm[1], dm[2]};
            const float* ch3[3] = {dch[0], dch[1], dch[2]};
            const float* gf3[3] = {dgf[0], dgf[1], dgf[2]};
            dout[0] = d1; dout[1] = d2; dout[2] = din0;
            ink_control3_gate(c3, m3, ch3, gf3, dout, count, n_embd, rank, rows, nullptr);
            // Also test count=1 on d1
            float* ds_out = dev(std::vector<float>((size_t) rows * n_embd, 0.f));
            float* d_single_c = dev(in0_h);
            ink_control3_gate(&d_single_c, &dm[0], &dch[0], &dgf[0], &ds_out, 1, n_embd, rank, rows, nullptr);
            cudaDeviceSynchronize();
            for (int j = 0; j < count; ++j) {
                char lbl[64];
                std::snprintf(lbl, sizeof(lbl), "F3 control3_gate r=%d j=%d", rows, j);
                bad += report(lbl, host(dout[j], ref_h[j].size()), ref_h[j], 1e-5);
            }
            bad += report("F3 control3_gate single", host(ds_out, ref_h[0].size()), ref_h[0], 1e-5);
            cudaFree(din0); cudaFree(d1); cudaFree(d2); cudaFree(ds_out); cudaFree(d_single_c);
            for (int j = 0; j < count; ++j) { cudaFree(dm[j]); cudaFree(dch[j]); cudaFree(dgf[j]); }
        }
    }

    // F4. inkling_conv4_stream against inkling_conv4 + copy/write/add stream
    {
        auto run_f4 = [&](const char* label, int rows, int pre_copy_dst, bool replace_s0, int post_op,
                          unsigned snaps) {
            const int n_streams = 6, n_embd = 512;
            auto x = random_vec((size_t) rows * n_embd, 0.8f);
            auto st = random_vec((size_t) n_embd * 3, 0.6f);
            auto w = random_vec((size_t) n_embd * 4, 0.4f);
            auto r0 = random_vec((size_t) rows * n_streams * n_embd, 0.7f);
            const int n_snaps = 4;
            const size_t stride = (size_t) n_embd * 3;
            std::vector<float> snap_z((size_t) n_snaps * stride, 0.f);

            float *dx = dev(x), *dw = dev(w);
            float *dst_ref = dev(st), *dst_got = dev(st);
            float *dr_ref = dev(r0), *dr_got = dev(r0);
            float *dao = dev(std::vector<float>(x.size(), 0.f));
            float *dsnap_ref = snaps ? dev(snap_z) : nullptr;
            float *dsnap_got = snaps ? dev(snap_z) : nullptr;

            inkling_conv4(dx, dst_ref, dw, nullptr, 0, dao, n_embd, rows, nullptr, dsnap_ref, stride, snaps);
            if (pre_copy_dst >= 0) inkling_copy_stream(dr_ref, n_streams, pre_copy_dst, 0, n_embd, rows, nullptr);
            if (replace_s0) inkling_write_stream(dao, dr_ref, n_streams, 0, n_embd, rows, nullptr);
            else inkling_add_to_stream(nullptr, dao, dr_ref, n_streams, 0, n_embd, rows, nullptr);
            if (post_op == 1) inkling_copy_stream(dr_ref, n_streams, 2, 0, n_embd, rows, nullptr);
            else if (post_op == 2) {
                inkling_copy_stream(dr_ref, n_streams, 4, 3, n_embd, rows, nullptr);
                inkling_copy_stream(dr_ref, n_streams, 5, 0, n_embd, rows, nullptr);
                inkling_copy_stream(dr_ref, n_streams, 0, 2, n_embd, rows, nullptr);
            }

            inkling_conv4_stream(dx, dst_got, dw, dr_got, n_streams, n_embd, rows, pre_copy_dst, replace_s0, post_op,
                                 nullptr, dsnap_got, stride, snaps);
            cudaDeviceSynchronize();
            bad += report(label, host(dr_got, r0.size()), host(dr_ref, r0.size()), 1e-6);
            bad += report(label, host(dst_got, st.size()), host(dst_ref, st.size()), 1e-6);
            if (snaps) bad += report(label, host(dsnap_got, snap_z.size()), host(dsnap_ref, snap_z.size()), 1e-6);

            cudaFree(dx); cudaFree(dw); cudaFree(dst_ref); cudaFree(dst_got);
            cudaFree(dr_ref); cudaFree(dr_got); cudaFree(dao); cudaFree(dsnap_ref); cudaFree(dsnap_got);
        };
        run_f4("F4 conv4_stream replace pre=3 r=4", 4, 3, true, 0, 0);
        run_f4("F4 conv4_stream add post=1 r=1", 1, -1, false, 1, 0);
        run_f4("F4 conv4_stream add post=2 snap r=4", 4, -1, false, 2, 0b1010u);
    }

    // F5. inkling_mix_norm_q8 against inkling_mix + rms_norm + reference Q8_1
    {
        for (int n_slots : {2, 3, 5}) {
            const int rows = 4, n_streams = 6, n_embd = 2048;
            const MixSlots slots{{0, 1, 2, 3, 4, 0}};
            const float mix_eps = 1e-6f, rms_eps = 1e-6f;
            auto r = random_vec((size_t) rows * n_streams * n_embd, 0.8f);
            auto q = random_vec((size_t) n_embd, 0.4f);
            auto nw = random_vec((size_t) n_embd, 1.0f);

            float *dr = dev(r), *dq = dev(q), *dnw = dev(nw);
            float *dmix = dev(std::vector<float>((size_t) rows * n_embd, 0.f));
            float *dxn_ref = dev(std::vector<float>((size_t) rows * n_embd, 0.f));
            float *dxn_got = dev(std::vector<float>((size_t) rows * n_embd, 0.f));
            const size_t q8_bytes = (size_t) rows * (n_embd / 32) * 36;
            uint8_t* dxq = dev(std::vector<uint8_t>(q8_bytes, 0));

            inkling_mix(dr, n_streams, slots, n_slots, dq, 1, dmix, n_embd, rows, mix_eps, nullptr);
            rms_norm(dmix, dnw, dxn_ref, n_embd, rows, rms_eps, nullptr);
            const bool ok = inkling_mix_norm_q8(dr, n_streams, slots, n_slots, dq, dnw, dxn_got, dxq, n_embd, rows,
                                                mix_eps, rms_eps, nullptr);
            cudaDeviceSynchronize();
            if (!ok) { std::printf("FAIL inkling_mix_norm_q8 returned false\n"); ++bad; }

            auto got_xn = host(dxn_got, (size_t) rows * n_embd);
            auto ref_xn = host(dxn_ref, (size_t) rows * n_embd);
            char lbl[64];
            std::snprintf(lbl, sizeof(lbl), "F5 mix_norm_q8 xn slots=%d", n_slots);
            bad += report(lbl, got_xn, ref_xn, 1e-5);

            // Verify Q8_1 block codes and (d, sum) scales match dxn_got
            auto got_xq = host(dxq, q8_bytes);
            std::vector<float> deq_got((size_t) rows * n_embd), deq_want((size_t) rows * n_embd);
            const int nb = n_embd / 32;
            for (int row = 0; row < rows; ++row) {
                for (int b = 0; b < nb; ++b) {
                    const uint8_t* blk = got_xq.data() + ((size_t) row * nb + b) * 36;
                    uint16_t dh, sh;
                    std::memcpy(&dh, blk, 2);
                    std::memcpy(&sh, blk + 2, 2);
                    const float df = __half2float(__ushort_as_half(dh));
                    const float sf = __half2float(__ushort_as_half(sh));
                    float amax = 0.f, bsum = 0.f;
                    for (int j = 0; j < 32; ++j) {
                        const float v = got_xn[(size_t) row * n_embd + b * 32 + j];
                        amax = std::max(amax, std::fabs(v));
                        bsum += v;
                    }
                    const float d_want = std::min(65504.f, amax / 127.0f);
                    const float id = d_want > 0.f ? 1.0f / d_want : 0.f;
                    for (int j = 0; j < 32; ++j) {
                        const int8_t qg = (int8_t) blk[4 + j];
                        const float v = got_xn[(size_t) row * n_embd + b * 32 + j];
                        const int qw = amax == 0.f ? 0 : std::clamp((int) std::lround(v * id), -127, 127);
                        deq_got[(size_t) row * n_embd + b * 32 + j] = (float) qg * df;
                        deq_want[(size_t) row * n_embd + b * 32 + j] = (float) qw * __half2float(__float2half(d_want));
                    }
                    if (std::fabs(sf - bsum) > 0.05f * (1.0f + std::fabs(bsum))) {
                        std::printf("FAIL Q8_1 block sum row=%d b=%d got=%.3f want=%.3f\n", row, b, sf, bsum);
                        ++bad;
                    }
                }
            }
            std::snprintf(lbl, sizeof(lbl), "F5 mix_norm_q8 deq slots=%d", n_slots);
            bad += report(lbl, deq_got, deq_want, 1e-4);

            // Also verify null xq_out (output_rows path)
            cudaMemset(dxn_got, 0, (size_t) rows * n_embd * sizeof(float));
            const bool ok_noq = inkling_mix_norm_q8(dr, n_streams, slots, n_slots, dq, dnw, dxn_got, nullptr, n_embd,
                                                    rows, mix_eps, rms_eps, nullptr);
            cudaDeviceSynchronize();
            if (!ok_noq) ++bad;
            std::snprintf(lbl, sizeof(lbl), "F5 mix_norm_q8 xn(noq) slots=%d", n_slots);
            bad += report(lbl, host(dxn_got, (size_t) rows * n_embd), ref_xn, 1e-5);

            cudaFree(dr); cudaFree(dq); cudaFree(dnw); cudaFree(dmix);
            cudaFree(dxn_ref); cudaFree(dxn_got); cudaFree(dxq);
        }
    }

    auto verify_q8_1 = [&](const char* label, const uint8_t* q8_bytes, const std::vector<float>& src, int rows,
                           int cols) {
        const int nb = cols / 32;
        std::vector<float> deq_got((size_t) rows * cols), deq_want((size_t) rows * cols);
        for (int row = 0; row < rows; ++row) {
            for (int b = 0; b < nb; ++b) {
                const uint8_t* blk = q8_bytes + ((size_t) row * nb + b) * 36;
                uint16_t dh, sh;
                std::memcpy(&dh, blk, 2);
                std::memcpy(&sh, blk + 2, 2);
                const float df = __half2float(__ushort_as_half(dh));
                const float sf = __half2float(__ushort_as_half(sh));
                float amax = 0.f, bsum = 0.f;
                for (int j = 0; j < 32; ++j) {
                    const float v = src[(size_t) row * cols + b * 32 + j];
                    amax = std::max(amax, std::fabs(v));
                    bsum += v;
                }
                const float d_want = std::min(65504.f, amax / 127.0f);
                const float id = d_want > 0.f ? 1.0f / d_want : 0.f;
                for (int j = 0; j < 32; ++j) {
                    const int8_t qg = (int8_t) blk[4 + j];
                    const float v = src[(size_t) row * cols + b * 32 + j];
                    const int qw = amax == 0.f ? 0 : std::clamp((int) std::lround(v * id), -127, 127);
                    deq_got[(size_t) row * cols + b * 32 + j] = (float) qg * df;
                    deq_want[(size_t) row * cols + b * 32 + j] = (float) qw * __half2float(__float2half(d_want));
                }
                if (std::fabs(sf - bsum) > 0.05f * (1.0f + std::fabs(bsum))) {
                    std::printf("FAIL %s block sum row=%d b=%d got=%.3f want=%.3f\n", label, row, b, sf, bsum);
                    ++bad;
                }
            }
        }
        bad += report(label, deq_got, deq_want, 1e-4);
    };

    // F6. inkling_attend(..., out = nullptr, out_q8 != nullptr)
    {
        const int rows = 4, n_head = 24, kv_heads = 8, dim = 128, n_rel = 16, window = 1024, pos = 700;
        const int width = kv_heads * dim, cols = n_head * dim;
        const float scale = 1.0f / std::sqrt((float) dim);
        auto q = random_vec((size_t) rows * cols, 0.7f);
        auto r_h = random_vec((size_t) rows * n_head * n_rel, 0.5f);
        auto rb_h = random_vec((size_t) window * n_rel, 0.3f);
        auto snk = random_vec((size_t) rows * n_head, 0.3f);
        auto gate = random_vec((size_t) rows * cols, 1.2f);
        auto k_raw = random_vec((size_t) window * width, 0.6f);
        auto v_raw = random_vec((size_t) window * width, 0.6f);

        float *d_q = dev(q), *d_r = dev(r_h), *d_rb = dev(rb_h), *d_snk = dev(snk), *d_g = dev(gate);
        float *d_kr = dev(k_raw), *d_vr = dev(v_raw);
        int8_t* d_kc = nullptr; uint8_t* d_v4 = nullptr;
        uint16_t *d_ks = nullptr, *d_vs = nullptr;
        cudaMalloc(&d_kc, (size_t) window * width);
        cudaMalloc(&d_v4, (size_t) window * (width / 2));
        cudaMalloc(&d_ks, (size_t) window * (width / 32) * sizeof(uint16_t));
        cudaMalloc(&d_vs, (size_t) window * (width / 32) * sizeof(uint16_t));
        store_q8(d_kr, d_kc, d_ks, width, 0, window, nullptr);
        store_q4(d_vr, d_v4, d_vs, width, 0, window, nullptr);

        float* d_sc = dev(std::vector<float>((size_t) rows * n_head * 64 * (dim + 2), 0.f));
        float* d_ref = dev(std::vector<float>((size_t) rows * cols, 0.f));
        const size_t q8_bytes = (size_t) rows * (cols / 32) * 36;
        uint8_t* d_q8 = dev(std::vector<uint8_t>(q8_bytes, 0));

        inkling_attend(d_q, nullptr, nullptr, nullptr, d_snk, window, pos, rows, d_sc, rows, d_ref, dim, n_head,
                       kv_heads, scale, nullptr, window, window, nullptr, 0, {d_kc, d_ks}, {}, {}, {d_v4, d_vs}, d_r,
                       d_rb, n_rel, d_g, nullptr);
        inkling_attend(d_q, nullptr, nullptr, nullptr, d_snk, window, pos, rows, d_sc, rows, nullptr, dim, n_head,
                       kv_heads, scale, nullptr, window, window, nullptr, 0, {d_kc, d_ks}, {}, {}, {d_v4, d_vs}, d_r,
                       d_rb, n_rel, d_g, d_q8);
        cudaDeviceSynchronize();
        verify_q8_1("F6 attend out=null out_q8 K8V4", host(d_q8, q8_bytes).data(), host(d_ref, (size_t) rows * cols),
                    rows, cols);

        cudaFree(d_q); cudaFree(d_r); cudaFree(d_rb); cudaFree(d_snk); cudaFree(d_g);
        cudaFree(d_kr); cudaFree(d_vr); cudaFree(d_kc); cudaFree(d_v4);
        cudaFree(d_ks); cudaFree(d_vs); cudaFree(d_sc); cudaFree(d_ref); cudaFree(d_q8);
    }

    // F7. situ_rows_q8_1 with ExpertScales against situ_rows
    {
        const int rows = 8, n_ff = 256;
        const float beta = 4.0f, lbeta = 25.0f;
        auto gu = random_vec((size_t) rows * 2 * n_ff, 3.0f);
        auto sg = random_vec(8, 0.5f);
        auto su = random_vec(8, 0.5f);
        std::vector<int32_t> h_ids = {2, 0, 5, 1, 7, 3, 4, 6};
        float *d_gu = dev(gu), *d_sg = dev(sg), *d_su = dev(su),
              *d_out = dev(std::vector<float>((size_t) rows * n_ff, 0.f));
        int32_t* d_ids = dev(h_ids);
        const size_t q8_bytes = (size_t) rows * (n_ff / 32) * 36;
        uint8_t* d_q8 = dev(std::vector<uint8_t>(q8_bytes, 0));
        ExpertScales es{d_sg, d_su, d_ids};

        situ_rows(d_gu, d_out, n_ff, rows, beta, lbeta, nullptr, &es);
        situ_rows_q8_1(d_gu, d_q8, n_ff, rows, beta, lbeta, nullptr, &es);
        cudaDeviceSynchronize();
        verify_q8_1("F7 situ_rows_q8_1 with ExpertScales", host(d_q8, q8_bytes).data(),
                    host(d_out, (size_t) rows * n_ff), rows, n_ff);

        cudaFree(d_gu); cudaFree(d_sg); cudaFree(d_su); cudaFree(d_out); cudaFree(d_ids); cudaFree(d_q8);
    }

    // F8. Shared-KV MMA multi-row decode attention (inkling_attend table == nullptr vs scalar table != nullptr)
    {
        auto run_f8 = [&](const char* label, int rows, int pos, int window, int ring, int n_extent, bool use_at,
                          int kv_mode) {
            const int n_head = 24, kv_heads = 8, dim = 128, n_rel = 16;
            const int width = kv_heads * dim;
            const float scale = 1.0f / std::sqrt((float) dim);
            const int kv_len = ring > 0 ? ring : (pos + rows);

            auto q = random_vec((size_t) rows * n_head * dim, 0.7f);
            auto r_h = random_vec((size_t) rows * n_head * n_rel, 0.5f);
            auto rb_h = random_vec((size_t) n_extent * n_rel, 0.3f);
            auto snk = random_vec((size_t) rows * n_head, 0.3f);
            auto gate = random_vec((size_t) rows * n_head * dim, 1.2f);
            auto k_raw = random_vec((size_t) kv_len * width, 0.6f);
            auto v_raw = random_vec((size_t) kv_len * width, 0.6f);

            float *d_q = dev(q), *d_r = dev(r_h), *d_rb = dev(rb_h), *d_snk = dev(snk), *d_g = dev(gate);
            float *d_kr = dev(k_raw), *d_vr = dev(v_raw);
            uint16_t *d_k16 = nullptr, *d_v16 = nullptr;
            int8_t *d_kc = nullptr, *d_vc = nullptr;
            uint8_t *d_v4 = nullptr;
            uint16_t *d_ks = nullptr, *d_vs = nullptr;
            if (kv_mode == 0) {
                cudaMalloc(&d_k16, (size_t) kv_len * width * sizeof(uint16_t));
                cudaMalloc(&d_v16, (size_t) kv_len * width * sizeof(uint16_t));
                store_f16(d_kr, d_k16, width, 0, kv_len, nullptr);
                store_f16(d_vr, d_v16, width, 0, kv_len, nullptr);
            } else {
                cudaMalloc(&d_kc, (size_t) kv_len * width);
                cudaMalloc(&d_ks, (size_t) kv_len * (width / 32) * sizeof(uint16_t));
                cudaMalloc(&d_vs, (size_t) kv_len * (width / 32) * sizeof(uint16_t));
                store_q8(d_kr, d_kc, d_ks, width, 0, kv_len, nullptr);
                if (kv_mode == 1) {
                    cudaMalloc(&d_vc, (size_t) kv_len * width);
                    store_q8(d_vr, d_vc, d_vs, width, 0, kv_len, nullptr);
                } else {
                    cudaMalloc(&d_v4, (size_t) kv_len * (width / 2));
                    store_q4(d_vr, d_v4, d_vs, width, 0, kv_len, nullptr);
                }
            }
            Kvq8 kq8 = kv_mode >= 1 ? Kvq8{d_kc, d_ks} : Kvq8{};
            Kvq8 vq8 = kv_mode == 1 ? Kvq8{d_vc, d_vs} : Kvq8{};
            Kvq4 vq4 = kv_mode == 2 ? Kvq4{d_v4, d_vs} : Kvq4{};

            const size_t score_n = (size_t) rows * n_head * 64 * (dim + 2);
            float* d_tbl = dev(std::vector<float>((size_t) rows * n_head * n_extent, 0.f));
            float* d_sc = dev(std::vector<float>(score_n, 0.f));
            float* d_ref = dev(std::vector<float>(q.size(), 0.f));
            float* d_got = dev(std::vector<float>(q.size(), 0.f));
            std::vector<long long> at_v((size_t) rows * 2);
            for (int t = 0; t < rows; ++t) at_v[2 * t] = at_v[2 * t + 1] = pos + t;
            long long* d_at = use_at ? dev(at_v) : nullptr;
            const int rm = d_at ? (window > 0 ? std::min(window + rows - 1, pos + rows) : pos + rows) : 0;

            inkling_rel_table(d_r, d_rb, d_snk, d_tbl, n_rel, n_extent, n_head, rows, nullptr, false);
            inkling_attend(d_q, d_k16, d_v16, d_tbl, d_snk, n_extent, pos, rows, d_sc, rows, d_ref, dim, n_head,
                           kv_heads, scale, nullptr, window, ring, d_at, rm, kq8, vq8, {}, vq4);
            mul_sigmoid(d_ref, d_g, n_head * dim, rows, n_head * dim, false, nullptr);

            inkling_attend(d_q, d_k16, d_v16, nullptr, d_snk, n_extent, pos, rows, d_sc, rows, d_got, dim, n_head,
                           kv_heads, scale, nullptr, window, ring, d_at, rm, kq8, vq8, {}, vq4, d_r, d_rb, n_rel,
                           d_g);
            cudaDeviceSynchronize();
            bad += report(label, host(d_got, q.size()), host(d_ref, q.size()), 2e-4);

            cudaFree(d_q); cudaFree(d_r); cudaFree(d_rb); cudaFree(d_snk); cudaFree(d_g);
            cudaFree(d_kr); cudaFree(d_vr); cudaFree(d_k16); cudaFree(d_v16);
            cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_v4); cudaFree(d_ks); cudaFree(d_vs);
            cudaFree(d_tbl); cudaFree(d_sc); cudaFree(d_ref); cudaFree(d_got); cudaFree(d_at);
        };
        run_f8("F8 MMA attend SWA wrap K8V4 at r=4", 4, 1022, 1024, 1024, 1024, true, 2);
        run_f8("F8 MMA attend Global K8V4 r=4", 4, 2050, 0, 0, 1024, false, 2);
        run_f8("F8 MMA attend Global FP16 r=5", 5, 1500, 0, 0, 1024, false, 0);
        run_f8("F8 MMA attend small SWA Q8 r=2", 2, 90, 1024, 1024, 1024, false, 1);
    }
    return bad;
}

int test_dspark_head_proxy() {
    int bad = 0;
    const int n_vocab = 4096, n_in = 2304, rank = 256, rows = 3;
    const float cap = 30.0f;

    auto deq_q8_1_row = [](const uint8_t* row_bytes, int k) {
        std::vector<float> out((size_t) k);
        const int nb = k / 32;
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = row_bytes + (size_t) b * 36;
            const __half* ds = reinterpret_cast<const __half*>(blk);
            const float d = __half2float(ds[0]);
            const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 4);
            for (int j = 0; j < 32; ++j) out[(size_t) b * 32 + j] = d * (float) qs[j];
        }
        return out;
    };

    auto quant_q8_1_rows = [](const std::vector<float>& x, int rows_cnt, int k) {
        const size_t row_bytes = (size_t) (k / 32) * 36;
        std::vector<uint8_t> out((size_t) rows_cnt * row_bytes, 0);
        for (int r = 0; r < rows_cnt; ++r) {
            for (int b = 0; b < k / 32; ++b) {
                const float* src = x.data() + (size_t) r * k + (size_t) b * 32;
                float amax = 0.0f;
                for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(src[j]));
                const float d = amax / 127.0f;
                const float id = d > 0.0f ? 1.0f / d : 0.0f;
                uint8_t* blk = out.data() + (size_t) r * row_bytes + (size_t) b * 36;
                int8_t* qs = reinterpret_cast<int8_t*>(blk + 4);
                float sum = 0.0f;
                for (int j = 0; j < 32; ++j) {
                    const int q = std::max(-127, std::min(127, (int) std::lrintf(src[j] * id)));
                    qs[j] = (int8_t) q;
                    sum += d * (float) q;
                }
                __half* ds = reinterpret_cast<__half*>(blk);
                ds[0] = __float2half(d);
                ds[1] = __float2half(sum);
            }
        }
        return out;
    };

    auto run_proxy_case = [&](const char* label, int head_type) {
        auto h_f = random_vec((size_t) rows * n_in, 0.25f);
        auto lat_f = random_vec((size_t) rows * rank, 0.25f);
        auto fold_b = random_vec(n_vocab, 0.2f);

        auto wh_f = random_vec((size_t) n_vocab * n_in, 0.12f);
        auto wf_f = random_vec((size_t) n_vocab * rank, 0.12f);
        const int winners[rows] = {13, 1987, 4021};
        for (int r = 0; r < rows; ++r) {
            const int tok = winners[r];
            for (int j = 0; j < n_in; ++j) wh_f[(size_t) tok * n_in + j] += 0.22f * h_f[(size_t) r * n_in + j];
            for (int j = 0; j < rank; ++j) wf_f[(size_t) tok * rank + j] += 0.22f * lat_f[(size_t) r * rank + j];
        }

        const ggml_type ht = (ggml_type) head_type;
        const size_t wh_bytes = ggml_row_size(ht, n_in) * (size_t) n_vocab;
        const size_t wf_bytes = ggml_row_size(GGML_TYPE_Q8_0, rank) * (size_t) n_vocab;
        std::vector<uint8_t> wh_q(wh_bytes), wf_q(wf_bytes);
        ggml_quantize_chunk(ht, wh_f.data(), wh_q.data(), 0, n_vocab, n_in, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, wf_f.data(), wf_q.data(), 0, n_vocab, rank, nullptr);

        // Allocate fold & w1 in pinned mapped host memory just like production Inkling2.
        const size_t w1_bytes = (size_t) n_vocab * rank * sizeof(uint16_t);
        uint16_t *h_w1 = nullptr, *d_w1_dev = nullptr;
        uint8_t *h_fold = nullptr, *d_fold_dev = nullptr;
        cudaHostAlloc(&h_w1, w1_bytes, cudaHostAllocMapped);
        cudaHostAlloc(&h_fold, wf_bytes, cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_w1_dev, h_w1, 0);
        cudaHostGetDevicePointer(&d_fold_dev, h_fold, 0);
        std::memcpy(h_fold, wf_q.data(), wf_bytes);
        for (size_t i = 0; i < (size_t) n_vocab * rank; ++i) h_w1[i] = (uint16_t) (0x3c00u + (i * 131u) % 0x1000u);

        uint8_t* d_head = dev(wh_q);
        float* d_fb = dev(fold_b);
        const size_t head_q8_row = (size_t) (n_in / 32) * 36;
        const size_t lat_q8_row = (size_t) (rank / 32) * 36;
        const auto hq_host = quant_q8_1_rows(h_f, rows, n_in);
        const auto lq_host = quant_q8_1_rows(lat_f, rows, rank);
        uint8_t* d_hq = dev(hq_host);
        uint8_t* d_latq = dev(lq_host);

        void *d_head_sc = nullptr, *d_fold_sc = nullptr, *d_scr = nullptr;
        float *d_fb_p = nullptr, *d_approx = nullptr;
        uint16_t* d_w1_row = nullptr;
        Likeliest* d_picks = nullptr;
        cudaMalloc(&d_head_sc, inkling_dspark_screen_bytes(n_in, n_vocab));
        cudaMalloc(&d_fold_sc, inkling_dspark_screen_bytes(rank, n_vocab));
        cudaMalloc(&d_fb_p, (size_t) inkling_dspark_screen_rows(n_vocab) * sizeof(float));
        cudaMalloc(&d_approx, (size_t) rows * (size_t) inkling_dspark_screen_rows(n_vocab) * sizeof(float));
        cudaMalloc(&d_scr, inkling_dspark_head_scratch_bytes(n_vocab));
        cudaMalloc(&d_w1_row, (size_t) rank * sizeof(uint16_t));
        cudaMalloc(&d_picks, (size_t) rows * sizeof(Likeliest));

        inkling_dspark_screen_pack(d_head, head_type, n_in, n_vocab, d_head_sc, nullptr);
        inkling_dspark_screen_pack(d_fold_dev, (int) GGML_TYPE_Q8_0, rank, n_vocab, d_fold_sc, nullptr);
        inkling_dspark_screen_order(d_fb, n_vocab, d_fb_p, nullptr);
        inkling_dspark_head_base(d_head_sc, d_hq, rows, n_vocab, d_approx, nullptr);

        DsparkHeadArgs hargs{d_head, head_type, d_head_sc, d_fold_dev, d_fold_sc, d_fb, d_fb_p, n_vocab, cap};
        const auto* h_traits = ggml_get_type_traits(ht);
        const auto* q8_traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
        std::vector<float> wh_deq((size_t) n_vocab * n_in), wf_deq((size_t) n_vocab * rank);
        h_traits->to_float(wh_q.data(), wh_deq.data(), (int64_t) wh_deq.size());
        q8_traits->to_float(wf_q.data(), wf_deq.data(), (int64_t) wf_deq.size());

        double max_prob_err = 0.0;
        for (int r = 0; r < rows; ++r) {
            inkling_dspark_head_pick(hargs, d_hq, d_latq + (size_t) r * lat_q8_row, d_approx, r, d_scr, d_picks + r,
                                     d_w1_dev, d_w1_row, nullptr);
            cudaDeviceSynchronize();
            const cudaError_t ce = cudaGetLastError();
            if (ce != cudaSuccess) {
                std::printf("%-42s CUDA ERR: %s\n", label, cudaGetErrorString(ce));
                ++bad;
                break;
            }
            const Likeliest got = host(d_picks + r, 1)[0];
            const auto w1_got = host(d_w1_row, (size_t) rank);

            // CPU exact reference using Q8_1 quantized vectors & dequantized quantized weights.
            const auto hr = deq_q8_1_row(hq_host.data() + (size_t) r * head_q8_row, n_in);
            const auto lr = deq_q8_1_row(lq_host.data() + (size_t) r * lat_q8_row, rank);
            std::vector<double> logits(n_vocab);
            int ref_tok = 0;
            double ref_max = -1e30;
            for (int v = 0; v < n_vocab; ++v) {
                double s = fold_b[v];
                const float* whv = wh_deq.data() + (size_t) v * n_in;
                const float* wfv = wf_deq.data() + (size_t) v * rank;
                for (int j = 0; j < n_in; ++j) s += (double) whv[j] * (double) hr[j];
                for (int j = 0; j < rank; ++j) s += (double) wfv[j] * (double) lr[j];
                s = (double) cap * std::tanh(s / (double) cap);
                logits[v] = s;
                if (s > ref_max) { ref_max = s; ref_tok = v; }
            }
            double sum_exp = 0.0;
            for (int v = 0; v < n_vocab; ++v) sum_exp += std::exp(logits[v] - ref_max);
            const double ref_prob = 1.0 / sum_exp;
            max_prob_err = std::max(max_prob_err, std::abs((double) got.prob - ref_prob));

            const bool w1_ok = std::memcmp(w1_got.data(), h_w1 + (size_t) ref_tok * rank,
                                           (size_t) rank * sizeof(uint16_t)) == 0;
            if (got.token != ref_tok || !w1_ok || std::abs((double) got.prob - ref_prob) > 0.05) {
                std::printf("%-42s r=%d FAIL tok=%d(ref=%d) prob=%.5f(ref=%.5f) w1=%d\n", label, r, got.token,
                            ref_tok, got.prob, ref_prob, (int) w1_ok);
                ++bad;
            }
        }
        std::printf("%-42s %d rows  max_prob_err=%.4f  %s\n", label, rows, max_prob_err,
                    max_prob_err <= 0.05 ? "PASS" : "FAIL");

        // Also test prep_rnn_step_row against w1_row
        auto state_f = random_vec(rank, 0.5f);
        float *d_state = dev(state_f), *d_rnn_out = dev(std::vector<float>(2 * rank, 0.f));
        inkling_dspark_prep_rnn_step_row(d_state, d_w1_row, d_rnn_out, rank, nullptr);
        cudaDeviceSynchronize();
        const auto rnn_got = host(d_rnn_out, (size_t) (2 * rank));
        const auto w1_ref = host(d_w1_row, (size_t) rank);
        for (int j = 0; j < rank; ++j) {
            uint32_t bits = (uint32_t) w1_ref[j] << 16;
            float wf = 0.f;
            std::memcpy(&wf, &bits, sizeof(float));
            if (rnn_got[j] != state_f[j] || rnn_got[rank + j] != wf) {
                ++bad;
                break;
            }
        }
        cudaFree(d_state); cudaFree(d_rnn_out);
        cudaFree(d_head); cudaFree(d_fb);
        cudaFree(d_hq); cudaFree(d_latq); cudaFree(d_head_sc); cudaFree(d_fold_sc);
        cudaFree(d_fb_p); cudaFree(d_approx); cudaFree(d_scr); cudaFree(d_w1_row); cudaFree(d_picks);
        cudaFreeHost(h_w1);
        cudaFreeHost(h_fold);
    };

    run_proxy_case("DSpark 2-bit head proxy Q8_0", (int) GGML_TYPE_Q8_0);
    run_proxy_case("DSpark 2-bit head proxy Q6_K", (int) GGML_TYPE_Q6_K);
    return bad;
}

}  // namespace

int main() {
    int bad = 0;
    bad += test_situ_gpu_and_cpu();
    bad += test_mix_and_control();
    bad += test_ctrl_down();
    bad += test_conv4();
    bad += test_l2_clip();
    bad += test_rel_and_attend();
    bad += test_dspark_kernels();
    bad += test_fused_and_predequant();
    bad += test_decode_fusions();
    bad += test_dspark_head_proxy();
    std::printf("\n%s\n", bad ? "FAILED" : "ALL PASSED");
    return bad ? 1 : 0;
}
