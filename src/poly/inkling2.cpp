// src/poly/inkling2.cpp - Thinking Machines Lab's Inkling 2 Mini (inkling2_loop and inkling2).
//
// Reference: mainline llama.cpp src/models/inkling2.cpp and src/models/inkling2-loop.cpp.
#include "inkling2.hpp"

#include "ggml.h"
#include "runtime.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;
constexpr size_t kQ8WorkRows = 2048;

const Names kInkling2MetaAliases[] = {
    {"loop.start", {"loop.core_start", nullptr}},
    {"loop.end", {"loop.core_end", nullptr}},
    {"loop.count", {"num_loops", nullptr}},
};

const Names kInkling2TensorAliases[] = {
    {"loop_q_ctrl_down.weight", {"loop_attn_down.weight", nullptr}},
    {"loop_q_ctrl_gain.weight", {"loop_attn_gain.weight", nullptr}},
    {"loop_q_ctrl_forget.weight", {"loop_attn_forget.weight", nullptr}},
    {"loop_k_ctrl_down.weight", {"loop_k_down.weight", nullptr}},
    {"loop_k_ctrl_gain.weight", {"loop_k_gain.weight", nullptr}},
    {"loop_k_ctrl_forget.weight", {"loop_k_forget.weight", nullptr}},
    {"loop_v_ctrl_down.weight", {"loop_v_down.weight", nullptr}},
    {"loop_v_ctrl_gain.weight", {"loop_v_gain.weight", nullptr}},
    {"loop_v_ctrl_forget.weight", {"loop_v_forget.weight", nullptr}},
    {"loop_ffn_ctrl_down.weight", {"loop_mlp_down.weight", nullptr}},
    {"loop_ffn_ctrl_gain.weight", {"loop_mlp_gain.weight", nullptr}},
    {"loop_ffn_ctrl_forget.weight", {"loop_mlp_forget.weight", nullptr}},
};

uint64_t val_u(const MetaValue& v) {
    return v.type == MetaType::ARRAY ? (v.items.empty() ? 0 : v.items[0].u) : v.u;
}

}  // namespace

const Names* Inkling2::meta_alias(const char* key) const {
    return root_layout_ ? names_of(kInkling2MetaAliases, key) : nullptr;
}

const Names* Inkling2::tensor_alias(const char* name) const {
    return root_layout_ ? names_of(kInkling2TensorAliases, name) : nullptr;
}

void Inkling2::begin_pass() {
    blocks_.assign((size_t) n_layer_, Block{});
}

void Inkling2::load_geometry() {
    const MetaValue* arch = gg_->get("general.architecture");
    root_layout_ = arch && arch->s == "inkling2";

    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported inkling2 model: ") + what);
    };
    auto must_u = [&](const char* key) -> int64_t {
        const MetaValue* v = find(key);
        need(v != nullptr, key);
        if (v->type == MetaType::ARRAY) {
            need(!v->items.empty(), key);
            for (const MetaValue& it : v->items) need(it.u == v->items[0].u, key);
            return (int64_t) v->items[0].u;
        }
        return (int64_t) v->u;
    };
    auto must_f = [&](const char* key) -> float {
        const MetaValue* v = find(key);
        need(v != nullptr, key);
        return (float) v->num();
    };
    auto opt_f = [&](const char* key, float fallback) -> float {
        const MetaValue* v = find(key);
        return v ? (float) v->num() : fallback;
    };

    n_ff_exp_ = (int) must_u("expert_feed_forward_length");
    n_head_ = (int) must_u("attention.head_count");
    n_kv_ = (int) must_u("attention.head_count_kv");
    head_dim_ = (int) must_u("attention.key_length");
    need((int) must_u("attention.value_length") == head_dim_, "key_length != value_length");
    need(n_head_ > 0 && n_kv_ > 0 && n_head_ % n_kv_ == 0 && n_head_ <= 256, "bad head counts");
    need(head_dim_ % 32 == 0 && head_dim_ <= 128, "bad head_dim");

    n_rel_ = (int) must_u("attention.relative_length");
    n_rel_extent_ = (int) must_u("attention.relative_extent");
    window_ = (int) must_u("attention.sliding_window");
    qk_clip_ = must_f("attention.qk_clip");
    rel_clip_ = must_f("attention.relative_clip");
    log_floor_ = (int) must_u("attention.log_scale_floor");
    log_alpha_ = must_f("attention.log_scale_alpha");
    block_size_ = (int) must_u("attn_res.block_size");
    l_conv_ = (int) must_u("shortconv.l_cache");
    need(n_rel_ > 0 && n_rel_ <= 16 && n_rel_extent_ > 0 && window_ > 0, "bad relative/window params");
    need(l_conv_ == 4, "shortconv.l_cache must be 4");
    need(block_size_ > 0, "bad attn_res.block_size");
    need(log_floor_ > 0 && log_alpha_ >= 0.f, "bad log scale params");

    const MetaValue* unpad = find("vocab_size_unpadded");
    need(unpad != nullptr, "vocab_size_unpadded");
    const int unpadded = (int) val_u(*unpad);
    need(unpadded > 0 && unpadded <= n_vocab_, "bad vocab_size_unpadded");
    n_vocab_padded_ = n_vocab_;
    n_vocab_ = unpadded;

    exp_act_ = ExpAct::kSiTU;
    situ_beta_ = must_f("activation.situ_beta");
    situ_linear_beta_ = must_f("activation.situ_linear_beta");
    need(situ_beta_ > 0.f && situ_linear_beta_ > 0.f, "bad situ parameters");
    w_scale_ = opt_f("expert_weights_scale", 8.f);
    final_logit_softcapping_ = opt_f("final_logit_softcapping", 0.f);
    half_sum_guard_ = true;

    core_start_ = (int) must_u("loop.start");
    core_end_ = (int) must_u("loop.end");
    n_passes_ = (int) must_u("loop.count");
    gate_rank_ = (int) must_u("loop.gate_rank");
    if (root_layout_) {
        const int n_phys = n_layer_ - (n_passes_ - 1) * (core_end_ - core_start_);
        need(n_phys > 0, "bad inkling2 loop core bounds");
        n_layer_ = n_phys;
    }
    need(n_passes_ >= 2 && gate_rank_ > 0 && core_start_ > 0 && core_start_ < core_end_ && core_end_ <= n_layer_ &&
             core_start_ % block_size_ == 0 && core_end_ % block_size_ == 0,
         "bad loop core geometry");
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;

    const MetaValue* swa = find("attention.sliding_window_pattern");
    need(swa != nullptr && swa->type == MetaType::ARRAY && (int) swa->items.size() >= n_layer_,
         "bad attention.sliding_window_pattern");

    schedule_.clear();
    for (int ip = 0; ip < core_start_; ++ip) schedule_.push_back({ip, (int) schedule_.size(), 0});
    for (int pass = 0; pass < n_passes_; ++pass)
        for (int ip = core_start_; ip < core_end_; ++ip) schedule_.push_back({ip, (int) schedule_.size(), pass});
    for (int ip = core_end_; ip < n_layer_; ++ip) schedule_.push_back({ip, (int) schedule_.size(), 0});

    n_swa_slots_ = 0;
    n_global_slots_ = 0;
    for (const Step& s : schedule_) {
        if (swa->items[(size_t) s.phys].u != 0) ++n_swa_slots_;
        else ++n_global_slots_;
    }

    const size_t kv_width = (size_t) n_kv_ * head_dim_;
    off_k_conv_ = 0;
    off_v_conv_ = (size_t) (l_conv_ - 1) * kv_width;
    off_attn_post_ = 2 * off_v_conv_;
    off_ffn_post_ = off_attn_post_ + (size_t) (l_conv_ - 1) * n_embd_;
    slot_conv_floats_ = off_ffn_post_ + (size_t) (l_conv_ - 1) * n_embd_;
}

void Inkling2::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };

    if (il == 0) {
        const TensorInfo* tn = gg_->find("token_embd_norm.weight");
        tok_norm_dev_ = upload(tn, "token_embd_norm.weight", n_embd_, 1, kF32);
        if (dense_) {
            tok_norm_host_.resize((size_t) n_embd_);
            const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) tn->type);
            if (tn->type == kTypeF32)
                std::memcpy(tok_norm_host_.data(), home(*tn).data(*tn), (size_t) n_embd_ * sizeof(float));
            else if (tr && tr->to_float)
                tr->to_float(home(*tn).data(*tn), tok_norm_host_.data(), n_embd_);
            else throw std::runtime_error("token_embd_norm.weight has unsupported type");
        }
        output_res_query_ = upload(gg_->find("output_res_query.weight"), "output_res_query.weight", n_embd_, 1, kF32);
    }

    const MetaValue* swa = meta_in(*gg_, "attention.sliding_window_pattern");
    B.swa = swa->items[(size_t) il].u != 0;
    const int64_t n_extent = B.swa ? window_ : n_rel_extent_;
    const int64_t q_dim = (int64_t) n_head_ * head_dim_;
    const int64_t kv_dim = (int64_t) n_kv_ * head_dim_;

    L.fixed = true;
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    B.wq = mat("attn_q.weight", n_embd_, q_dim);
    B.wk = mat("attn_k.weight", n_embd_, kv_dim);
    B.wv = mat("attn_v.weight", n_embd_, kv_dim);
    B.wo = mat("attn_output.weight", q_dim, n_embd_);
    B.wqkv_gate = mat("attn_gate.weight", n_embd_, q_dim);
    B.attn_q_clip = f32("attn_q_clip.weight", head_dim_, n_head_);
    B.attn_sink_proj = mat("attn_sink_proj.weight", n_embd_, n_head_);
    B.attn_rel = mat("attn_rel.weight", n_embd_, (int64_t) n_rel_ * n_head_);
    B.attn_rel_bias = f32("attn_rel_bias.weight", n_rel_, n_extent);
    B.attn_rel_clip = f32("attn_rel_clip.weight", n_rel_, n_head_);
    B.attn_k_conv = f32("attn_k_conv.weight", l_conv_, kv_dim);
    B.attn_v_conv = f32("attn_v_conv.weight", l_conv_, kv_dim);
    B.attn_post_conv = f32("attn_post_conv.weight", l_conv_, n_embd_);
    B.ffn_post_conv = f32("ffn_post_conv.weight", l_conv_, n_embd_);

    if (B.swa) {
        L.attn_norm = f32("attn_norm.weight", n_embd_);
    } else {
        B.q_inp_norm = f32("attn_q_inp_norm.weight", n_embd_);
        B.k_inp_norm = f32("attn_k_inp_norm.weight", n_embd_);
        B.v_inp_norm = f32("attn_v_inp_norm.weight", n_embd_);
        B.k_conv_proj = mat("attn_k_conv_proj.weight", n_embd_, (int64_t) l_conv_ * n_kv_);
        B.v_conv_proj = mat("attn_v_conv_proj.weight", n_embd_, (int64_t) l_conv_ * n_kv_);
    }

    if (il > 0) B.attn_res_query = f32("attn_res_query.weight", n_embd_, B.swa ? 1 : 3);
    B.ffn_res_query = f32("ffn_res_query.weight", n_embd_);

    if (il >= core_start_ && il < core_end_) {
        if (root_layout_) {
            const size_t bytes = (size_t) n_embd_ * 4 * sizeof(float);
            B.loop_msg_query = Dev{dense_ ? dense_ + dense_at_ : nullptr, kTypeF32, n_embd_, 4, 1.f};
            dense_at_ += pad(bytes) + kTail;
            if (dense_) {
                const TensorInfo* lr = gg_->find("loop_res_query.weight");
                const TensorInfo* lkv = gg_->find("loop_kv_res_query.weight");
                const int64_t rows = 2 * (core_end_ - core_start_);
                if (!lr || !lkv || (int64_t) lr->shape[0] != n_embd_ || (int64_t) lr->shape[1] != rows ||
                    (int64_t) lkv->shape[0] != n_embd_ || (int64_t) lkv->shape[1] != rows)
                    throw std::runtime_error("missing or malformed loop_res_query / loop_kv_res_query");
                auto read_row = [&](const TensorInfo& t, int64_t r, float* dst) {
                    const size_t row_bytes = home(t).payload(t) / (size_t) rows;
                    const uint8_t* p = home(t).data(t) + (size_t) r * row_bytes;
                    const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) t.type);
                    if (t.type == kTypeF32) std::memcpy(dst, p, (size_t) n_embd_ * sizeof(float));
                    else if (tr && tr->to_float) tr->to_float(p, dst, n_embd_);
                    else throw std::runtime_error(t.name + " has unsupported type");
                };
                const int k = il - core_start_;
                std::vector<float> buf((size_t) n_embd_ * 4);
                read_row(*lr, 2 * k + 0, buf.data() + 0 * n_embd_);
                read_row(*lkv, 2 * k + 0, buf.data() + 1 * n_embd_);
                read_row(*lkv, 2 * k + 1, buf.data() + 2 * n_embd_);
                read_row(*lr, 2 * k + 1, buf.data() + 3 * n_embd_);
                h2d(B.loop_msg_query.p, buf.data(), bytes);
            }
        } else {
            B.loop_msg_query = f32("loop_msg_query.weight", n_embd_, 4);
        }
        const char* down_names[4] = {"loop_q_ctrl_down.weight", "loop_k_ctrl_down.weight",
                                     "loop_v_ctrl_down.weight", "loop_ffn_ctrl_down.weight"};
        const char* gain_names[4] = {"loop_q_ctrl_gain.weight", "loop_k_ctrl_gain.weight",
                                     "loop_v_ctrl_gain.weight", "loop_ffn_ctrl_gain.weight"};
        const char* forget_names[4] = {"loop_q_ctrl_forget.weight", "loop_k_ctrl_forget.weight",
                                       "loop_v_ctrl_forget.weight", "loop_ffn_ctrl_forget.weight"};
        for (int c = 0; c < 4; ++c) {
            B.ctrl_down[c] = mat(down_names[c], 2 * n_embd_, gate_rank_);
            B.ctrl_gain[c] = mat(gain_names[c], gate_rank_, n_embd_);
            B.ctrl_forget[c] = mat(forget_names[c], gate_rank_, n_embd_);
        }
    }

    load_moe(il, false);
    B.probs_b.assign((size_t) n_expert_, 0.f);
    if (const TensorInfo* b = tensor(il, "exp_probs_b.bias")) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(blk(il, "exp_probs_b.bias") + " must be F32 [" + std::to_string(n_expert_) + "]");
        home(*b).payload(*b);
        std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
        B.probs_b_dev = upload(b, blk(il, "exp_probs_b.bias"), n_expert_, 1, kF32);
    }
}

void Inkling2::norm_text_embed(float* dst) const {
    if (tok_norm_host_.empty()) return;
    double sq = 0.0;
    for (int i = 0; i < n_embd_; ++i) sq += (double) dst[i] * (double) dst[i];
    const float inv = 1.0f / std::sqrt((float) (sq / (double) n_embd_) + rms_eps_);
    for (int i = 0; i < n_embd_; ++i) dst[i] = dst[i] * inv * tok_norm_host_[(size_t) i];
}

size_t Inkling2::recurrent_floats() const {
    return schedule_.size() * slot_conv_floats_;
}

size_t Inkling2::kv_code_row_bytes() const {
    const size_t row = (size_t) n_kv_ * head_dim_;
    if (opt_.kv_format == KvFmt::kQ4_0) return row / 2;
    if (opt_.kv_format == KvFmt::kQ8_0) return row;
    return row * sizeof(uint16_t);
}

size_t Inkling2::kv_scale_row_bytes() const {
    const bool quant = opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kQ4_0;
    return quant ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
}

uint8_t* Inkling2::slot_k_codes(const SlotCache& sc) const {
    if (opt_.kv_format == KvFmt::kQ4_0) return sc.k_q4;
    if (opt_.kv_format == KvFmt::kQ8_0) return (uint8_t*) sc.k_codes;
    return (uint8_t*) sc.keys;
}

uint8_t* Inkling2::slot_v_codes(const SlotCache& sc) const {
    if (opt_.kv_format == KvFmt::kQ4_0) return sc.v_q4;
    if (opt_.kv_format == KvFmt::kQ8_0) return (uint8_t*) sc.v_codes;
    return (uint8_t*) sc.vals;
}

void Inkling2::alloc_caches() {
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const bool q4 = opt_.kv_format == KvFmt::kQ4_0;
    const bool quant = q8 || q4;
    const size_t row = (size_t) n_kv_ * head_dim_;
    const size_t code_row = kv_code_row_bytes();
    const size_t scale_row = kv_scale_row_bytes();
    if (quant && again_ == SIZE_MAX)
        std::fprintf(stderr, "strata-poly: inkling2 %s KV cache: %zu bytes/row (%d global, %d SWA)\n",
                     q4 ? "q4_0" : "q8_0", code_row + scale_row, n_global_slots_, n_swa_slots_);

    const size_t swa_codes = 2 * (size_t) n_swa_slots_ * (size_t) ring_ * code_row;
    const size_t swa_scales = quant ? 2 * (size_t) n_swa_slots_ * (size_t) ring_ * scale_row : 0;

    uint8_t* rc = quant ? (uint8_t*) seq_alloc(swa_codes, "the attention cache") : nullptr;
    uint8_t* rs = quant ? (uint8_t*) seq_alloc(swa_scales, "the attention cache") : nullptr;
    uint16_t* kv = !quant ? (uint16_t*) seq_alloc(2 * (size_t) n_swa_slots_ * (size_t) ring_ * row * sizeof(uint16_t),
                                                   "the attention cache")
                          : nullptr;

    slots_.assign(schedule_.size(), SlotCache{});
    int swa_idx = 0;
    for (size_t s = 0; s < schedule_.size(); ++s) {
        SlotCache& sc = slots_[s];
        const Block& B = blocks_[(size_t) schedule_[s].phys];
        sc.swa = B.swa;
        sc.state_off = s * slot_conv_floats_;
        if (B.swa) {
            sc.swa_idx = swa_idx++;
            if (q4) {
                sc.k_q4 = rc;
                rc += (size_t) ring_ * code_row;
                sc.k_scales = (uint16_t*) rs;
                rs += (size_t) ring_ * scale_row;
                sc.v_q4 = rc;
                rc += (size_t) ring_ * code_row;
                sc.v_scales = (uint16_t*) rs;
                rs += (size_t) ring_ * scale_row;
            } else if (q8) {
                sc.k_codes = (int8_t*) rc;
                rc += (size_t) ring_ * code_row;
                sc.k_scales = (uint16_t*) rs;
                rs += (size_t) ring_ * scale_row;
                sc.v_codes = (int8_t*) rc;
                rc += (size_t) ring_ * code_row;
                sc.v_scales = (uint16_t*) rs;
                rs += (size_t) ring_ * scale_row;
            } else {
                sc.keys = kv;
                kv += (size_t) ring_ * row;
                sc.vals = kv;
                kv += (size_t) ring_ * row;
            }
        } else if (q4) {
            sc.k_q4 = (uint8_t*) kv_alloc(code_row, 1);
            sc.k_scales = (uint16_t*) kv_alloc(scale_row, 1);
            sc.v_q4 = (uint8_t*) kv_alloc(code_row, 1);
            sc.v_scales = (uint16_t*) kv_alloc(scale_row, 1);
        } else if (q8) {
            sc.k_codes = (int8_t*) kv_alloc(code_row, 1);
            sc.k_scales = (uint16_t*) kv_alloc(scale_row, 1);
            sc.v_codes = (int8_t*) kv_alloc(code_row, 1);
            sc.v_scales = (uint16_t*) kv_alloc(scale_row, 1);
        } else {
            sc.keys = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
            sc.vals = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
        }
    }

    const int n_splits_max = std::clamp((opt_.max_context + 511) / 512, 1, 64);
    chunk_batch_ = std::min(rows_, 64);
    chunk_dec_ = (int) std::clamp<size_t>(
        (size_t(8) << 20) / std::max<size_t>(1, (size_t) n_head_ * n_splits_max * (head_dim_ + 2)), 1,
        (size_t) std::min(rows_, kStageRows - 1));
}

int Inkling2::widest() const {
    return std::max({2 * n_embd_, n_head_ * head_dim_});
}

void Inkling2::assign_scratch() {
    const size_t n = (size_t) rows_;
    const size_t heads = n * n_head_ * head_dim_;
    const size_t kv = n * n_kv_ * head_dim_;
    inputs_buf_ = fbuf(3 * n * n_embd_);
    msg_ = fbuf(4 * n * n_embd_);
    ctrl_cat_ = fbuf(n * 2 * n_embd_);
    ctrl_r_ = fbuf(n * gate_rank_);
    ctrl_g_ = fbuf(n * n_embd_);
    ctrl_f_ = fbuf(n * n_embd_);
    inp_sa_ = fbuf(n * n_embd_);

    qraw_ = fbuf(heads);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    r_raw_ = fbuf(n * n_head_ * n_rel_);
    r_cur_ = fbuf(n * n_head_ * n_rel_);
    dyn_ = fbuf(n * l_conv_ * n_kv_);
    sinks_ = fbuf(n * n_head_);
    gate_ = fbuf(heads);
    att_ = fbuf(heads);

    const size_t max_ext = (size_t) std::max(window_, n_rel_extent_);
    table_ = fbuf((size_t) chunk_batch_ * n_head_ * max_ext);

    const size_t n_splits_max = (size_t) std::clamp((opt_.max_context + 511) / 512, 1, 64);
    score_n_ = std::max((size_t) chunk_dec_ * n_head_ * n_splits_max * (head_dim_ + 2),
                        (size_t) chunk_batch_ * n_head_ * 1536);
    score_ = fbuf(score_n_);

    if (rows_ >= kStageRows) {
        const size_t ch = (size_t) chunk_batch_ * n_head_;
        awork_ = fbuf(ch * (head_dim_ + head_dim_ / 2 + 3) + 1);
    }
    if (rows_ >= kStageRows && (opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kQ4_0)) {
        work_halves_ = kQ8WorkRows * (size_t) n_kv_ * head_dim_;
        q8w_ = fbuf((2 * work_halves_ + 1) / 2);
    }
}

size_t Inkling2::card_row_bytes() const {
    const size_t code_row = kv_code_row_bytes();
    const size_t scale_row = kv_scale_row_bytes();
    size_t layers = 0;
    for (const SlotCache& sc : slots_)
        if (!sc.swa && slot_k_codes(sc) != nullptr) ++layers;
    return layers * 2 * (code_row + scale_row);
}

void Inkling2::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const bool quant = opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kQ4_0;
    const size_t code_row = kv_code_row_bytes();
    const size_t scale_row = kv_scale_row_bytes();
    const size_t at_c = (size_t) first * code_row, at_s = (size_t) first * scale_row;
    const size_t n_c = (size_t) n * code_row, n_s = (size_t) n * scale_row;
    for (const SlotCache& sc : slots_) {
        const uint8_t* kptr = slot_k_codes(sc);
        const uint8_t* vptr = slot_v_codes(sc);
        if (sc.swa || n == 0 || kptr == nullptr) continue;
        if (quant) {
            ck(cudaMemcpy(dst, kptr + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
            ck(cudaMemcpy(dst, (const uint8_t*) sc.k_scales + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
            ck(cudaMemcpy(dst, vptr + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
            ck(cudaMemcpy(dst, (const uint8_t*) sc.v_scales + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
            continue;
        }
        for (const uint8_t* cache : {kptr, vptr}) {
            ck(cudaMemcpy(dst, cache + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
        }
    }
}

void Inkling2::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const bool quant = opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kQ4_0;
    const size_t code_row = kv_code_row_bytes();
    const size_t scale_row = kv_scale_row_bytes();
    const size_t at_c = (size_t) first * code_row, at_s = (size_t) first * scale_row;
    const size_t n_c = (size_t) n * code_row, n_s = (size_t) n * scale_row;
    const size_t sk_c = (size_t) (first - from) * code_row, sk_s = (size_t) (first - from) * scale_row;
    const size_t all_c = (size_t) count * code_row, all_s = (size_t) count * scale_row;
    for (const SlotCache& sc : slots_) {
        uint8_t* kptr = slot_k_codes(sc);
        uint8_t* vptr = slot_v_codes(sc);
        if (sc.swa || n == 0 || kptr == nullptr) continue;
        if (quant) {
            ck(cudaMemcpy(kptr + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
            ck(cudaMemcpy((uint8_t*) sc.k_scales + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
            ck(cudaMemcpy(vptr + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
            ck(cudaMemcpy((uint8_t*) sc.v_scales + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
            continue;
        }
        for (uint8_t* cache : {kptr, vptr}) {
            ck(cudaMemcpy(cache + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
        }
    }
}

template <class F>
void Inkling2::window_runs(int64_t pos, F&& f) const {
    for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
        const int64_t at = p % ring_, m = std::min<int64_t>(pos - p, ring_ - at);
        f((size_t) (p - (pos - keep_)), (size_t) at, (size_t) m);
        p += m;
    }
}

size_t Inkling2::extra_bytes() const {
    return (size_t) n_swa_slots_ * 2 * (size_t) keep_ * (kv_code_row_bytes() + kv_scale_row_bytes());
}

void Inkling2::save_layer(int slot, int64_t pos, uint8_t* dst) {
    const SlotCache& sc = slots_[(size_t) slot];
    ck(cudaMemcpy(dst + sc.state_off * sizeof(float), state_ + sc.state_off,
                  slot_conv_floats_ * sizeof(float), cudaMemcpyDeviceToHost),
       "save state");
    if (!sc.swa) return;
    const bool quant = opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kQ4_0;
    const size_t code_row = kv_code_row_bytes();
    const size_t scale_row = kv_scale_row_bytes();
    const size_t whole = 2 * (size_t) keep_ * (code_row + scale_row);
    dst += state_n_ * sizeof(float) + (size_t) sc.swa_idx * whole;
    auto save_ring = [&](const uint8_t* cache, size_t one_row) {
        std::memset(dst, 0, (size_t) keep_ * one_row);
        window_runs(pos, [&](size_t s_at, size_t at, size_t m) {
            ck(cudaMemcpy(dst + s_at * one_row, cache + at * one_row, m * one_row, cudaMemcpyDeviceToHost),
               "save state");
        });
        dst += (size_t) keep_ * one_row;
    };
    const uint8_t* kptr = slot_k_codes(sc);
    const uint8_t* vptr = slot_v_codes(sc);
    if (quant) {
        save_ring(kptr, code_row);
        save_ring((const uint8_t*) sc.k_scales, scale_row);
        save_ring(vptr, code_row);
        save_ring((const uint8_t*) sc.v_scales, scale_row);
        return;
    }
    save_ring(kptr, code_row);
    save_ring(vptr, code_row);
}

void Inkling2::load_extra(const uint8_t* src, int64_t pos) {
    const bool quant = opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kQ4_0;
    const size_t code_row = kv_code_row_bytes();
    const size_t scale_row = kv_scale_row_bytes();
    for (const SlotCache& sc : slots_) {
        if (!sc.swa) continue;
        auto load_ring = [&](uint8_t* cache, size_t one_row) {
            window_runs(pos, [&](size_t s_at, size_t at, size_t m) {
                ck(cudaMemcpy(cache + at * one_row, src + s_at * one_row, m * one_row, cudaMemcpyHostToDevice),
                   "load state");
            });
            src += (size_t) keep_ * one_row;
        };
        uint8_t* kptr = slot_k_codes(sc);
        uint8_t* vptr = slot_v_codes(sc);
        if (quant) {
            load_ring(kptr, code_row);
            load_ring((uint8_t*) sc.k_scales, scale_row);
            load_ring(vptr, code_row);
            load_ring((uint8_t*) sc.v_scales, scale_row);
            continue;
        }
        load_ring(kptr, code_row);
        load_ring(vptr, code_row);
    }
}

int Inkling2::head_key(int slot, int n) const {
    return slots_[(size_t) slot].swa ? 0 : room_key(n);
}

void Inkling2::apply_ctrl(const float* cur, const float* msg, const Block& B, int ctrl, float* out, int n) {
    inkling_control_prep(cur, msg, ctrl_cat_, n_embd_, n, 1e-6f, st_);
    lin(B.ctrl_down[ctrl], ctrl_cat_, ctrl_r_, n);
    lin(B.ctrl_gain[ctrl], ctrl_r_, ctrl_g_, n);
    lin(B.ctrl_forget[ctrl], ctrl_r_, ctrl_f_, n);
    inkling_control_apply(cur, msg, ctrl_g_, ctrl_f_, out, n_embd_, n, st_);
}

void Inkling2::store(const float* x, const SlotCache& sc, int64_t pos, int n, bool value) {
    const int width = n_kv_ * head_dim_;
    if (opt_.kv_format == KvFmt::kQ4_0) {
        uint8_t* codes = value ? sc.v_q4 : sc.k_q4;
        uint16_t* scales = value ? sc.v_scales : sc.k_scales;
        if (recording_) return store_q4_at(x, codes, scales, width, at_, n, sc.swa ? ring_ : 0, st_);
        if (!sc.swa) return store_q4(x, codes, scales, width, pos, n, st_);
        const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
        store_q4(x, codes, scales, width, at, head, st_);
        if (head < n) store_q4(x + (size_t) head * width, codes, scales, width, 0, n - head, st_);
        return;
    }
    if (opt_.kv_format == KvFmt::kQ8_0) {
        int8_t* codes = value ? sc.v_codes : sc.k_codes;
        uint16_t* scales = value ? sc.v_scales : sc.k_scales;
        if (recording_) return store_q8_at(x, codes, scales, width, at_, n, sc.swa ? ring_ : 0, st_);
        if (!sc.swa) return store_q8(x, codes, scales, width, pos, n, st_);
        const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
        store_q8(x, codes, scales, width, at, head, st_);
        if (head < n) store_q8(x + (size_t) head * width, codes, scales, width, 0, n - head, st_);
        return;
    }
    uint16_t* cache = value ? sc.vals : sc.keys;
    if (recording_) return store_f16_at(x, cache, width, at_, n, sc.swa ? ring_ : 0, st_);
    if (!sc.swa) return store_f16(x, cache, width, pos, n, st_);
    const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
    store_f16(x, cache, width, at, head, st_);
    if (head < n) store_f16(x + (size_t) head * width, cache, width, 0, n - head, st_);
}

void Inkling2::attn(const Step& st, Block& B, SlotCache& sc, int64_t pos, int n) {
    const int ip = st.phys, pass = st.pass;
    const long long* at = recording_ ? at_ : nullptr;
    const int threads = sum_threads(n_embd_);
    const Layer& L = layers_[(size_t) ip];
    float* in0 = inputs_buf_;
    float* in1 = inputs_buf_ + (size_t) n * n_embd_;
    float* in2 = inputs_buf_ + 2 * (size_t) n * n_embd_;

    // 1. Attention Residual mix over blocks so far + current stream 0
    if (ip == 0) {
        inkling_read_stream(r_, 6, 0, in0, n_embd_, n, st_);
    } else {
        MixSlots ms{};
        int ns = 2;
        if (ip <= core_start_) {
            ms = {{0, 1, 0, 0, 0, 0}};
            ns = 2;
        } else if (ip <= core_start_ + block_size_) {
            ms = {{0, 1, 2, 0, 0, 0}};
            ns = 3;
        } else if (ip <= core_end_) {
            ms = {{0, 1, 2, 3, 0, 0}};
            ns = 4;
        } else {
            ms = {{0, 1, 2, 3, 4, 0}};
            ns = 5;
        }
        if (B.swa) {
            inkling_mix(r_, 6, ms, ns, fp(B.attn_res_query), 1, in0, n_embd_, n, 1e-6f, st_);
            if (pass > 0) {
                ck(cudaMemcpyAsync(in1, in0, (size_t) n * n_embd_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
                   "copy input");
                ck(cudaMemcpyAsync(in2, in0, (size_t) n * n_embd_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
                   "copy input");
            }
        } else {
            inkling_mix(r_, 6, ms, ns, fp(B.attn_res_query), 3, in0, n_embd_, n, 1e-6f, st_);
        }
    }

    // 2. Cross-pass loop controls (pass > 0)
    const bool share_swa = B.swa && pass == 0;
    if (pass > 0) {
        apply_ctrl(in0, msg_ + 0 * (size_t) n * n_embd_, B, 0, in0, n);
        apply_ctrl(in1, msg_ + 1 * (size_t) n * n_embd_, B, 1, in1, n);
        apply_ctrl(in2, msg_ + 2 * (size_t) n * n_embd_, B, 2, in2, n);
    }

    // 3. Input norm + linear projections -> Q, K, V, R, sinks, gate
    const Dev& nq = B.swa ? L.attn_norm : B.q_inp_norm;
    const Dev& nk = B.swa ? L.attn_norm : B.k_inp_norm;
    const Dev& nv = B.swa ? L.attn_norm : B.v_inp_norm;

    kernels::native_add_rms_norm_q8_1(in0, nullptr, fp(nq), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    lin(B.wq, xn_, qraw_, n, xq_);
    lin(B.attn_rel, xn_, r_raw_, n, xq_);
    lin(B.attn_sink_proj, xn_, sinks_, n, xq_);
    lin(B.wqkv_gate, xn_, gate_, n, xq_);

    if (!share_swa) {
        kernels::native_add_rms_norm_q8_1(in1, nullptr, fp(nk), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    }
    lin(B.wk, xn_, kraw_, n, xq_);
    if (!B.swa) lin(B.k_conv_proj, xn_, dyn_, n, xq_);
    inkling_conv4(kraw_, state_ + sc.state_off + off_k_conv_, fp(B.attn_k_conv), B.swa ? nullptr : dyn_, head_dim_,
                  k_, n_kv_ * head_dim_, n, st_, snap_bits_ ? snap_at_ + sc.state_off + off_k_conv_ : nullptr, rec_n_,
                  snap_bits_);

    if (!share_swa) {
        kernels::native_add_rms_norm_q8_1(in2, nullptr, fp(nv), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    }
    lin(B.wv, xn_, kraw_, n, xq_);
    if (!B.swa) lin(B.v_conv_proj, xn_, dyn_, n, xq_);
    inkling_conv4(kraw_, state_ + sc.state_off + off_v_conv_, fp(B.attn_v_conv), B.swa ? nullptr : dyn_, head_dim_,
                  v_, n_kv_ * head_dim_, n, st_, snap_bits_ ? snap_at_ + sc.state_off + off_v_conv_ : nullptr, rec_n_,
                  snap_bits_);

    // 4. L2 clip + global tau(pos) on Q & R, L2 clip on K, parameterless RMSNorm on V
    inkling_l2_clip(qraw_, fp(B.attn_q_clip), q_, head_dim_, n_head_, qk_clip_, pos, !B.swa, (float) log_floor_,
                    log_alpha_, n, at, st_);
    inkling_l2_clip(k_, nullptr, k_, head_dim_, n_kv_, qk_clip_, pos, false, (float) log_floor_, 0.f, n, nullptr,
                    st_);
    rms_norm(v_, nullptr, v_, head_dim_, n * n_kv_, rms_eps_, st_);
    inkling_l2_clip(r_raw_, fp(B.attn_rel_clip), r_cur_, n_rel_, n_head_, rel_clip_, pos, !B.swa, (float) log_floor_,
                    log_alpha_, n, at, st_);

    // 5. Store K and V into slot cache
    store(k_, sc, pos, n, false);
    store(v_, sc, pos, n, true);

    // 6. Relative-bias attention + virtual zero sink
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    const int n_extent = B.swa ? window_ : n_rel_extent_;
    const bool q8 = opt_.kv_format == KvFmt::kQ8_0;
    const bool q4 = opt_.kv_format == KvFmt::kQ4_0;
    const bool quant = q8 || q4;
    const Kvq8 kq = q8 ? Kvq8{sc.k_codes, sc.k_scales} : Kvq8{};
    const Kvq8 vq = q8 ? Kvq8{sc.v_codes, sc.v_scales} : Kvq8{};
    const Kvq4 kq4 = q4 ? Kvq4{sc.k_q4, sc.k_scales} : Kvq4{};
    const Kvq4 vq4 = q4 ? Kvq4{sc.v_q4, sc.v_scales} : Kvq4{};
    const int c_step = n >= kStageRows ? chunk_batch_ : chunk_dec_;
    for (int first = 0; first < n; first += c_step) {
        const int c = std::min(c_step, n - first);
        const float* r_sub = r_cur_ + (size_t) first * n_head_ * n_rel_;
        const float* s_sub = sinks_ + (size_t) first * n_head_;
        const float* q_sub = q_ + (size_t) first * n_head_ * head_dim_;
        float* att_sub = att_ + (size_t) first * n_head_ * head_dim_;
        inkling_rel_table(r_sub, fp(B.attn_rel_bias), s_sub, table_, n_rel_, n_extent, n_head_, c, st_);
        if (n >= kStageRows) {
            uint16_t* kw = quant ? (uint16_t*) q8w_ : nullptr;
            uint16_t* vw = quant ? kw + work_halves_ : nullptr;
            inkling_attend_batch(blas_, q_sub, quant ? nullptr : sc.keys, quant ? nullptr : sc.vals, table_, s_sub,
                                 n_extent, pos + first, c, score_, score_n_, awork_, att_sub, head_dim_, n_head_,
                                 n_kv_, scale, st_, window, ring, kq, vq, kw, vw, quant ? work_halves_ : 0, kq4,
                                 vq4);
        } else {
            const long long* at_sub = at ? at + 2 * first : nullptr;
            const int rm = !at ? 0 : B.swa ? std::min(window_ + n - 1, opt_.max_context) : room(n);
            inkling_attend(q_sub, quant ? nullptr : sc.keys, quant ? nullptr : sc.vals, table_, s_sub, n_extent,
                           pos + first, c, score_, chunk_dec_, att_sub, head_dim_, n_head_, n_kv_, scale, st_,
                           window, ring, at_sub, rm, kq, vq, kq4, vq4);
        }
    }

    // 7. Sigmoid gate -> wo -> post-conv4
    mul_sigmoid(att_, gate_, n_head_ * head_dim_, n, n_head_ * head_dim_, false, st_);
    lin(B.wo, att_, cur_, n);
    inkling_conv4(cur_, state_ + sc.state_off + off_attn_post_, fp(B.attn_post_conv), nullptr, 0, ao_, n_embd_, n, st_,
                  snap_bits_ ? snap_at_ + sc.state_off + off_attn_post_ : nullptr, rec_n_, snap_bits_);
}

void Inkling2::layer_head(Layer& L, int slot, int n) {
    const Step& st = schedule_[(size_t) slot];
    const int ip = st.phys, pass = st.pass;
    Block& B = blocks_[(size_t) ip];
    SlotCache& sc = slots_[(size_t) slot];

    if (pass > 0) {
        const MixSlots bank_slots{{5, 4, 0, 0, 0, 0}};
        inkling_mix(r_, 6, bank_slots, 2, fp(B.loop_msg_query), 4, msg_, n_embd_, n, 1e-6f, st_);
    }

    attn(st, B, sc, pos_, n);
    dbg("attn", slot, ao_, n_embd_);

    // Block AttnRes update: at ip % block_size_ == 0 push inpSA (stream 0) and replace stream 0 = ao_ (no add).
    if (ip == 12) inkling_copy_stream(r_, 6, 3, 0, n_embd_, n, st_);
    else if (ip == core_end_) inkling_copy_stream(r_, 6, 4, 0, n_embd_, n, st_);

    if (ip % block_size_ == 0) {
        inkling_write_stream(ao_, r_, 6, 0, n_embd_, n, st_);
    } else {
        inkling_read_stream(r_, 6, 0, inp_sa_, n_embd_, n, st_);
        inkling_add_to_stream(inp_sa_, ao_, r_, 6, 0, n_embd_, n, st_);
    }

    // FFN Block AttnRes mix + optional loop control + ffn_norm -> xn_, xq_
    MixSlots ffn_slots{};
    int ffn_n = 2;
    if (ip < core_start_) { ffn_slots = {{0, 1, 0, 0, 0, 0}}; ffn_n = 2; }
    else if (ip < 12) { ffn_slots = {{0, 1, 2, 0, 0, 0}}; ffn_n = 3; }
    else if (ip < core_end_) { ffn_slots = {{0, 1, 2, 3, 0, 0}}; ffn_n = 4; }
    else { ffn_slots = {{0, 1, 2, 3, 4, 0}}; ffn_n = 5; }

    inkling_mix(r_, 6, ffn_slots, ffn_n, fp(B.ffn_res_query), 1, cur_, n_embd_, n, 1e-6f, st_);
    if (pass > 0) apply_ctrl(cur_, msg_ + 3 * (size_t) n * n_embd_, B, 3, cur_, n);

    const int threads = sum_threads(n_embd_);
    kernels::native_add_rms_norm_q8_1(cur_, nullptr, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("ffn_in", slot, xn_, n_embd_);
    if (!batched(L, n)) moe_router(L, n);
}

void Inkling2::layer_tail(Layer&, int slot, int n) {
    const Step& st = schedule_[(size_t) slot];
    const int ip = st.phys, pass = st.pass;
    Block& B = blocks_[(size_t) ip];
    SlotCache& sc = slots_[(size_t) slot];

    inkling_conv4(fo_, state_ + sc.state_off + off_ffn_post_, fp(B.ffn_post_conv), nullptr, 0, cur_, n_embd_, n, st_,
                  snap_bits_ ? snap_at_ + sc.state_off + off_ffn_post_ : nullptr, rec_n_, snap_bits_);
    inkling_read_stream(r_, 6, 0, inp_sa_, n_embd_, n, st_);
    inkling_add_to_stream(inp_sa_, cur_, r_, 6, 0, n_embd_, n, st_);

    if (ip == core_start_ - 1) {
        inkling_copy_stream(r_, 6, 2, 0, n_embd_, n, st_);
    } else if (ip == core_end_ - 1 && pass < n_passes_ - 1) {
        inkling_copy_stream(r_, 6, 4, 3, n_embd_, n, st_);
        inkling_copy_stream(r_, 6, 5, 0, n_embd_, n, st_);
        inkling_copy_stream(r_, 6, 0, 2, n_embd_, n, st_);
    }
}

void Inkling2::output_rows(int first, int rows, float* h) {
    const MixSlots out_slots{{0, 1, 2, 3, 4, 0}};
    inkling_mix(r_ + (size_t) first * 6 * n_embd_, 6, out_slots, 5, fp(output_res_query_), 1, cur_, n_embd_, rows,
                1e-6f, st_);
    rms_norm(cur_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

void Inkling2::finish_logits(float* logits, int rows) {
    if (final_logit_softcapping_ > 0.f) softcap(logits, (size_t) rows * n_vocab_, final_logit_softcapping_, st_);
}

void Inkling2::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];
    }
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = probs[best];
        sum += w[k];
    }
    const float denom = std::max(sum, kMinWeightSum);
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

bool Inkling2::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, fp(B.probs_b_dev), nullptr, true, w_scale_, ids, w, st_);
    return true;
}

void Inkling2::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
}

}  // namespace strata::poly
