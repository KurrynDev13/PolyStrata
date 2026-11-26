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
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;
constexpr size_t kQ8WorkRows = 2048;

double env_num(const char* key, double fallback) {
    const char* v = std::getenv(key);
    return v && *v ? std::atof(v) : fallback;
}

std::string zha_file() {
    const char* v = std::getenv("STRATA_INK_ZHA");
    return v ? v : "";
}

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
    std::unique_ptr<Shards> other = std::move(dg_);
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

    const char* env_fb = std::getenv("STRATA_FAST_REL_BIAS");
    fast_rel_bias_ = n_rel_ == 16 && env_fb &&
                     (std::strcmp(env_fb, "1") == 0 || std::strcmp(env_fb, "true") == 0 ||
                      std::strcmp(env_fb, "on") == 0);
    if (fast_rel_bias_) {
        std::fprintf(stderr, "strata-poly: Inkling2 fast transposed relative bias enabled\n");
    }

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
#if !defined(STRATA_GPU_METAL) && !defined(STRATA_USE_HIP)
    {
        const char* v = std::getenv("STRATA_INK_KERNEL_TABLES");
        kernel_tables_ = !(v && std::string(v) == "0");
    }
#endif

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

    if (other) {
        if (split_) throw std::runtime_error("Inkling2 DSpark does not support --layer-split");
        const MetaValue* darch = other->get("general.architecture");
        if (!darch || darch->s != "inkling2_dspark")
            throw std::runtime_error(opt_.draft_model + " is not an inkling2_dspark draft GGUF");
        auto d_u = [&](const char* key) -> int64_t {
            const MetaValue* v = other->get(std::string("inkling2_dspark.") + key);
            need(v != nullptr, key);
            return (int64_t) v->u;
        };
        auto d_f = [&](const char* key) -> float {
            const MetaValue* v = other->get(std::string("inkling2_dspark.") + key);
            need(v != nullptr, key);
            return (float) v->num();
        };
        dspark_layers_ = (int) d_u("block_count");
        need(dspark_layers_ == 3, "inkling2_dspark.block_count must be 3");
        need((int) d_u("embedding_length") == n_embd_, "inkling2_dspark embedding_length mismatch");
        dspark_ff_ = (int) d_u("feed_forward_length");
        dspark_heads_ = (int) d_u("attention.head_count");
        dspark_kv_ = (int) d_u("attention.head_count_kv");
        dspark_head_dim_ = (int) d_u("attention.key_length");
        need((int) d_u("attention.value_length") == dspark_head_dim_, "inkling2_dspark key_length != value_length");
        need(dspark_heads_ == 24 && dspark_kv_ == 8 && dspark_head_dim_ == 128 && dspark_ff_ > 0 && dspark_ff_ % 32 == 0,
             "unsupported inkling2_dspark attention/ffn dimensions");
        dspark_window_ = (int) d_u("attention.sliding_window");
        dspark_eps_ = d_f("attention.layer_norm_rms_epsilon");
        dspark_rope_theta_ = d_f("rope.freq_base");
        dspark_block_ = (int) d_u("block_size");
        dspark_rank_ = (int) d_u("markov_rank");
        dspark_w1_vocab_ = (int) d_u("vocab_size");
        need(dspark_window_ > 0 && dspark_block_ == 8 && dspark_rank_ == 256 && dspark_w1_vocab_ >= n_vocab_,
             "unsupported inkling2_dspark block/markov geometry");
        const MetaValue* htype = other->get("inkling2_dspark.markov_head_type");
        need(htype != nullptr && htype->s == "rnn_hidden", "inkling2_dspark.markov_head_type must be rnn_hidden");
        const MetaValue* taps = other->get("inkling2_dspark.target_layer_ids");
        need(taps != nullptr && taps->type == MetaType::ARRAY && taps->items.size() == 4,
             "inkling2_dspark.target_layer_ids must have 4 entries");
        for (int i = 0; i < 3; ++i) {
            dspark_taps_[i] = (int) taps->items[(size_t) i].u;
            need(dspark_taps_[i] >= 0 && dspark_taps_[i] < exec_steps() &&
                     (i == 0 || dspark_taps_[i] > dspark_taps_[i - 1]),
                 "bad inkling2_dspark.target_layer_ids");
        }
        need((int) taps->items[3].u == exec_steps(), "inkling2_dspark last target_layer_id must equal exec_steps");
        dspark_l_.assign((size_t) dspark_layers_, DsparkLayer{});
        const MetaValue* locs = other->get("inkling2_dspark.local_layer_ids");
        need(locs != nullptr && locs->type == MetaType::ARRAY, "bad inkling2_dspark.local_layer_ids");
        for (const MetaValue& it : locs->items) {
            need((int64_t) it.u >= 0 && (int64_t) it.u < dspark_layers_, "local_layer_id out of range");
            dspark_l_[(size_t) it.u].local = true;
        }
        dspark_ring_ = dspark_window_ + dspark_block_;
        dspark_theta_scale_ = std::pow(dspark_rope_theta_, -2.0f / (float) dspark_head_dim_);
        dspark_ = true;
        draft_ = true;
        block_layer_ = false;
        dg_ = std::move(other);
        block_ = dg_.get();
        std::fprintf(stderr,
                     "strata-poly: Inkling2 DSpark MTP loaded (block=%d, rank=%d, window=%d, taps=[%d,%d,%d,%d])\n",
                     dspark_block_, dspark_rank_, dspark_window_, dspark_taps_[0], dspark_taps_[1], dspark_taps_[2],
                     exec_steps());
    }
    zha_load();
    if (zha_on_ && dspark_) {
        if (opt_.window == 0) opt_.window = 4;
        need(opt_.window <= 4, "the ZHA index (STRATA_INK_ZHA) with --draft-model supports --window 1 to 4");
    }
    need(!zha_on_ || opt_.batch <= 0, "the ZHA index (STRATA_INK_ZHA) runs on one sequence (--batch 0)");
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
    const int64_t rel_dim = (int64_t) n_rel_ * n_head_;

    auto can_fuse_as_is = [&](const TensorInfo* t, int64_t a, int64_t b) {
        return t && t->shape.size() == 2 && (int64_t) t->shape[0] == a && (int64_t) t->shape[1] == b &&
               global_scale(*t) == 1.f &&
               (t->type == kTypeF32 || t->type == kTypeBF16 || kernels::native_mmvq_supported((int) t->type));
    };
    auto concat_same_type = [&](std::initializer_list<std::pair<const char*, int64_t>> names, int64_t n_in,
                                Dev& fused, std::initializer_list<Dev*> views) {
        std::vector<const TensorInfo*> ts;
        std::vector<size_t> bytes;
        int64_t total_out = 0;
        size_t total_bytes = 0;
        uint32_t type = 0;
        for (const auto& [name, n_out] : names) {
            const TensorInfo* t = tensor(il, name);
            if (!t) throw std::runtime_error("the GGUF has no tensor " + blk(il, name));
            if (ts.empty()) type = t->type;
            const size_t b = home(*t).payload(*t);
            ts.push_back(t);
            bytes.push_back(b);
            total_out += n_out;
            total_bytes += b;
        }
        if (type == kTypeBF16) bf16_max_ = std::max(bf16_max_, (size_t) n_in * (size_t) total_out);
        fused = Dev{dense_ ? dense_ + dense_at_ : nullptr, type, (int) n_in, (int) total_out, 1.f};
        dense_at_ += pad(total_bytes) + kTail;
        if (dense_ && dense_at_ > dense_bytes_)
            throw std::logic_error("the dense arena is too small for fused tensor " + ts[0]->name);
        size_t off = 0;
        size_t idx = 0;
        auto v_it = views.begin();
        auto n_it = names.begin();
        for (; idx < ts.size(); ++idx, ++v_it, ++n_it) {
            Dev* vp = *v_it;
            *vp = Dev{dense_ ? (uint8_t*) fused.p + off : nullptr, type, (int) n_in, (int) n_it->second, 1.f};
            if (dense_) {
                h2d(vp->p, home(*ts[idx]).data(*ts[idx]), bytes[idx]);
                uploaded_.emplace_back(home(*ts[idx]).data(*ts[idx]), bytes[idx]);
            }
            off += bytes[idx];
        }
    };

    L.fixed = !zha_full(B);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    {
        const TensorInfo* tq = tensor(il, "attn_q.weight");
        const TensorInfo* tg = tensor(il, "attn_gate.weight");
        const TensorInfo* tr = tensor(il, "attn_rel.weight");
        const TensorInfo* ts = tensor(il, "attn_sink_proj.weight");
        const bool fuse_qg = can_fuse_as_is(tq, n_embd_, q_dim) && can_fuse_as_is(tg, n_embd_, q_dim) &&
                             tq->type == tg->type;
        if (fuse_qg) {
            const bool fuse_r = can_fuse_as_is(tr, n_embd_, rel_dim) && tr->type == tq->type;
            const bool fuse_s = can_fuse_as_is(ts, n_embd_, n_head_) && ts->type == tq->type;
            B.wq_fuse_flags = 1 | (fuse_r ? 2 : 0) | (fuse_s ? 4 : 0);
            int off_rows = (int) (2 * q_dim);
            if (fuse_r && fuse_s) {
                B.wq_rel_off = off_rows;
                B.wq_sink_off = off_rows + (int) rel_dim;
                concat_same_type({{"attn_q.weight", q_dim}, {"attn_gate.weight", q_dim},
                                  {"attn_rel.weight", rel_dim}, {"attn_sink_proj.weight", n_head_}},
                                 n_embd_, B.wq_fused, {&B.wq, &B.wqkv_gate, &B.attn_rel, &B.attn_sink_proj});
            } else if (fuse_r) {
                B.wq_rel_off = off_rows;
                concat_same_type({{"attn_q.weight", q_dim}, {"attn_gate.weight", q_dim},
                                  {"attn_rel.weight", rel_dim}},
                                 n_embd_, B.wq_fused, {&B.wq, &B.wqkv_gate, &B.attn_rel});
                B.attn_sink_proj = mat("attn_sink_proj.weight", n_embd_, n_head_);
            } else if (fuse_s) {
                B.wq_sink_off = off_rows;
                concat_same_type({{"attn_q.weight", q_dim}, {"attn_gate.weight", q_dim},
                                  {"attn_sink_proj.weight", n_head_}},
                                 n_embd_, B.wq_fused, {&B.wq, &B.wqkv_gate, &B.attn_sink_proj});
                B.attn_rel = mat("attn_rel.weight", n_embd_, rel_dim);
            } else {
                concat_same_type({{"attn_q.weight", q_dim}, {"attn_gate.weight", q_dim}},
                                 n_embd_, B.wq_fused, {&B.wq, &B.wqkv_gate});
                B.attn_rel = mat("attn_rel.weight", n_embd_, rel_dim);
                B.attn_sink_proj = mat("attn_sink_proj.weight", n_embd_, n_head_);
            }
        } else {
            B.wq = mat("attn_q.weight", n_embd_, q_dim);
            B.wqkv_gate = mat("attn_gate.weight", n_embd_, q_dim);
            B.attn_sink_proj = mat("attn_sink_proj.weight", n_embd_, n_head_);
            B.attn_rel = mat("attn_rel.weight", n_embd_, rel_dim);
        }
    }
    if (B.swa) {
        const TensorInfo* tk = tensor(il, "attn_k.weight");
        const TensorInfo* tv = tensor(il, "attn_v.weight");
        if (can_fuse_as_is(tk, n_embd_, kv_dim) && can_fuse_as_is(tv, n_embd_, kv_dim) && tk->type == tv->type) {
            concat_same_type({{"attn_k.weight", kv_dim}, {"attn_v.weight", kv_dim}},
                             n_embd_, B.wkv_fused, {&B.wk, &B.wv});
        } else {
            B.wk = mat("attn_k.weight", n_embd_, kv_dim);
            B.wv = mat("attn_v.weight", n_embd_, kv_dim);
        }
    } else {
        B.wk = mat("attn_k.weight", n_embd_, kv_dim);
        B.wv = mat("attn_v.weight", n_embd_, kv_dim);
    }
    B.wo = mat("attn_output.weight", q_dim, n_embd_);
    B.attn_q_clip = f32("attn_q_clip.weight", head_dim_, n_head_);
    B.attn_rel_bias = f32("attn_rel_bias.weight", n_rel_, n_extent);
    if (fast_rel_bias_ && dense_ && B.attn_rel_bias.p && n_rel_ == 16) {
        const TensorInfo* tb = tensor(il, "attn_rel_bias.weight");
        if (tb && tb->type == kTypeF32) {
            std::vector<float> tr((size_t) n_rel_ * (size_t) n_extent);
            const float* src = (const float*) home(*tb).data(*tb);
            for (int d = 0; d < n_extent; ++d) {
                for (int k = 0; k < n_rel_; ++k) {
                    tr[(size_t) k * (size_t) n_extent + d] = src[(size_t) d * (size_t) n_rel_ + k];
                }
            }
            void* p = nullptr;
            ck(cudaMalloc(&p, tr.size() * sizeof(float)), "attn_rel_bias_trans");
            dev_allocs_.push_back(p);
            B.attn_rel_bias_trans.p = p;
            B.attn_rel_bias_trans.type = kTypeF32;
            h2d(p, tr.data(), tr.size() * sizeof(float));
        }
    }
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
            const TensorInfo* tgn = tensor(il, gain_names[c]);
            const TensorInfo* tfg = tensor(il, forget_names[c]);
            if (can_fuse_as_is(tgn, gate_rank_, n_embd_) && can_fuse_as_is(tfg, gate_rank_, n_embd_) &&
                tgn->type == tfg->type) {
                concat_same_type({{gain_names[c], n_embd_}, {forget_names[c], n_embd_}},
                                 gate_rank_, B.ctrl_gf[c], {&B.ctrl_gain[c], &B.ctrl_forget[c]});
            } else {
                B.ctrl_gain[c] = mat(gain_names[c], gate_rank_, n_embd_);
                B.ctrl_forget[c] = mat(forget_names[c], gate_rank_, n_embd_);
            }
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

Inkling2::~Inkling2() {
    if (device_ >= 0) cudaSetDevice(device_);
    if (st_) cudaStreamSynchronize(cs(st_));
    for (ZSlot& z : zs_) {
        for (void* p : {(void*) z.zk, (void*) z.zv, (void*) z.zb})
            if (p) cudaFree(p);
        for (void* p : {(void*) z.kc, (void*) z.ksc, (void*) z.vc, (void*) z.vsc})
            if (p) cudaFreeHost(p);
    }
    for (Mapped& m : zmap_) vmm_free(m);
    if (h_dspark_w1_) cudaFreeHost(h_dspark_w1_);
    if (h_dspark_fold_w_) cudaFreeHost(h_dspark_fold_w_);
}

size_t Inkling2::kv_code_row_bytes(bool value) const {
    const size_t row = (size_t) n_kv_ * head_dim_;
    if (opt_.kv_format == KvFmt::kQ4_0 || (value && opt_.kv_format == KvFmt::kK8V4)) return row / 2;
    if (opt_.kv_format == KvFmt::kQ8_0 || (!value && opt_.kv_format == KvFmt::kK8V4)) return row;
    return row * sizeof(uint16_t);
}

size_t Inkling2::kv_scale_row_bytes() const {
    const bool quant =
        opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kQ4_0 || opt_.kv_format == KvFmt::kK8V4;
    return quant ? (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t) : 0;
}

size_t Inkling2::dspark_kv_code_row_bytes(bool value) const {
    const size_t row = (size_t) dspark_kv_ * dspark_head_dim_;
    if (opt_.kv_format == KvFmt::kQ4_0 || (value && opt_.kv_format == KvFmt::kK8V4)) return row / 2;
    if (opt_.kv_format == KvFmt::kQ8_0 || (!value && opt_.kv_format == KvFmt::kK8V4)) return row;
    return row * sizeof(uint16_t);
}

size_t Inkling2::dspark_kv_scale_row_bytes() const {
    const bool quant =
        opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kQ4_0 || opt_.kv_format == KvFmt::kK8V4;
    return quant ? (size_t) dspark_kv_ * (dspark_head_dim_ / 32) * sizeof(uint16_t) : 0;
}

uint8_t* Inkling2::slot_k_codes(const SlotCache& sc) const {
    if (opt_.kv_format == KvFmt::kQ4_0) return sc.k_q4;
    if (opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kK8V4) return (uint8_t*) sc.k_codes;
    return (uint8_t*) sc.keys;
}

uint8_t* Inkling2::slot_v_codes(const SlotCache& sc) const {
    if (opt_.kv_format == KvFmt::kQ4_0 || opt_.kv_format == KvFmt::kK8V4) return sc.v_q4;
    if (opt_.kv_format == KvFmt::kQ8_0) return (uint8_t*) sc.v_codes;
    return (uint8_t*) sc.vals;
}

uint8_t* Inkling2::dspark_k_codes(const DsparkLayer& dl) const {
    if (dl.local) return (uint8_t*) dl.dk;
    if (opt_.kv_format == KvFmt::kQ4_0) return dl.dk_q4;
    if (opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kK8V4) return (uint8_t*) dl.dk_codes;
    return (uint8_t*) dl.dk;
}

uint8_t* Inkling2::dspark_v_codes(const DsparkLayer& dl) const {
    if (dl.local) return (uint8_t*) dl.dv;
    if (opt_.kv_format == KvFmt::kQ4_0 || opt_.kv_format == KvFmt::kK8V4) return dl.dv_q4;
    if (opt_.kv_format == KvFmt::kQ8_0) return (uint8_t*) dl.dv_codes;
    return (uint8_t*) dl.dv;
}

void Inkling2::alloc_caches() {
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    const bool k_q8 = opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kK8V4;
    const bool k_q4 = opt_.kv_format == KvFmt::kQ4_0;
    const bool v_q8 = opt_.kv_format == KvFmt::kQ8_0;
    const bool v_q4 = opt_.kv_format == KvFmt::kQ4_0 || opt_.kv_format == KvFmt::kK8V4;
    const bool quant = k_q8 || k_q4 || v_q8 || v_q4;
    const size_t row = (size_t) n_kv_ * head_dim_;
    const size_t k_code = kv_code_row_bytes(false);
    const size_t v_code = kv_code_row_bytes(true);
    const size_t scale_row = kv_scale_row_bytes();
    if (quant && again_ == SIZE_MAX) {
        if (opt_.kv_format == KvFmt::kK8V4)
            std::fprintf(stderr, "strata-poly: inkling2 k8v4 KV cache: %zu K + %zu V bytes/row (%d global, %d SWA)\n",
                         k_code + scale_row, v_code + scale_row, n_global_slots_, n_swa_slots_);
        else
            std::fprintf(stderr, "strata-poly: inkling2 %s KV cache: %zu bytes/row (%d global, %d SWA)\n",
                         k_q4 ? "q4_0" : "q8_0", k_code + scale_row, n_global_slots_, n_swa_slots_);
    }

    const size_t swa_codes = (size_t) n_swa_slots_ * (size_t) ring_ * (k_code + v_code);
    const size_t swa_scales = quant ? 2 * (size_t) n_swa_slots_ * (size_t) ring_ * scale_row : 0;

    uint8_t* rc = quant ? (uint8_t*) seq_alloc(swa_codes, "the attention cache") : nullptr;
    uint8_t* rs = quant ? (uint8_t*) seq_alloc(swa_scales, "the attention cache") : nullptr;
    uint16_t* kv = !quant ? (uint16_t*) seq_alloc(2 * (size_t) n_swa_slots_ * (size_t) ring_ * row * sizeof(uint16_t),
                                                   "the attention cache")
                          : nullptr;

    slots_.assign(schedule_.size(), SlotCache{});
    if (zha_on_) {
        zring_ = std::max<int64_t>(zring_, (int64_t) zha_.recent + 2 * zha_.C + std::max(rows_, 64) + 64);
        if (zs_.size() != slots_.size()) zs_.assign(slots_.size(), ZSlot{});
    }
    int swa_idx = 0;
    for (size_t s = 0; s < schedule_.size(); ++s) {
        SlotCache& sc = slots_[s];
        const Block& B = blocks_[(size_t) schedule_[s].phys];
        sc.swa = B.swa;
        sc.state_off = s * slot_conv_floats_;
        if (B.swa) {
            sc.swa_idx = swa_idx++;
            if (quant) {
                if (k_q4) sc.k_q4 = rc;
                else sc.k_codes = (int8_t*) rc;
                rc += (size_t) ring_ * k_code;
                sc.k_scales = (uint16_t*) rs;
                rs += (size_t) ring_ * scale_row;
                if (v_q4) sc.v_q4 = rc;
                else sc.v_codes = (int8_t*) rc;
                rc += (size_t) ring_ * v_code;
                sc.v_scales = (uint16_t*) rs;
                rs += (size_t) ring_ * scale_row;
            } else {
                sc.keys = kv;
                kv += (size_t) ring_ * row;
                sc.vals = kv;
                kv += (size_t) ring_ * row;
            }
        } else if (zha_on_) {
            zha_alloc((int) s, schedule_[s].phys, schedule_[s].pass);
        } else if (quant) {
            if (k_q4) sc.k_q4 = (uint8_t*) kv_alloc(k_code, 1);
            else sc.k_codes = (int8_t*) kv_alloc(k_code, 1);
            sc.k_scales = (uint16_t*) kv_alloc(scale_row, 1);
            if (v_q4) sc.v_q4 = (uint8_t*) kv_alloc(v_code, 1);
            else sc.v_codes = (int8_t*) kv_alloc(v_code, 1);
            sc.v_scales = (uint16_t*) kv_alloc(scale_row, 1);
        } else {
            sc.keys = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
            sc.vals = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
        }
    }

    if (dspark_) {
        const size_t d_row = (size_t) dspark_kv_ * dspark_head_dim_ * sizeof(uint16_t);
        const size_t dk_code = dspark_kv_code_row_bytes(false);
        const size_t dv_code = dspark_kv_code_row_bytes(true);
        const size_t d_scale_row = dspark_kv_scale_row_bytes();
        for (DsparkLayer& dl : dspark_l_) {
            if (dl.local) {
                dl.dk = (uint16_t*) seq_alloc((size_t) dspark_ring_ * d_row, "dspark swa k");
                dl.dv = (uint16_t*) seq_alloc((size_t) dspark_ring_ * d_row, "dspark swa v");
            } else if (quant) {
                if (k_q4) dl.dk_q4 = (uint8_t*) kv_alloc(dk_code, 1);
                else dl.dk_codes = (int8_t*) kv_alloc(dk_code, 1);
                dl.dk_scales = (uint16_t*) kv_alloc(d_scale_row, 1);
                if (v_q4) dl.dv_q4 = (uint8_t*) kv_alloc(dv_code, 1);
                else dl.dv_codes = (int8_t*) kv_alloc(dv_code, 1);
                dl.dv_scales = (uint16_t*) kv_alloc(d_scale_row, 1);
            } else {
                dl.dk = (uint16_t*) kv_alloc(d_row, 1);
                dl.dv = (uint16_t*) kv_alloc(d_row, 1);
            }
        }
    }

    const int n_splits_max = std::clamp((opt_.max_context + 127) / 128, 1, 64);
    chunk_batch_ = std::min(rows_, 64);
    chunk_dec_ = (int) std::clamp<size_t>(
        (size_t(8) << 20) / std::max<size_t>(1, (size_t) n_head_ * n_splits_max * (head_dim_ + 2)), 1,
        (size_t) std::min({rows_, kStageRows - 1, 16}));
}

int Inkling2::widest() const {
    int w = std::max({2 * n_embd_, n_head_ * head_dim_});
    if (dspark_) w = std::max({w, dspark_ff_, 2 * dspark_rank_ + 2 * n_embd_});
    return w;
}

void Inkling2::assign_scratch() {
    const size_t n = (size_t) scratch_rows();
    const bool prompt_mode = !prompt_loan_ || binding_prompt();
    const size_t n_d = dspark_ ? std::max(n, (size_t) dspark_block_) : n;
    const size_t heads = n_d * n_head_ * head_dim_;
    const size_t kv = n_d * n_kv_ * head_dim_;
    inputs_buf_ = sbuf(3 * n * n_embd_);
    msg_ = sbuf(4 * n * n_embd_);
    const size_t max_fused = std::min<size_t>(n, (size_t) kStageRows - 1);
    const size_t qf_cols = 2 * (size_t) n_head_ * head_dim_ + (size_t) n_rel_ * n_head_ + (size_t) n_head_;
    const size_t qf_stride = (qf_cols + 63) & ~size_t(63);
    const size_t ctrl_n = std::max(4 * n * n_embd_, max_fused * (qf_stride + 2 * (size_t) n_kv_ * head_dim_));
    float* ctrl_arena = sbuf(ctrl_n);
    ctrl_cat_ = ctrl_arena;
    ctrl_g_ = ctrl_arena + n * 2 * n_embd_;
    ctrl_r_ = sbuf(3 * n * gate_rank_);
    ctrl_f_ = sbuf(n * n_embd_);
    q_fused_buf_ = ctrl_arena;
    kv_fused_buf_ = ctrl_arena + max_fused * qf_stride;

    qraw_ = sbuf(heads);
    q_ = sbuf(heads);
    kraw_ = sbuf(kv);
    k_ = sbuf(kv);
    v_ = sbuf(kv);
    r_raw_ = sbuf(n * n_head_ * n_rel_);
    r_cur_ = sbuf(n * n_head_ * n_rel_);
    dyn_ = sbuf(n * l_conv_ * n_kv_);
    sinks_ = sbuf(n * n_head_);
    gate_ = sbuf(heads);
    att_ = sbuf(heads);

    const size_t max_ext = (size_t) std::max(window_, n_rel_extent_);
    const bool direct_only = head_dim_ == 128 && n_head_ == n_kv_ * 3 && n_rel_ == 16;
    if (prompt_mode) table_ = sbuf((size_t) chunk_batch_ * n_head_ * max_ext);
    else if (!direct_only) table_ = sbuf((size_t) chunk_dec_ * n_head_ * max_ext);
    else table_ = nullptr;

    const int64_t att_ctx = zha_on_ ? std::min<int64_t>(opt_.max_context, (int64_t) window_ + rows_) : opt_.max_context;
    const size_t n_splits_max = (size_t) std::clamp((att_ctx + 127) / 128, (int64_t) 1, (int64_t) 64);
    score_n_ = (size_t) chunk_dec_ * n_head_ * n_splits_max * (head_dim_ + 2);
    if (prompt_mode) score_n_ = std::max(score_n_, (size_t) chunk_batch_ * n_head_ * 1536);
    if (dspark_) {
        const size_t n_splits_dspark = (size_t) std::clamp((opt_.max_context + 127) / 128, 1, 64);
        score_n_ = std::max(score_n_, (size_t) dspark_block_ * dspark_heads_ *
                                          std::max<size_t>(1536, n_splits_dspark * (dspark_head_dim_ + 2)));
    }
    score_ = sbuf(score_n_);

    const size_t ch_base = (prompt_mode && rows_ >= kStageRows) ? (size_t) chunk_batch_ * n_head_ : 0;
    const size_t ch = dspark_ ? std::max(ch_base, (size_t) dspark_block_ * dspark_heads_) : ch_base;
    awork_ = ch > 0 ? sbuf(ch * (head_dim_ + head_dim_ / 2 + 3) + 1) : nullptr;
    if (prompt_mode && rows_ >= kStageRows &&
        (opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kQ4_0 || opt_.kv_format == KvFmt::kK8V4)) {
        work_halves_ = kQ8WorkRows * (size_t) n_kv_ * head_dim_;
        q8w_ = sbuf((2 * work_halves_ + 1) / 2);
    } else {
        work_halves_ = 0;
        q8w_ = nullptr;
    }
    if (zha_on_) zha_scratch();
    if (dspark_) {
        const size_t blk = (size_t) dspark_block_;
        const size_t q_dim = (size_t) dspark_heads_ * dspark_head_dim_;
        const size_t kv_dim = (size_t) dspark_kv_ * dspark_head_dim_;
        const int dv = draft_rows();
        d_taps_ = sbuf((size_t) std::max(n, (size_t) opt_.window) * 3 * n_embd_);
        if (!binding_prompt()) {
            d_f_rnn_ = fbuf((size_t) n_embd_);
            d_blk_x_ = fbuf(blk * n_embd_);
            d_blk_xn_ = fbuf(blk * n_embd_);
            d_blk_h_ = fbuf((blk - 1) * n_embd_);
            d_blk_qkv_ = fbuf(blk * (q_dim + 2 * kv_dim));
            d_blk_qraw_ = fbuf(blk * q_dim);
            d_blk_kraw_ = fbuf(blk * kv_dim);
            d_blk_attn_ = fbuf(blk * n_embd_);
            d_blk_gate_ = fbuf(blk * 2 * dspark_ff_);
            dspark_mask_tiled_ = fbuf((blk - 1) * n_embd_);
            if (dspark_proxy_) {
                d_head_approx_ = fbuf((blk - 1) * (size_t) inkling_dspark_screen_rows(dv));
                d_head_scr_ = (void*) fbuf((inkling_dspark_head_scratch_bytes(dv) + 3) / 4);
                d_rnn_lat_q_ = (void*) fbuf(((blk - 1) * kernels::native_q8_1_bytes(dspark_rank_) + 3) / 4);
                d_w1_row_ = (uint16_t*) fbuf(((size_t) dspark_rank_ * sizeof(uint16_t) + 3) / 4);
            } else {
                d_base_logits_ = fbuf((blk - 1) * (size_t) dv);
                d_markov_bias_ = fbuf((size_t) dv);
            }
            d_rnn_in_ = fbuf((size_t) (2 * dspark_rank_));
            d_rnn_gates_ = fbuf((size_t) (3 * dspark_rank_));
            d_rnn_h_gates_ = fbuf((blk - 1) * (size_t) (3 * dspark_rank_));
            d_rnn_f_gates_ = fbuf((size_t) (3 * dspark_rank_));
            d_rnn_state_ = fbuf((size_t) dspark_rank_);
            d_rnn_latent_ = fbuf((size_t) dspark_rank_);
            d_blk_xq_ = (void*) fbuf((blk * kernels::native_q8_1_bytes(n_embd_) + 3) / 4);
            d_blk_q8ff_ = (void*) fbuf((blk * kernels::native_q8_1_bytes(dspark_ff_) + 3) / 4);
            d_block_picks_ = (Likeliest*) fbuf(2 * blk);
        }
    }
}

size_t Inkling2::card_row_bytes() const {
    size_t bytes = 0;
    if (zha_on_) {
        size_t layers = 0;
        for (const SlotCache& sc : slots_)
            if (!sc.swa) ++layers;
        bytes = layers * 2 * (zha_leaf_codes() + zha_leaf_scales());
    } else {
        const size_t k_code = kv_code_row_bytes(false);
        const size_t v_code = kv_code_row_bytes(true);
        const size_t scale_row = kv_scale_row_bytes();
        size_t layers = 0;
        for (const SlotCache& sc : slots_)
            if (!sc.swa && slot_k_codes(sc) != nullptr) ++layers;
        bytes = layers * (k_code + v_code + 2 * scale_row);
    }
    if (dspark_) {
        const size_t dk_code = dspark_kv_code_row_bytes(false);
        const size_t dv_code = dspark_kv_code_row_bytes(true);
        const size_t d_scale = dspark_kv_scale_row_bytes();
        for (const DsparkLayer& dl : dspark_l_)
            if (!dl.local && dspark_k_codes(dl) != nullptr) bytes += dk_code + dv_code + 2 * d_scale;
    }
    return bytes;
}

void Inkling2::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const bool quant =
        opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kQ4_0 || opt_.kv_format == KvFmt::kK8V4;
    const size_t k_code = kv_code_row_bytes(false);
    const size_t v_code = kv_code_row_bytes(true);
    const size_t scale_row = kv_scale_row_bytes();
    const size_t at_kc = (size_t) first * k_code, at_vc = (size_t) first * v_code, at_s = (size_t) first * scale_row;
    const size_t n_kc = (size_t) n * k_code, n_vc = (size_t) n * v_code, n_s = (size_t) n * scale_row;
    for (size_t i = 0; i < slots_.size() && zha_on_ && n > 0; ++i) {
        const ZSlot& z = zs_[i];
        if (slots_[i].swa || !z.kc) continue;
        if (first + n > z.zleaf_cap)
            throw std::logic_error("save_rows: the rows are outside the ZHA leaf capacity");
        const size_t codes = zha_leaf_codes(), scales = zha_leaf_scales();
        const std::pair<const int8_t*, const uint16_t*> parts[2] = {{z.kc, z.ksc}, {z.vc, z.vsc}};
        for (const auto& [c, sc_ptr] : parts) {
            std::memcpy(dst, c + (size_t) first * codes, (size_t) n * codes);
            dst += (size_t) n * codes;
            std::memcpy(dst, (const uint8_t*) sc_ptr + (size_t) first * scales, (size_t) n * scales);
            dst += (size_t) n * scales;
        }
    }
    for (const SlotCache& sc : slots_) {
        const uint8_t* kptr = slot_k_codes(sc);
        const uint8_t* vptr = slot_v_codes(sc);
        if (sc.swa || n == 0 || kptr == nullptr) continue;
        if (quant) {
            ck(cudaMemcpy(dst, kptr + at_kc, n_kc, cudaMemcpyDeviceToHost), "save rows");
            dst += n_kc;
            ck(cudaMemcpy(dst, (const uint8_t*) sc.k_scales + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
            ck(cudaMemcpy(dst, vptr + at_vc, n_vc, cudaMemcpyDeviceToHost), "save rows");
            dst += n_vc;
            ck(cudaMemcpy(dst, (const uint8_t*) sc.v_scales + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
            continue;
        }
        ck(cudaMemcpy(dst, kptr + at_kc, n_kc, cudaMemcpyDeviceToHost), "save rows");
        dst += n_kc;
        ck(cudaMemcpy(dst, vptr + at_vc, n_vc, cudaMemcpyDeviceToHost), "save rows");
        dst += n_vc;
    }
    if (dspark_ && n > 0) {
        const size_t dk_code = dspark_kv_code_row_bytes(false);
        const size_t dv_code = dspark_kv_code_row_bytes(true);
        const size_t d_scale = dspark_kv_scale_row_bytes();
        const size_t dat_kc = (size_t) first * dk_code, dat_vc = (size_t) first * dv_code, dat_s = (size_t) first * d_scale;
        const size_t dn_kc = (size_t) n * dk_code, dn_vc = (size_t) n * dv_code, dn_s = (size_t) n * d_scale;
        for (const DsparkLayer& dl : dspark_l_) {
            const uint8_t* dkptr = dspark_k_codes(dl);
            const uint8_t* dvptr = dspark_v_codes(dl);
            if (dl.local || dkptr == nullptr) continue;
            if (quant) {
                ck(cudaMemcpy(dst, dkptr + dat_kc, dn_kc, cudaMemcpyDeviceToHost), "save dspark rows");
                dst += dn_kc;
                ck(cudaMemcpy(dst, (const uint8_t*) dl.dk_scales + dat_s, dn_s, cudaMemcpyDeviceToHost), "save dspark rows");
                dst += dn_s;
                ck(cudaMemcpy(dst, dvptr + dat_vc, dn_vc, cudaMemcpyDeviceToHost), "save dspark rows");
                dst += dn_vc;
                ck(cudaMemcpy(dst, (const uint8_t*) dl.dv_scales + dat_s, dn_s, cudaMemcpyDeviceToHost), "save dspark rows");
                dst += dn_s;
            } else {
                ck(cudaMemcpy(dst, dkptr + dat_kc, dn_kc, cudaMemcpyDeviceToHost), "save dspark rows");
                dst += dn_kc;
                ck(cudaMemcpy(dst, dvptr + dat_vc, dn_vc, cudaMemcpyDeviceToHost), "save dspark rows");
                dst += dn_vc;
            }
        }
    }
}

void Inkling2::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    for (ZSlot& z : zs_) z.zpos = -1;
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const bool quant =
        opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kQ4_0 || opt_.kv_format == KvFmt::kK8V4;
    const size_t k_code = kv_code_row_bytes(false);
    const size_t v_code = kv_code_row_bytes(true);
    const size_t scale_row = kv_scale_row_bytes();
    const size_t at_kc = (size_t) first * k_code, at_vc = (size_t) first * v_code, at_s = (size_t) first * scale_row;
    const size_t n_kc = (size_t) n * k_code, n_vc = (size_t) n * v_code, n_s = (size_t) n * scale_row;
    const size_t sk_kc = (size_t) (first - from) * k_code, sk_vc = (size_t) (first - from) * v_code,
                 sk_s = (size_t) (first - from) * scale_row;
    const size_t all_kc = (size_t) count * k_code, all_vc = (size_t) count * v_code, all_s = (size_t) count * scale_row;
    for (size_t i = 0; i < slots_.size() && zha_on_ && n > 0; ++i) {
        ZSlot& z = zs_[i];
        if (slots_[i].swa || !z.kc) continue;
        zha_leaf_room(z, first + n);
        z.zleaf_used = std::max<int64_t>(z.zleaf_used, first + n);
        const size_t codes = zha_leaf_codes(), scales = zha_leaf_scales();
        const std::pair<int8_t*, uint16_t*> parts[2] = {{z.kc, z.ksc}, {z.vc, z.vsc}};
        for (const auto& [c, sc_ptr] : parts) {
            std::memcpy(c + (size_t) first * codes, src + (size_t) (first - from) * codes, (size_t) n * codes);
            src += (size_t) count * codes;
            std::memcpy((uint8_t*) sc_ptr + (size_t) first * scales, src + (size_t) (first - from) * scales,
                        (size_t) n * scales);
            src += (size_t) count * scales;
        }
        z.zbuilt = std::min(z.zbuilt, (int) (first / zha_.C));
    }
    for (const SlotCache& sc : slots_) {
        uint8_t* kptr = slot_k_codes(sc);
        uint8_t* vptr = slot_v_codes(sc);
        if (sc.swa || n == 0 || kptr == nullptr) continue;
        if (quant) {
            ck(cudaMemcpy(kptr + at_kc, src + sk_kc, n_kc, cudaMemcpyHostToDevice), "load rows");
            src += all_kc;
            ck(cudaMemcpy((uint8_t*) sc.k_scales + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
            ck(cudaMemcpy(vptr + at_vc, src + sk_vc, n_vc, cudaMemcpyHostToDevice), "load rows");
            src += all_vc;
            ck(cudaMemcpy((uint8_t*) sc.v_scales + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
            continue;
        }
        ck(cudaMemcpy(kptr + at_kc, src + sk_kc, n_kc, cudaMemcpyHostToDevice), "load rows");
        src += all_kc;
        ck(cudaMemcpy(vptr + at_vc, src + sk_vc, n_vc, cudaMemcpyHostToDevice), "load rows");
        src += all_vc;
    }
    if (dspark_ && n > 0) {
        const size_t dk_code = dspark_kv_code_row_bytes(false);
        const size_t dv_code = dspark_kv_code_row_bytes(true);
        const size_t d_scale = dspark_kv_scale_row_bytes();
        const size_t dat_kc = (size_t) first * dk_code, dat_vc = (size_t) first * dv_code, dat_s = (size_t) first * d_scale;
        const size_t dn_kc = (size_t) n * dk_code, dn_vc = (size_t) n * dv_code, dn_s = (size_t) n * d_scale;
        const size_t dsk_kc = (size_t) (first - from) * dk_code, dsk_vc = (size_t) (first - from) * dv_code,
                     dsk_s = (size_t) (first - from) * d_scale;
        const size_t dall_kc = (size_t) count * dk_code, dall_vc = (size_t) count * dv_code,
                     dall_s = (size_t) count * d_scale;
        for (DsparkLayer& dl : dspark_l_) {
            uint8_t* dkptr = dspark_k_codes(dl);
            uint8_t* dvptr = dspark_v_codes(dl);
            if (dl.local || dkptr == nullptr) continue;
            if (quant) {
                ck(cudaMemcpy(dkptr + dat_kc, src + dsk_kc, dn_kc, cudaMemcpyHostToDevice), "load dspark rows");
                src += dall_kc;
                ck(cudaMemcpy((uint8_t*) dl.dk_scales + dat_s, src + dsk_s, dn_s, cudaMemcpyHostToDevice), "load dspark rows");
                src += dall_s;
                ck(cudaMemcpy(dvptr + dat_vc, src + dsk_vc, dn_vc, cudaMemcpyHostToDevice), "load dspark rows");
                src += dall_vc;
                ck(cudaMemcpy((uint8_t*) dl.dv_scales + dat_s, src + dsk_s, dn_s, cudaMemcpyHostToDevice), "load dspark rows");
                src += dall_s;
            } else {
                ck(cudaMemcpy(dkptr + dat_kc, src + dsk_kc, dn_kc, cudaMemcpyHostToDevice), "load dspark rows");
                src += dall_kc;
                ck(cudaMemcpy(dvptr + dat_vc, src + dsk_vc, dn_vc, cudaMemcpyHostToDevice), "load dspark rows");
                src += dall_vc;
            }
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
    size_t bytes = (size_t) n_swa_slots_ * (size_t) keep_ *
                   (kv_code_row_bytes(false) + kv_code_row_bytes(true) + 2 * kv_scale_row_bytes());
    if (dspark_) {
        const size_t kv_row = (size_t) dspark_kv_ * dspark_head_dim_ * sizeof(uint16_t);
        size_t n_local = 0;
        for (const DsparkLayer& dl : dspark_l_)
            if (dl.local) ++n_local;
        bytes += (size_t) n_embd_ * sizeof(float) + n_local * 2 * (size_t) dspark_window_ * kv_row;
    }
    return bytes;
}

void Inkling2::save_layer(int slot, int64_t pos, uint8_t* dst) {
    const SlotCache& sc = slots_[(size_t) slot];
    ck(cudaMemcpy(dst + sc.state_off * sizeof(float), state_ + sc.state_off,
                  slot_conv_floats_ * sizeof(float), cudaMemcpyDeviceToHost),
       "save state");
    if (!sc.swa) return;
    const bool quant =
        opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kQ4_0 || opt_.kv_format == KvFmt::kK8V4;
    const size_t k_code = kv_code_row_bytes(false);
    const size_t v_code = kv_code_row_bytes(true);
    const size_t scale_row = kv_scale_row_bytes();
    const size_t whole = (size_t) keep_ * (k_code + v_code + 2 * scale_row);
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
        save_ring(kptr, k_code);
        save_ring((const uint8_t*) sc.k_scales, scale_row);
        save_ring(vptr, v_code);
        save_ring((const uint8_t*) sc.v_scales, scale_row);
        return;
    }
    save_ring(kptr, k_code);
    save_ring(vptr, v_code);
}

void Inkling2::load_extra(const uint8_t* src, int64_t pos) {
    for (ZSlot& z : zs_) z.zpos = -1;
    const bool quant =
        opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kQ4_0 || opt_.kv_format == KvFmt::kK8V4;
    const size_t k_code = kv_code_row_bytes(false);
    const size_t v_code = kv_code_row_bytes(true);
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
            load_ring(kptr, k_code);
            load_ring((uint8_t*) sc.k_scales, scale_row);
            load_ring(vptr, v_code);
            load_ring((uint8_t*) sc.v_scales, scale_row);
            continue;
        }
        load_ring(kptr, k_code);
        load_ring(vptr, v_code);
    }
    if (!dspark_) return;
    const size_t f_bytes = (size_t) n_embd_ * sizeof(float);
    ck(cudaMemcpyAsync(d_f_rnn_, src, f_bytes, cudaMemcpyHostToDevice, cs(st_)), "load dspark f_rnn");
    src += f_bytes;
    lin(dspark_rnn_f_, d_f_rnn_, d_rnn_f_gates_, 1);
    const size_t kv_row = (size_t) dspark_kv_ * dspark_head_dim_ * sizeof(uint16_t);
    const size_t keep_d = (size_t) dspark_window_;
    const int64_t lo = std::max<int64_t>(0, pos - dspark_window_);
    for (const DsparkLayer& dl : dspark_l_) {
        if (!dl.local) continue;
        auto load_d_ring = [&](uint16_t* cache) {
            for (int64_t p = lo; p < pos;) {
                const int64_t at = p % dspark_ring_, m = std::min<int64_t>(pos - p, dspark_ring_ - at);
                const size_t s_at = (size_t) (p - (pos - dspark_window_));
                ck(cudaMemcpyAsync((uint8_t*) cache + (size_t) at * kv_row, src + s_at * kv_row, (size_t) m * kv_row,
                                   cudaMemcpyHostToDevice, cs(st_)),
                   "load dspark kv");
                p += m;
            }
            src += keep_d * kv_row;
        };
        load_d_ring(dl.dk);
        load_d_ring(dl.dv);
    }
    ck(cudaStreamSynchronize(cs(st_)), "load dspark extra");
}

int Inkling2::head_key(int slot, int n) const {
    return slots_[(size_t) slot].swa ? 0 : room_key(n);
}

void Inkling2::apply_ctrl(const float* cur, const float* msg, const Block& B, int ctrl, float* out, int n,
                          const float* ctrl_r_in) {
    const float* r_ptr = ctrl_r_in;
    if (!r_ptr) {
        bool fused = false;
        if (n < kStageRows && B.ctrl_down[ctrl].type == kTypeF32 && B.ctrl_down[ctrl].scale == 1.0f) {
            const float* c = cur;
            const float* m = msg;
            const float* w = fp(B.ctrl_down[ctrl]);
            float* y = ctrl_r_;
            fused = ink_ctrl_down(&c, &m, &w, &y, 1, n_embd_, gate_rank_, n, st_);
        }
        if (!fused) {
            inkling_control_prep(cur, msg, ctrl_cat_, n_embd_, n, 1e-6f, st_);
            lin(B.ctrl_down[ctrl], ctrl_cat_, ctrl_r_, n);
        }
        r_ptr = ctrl_r_;
    }
    if (B.ctrl_gf[ctrl].p) {
        if (n < kStageRows && B.ctrl_gf[ctrl].type == kTypeF32 && B.ctrl_gf[ctrl].scale == 1.0f) {
            const float* gf = fp(B.ctrl_gf[ctrl]);
            if (ink_control3_gate(&cur, &msg, &r_ptr, &gf, &out, 1, n_embd_, gate_rank_, n, st_)) return;
        }
        lin(B.ctrl_gf[ctrl], r_ptr, ctrl_g_, n);
        inkling_control_apply(cur, msg, ctrl_g_, ctrl_g_ + n_embd_, out, n_embd_, n, st_, 2 * n_embd_);
        return;
    }
    lin(B.ctrl_gain[ctrl], r_ptr, ctrl_g_, n);
    lin(B.ctrl_forget[ctrl], r_ptr, ctrl_f_, n);
    inkling_control_apply(cur, msg, ctrl_g_, ctrl_f_, out, n_embd_, n, st_);
}

void Inkling2::store(const float* x, const SlotCache& sc, int64_t pos, int n, bool value) {
    const int width = n_kv_ * head_dim_;
    const bool use_q4 = opt_.kv_format == KvFmt::kQ4_0 || (value && opt_.kv_format == KvFmt::kK8V4);
    const bool use_q8 = opt_.kv_format == KvFmt::kQ8_0 || (!value && opt_.kv_format == KvFmt::kK8V4);
    if (use_q4) {
        uint8_t* codes = value ? sc.v_q4 : sc.k_q4;
        uint16_t* scales = value ? sc.v_scales : sc.k_scales;
        if (recording_) return store_q4_at(x, codes, scales, width, at_, n, sc.swa ? ring_ : 0, st_);
        if (!sc.swa) return store_q4(x, codes, scales, width, pos, n, st_);
        const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
        store_q4(x, codes, scales, width, at, head, st_);
        if (head < n) store_q4(x + (size_t) head * width, codes, scales, width, 0, n - head, st_);
        return;
    }
    if (use_q8) {
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

void Inkling2::attn_part(const Step& st, Block& B, SlotCache& sc, int64_t pos, int n, bool fuse_post, int part) {
    const int ip = st.phys, pass = st.pass;
    const long long* at = recording_ ? at_ : nullptr;
    const int threads = sum_threads(n_embd_);
    const Layer& L = layers_[(size_t) ip];
    float* in0 = inputs_buf_;
    float* in1 = inputs_buf_ + (size_t) n * n_embd_;
    float* in2 = inputs_buf_ + 2 * (size_t) n * n_embd_;

    const int ns_all = streams();
    const bool share_swa = B.swa && pass == 0;
    const float *qraw_p = qraw_, *gate_p = gate_, *r_raw_p = r_raw_, *sinks_p = sinks_;
    if (n == 1 && B.wq_fused.p) {
        qraw_p = q_fused_buf_;
        gate_p = q_fused_buf_ + (size_t) n_head_ * head_dim_;
        if (B.wq_fuse_flags & 2) r_raw_p = q_fused_buf_ + B.wq_rel_off;
        if (B.wq_fuse_flags & 4) sinks_p = q_fused_buf_ + B.wq_sink_off;
    }

    if (part & 1) {
    bool normed = false;
    // 1. Attention Residual mix over blocks so far + current stream 0
    if (ip == 0) {
        inkling_read_stream(r_, ns_all, 0, in0, n_embd_, n, st_);
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
        if (share_swa) {
            normed = inkling_mix_norm_q8(r_, ns_all, ms, ns, fp(B.attn_res_query), fp(L.attn_norm), xn_,
                                         n < kStageRows ? xq_ : nullptr, n_embd_, n, 1e-6f, rms_eps_, st_);
        }
        if (!normed) {
            if (B.swa) {
                inkling_mix(r_, ns_all, ms, ns, fp(B.attn_res_query), 1, in0, n_embd_, n, 1e-6f, st_);
            } else {
                inkling_mix(r_, ns_all, ms, ns, fp(B.attn_res_query), 3, in0, n_embd_, n, 1e-6f, st_);
            }
        }
    }

    // 2. Cross-pass loop controls (pass > 0)
    if (pass > 0) {
        const float* c3[3] = {B.swa ? in0 : in1, B.swa ? in0 : in2, in0};
        const float* m3[3] = {msg_ + 1 * (size_t) n * n_embd_, msg_ + 2 * (size_t) n * n_embd_, msg_};
        const float* w3[3] = {
            B.ctrl_down[1].type == kTypeF32 && B.ctrl_down[1].scale == 1.0f ? fp(B.ctrl_down[1]) : nullptr,
            B.ctrl_down[2].type == kTypeF32 && B.ctrl_down[2].scale == 1.0f ? fp(B.ctrl_down[2]) : nullptr,
            B.ctrl_down[0].type == kTypeF32 && B.ctrl_down[0].scale == 1.0f ? fp(B.ctrl_down[0]) : nullptr,
        };
        float* y3[3] = {ctrl_r_, ctrl_r_ + (size_t) n * gate_rank_, ctrl_r_ + 2 * (size_t) n * gate_rank_};
        float* o3[3] = {in1, in2, in0};
        const int idx3[3] = {1, 2, 0};
        const bool all_fused =
            n < kStageRows && w3[0] && w3[1] && w3[2] && ink_ctrl_down(c3, m3, w3, y3, 3, n_embd_, gate_rank_, n, st_);
        const float* gf3[3] = {
            B.ctrl_gf[1].p && B.ctrl_gf[1].type == kTypeF32 && B.ctrl_gf[1].scale == 1.0f ? fp(B.ctrl_gf[1]) : nullptr,
            B.ctrl_gf[2].p && B.ctrl_gf[2].type == kTypeF32 && B.ctrl_gf[2].scale == 1.0f ? fp(B.ctrl_gf[2]) : nullptr,
            B.ctrl_gf[0].p && B.ctrl_gf[0].type == kTypeF32 && B.ctrl_gf[0].scale == 1.0f ? fp(B.ctrl_gf[0]) : nullptr,
        };
        const float* y3_c[3] = {y3[0], y3[1], y3[2]};
        if (all_fused && gf3[0] && gf3[1] && gf3[2] &&
            ink_control3_gate(c3, m3, y3_c, gf3, o3, 3, n_embd_, gate_rank_, n, st_)) {
            // Fused 3-controller up-projection + gate applied directly into o3
        } else {
            for (int k = 0; k < 3; ++k) apply_ctrl(c3[k], m3[k], B, idx3[k], o3[k], n, all_fused ? y3[k] : nullptr);
        }
    }

    // 3. Input norm + linear projections -> Q, K, V, R, sinks, gate
    const Dev& nq = B.swa ? L.attn_norm : B.q_inp_norm;
    const Dev& nk = B.swa ? L.attn_norm : B.k_inp_norm;
    const Dev& nv = B.swa ? L.attn_norm : B.v_inp_norm;

    if (!normed) {
        kernels::native_add_rms_norm_q8_1(in0, nullptr, fp(nq), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    }

    if (n < kStageRows && B.wq_fused.p) {
        lin(B.wq_fused, xn_, q_fused_buf_, n, xq_);
        if (n == 1) {
            if (!(B.wq_fuse_flags & 2)) lin(B.attn_rel, xn_, r_raw_, 1, xq_);
            if (!(B.wq_fuse_flags & 4)) lin(B.attn_sink_proj, xn_, sinks_, 1, xq_);
        } else {
            inkling_unpack_q_fused(q_fused_buf_, qraw_, gate_, (B.wq_fuse_flags & 2) ? r_raw_ : nullptr,
                                   (B.wq_fuse_flags & 4) ? sinks_ : nullptr, n_head_ * head_dim_, n_head_ * n_rel_,
                                   n_head_, B.wq_rel_off, B.wq_sink_off, B.wq_fused.n_out, n, st_);
            if (!(B.wq_fuse_flags & 2)) lin(B.attn_rel, xn_, r_raw_, n, xq_);
            if (!(B.wq_fuse_flags & 4)) lin(B.attn_sink_proj, xn_, sinks_, n, xq_);
        }
    } else {
        lin(B.wq, xn_, qraw_, n, xq_);
        lin(B.attn_rel, xn_, r_raw_, n, xq_);
        lin(B.attn_sink_proj, xn_, sinks_, n, xq_);
        lin(B.wqkv_gate, xn_, gate_, n, xq_);
    }

    if (share_swa && n < kStageRows && B.wkv_fused.p) {
        lin(B.wkv_fused, xn_, kv_fused_buf_, n, xq_);
        InkConv ck, cv;
        ck.x = kv_fused_buf_;
        ck.x_stride = 2 * n_kv_ * head_dim_;
        ck.state = state_ + sc.state_off + off_k_conv_;
        ck.w = fp(B.attn_k_conv);
        ck.dyn = nullptr;
        ck.out = k_;
        ck.snap = snap_bits_ ? snap_at_ + sc.state_off + off_k_conv_ : nullptr;

        cv.x = kv_fused_buf_ + (size_t) n_kv_ * head_dim_;
        cv.x_stride = 2 * n_kv_ * head_dim_;
        cv.state = state_ + sc.state_off + off_v_conv_;
        cv.w = fp(B.attn_v_conv);
        cv.dyn = nullptr;
        cv.out = v_;
        cv.snap = snap_bits_ ? snap_at_ + sc.state_off + off_v_conv_ : nullptr;

        ink_conv_few(ck, cv, n_kv_ * head_dim_, head_dim_, n, st_, nullptr, rec_n_, snap_bits_);
    } else {
        if (!share_swa) {
            kernels::native_add_rms_norm_q8_1(in1, nullptr, fp(nk), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
        }
        lin(B.wk, xn_, kraw_, n, xq_);
        if (!B.swa) lin(B.k_conv_proj, xn_, dyn_, n, xq_);
        inkling_conv4(kraw_, state_ + sc.state_off + off_k_conv_, fp(B.attn_k_conv), B.swa ? nullptr : dyn_, head_dim_,
                      k_, n_kv_ * head_dim_, n, st_, snap_bits_ ? snap_at_ + sc.state_off + off_k_conv_ : nullptr,
                      rec_n_, snap_bits_);

        if (!share_swa) {
            kernels::native_add_rms_norm_q8_1(in2, nullptr, fp(nv), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
        }
        lin(B.wv, xn_, kraw_, n, xq_);
        if (!B.swa) lin(B.v_conv_proj, xn_, dyn_, n, xq_);
        inkling_conv4(kraw_, state_ + sc.state_off + off_v_conv_, fp(B.attn_v_conv), B.swa ? nullptr : dyn_, head_dim_,
                      v_, n_kv_ * head_dim_, n, st_, snap_bits_ ? snap_at_ + sc.state_off + off_v_conv_ : nullptr,
                      rec_n_, snap_bits_);
    }

    // 4-5. L2 clip + global tau(pos) on Q & R, L2 clip on K + store K, RMSNorm on V + store V
    if (!zha_full(B) && head_dim_ == 128 && n_rel_ <= 32) {
        InkClipsStore cs;
        cs.q_in = qraw_p;
        cs.q_scale = fp(B.attn_q_clip);
        cs.q_out = q_;
        cs.k_in = k_;
        cs.v_in = v_;
        cs.r_in = r_raw_p;
        cs.r_scale = fp(B.attn_rel_clip);
        cs.r_out = r_cur_;
        if (opt_.kv_format == KvFmt::kQ4_0) {
            cs.k_q4 = sc.k_q4;
            cs.k_scales = sc.k_scales;
            cs.v_q4 = sc.v_q4;
            cs.v_scales = sc.v_scales;
        } else if (opt_.kv_format == KvFmt::kK8V4) {
            cs.k_q8 = sc.k_codes;
            cs.k_scales = sc.k_scales;
            cs.v_q4 = sc.v_q4;
            cs.v_scales = sc.v_scales;
        } else if (opt_.kv_format == KvFmt::kQ8_0) {
            cs.k_q8 = sc.k_codes;
            cs.k_scales = sc.k_scales;
            cs.v_q8 = sc.v_codes;
            cs.v_scales = sc.v_scales;
        } else {
            cs.k_fp16 = sc.keys;
            cs.v_fp16 = sc.vals;
        }
        cs.n_head = n_head_;
        cs.n_kv = n_kv_;
        cs.dim = head_dim_;
        cs.n_rel = n_rel_;
        cs.rows = n;
        cs.ring = sc.swa ? ring_ : 0;
        cs.pos = pos;
        cs.at = at;
        cs.qk_clip = qk_clip_;
        cs.rel_clip = rel_clip_;
        cs.rms_eps = rms_eps_;
        cs.is_global = !B.swa;
        cs.log_floor = (float) log_floor_;
        cs.log_alpha = log_alpha_;
        inkling_clips_store(cs, st_);
    } else {
        inkling_l2_clip(qraw_p, fp(B.attn_q_clip), q_, head_dim_, n_head_, qk_clip_, pos, !B.swa, (float) log_floor_,
                        log_alpha_, n, at, st_);
        inkling_l2_clip(k_, nullptr, k_, head_dim_, n_kv_, qk_clip_, pos, false, (float) log_floor_, 0.f, n, nullptr,
                        st_);
        rms_norm(v_, nullptr, v_, head_dim_, n * n_kv_, rms_eps_, st_);
        inkling_l2_clip(r_raw_p, fp(B.attn_rel_clip), r_cur_, n_rel_, n_head_, rel_clip_, pos, !B.swa,
                        (float) log_floor_, log_alpha_, n, at, st_);

        // 5. Store K and V into slot cache
        if (!zha_full(B)) {
            store(k_, sc, pos, n, false);
            store(v_, sc, pos, n, true);
        }
    }
    }   // part & 1
    if (!(part & 2)) return;

    // 6. Relative-bias attention + virtual zero sink
    bool quant_wo = false;
    if (part == 2) {
        mul_sigmoid(att_, gate_p, n_head_ * head_dim_, n, n_head_ * head_dim_, false, st_);
    } else if (zha_full(B)) {
        zha_attention(st.slot, n, pos, sinks_p);
        mul_sigmoid(att_, gate_p, n_head_ * head_dim_, n, n_head_ * head_dim_, false, st_);
    } else {
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    const int n_extent = B.swa ? window_ : n_rel_extent_;
    const bool k_q8 = opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kK8V4;
    const bool k_q4 = opt_.kv_format == KvFmt::kQ4_0;
    const bool v_q8 = opt_.kv_format == KvFmt::kQ8_0;
    const bool v_q4 = opt_.kv_format == KvFmt::kQ4_0 || opt_.kv_format == KvFmt::kK8V4;
    const bool quant = k_q8 || k_q4 || v_q8 || v_q4;
    const Kvq8 kq = k_q8 ? Kvq8{sc.k_codes, sc.k_scales} : Kvq8{};
    const Kvq8 vq = v_q8 ? Kvq8{sc.v_codes, sc.v_scales} : Kvq8{};
    const Kvq4 kq4 = k_q4 ? Kvq4{sc.k_q4, sc.k_scales} : Kvq4{};
    const Kvq4 vq4 = v_q4 ? Kvq4{sc.v_q4, sc.v_scales} : Kvq4{};
    const int kv_width = n_kv_ * head_dim_;
    const int work_rows = kv_width > 0 ? (int) (work_halves_ / (size_t) kv_width) : 0;
    const int64_t lo_all = (window > 0 && pos >= window) ? (pos - window + 1) : 0;
    const int64_t keys_all_64 = (pos + n) - lo_all;
    const bool pre_dq = n >= kStageRows && quant && keys_all_64 > 0 && keys_all_64 <= work_rows;
    uint16_t* kw = quant ? (uint16_t*) q8w_ : nullptr;
    uint16_t* vw = quant ? kw + work_halves_ : nullptr;
    bool flash_done = false;
    if (n >= kStageRows) {
        InkAttn ia;
        ia.q = q_;
        ia.r = r_cur_;
        ia.sink = sinks_p;
        ia.keys = quant ? nullptr : sc.keys;
        ia.vals = quant ? nullptr : sc.vals;
        ia.rel_bias = fp(B.attn_rel_bias);
        ia.gate = gate_p;
        ia.out = att_;
        ia.pos = pos;
        ia.rows = n;
        ia.n_head = n_head_;
        ia.n_kv = n_kv_;
        ia.dim = head_dim_;
        ia.n_rel = n_rel_;
        ia.extent = n_extent;
        ia.window = window;
        ia.ring = ring;
        ia.scale = scale;
        ia.kq8 = kq;
        ia.vq8 = vq;
        ia.kq4 = kq4;
        ia.vq4 = vq4;
        flash_done = ink_attend_batch(ia, st_);
    }
    if (!flash_done) {
        if (pre_dq) {
            inkling_dequant_kv_span(lo_all, (int) keys_all_64, kv_width, ring, kq, vq, kq4, vq4, kw, vw, st_);
        }
        const bool direct_dec = n < kStageRows && head_dim_ == 128 && n_head_ == n_kv_ * 3 && n_rel_ == 16;
        quant_wo = direct_dec && n <= chunk_dec_ && B.wo.type != kTypeF32 && B.wo.type != kTypeBF16 && B.wo.scale == 1.0f;
        const int c_step = n >= kStageRows ? chunk_batch_ : chunk_dec_;
        for (int first = 0; first < n; first += c_step) {
            const int c = std::min(c_step, n - first);
            const float* r_sub = r_cur_ + (size_t) first * n_head_ * n_rel_;
            const float* s_sub = sinks_p + (size_t) first * n_head_;
            const float* q_sub = q_ + (size_t) first * n_head_ * head_dim_;
            float* att_sub = att_ + (size_t) first * n_head_ * head_dim_;
            if (!direct_dec) {
                inkling_rel_table(r_sub, fp(B.attn_rel_bias_trans.p ? B.attn_rel_bias_trans : B.attn_rel_bias), s_sub,
                                  table_, n_rel_, n_extent, n_head_, c, st_, fast_rel_bias_);
            }
            if (n >= kStageRows) {
                if (pre_dq) {
                    inkling_attend_batch(blas_, q_sub, kw, vw, table_, s_sub, n_extent, (pos - lo_all) + first, c,
                                         score_, score_n_, awork_, att_sub, head_dim_, n_head_, n_kv_, scale, st_,
                                         window, 0, Kvq8{}, Kvq8{}, nullptr, nullptr, 0, Kvq4{}, Kvq4{});
                } else {
                    inkling_attend_batch(blas_, q_sub, quant ? nullptr : sc.keys, quant ? nullptr : sc.vals, table_,
                                         s_sub, n_extent, pos + first, c, score_, score_n_, awork_, att_sub, head_dim_,
                                         n_head_, n_kv_, scale, st_, window, ring, kq, vq, kw, vw,
                                         quant ? work_halves_ : 0, kq4, vq4);
                }
            } else {
                const long long* at_sub = at ? at + 2 * first : nullptr;
                const int rm = !at ? 0 : B.swa ? std::min(window_ + n - 1, opt_.max_context) : room(n);
                if (direct_dec) {
                    const float* gate_sub = gate_p + (size_t) first * n_head_ * head_dim_;
                    inkling_attend(q_sub, quant ? nullptr : sc.keys, quant ? nullptr : sc.vals, nullptr, s_sub,
                                   n_extent, pos + first, c, score_, chunk_dec_, quant_wo ? nullptr : att_sub,
                                   head_dim_, n_head_, n_kv_, scale, st_, window, ring, at_sub, rm, kq, vq, kq4, vq4,
                                   r_sub, fp(B.attn_rel_bias), n_rel_, gate_sub, quant_wo ? q8_ : nullptr);
                } else {
                    inkling_attend(q_sub, quant ? nullptr : sc.keys, quant ? nullptr : sc.vals, table_, s_sub, n_extent,
                                   pos + first, c, score_, chunk_dec_, att_sub, head_dim_, n_head_, n_kv_, scale, st_,
                                   window, ring, at_sub, rm, kq, vq, kq4, vq4);
                }
            }
        }
        // 7. Sigmoid gate -> wo -> post-conv4
        if (!direct_dec) mul_sigmoid(att_, gate_p, n_head_ * head_dim_, n, n_head_ * head_dim_, false, st_);
    }
    }

    lin(B.wo, att_, cur_, n, quant_wo ? q8_ : nullptr);
    if (fuse_post && n < 16) {
        const int pre_copy = (ip == 12) ? 3 : (ip == core_end_ ? 4 : -1);
        inkling_conv4_stream(cur_, state_ + sc.state_off + off_attn_post_, fp(B.attn_post_conv), r_, ns_all, n_embd_,
                             n, pre_copy, ip % block_size_ == 0, 0, st_,
                             snap_bits_ ? snap_at_ + sc.state_off + off_attn_post_ : nullptr, rec_n_, snap_bits_);
    } else {
        inkling_conv4(cur_, state_ + sc.state_off + off_attn_post_, fp(B.attn_post_conv), nullptr, 0, ao_, n_embd_, n,
                      st_, snap_bits_ ? snap_at_ + sc.state_off + off_attn_post_ : nullptr, rec_n_, snap_bits_);
    }
}

void Inkling2::layer_head(Layer& L, int slot, int n) {
    const Step& st = schedule_[(size_t) slot];
    Block& B = blocks_[(size_t) st.phys];
    const bool use_zgraph =
        zha_full(B) && n >= 1 && n <= 4 && !rows_of_ && opt_.graphs && !opt_.dump && !recording_ && !prompt_lent_ &&
        !in_read_ && !image_at(pos_) &&
        (n == 1 ? snap_bits_ == 0 : (windowed_ && snap_bits_ == snap_tokens(n) && snap_at_ == snap_));
    if (use_zgraph) {
        if (zgraph_.size() != schedule_.size() * 8) zgraph_.assign(schedule_.size() * 8, nullptr);
        const size_t at = ((size_t) slot * 4 + (size_t) (n - 1)) * 2;
        if (!zgraph_[at]) zgraph_[at] = record_fn([&] { head_part(L, slot, n, 1); });
        ck(cudaGraphLaunch((cudaGraphExec_t) zgraph_[at], cs(st_)), "zha head part 1");
        const float* sinks_p = (n == 1 && B.wq_fused.p && (B.wq_fuse_flags & 4)) ? (q_fused_buf_ + B.wq_sink_off) : sinks_;
        zha_attention(slot, n, pos_, sinks_p);
        if (!zgraph_[at + 1]) zgraph_[at + 1] = record_fn([&] { head_part(L, slot, n, 2); });
        ck(cudaGraphLaunch((cudaGraphExec_t) zgraph_[at + 1], cs(st_)), "zha head part 2");
        return;
    }
    head_part(L, slot, n, 3);
}

void Inkling2::head_part(Layer& L, int slot, int n, int part) {
    const int ns_all = streams();
    if ((part & 1) && dspark_) {
        for (int t = 0; t < 3; ++t)
            if (slot == dspark_taps_[t]) inkling_copy_stream(r_, ns_all, 6 + t, 0, n_embd_, n, st_);
    }
    const Step& st = schedule_[(size_t) slot];
    const int ip = st.phys, pass = st.pass;
    Block& B = blocks_[(size_t) ip];
    SlotCache& sc = slots_[(size_t) slot];

    if ((part & 1) && pass > 0) {
        const MixSlots bank_slots{{5, 4, 0, 0, 0, 0}};
        inkling_mix(r_, ns_all, bank_slots, 2, fp(B.loop_msg_query), 4, msg_, n_embd_, n, 1e-6f, st_);
    }

    const bool fuse_post = !opt_.dump && n < 16;
    attn_part(st, B, sc, pos_, n, fuse_post, part);
    if (!(part & 2)) return;
    dbg("attn", slot, ao_, n_embd_);

    // Block AttnRes update: at ip % block_size_ == 0 push inpSA (stream 0) and replace stream 0 = ao_ (no add).
    if (!fuse_post) {
        if (ip == 12) inkling_copy_stream(r_, ns_all, 3, 0, n_embd_, n, st_);
        else if (ip == core_end_) inkling_copy_stream(r_, ns_all, 4, 0, n_embd_, n, st_);

        if (ip % block_size_ == 0) {
            inkling_write_stream(ao_, r_, ns_all, 0, n_embd_, n, st_);
        } else {
            inkling_add_to_stream(nullptr, ao_, r_, ns_all, 0, n_embd_, n, st_);
        }
    }

    // FFN Block AttnRes mix + optional loop control + ffn_norm -> xn_, xq_
    MixSlots ffn_slots{};
    int ffn_n = 2;
    if (ip < core_start_) { ffn_slots = {{0, 1, 0, 0, 0, 0}}; ffn_n = 2; }
    else if (ip < 12) { ffn_slots = {{0, 1, 2, 0, 0, 0}}; ffn_n = 3; }
    else if (ip < core_end_) { ffn_slots = {{0, 1, 2, 3, 0, 0}}; ffn_n = 4; }
    else { ffn_slots = {{0, 1, 2, 3, 4, 0}}; ffn_n = 5; }

    const int threads = sum_threads(n_embd_);
    const bool ffn_normed =
        pass == 0 && inkling_mix_norm_q8(r_, ns_all, ffn_slots, ffn_n, fp(B.ffn_res_query), fp(L.ffn_norm), xn_,
                                         n < kStageRows ? xq_ : nullptr, n_embd_, n, 1e-6f, rms_eps_, st_);
    if (!ffn_normed) {
        inkling_mix(r_, ns_all, ffn_slots, ffn_n, fp(B.ffn_res_query), 1, cur_, n_embd_, n, 1e-6f, st_);
        if (pass > 0) apply_ctrl(cur_, msg_ + 3 * (size_t) n * n_embd_, B, 3, cur_, n);
        kernels::native_add_rms_norm_q8_1(cur_, nullptr, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    }
    dbg("ffn_in", slot, xn_, n_embd_);
    if (!batched(L, n)) moe_router(L, n);
}

void Inkling2::layer_tail(Layer&, int slot, int n) {
    const Step& st = schedule_[(size_t) slot];
    const int ip = st.phys, pass = st.pass;
    Block& B = blocks_[(size_t) ip];
    SlotCache& sc = slots_[(size_t) slot];

    const int ns_all = streams();
    if (n < 16) {
        const int post_op = (ip == core_start_ - 1) ? 1 : ((ip == core_end_ - 1 && pass < n_passes_ - 1) ? 2 : 0);
        inkling_conv4_stream(fo_, state_ + sc.state_off + off_ffn_post_, fp(B.ffn_post_conv), r_, ns_all, n_embd_, n,
                             -1, false, post_op, st_, snap_bits_ ? snap_at_ + sc.state_off + off_ffn_post_ : nullptr,
                             rec_n_, snap_bits_);
        return;
    }
    inkling_conv4(fo_, state_ + sc.state_off + off_ffn_post_, fp(B.ffn_post_conv), nullptr, 0, cur_, n_embd_, n, st_,
                  snap_bits_ ? snap_at_ + sc.state_off + off_ffn_post_ : nullptr, rec_n_, snap_bits_);
    inkling_add_to_stream(nullptr, cur_, r_, ns_all, 0, n_embd_, n, st_);

    if (ip == core_start_ - 1) {
        inkling_copy_stream(r_, ns_all, 2, 0, n_embd_, n, st_);
    } else if (ip == core_end_ - 1 && pass < n_passes_ - 1) {
        inkling_copy_stream(r_, ns_all, 4, 3, n_embd_, n, st_);
        inkling_copy_stream(r_, ns_all, 5, 0, n_embd_, n, st_);
        inkling_copy_stream(r_, ns_all, 0, 2, n_embd_, n, st_);
    }
}

void Inkling2::output_rows(int first, int rows, float* h) {
    const int ns_all = streams();
    const MixSlots out_slots{{0, 1, 2, 3, 4, 0}};
    if (!inkling_mix_norm_q8(r_ + (size_t) first * ns_all * n_embd_, ns_all, out_slots, 5, fp(output_res_query_),
                             fp(out_norm_), h, nullptr, n_embd_, rows, 1e-6f, rms_eps_, st_)) {
        inkling_mix(r_ + (size_t) first * ns_all * n_embd_, ns_all, out_slots, 5, fp(output_res_query_), 1, cur_,
                    n_embd_, rows, 1e-6f, st_);
        rms_norm(cur_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
    }
    if (dspark_) inkling_dspark_extract_taps(r_ + (size_t) first * ns_all * n_embd_, ns_all, 6, d_taps_, n_embd_, rows, st_);
}

void Inkling2::finish_logits(float* logits, int rows) {
    if (final_logit_softcapping_ > 0.f) softcap(logits, (size_t) rows * n_vocab_, final_logit_softcapping_, st_);
}

void Inkling2::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    route_by(logits, blocks_[(size_t) il].probs_b.data(), ids, w, probs);
}

// --cache-bias: F onto the pick of each expert in VRAM.  The pick is the sigmoid gate plus the layer's bias, so the
// experts in VRAM keep their order among themselves, as do the others, and only a close call between the two goes the
// other way.  The weights stay the bare gates'.
void Inkling2::cached_bias(const Layer& L, int il, float* b) const {
    const std::vector<float>& pb = blocks_[(size_t) il].probs_b;
    for (int e = 0; e < n_expert_; ++e) b[e] = pb[(size_t) e] + (L.slot[(size_t) e] ? opt_.cache_bias : 0.f);
}

void Inkling2::route_cached(const Layer& L, int il, const float* logits, int32_t, int* ids, float* w, float* probs) {
    cached_bias(L, il, bias_rl_.data());
    route_by(logits, bias_rl_.data(), ids, w, probs);
}

// route_card()'s bias for a few tokens, every layer's, from the experts card_moe() has just put on the card.
void Inkling2::slots_moved() {
    if (!(opt_.cache_bias > 0.f)) return;
    const size_t ne = (size_t) n_expert_;
    std::vector<float> b((size_t) n_layer_ * ne, 0.f);
    for (int il = 0; il < n_layer_; ++il)
        if (layers_[(size_t) il].moe) cached_bias(layers_[(size_t) il], il, b.data() + (size_t) il * ne);
    if (!cache_b_) cache_b_ = (float*) dmalloc(b.size() * sizeof(float), "scratch");
    ck(cudaMemcpy(cache_b_, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice), "cache bias");
}

void Inkling2::route_by(const float* logits, const float* bias, int* ids, float* w, float* probs) const {
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[e];
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
    const float* bias = B.probs_b_dev.p ? fp(B.probs_b_dev) : nullptr;
    if (cache_b_ && n < kStageRows) bias = cache_b_ + (size_t) il * n_expert_;
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, bias, nullptr, true, w_scale_, ids, w, st_);
    return true;
}

void Inkling2::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
}

void Inkling2::load_drafter() {
    if (!dspark_) return;
    auto dtensor = [&](const std::string& name) -> const TensorInfo* {
        const TensorInfo* t = dg_->find(name);
        if (!t) throw std::runtime_error(opt_.draft_model + ": missing DSpark tensor " + name);
        return t;
    };
    auto dmat = [&](const std::string& name, int64_t a, int64_t b) -> Dev {
        return upload(dtensor(name), name, a, b, kGuessMat);
    };
    auto dvec = [&](const std::string& name, int64_t a) -> Dev {
        return upload(dtensor(name), name, a, 1, kF32);
    };
    auto reserve_dense = [&](size_t bytes, uint32_t type, int n_in, int n_out) -> Dev {
        const size_t aligned = pad(bytes);
        const Dev d{dense_ ? dense_ + dense_at_ : nullptr, type, n_in, n_out, 1.f};
        dense_at_ += aligned + kTail;
        if (dense_ && dense_at_ > dense_bytes_)
            throw std::logic_error("the dense arena is too small for dspark weights");
        return d;
    };
    auto dmat_cat = [&](std::initializer_list<std::pair<std::string, int64_t>> parts, int64_t n_in, Dev& fused,
                        std::initializer_list<Dev*> views) {
        int64_t total_out = 0;
        for (const auto& kv : parts) total_out += kv.second;
        const size_t q_row = ggml_row_size(GGML_TYPE_Q4_0, n_in);
        const size_t total_bytes = q_row * (size_t) total_out;
        fused = reserve_dense(total_bytes, (uint32_t) GGML_TYPE_Q4_0, (int) n_in, (int) total_out);

        std::vector<uint8_t> q(dense_ ? total_bytes : 0);
        size_t off_rows = 0;
        auto p_it = parts.begin();
        auto v_it = views.begin();
        for (; p_it != parts.end(); ++p_it, ++v_it) {
            const TensorInfo* t = dtensor(p_it->first);
            const int64_t n_out = p_it->second;
            const uint64_t rest = t->elements() / std::max<uint64_t>(t->shape[0], 1);
            if ((int64_t) t->shape[0] != n_in || (int64_t) rest != n_out)
                throw std::runtime_error(t->name + " has unexpected shape");
            *(*v_it) = Dev{dense_ ? (uint8_t*) fused.p + off_rows * q_row : nullptr, (uint32_t) GGML_TYPE_Q4_0,
                           (int) n_in, (int) n_out, 1.f};
            if (dense_) {
                const size_t t_bytes = dg_->payload(*t);
                uploaded_.emplace_back(dg_->data(*t), t_bytes);
                const ggml_to_float_t to_float = ggml_get_type_traits((ggml_type) t->type)->to_float;
                if (!to_float) throw std::runtime_error(t->name + " unsupported type");
                std::vector<float> f((size_t) n_in * (size_t) n_out);
                to_float(dg_->data(*t), f.data(), (int64_t) f.size());
                ggml_quantize_chunk(GGML_TYPE_Q4_0, f.data(), q.data() + off_rows * q_row, 0, n_out, n_in, nullptr);
            }
            off_rows += (size_t) n_out;
        }
        if (dense_) h2d(fused.p, q.data(), total_bytes);
    };
    auto dmat_cols_q4 = [&](const TensorInfo* t, int64_t col_off, int64_t slice_in, int64_t n_out) -> Dev {
        const size_t q_row = ggml_row_size(GGML_TYPE_Q4_0, slice_in);
        const size_t out_bytes = q_row * (size_t) n_out;
        Dev d = reserve_dense(out_bytes, (uint32_t) GGML_TYPE_Q4_0, (int) slice_in, (int) n_out);
        if (!dense_) return d;
        const size_t in_row = dg_->payload(*t) / (size_t) n_out;
        const size_t elem_bytes = ggml_type_size((ggml_type) t->type);
        const ggml_to_float_t to_float = ggml_get_type_traits((ggml_type) t->type)->to_float;
        if (!to_float || ggml_blck_size((ggml_type) t->type) != 1)
            throw std::runtime_error(t->name + " must have unblocked type for column slicing");
        const uint8_t* src = (const uint8_t*) dg_->data(*t);
        std::vector<float> f((size_t) n_out * (size_t) slice_in);
        for (int64_t r = 0; r < n_out; ++r)
            to_float(src + (size_t) r * in_row + (size_t) col_off * elem_bytes, f.data() + (size_t) r * slice_in,
                     slice_in);
        std::vector<uint8_t> q(out_bytes);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, f.data(), q.data(), 0, n_out, slice_in, nullptr);
        h2d(d.p, q.data(), out_bytes);
        return d;
    };

    dspark_mask_ = dvec("mask_embed", n_embd_);
    dspark_norm_ = dvec("norm.weight", n_embd_);
    dspark_rnn_norm_ = dvec("markov_head.dflare_norm.weight", n_embd_);
    {
        const TensorInfo* rw = dtensor("markov_head.rnn_proj.weight");
        const int64_t rnn_in = 2 * (int64_t) dspark_rank_ + 2 * n_embd_;
        const int64_t rnn_out = 3 * (int64_t) dspark_rank_;
        if (rw->shape.size() != 2 || (int64_t) rw->shape[0] != rnn_in || (int64_t) rw->shape[1] != rnn_out)
            throw std::runtime_error("dspark markov_head.rnn_proj.weight has wrong shape");
        if (dense_) uploaded_.emplace_back(dg_->data(*rw), dg_->payload(*rw));
        dspark_rnn_sp_ = dmat_cols_q4(rw, 0, 2 * (int64_t) dspark_rank_, rnn_out);
        dspark_rnn_h_ = dmat_cols_q4(rw, 2 * (int64_t) dspark_rank_, n_embd_, rnn_out);
        dspark_rnn_f_ = dmat_cols_q4(rw, 2 * (int64_t) dspark_rank_ + n_embd_, n_embd_, rnn_out);
    }
    auto env_on = [](const char* name) {
        const char* v = std::getenv(name);
        return !(v && v[0] == '0' && v[1] == '\0');
    };
    const bool can_proxy = (output_.type == (uint32_t) GGML_TYPE_Q8_0 || output_.type == (uint32_t) GGML_TYPE_Q6_K) &&
                           n_embd_ == 2304 && dspark_rank_ == 256 && dspark_block_ == 8;
    dspark_proxy_ = can_proxy && env_on("STRATA_INK_DSPARK_PROXY");
    dspark_stage_graphs_ = env_on("STRATA_INK_DSPARK_GRAPHS");

    dspark_rnn_bias_ = dvec("markov_head.rnn_proj.bias", 3 * (int64_t) dspark_rank_);
    {
        const TensorInfo* w1 = dtensor("markov_head.markov_w1.weight");
        if (w1->type != kTypeBF16 || w1->shape.size() != 2 || (int64_t) w1->shape[0] != dspark_rank_ ||
            (int64_t) w1->shape[1] != dspark_w1_vocab_)
            throw std::runtime_error("dspark markov_w1.weight has wrong shape or type");
        const size_t w1_bytes = dg_->payload(*w1);
        if (dspark_proxy_) {
            if (dense_) {
                ck(cudaHostAlloc(&h_dspark_w1_, w1_bytes, cudaHostAllocMapped), "dspark w1 host alloc");
                std::memcpy(h_dspark_w1_, dg_->data(*w1), w1_bytes);
                void* dev_p = nullptr;
                ck(cudaHostGetDevicePointer(&dev_p, h_dspark_w1_, 0), "dspark w1 host dev ptr");
                d_dspark_w1_dev_ = (const uint16_t*) dev_p;
                dspark_w1_ = Dev{dev_p, kTypeBF16, dspark_rank_, dspark_w1_vocab_, 1.f};
                uploaded_.emplace_back(dg_->data(*w1), w1_bytes);
            }
        } else {
            dspark_w1_ = reserve_dense(w1_bytes, kTypeBF16, dspark_rank_, dspark_w1_vocab_);
            if (dense_) {
                h2d(dspark_w1_.p, dg_->data(*w1), w1_bytes);
                uploaded_.emplace_back(dg_->data(*w1), w1_bytes);
            }
        }
    }

    for (int i = 0; i < dspark_layers_; ++i) {
        DsparkLayer& dl = dspark_l_[(size_t) i];
        const std::string p = "layers." + std::to_string(i) + ".";
        const std::string a = p + "self_attn.";
        const int64_t q_dim = (int64_t) dspark_heads_ * dspark_head_dim_;
        const int64_t kv_dim = (int64_t) dspark_kv_ * dspark_head_dim_;
        dl.dflare_norm = dvec(p + "dflare_norm.weight", n_embd_);
        dl.input_norm = dvec(p + "input_layernorm.weight", n_embd_);
        dl.ff_norm = dvec(p + "post_attention_layernorm.weight", n_embd_);
        dl.q_norm = dvec(a + "q_norm.weight", dspark_head_dim_);
        dl.k_norm = dvec(a + "k_norm.weight", dspark_head_dim_);
        dl.k_norm_ctx = dvec(a + "k_norm_ctx.weight", dspark_head_dim_);
        dmat_cat({{a + "q_proj.weight", q_dim}, {a + "k_proj.weight", kv_dim}, {a + "v_proj.weight", kv_dim}}, n_embd_,
                 dl.qkv_proj, {&dl.q_proj, &dl.k_proj, &dl.v_proj});
        dmat_cat({{a + "k_proj_ctx.weight", kv_dim}, {a + "v_proj_ctx.weight", kv_dim}}, n_embd_, dl.kv_proj_ctx,
                 {&dl.k_proj_ctx, &dl.v_proj_ctx});
        dl.o_proj = dmat(a + "o_proj.weight", q_dim, n_embd_);
        dmat_cat({{p + "mlp.gate_proj.weight", dspark_ff_}, {p + "mlp.up_proj.weight", dspark_ff_}}, n_embd_,
                 dl.gate_up_proj, {&dl.gate_proj, &dl.up_proj});
        dl.down_proj = dmat(p + "mlp.down_proj.weight", dspark_ff_, n_embd_);
    }

    const size_t fold_w_bytes = ggml_row_size(GGML_TYPE_Q8_0, dspark_rank_) * (size_t) n_vocab_;
    const size_t fold_b_bytes = (size_t) n_vocab_ * sizeof(float);
    if (dspark_proxy_) {
        if (dense_) {
            ck(cudaHostAlloc(&h_dspark_fold_w_, fold_w_bytes, cudaHostAllocMapped), "dspark fold_w host alloc");
            ck(cudaHostGetDevicePointer(&d_dspark_fold_w_dev_, h_dspark_fold_w_, 0), "dspark fold_w host dev ptr");
            dspark_fold_w_ = Dev{d_dspark_fold_w_dev_, (uint32_t) GGML_TYPE_Q8_0, dspark_rank_, n_vocab_, 1.f};
        } else {
            dspark_fold_w_ = Dev{nullptr, (uint32_t) GGML_TYPE_Q8_0, dspark_rank_, n_vocab_, 1.f};
        }
        const int dv = draft_rows();
        dspark_head_sc_ = reserve_dense(inkling_dspark_screen_bytes(n_embd_, dv), kTypeF32, n_embd_, dv);
        dspark_fold_sc_ = reserve_dense(inkling_dspark_screen_bytes(dspark_rank_, dv), kTypeF32, dspark_rank_, dv);
        dspark_fold_bp_ = reserve_dense((size_t) inkling_dspark_screen_rows(dv) * sizeof(float), kTypeF32,
                                        (int) inkling_dspark_screen_rows(dv), 1);
    } else {
        dspark_fold_w_ = reserve_dense(fold_w_bytes, (uint32_t) GGML_TYPE_Q8_0, dspark_rank_, n_vocab_);
    }
    dspark_fold_b_ = reserve_dense(fold_b_bytes, kTypeF32, n_vocab_, 1);

    if (!dense_) return;

    const ggml_type_traits* bf16 = ggml_get_type_traits((ggml_type) kTypeBF16);
    if (!bf16 || !bf16->to_float) throw std::logic_error("ggml BF16 to_float unavailable");

    const TensorInfo* fw = dtensor("fuse_weight");
    if (fw->type != kTypeBF16 || fw->shape.size() != 2 || fw->shape[0] != 4 || fw->shape[1] != 4)
        throw std::runtime_error("dspark fuse_weight must be [4, 4] BF16");
    float fw_raw[16];
    bf16->to_float(dg_->data(*fw), fw_raw, 16);
    for (int r = 0; r < 4; ++r) {
        float mx = fw_raw[r * 4];
        for (int c = 1; c < 4; ++c) mx = std::max(mx, fw_raw[r * 4 + c]);
        double sum = 0.0;
        for (int c = 0; c < 4; ++c) {
            const float e = std::exp(fw_raw[r * 4 + c] - mx);
            dspark_fuse_[r][c] = e;
            sum += e;
        }
        for (int c = 0; c < 4; ++c) dspark_fuse_[r][c] = (float) (dspark_fuse_[r][c] / sum);
    }

    const TensorInfo* hw = dtensor("markov_head.hidden_out.weight");
    const TensorInfo* hb = dtensor("markov_head.hidden_out.bias");
    if (hw->type != kTypeBF16 || hw->shape.size() != 2 || (int64_t) hw->shape[0] != dspark_rank_ ||
        (int64_t) hw->shape[1] != n_embd_)
        throw std::runtime_error("dspark markov_head.hidden_out.weight has wrong shape or type");
    if (hb->type != kTypeBF16 || (int64_t) hb->elements() != n_embd_)
        throw std::runtime_error("dspark markov_head.hidden_out.bias has wrong shape or type");

    std::vector<float> hw_f((size_t) n_embd_ * dspark_rank_);
    bf16->to_float(dg_->data(*hw), hw_f.data(), (int64_t) hw_f.size());
    h_hidden_out_wt_.resize((size_t) dspark_rank_ * n_embd_);
    for (int j = 0; j < n_embd_; ++j)
        for (int k = 0; k < dspark_rank_; ++k)
            h_hidden_out_wt_[(size_t) k * n_embd_ + j] = hw_f[(size_t) j * dspark_rank_ + k];

    h_hidden_out_b_.resize((size_t) n_embd_);
    bf16->to_float(dg_->data(*hb), h_hidden_out_b_.data(), n_embd_);

    h_dspark_anchor_.resize((size_t) n_embd_);
}

void Inkling2::init_drafter() {
    if (!dspark_) return;
    float* tmp_in = (float*) dmalloc((size_t) 32 * n_embd_ * sizeof(float), "dspark fold in");
    float* tmp_out = (float*) dmalloc((size_t) 32 * n_vocab_ * sizeof(float), "dspark fold out");

    ck(cudaMemcpyAsync(tmp_in, h_hidden_out_b_.data(), (size_t) n_embd_ * sizeof(float), cudaMemcpyHostToDevice,
                       cs(st_)),
       "dspark fold bias");
    lin(output_, tmp_in, (float*) dspark_fold_b_.p, 1);

    const int n_blocks = dspark_rank_ / 32;
    const int step = std::min(rows_, 8);
    for (int b = 0; b < n_blocks; ++b) {
        ck(cudaMemcpyAsync(tmp_in, h_hidden_out_wt_.data() + (size_t) b * 32 * n_embd_,
                           (size_t) 32 * n_embd_ * sizeof(float), cudaMemcpyHostToDevice, cs(st_)),
           "dspark fold weight");
        for (int c0 = 0; c0 < 32;) {
            const int m = std::min(32 - c0, step);
            lin(output_, tmp_in + (size_t) c0 * n_embd_, tmp_out + (size_t) c0 * n_vocab_, m);
            c0 += m;
        }
        inkling_dspark_pack_q8_0_block(tmp_out, (uint8_t*) dspark_fold_w_.p, n_vocab_, b, n_blocks, st_);
    }
    for (int r = 0; r < dspark_block_ - 1; ++r)
        ck(cudaMemcpyAsync(dspark_mask_tiled_ + (size_t) r * n_embd_, dspark_mask_.p, (size_t) n_embd_ * sizeof(float),
                           cudaMemcpyDeviceToDevice, cs(st_)),
           "dspark mask tile");
    if (dspark_proxy_) {
        const int64_t dv = draft_rows();
        inkling_dspark_screen_pack(output_.p, (int) output_.type, n_embd_, dv, dspark_head_sc_.p, st_);
        inkling_dspark_screen_pack(dspark_fold_w_.p, (int) GGML_TYPE_Q8_0, dspark_rank_, dv, dspark_fold_sc_.p, st_);
        inkling_dspark_screen_order((const float*) dspark_fold_b_.p, dv, (float*) dspark_fold_bp_.p, st_);
        std::fprintf(stderr, "strata-poly: Inkling2 DSpark 2-bit head proxy enabled (type=%s, dv=%lld, w1+fold_w in pinned host RAM)\n",
                     output_.type == (uint32_t) GGML_TYPE_Q6_K ? "Q6_K" : "Q8_0", (long long) dv);
    }
    dspark_stage_graph_.assign((size_t) dspark_layers_, nullptr);
    ck(cudaStreamSynchronize(cs(st_)), "init_drafter");
    cudaFree(tmp_in);
    cudaFree(tmp_out);
    std::vector<float>().swap(h_hidden_out_wt_);
    std::vector<float>().swap(h_hidden_out_b_);
}

void Inkling2::reset_drafter() {
    if (!dspark_) return;
    if (d_f_rnn_) ck(cudaMemsetAsync(d_f_rnn_, 0, (size_t) n_embd_ * sizeof(float), cs(st_)), "reset dspark");
    if (d_rnn_f_gates_)
        ck(cudaMemsetAsync(d_rnn_f_gates_, 0, (size_t) (3 * dspark_rank_) * sizeof(float), cs(st_)), "reset dspark");
    block_n_ = 0;
}

void Inkling2::draft_commit(int64_t pos, int n) {
    if (!dspark_ || n <= 0) return;
    if (zha_on_ && !in_read_ && win_ > 0) {
        const int64_t committed = pos + n;
        for (ZSlot& z : zs_) {
            if (z.zpos > committed) {
                z.zpos = committed;
                z.zbuilt = std::min(z.zbuilt, (int) (committed / zha_.C));
                z.zleaf_used = std::min<int64_t>(z.zleaf_used, committed);
            }
        }
    }
    cover(pos + n);
    inkling_dspark_mix_norm(d_taps_ + (size_t) (n - 1) * 3 * n_embd_, hn_ + (size_t) n * n_embd_, dspark_fuse_[3][0],
                            dspark_fuse_[3][1], dspark_fuse_[3][2], dspark_fuse_[3][3], fp(dspark_rnn_norm_), d_f_rnn_,
                            n_embd_, 1, dspark_eps_, st_);
    lin(dspark_rnn_f_, d_f_rnn_, d_rnn_f_gates_, 1);

    const int kv_width = dspark_kv_ * dspark_head_dim_;
    auto store_d_kv = [&](const float* x, DsparkLayer& dl, bool is_v) {
        if (dl.local) {
            uint16_t* cache = is_v ? dl.dv : dl.dk;
            for (int done = 0; done < n;) {
                const int64_t at = (pos + done) % dspark_ring_;
                const int m = (int) std::min<int64_t>(n - done, dspark_ring_ - at);
                store_f16(x + (size_t) done * kv_width, cache, kv_width, at, m, st_);
                done += m;
            }
            return;
        }
        const bool use_q4 = opt_.kv_format == KvFmt::kQ4_0 || (is_v && opt_.kv_format == KvFmt::kK8V4);
        const bool use_q8 = opt_.kv_format == KvFmt::kQ8_0 || (!is_v && opt_.kv_format == KvFmt::kK8V4);
        if (use_q4) {
            store_q4(x, is_v ? dl.dv_q4 : dl.dk_q4, is_v ? dl.dv_scales : dl.dk_scales, kv_width, pos, n, st_);
            return;
        }
        if (use_q8) {
            store_q8(x, is_v ? dl.dv_codes : dl.dk_codes, is_v ? dl.dv_scales : dl.dk_scales, kv_width, pos, n, st_);
            return;
        }
        store_f16(x, is_v ? dl.dv : dl.dk, kv_width, pos, n, st_);
    };

    for (int l = 0; l < dspark_layers_; ++l) {
        DsparkLayer& dl = dspark_l_[(size_t) l];
        inkling_dspark_mix_norm(d_taps_, hn_ + n_embd_, dspark_fuse_[l][0], dspark_fuse_[l][1], dspark_fuse_[l][2],
                                dspark_fuse_[l][3], fp(dl.dflare_norm), cur_, n_embd_, n, dspark_eps_, st_);
        const void* xq_ctx = nullptr;
        if (n < kStageRows) {
            kernels::native_quantize_q8_1(cur_, xq_, n_embd_ * n, 1, st_);
            xq_ctx = xq_;
        }
        if (n == 1) {
            lin(dl.kv_proj_ctx, cur_, qraw_, 1, xq_ctx);
            head_norm_rope(qraw_, dspark_head_dim_, fp(dl.k_norm_ctx), k_, dspark_head_dim_, dspark_kv_,
                           dspark_head_dim_, dspark_theta_scale_, pos, 1, dspark_eps_, st_);
            store_d_kv(k_, dl, false);
            store_d_kv(qraw_ + kv_width, dl, true);
        } else {
            lin(dl.k_proj_ctx, cur_, q_, n, xq_ctx);
            head_norm_rope(q_, dspark_head_dim_, fp(dl.k_norm_ctx), k_, dspark_head_dim_, dspark_kv_, dspark_head_dim_,
                           dspark_theta_scale_, pos, n, dspark_eps_, st_);
            store_d_kv(k_, dl, false);
            lin(dl.v_proj_ctx, cur_, v_, n, xq_ctx);
            store_d_kv(v_, dl, true);
        }
    }
}

void Inkling2::draft_block(int32_t token) {
    if (!dspark_ || pos_ <= 0 || pos_ + dspark_block_ > opt_.max_context) {
        block_n_ = 0;
        return;
    }
    cover(pos_ + dspark_block_);

    embed(token, h_dspark_anchor_.data());
    norm_text_embed(h_dspark_anchor_.data());
    ck(cudaMemcpyAsync(d_blk_x_, h_dspark_anchor_.data(), (size_t) n_embd_ * sizeof(float), cudaMemcpyHostToDevice,
                       cs(st_)),
       "dspark anchor");
    ck(cudaMemcpyAsync(d_blk_x_ + n_embd_, dspark_mask_tiled_,
                       (size_t) (dspark_block_ - 1) * n_embd_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(st_)),
       "dspark mask");
    if (dspark_proxy_) {
        if (token >= 0 && token < dspark_w1_vocab_) {
            ck(cudaMemcpyAsync(d_w1_row_, h_dspark_w1_ + (size_t) token * (size_t) dspark_rank_,
                               (size_t) dspark_rank_ * sizeof(uint16_t), cudaMemcpyHostToDevice, cs(st_)),
               "dspark w1 anchor row");
        } else {
            ck(cudaMemsetAsync(d_w1_row_, 0, (size_t) dspark_rank_ * sizeof(uint16_t), cs(st_)),
               "dspark w1 anchor zero");
        }
    }

    const int B = dspark_block_;
    const int q_dim = dspark_heads_ * dspark_head_dim_;
    const int kv_width = dspark_kv_ * dspark_head_dim_;
    const int threads = sum_threads(n_embd_);
    const bool k_q8 = opt_.kv_format == KvFmt::kQ8_0 || opt_.kv_format == KvFmt::kK8V4;
    const bool k_q4 = opt_.kv_format == KvFmt::kQ4_0;
    const bool v_q8 = opt_.kv_format == KvFmt::kQ8_0;
    const bool v_q4 = opt_.kv_format == KvFmt::kQ4_0 || opt_.kv_format == KvFmt::kK8V4;
    const bool quant = k_q8 || k_q4 || v_q8 || v_q4;
    auto store_d_blk = [&](const float* x, DsparkLayer& dl, bool is_v) {
        if (dl.local) {
            uint16_t* cache = is_v ? dl.dv : dl.dk;
            for (int done = 0; done < B;) {
                const int64_t at = (pos_ + done) % dspark_ring_;
                const int m = (int) std::min<int64_t>(B - done, dspark_ring_ - at);
                store_f16(x + (size_t) done * kv_width, cache, kv_width, at, m, st_);
                done += m;
            }
            return;
        }
        const bool use_q4 = is_v ? v_q4 : k_q4;
        const bool use_q8 = is_v ? v_q8 : k_q8;
        if (use_q4) {
            store_q4(x, is_v ? dl.dv_q4 : dl.dk_q4, is_v ? dl.dv_scales : dl.dk_scales, kv_width, pos_, B, st_);
            return;
        }
        if (use_q8) {
            store_q8(x, is_v ? dl.dv_codes : dl.dk_codes, is_v ? dl.dv_scales : dl.dk_scales, kv_width, pos_, B, st_);
            return;
        }
        store_f16(x, is_v ? dl.dv : dl.dk, kv_width, pos_, B, st_);
    };

    const float scale = 1.0f / std::sqrt((float) dspark_head_dim_);
    const bool use_stage_graphs = opt_.graphs && !opt_.dump && dspark_stage_graphs_;
    auto pre_attn = [&](int l) {
        DsparkLayer& dl = dspark_l_[(size_t) l];
        kernels::native_add_rms_norm_q8_1(d_blk_x_, l == 0 ? nullptr : d_blk_attn_, fp(dl.input_norm), d_blk_xn_,
                                          d_blk_xq_, n_embd_, B, dspark_eps_, threads, st_);
        lin(dl.qkv_proj, d_blk_xn_, d_blk_qkv_, B, d_blk_xq_);
        inkling_dspark_split_qkv(d_blk_qkv_, d_blk_qraw_, d_blk_kraw_, v_, q_dim, kv_width, B, st_);
    };
    auto post_attn = [&](int l) {
        DsparkLayer& dl = dspark_l_[(size_t) l];
        lin(dl.o_proj, att_, d_blk_attn_, B);
        kernels::native_add_rms_norm_q8_1(d_blk_x_, d_blk_attn_, fp(dl.ff_norm), d_blk_xn_, d_blk_xq_, n_embd_, B,
                                          dspark_eps_, threads, st_);
        lin(dl.gate_up_proj, d_blk_xn_, d_blk_gate_, B, d_blk_xq_);
        inkling_swiglu_gate_up_q8_1(d_blk_gate_, d_blk_q8ff_, dspark_ff_, B, st_);
        lin(dl.down_proj, d_blk_gate_, d_blk_attn_, B, d_blk_q8ff_);
    };
    for (int l = 0; l < dspark_layers_; ++l) {
        auto run_stage = [&] {
            if (l > 0) post_attn(l - 1);
            pre_attn(l);
        };
        if (use_stage_graphs) {
            void*& g = dspark_stage_graph_[(size_t) l];
            if (!g) g = record_fn(run_stage);
            ck(cudaGraphLaunch((cudaGraphExec_t) g, cs(st_)), "dspark stage graph");
        } else {
            run_stage();
        }
        DsparkLayer& dl = dspark_l_[(size_t) l];
        head_norm_rope(d_blk_qraw_, dspark_head_dim_, fp(dl.q_norm), q_, dspark_head_dim_, dspark_heads_,
                       dspark_head_dim_, dspark_theta_scale_, pos_, B, dspark_eps_, st_);
        head_norm_rope(d_blk_kraw_, dspark_head_dim_, fp(dl.k_norm), k_, dspark_head_dim_, dspark_kv_, dspark_head_dim_,
                       dspark_theta_scale_, pos_, B, dspark_eps_, st_);
        store_d_blk(k_, dl, false);
        store_d_blk(v_, dl, true);

        const bool quant_g = !dl.local && quant;
        const Kvq8 kq = (!dl.local && k_q8) ? Kvq8{dl.dk_codes, dl.dk_scales} : Kvq8{};
        const Kvq8 vq = (!dl.local && v_q8) ? Kvq8{dl.dv_codes, dl.dv_scales} : Kvq8{};
        const Kvq4 kq4 = (!dl.local && k_q4) ? Kvq4{dl.dk_q4, dl.dk_scales} : Kvq4{};
        const Kvq4 vq4 = (!dl.local && v_q4) ? Kvq4{dl.dv_q4, dl.dv_scales} : Kvq4{};
        inkling_dspark_attend(q_, quant_g ? nullptr : dl.dk, quant_g ? nullptr : dl.dv, pos_, B, score_, att_,
                              dspark_head_dim_, dspark_heads_, dspark_kv_, scale, st_,
                              dl.local ? (dspark_window_ + 1) : 0, dl.local ? dspark_ring_ : 0, nullptr, 0, kq, vq,
                              kq4, vq4);
    }

    const int n_cand = B - 1;
    const int dv = draft_rows();
    const int steady_steps = std::min(n_cand, opt_.window - 1);
    const int n_steps = std::min(steady_steps, (int) std::max<int64_t>(0, opt_.max_context - pos_ - 1));
    if (n_steps <= 0) {
        post_attn(dspark_layers_ - 1);
        block_n_ = 0;
        return;
    }

    if (!dspark_proxy_) {
        h_dspark_anchor_pick_ = {token, 1.0f};
        ck(cudaMemcpyAsync(d_block_picks_ + (B - 1), &h_dspark_anchor_pick_, sizeof(Likeliest), cudaMemcpyHostToDevice,
                           cs(st_)),
           "dspark anchor pick");
    }

    auto run_head = [&](int steps) {
        post_attn(dspark_layers_ - 1);
        kernels::native_add_rms_norm_q8_1(d_blk_x_ + n_embd_, d_blk_attn_ + n_embd_, fp(dspark_norm_), d_blk_h_,
                                          d_blk_xq_, n_embd_, steps, dspark_eps_, threads, st_);
        lin(dspark_rnn_h_, d_blk_h_, d_rnn_h_gates_, steps, d_blk_xq_);
        ck(cudaMemsetAsync(d_rnn_state_, 0, (size_t) dspark_rank_ * sizeof(float), cs(st_)), "dspark state");
        if (dspark_proxy_) {
            inkling_dspark_head_base(dspark_head_sc_.p, d_blk_xq_, steps, dv, d_head_approx_, st_);
            const DsparkHeadArgs hargs{
                output_.p, (int) output_.type, dspark_head_sc_.p, dspark_fold_w_.p, dspark_fold_sc_.p,
                (const float*) dspark_fold_b_.p, (const float*) dspark_fold_bp_.p, dv, final_logit_softcapping_};
            const size_t lat_stride = kernels::native_q8_1_bytes(dspark_rank_);
            for (int i = 0; i < steps; ++i) {
                inkling_dspark_prep_rnn_step_row(d_rnn_state_, d_w1_row_, d_rnn_in_, dspark_rank_, st_);
                lin(dspark_rnn_sp_, d_rnn_in_, d_rnn_gates_, 1);
                inkling_dspark_rnn_gate(d_rnn_gates_, d_rnn_h_gates_ + (size_t) i * 3 * dspark_rank_, d_rnn_f_gates_,
                                        fp(dspark_rnn_bias_), d_rnn_state_, d_rnn_latent_, dspark_rank_, st_);
                void* latq = static_cast<char*>(d_rnn_lat_q_) + (size_t) i * lat_stride;
                kernels::native_quantize_q8_1(d_rnn_latent_, latq, dspark_rank_, 1, st_);
                inkling_dspark_head_pick(hargs, d_blk_xq_, latq, d_head_approx_, i, d_head_scr_, d_block_picks_ + i,
                                         d_dspark_w1_dev_, d_w1_row_, st_);
            }
        } else {
            Dev head = output_;
            head.n_out = dv;
            lin(head, d_blk_h_, d_base_logits_, steps, d_blk_xq_);
            for (int i = 0; i < steps; ++i) {
                inkling_dspark_prep_rnn_step(d_rnn_state_, (const uint16_t*) dspark_w1_.p, -1, d_block_picks_, i, B - 1,
                                             dspark_rank_, dspark_w1_vocab_, d_rnn_in_, st_);
                lin(dspark_rnn_sp_, d_rnn_in_, d_rnn_gates_, 1);
                inkling_dspark_rnn_gate(d_rnn_gates_, d_rnn_h_gates_ + (size_t) i * 3 * dspark_rank_, d_rnn_f_gates_,
                                        fp(dspark_rnn_bias_), d_rnn_state_, d_rnn_latent_, dspark_rank_, st_);
                Dev fhead = dspark_fold_w_;
                fhead.n_out = dv;
                lin(fhead, d_rnn_latent_, d_markov_bias_, 1);
                inkling_dspark_add_softcap(d_base_logits_ + (size_t) i * dv, d_markov_bias_,
                                           (const float*) dspark_fold_b_.p, logits_, dv, final_logit_softcapping_, st_);
                likeliest(logits_, dv, d_block_picks_ + i, st_);
            }
        }
    };

    if (use_stage_graphs && n_steps == steady_steps) {
        if (!dspark_head_graph_) dspark_head_graph_ = record_fn([&] { run_head(steady_steps); });
        ck(cudaGraphLaunch((cudaGraphExec_t) dspark_head_graph_, cs(st_)), "dspark head graph");
    } else {
        run_head(n_steps);
    }

    ck(cudaMemcpyAsync(h_block_, d_block_picks_, (size_t) n_steps * sizeof(Likeliest), cudaMemcpyDeviceToHost,
                       cs(st_)),
       "dspark picks");
    ck(cudaStreamSynchronize(cs(st_)), "dspark block");
    block_n_ = n_steps;
}

void Inkling2::save_draft_extra(int64_t pos, uint8_t* dst) {
    if (!dspark_) return;
    ck(cudaStreamSynchronize(cs(st_)), "save dspark extra");
    const size_t base_swa = (size_t) n_swa_slots_ * (size_t) keep_ *
                            (kv_code_row_bytes(false) + kv_code_row_bytes(true) + 2 * kv_scale_row_bytes());
    dst += state_n_ * sizeof(float) + base_swa;
    const size_t f_bytes = (size_t) n_embd_ * sizeof(float);
    ck(cudaMemcpy(dst, d_f_rnn_, f_bytes, cudaMemcpyDeviceToHost), "save dspark f_rnn");
    dst += f_bytes;

    const size_t kv_row = (size_t) dspark_kv_ * dspark_head_dim_ * sizeof(uint16_t);
    const size_t keep_d = (size_t) dspark_window_;
    const int64_t lo = std::max<int64_t>(0, pos - dspark_window_);
    for (const DsparkLayer& dl : dspark_l_) {
        if (!dl.local) continue;
        auto save_d_ring = [&](const uint16_t* cache) {
            std::memset(dst, 0, keep_d * kv_row);
            for (int64_t p = lo; p < pos;) {
                const int64_t at = p % dspark_ring_, m = std::min<int64_t>(pos - p, dspark_ring_ - at);
                const size_t s_at = (size_t) (p - (pos - dspark_window_));
                ck(cudaMemcpy(dst + s_at * kv_row, (const uint8_t*) cache + (size_t) at * kv_row, (size_t) m * kv_row,
                              cudaMemcpyDeviceToHost),
                   "save dspark kv");
                p += m;
            }
            dst += keep_d * kv_row;
        };
        save_d_ring(dl.dk);
        save_d_ring(dl.dv);
    }
}

// ---- ZHA index over global layers ----------------------------------------------------------------------------------

void Inkling2::zha_load() {
    const std::string path = zha_file();
    if (path.empty()) return;
#if defined(STRATA_GPU_METAL) || defined(STRATA_USE_HIP)
    throw std::runtime_error("STRATA_INK_ZHA: the ZHA index runs on NVIDIA cards only");
#else
    const auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(what);
    };
    {
        int dev = 0, major = 0, minor = 0;
        const bool known = cudaGetDevice(&dev) == cudaSuccess &&
                           cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                           cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess;
        need(known && major * 10 + minor >= 75, "STRATA_INK_ZHA: the ZHA index needs an NVIDIA card from Turing on");
    }
    need(!split_ && opt_.layer_split.empty(), "STRATA_INK_ZHA: the ZHA index runs on one card");
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("STRATA_INK_ZHA: cannot open " + path);
    int32_t h[4];
    if (std::fread(h, sizeof h, 1, f) != 1 || h[0] != 0x5A484152) {
        std::fclose(f);
        throw std::runtime_error("STRATA_INK_ZHA: not a ZHAR file");
    }
    zha_.C = h[1];
    zha_.R = h[2];
    if (head_dim_ != 128 || zha_.R != 4 || zha_.C != 64) {
        std::fclose(f);
        throw std::runtime_error("STRATA_INK_ZHA: heads of 128, chunks of 64 and 4 references");
    }
    const size_t per = (size_t) n_kv_ * zha_.R * head_dim_;
    zha_refs_.resize(per * (size_t) h[3]);
    for (int k = 0; k < h[3]; ++k) {
        int32_t id[2];
        if (std::fread(id, sizeof id, 1, f) != 1 ||
            std::fread(zha_refs_.data() + k * per, sizeof(float), per, f) != per) {
            std::fclose(f);
            throw std::runtime_error("STRATA_INK_ZHA: the file ends early");
        }
        zha_ids_.push_back({id[0], id[1]});
    }
    std::fclose(f);
    zha_.budget = (float) env_num("STRATA_INK_ZHA_BUDGET", 0.16);
    zha_.max_sel = (int) env_num("STRATA_INK_ZHA_MAX", 256);
    zha_pool_ = (int) env_num("STRATA_INK_ZHA_POOL", 600);
    need(zha_.budget > 0.f && zha_.budget <= 1.f && zha_.max_sel >= 1 && zha_pool_ >= 0,
         "STRATA_INK_ZHA: _BUDGET in (0, 1], _MAX 1 or more, _POOL 0 or more");
    zring_ = (int64_t) zha_.recent + 2 * zha_.C + std::max(opt_.prompt_batch, 64) + 64;
    zha_max_chunks_ = (int) ((opt_.max_context + zha_.C - 1) / zha_.C + 31) / 32 * 32;
    zha_on_ = true;
    std::fprintf(stderr,
                 "strata-poly: ZHA index: chunks of %d, %d references, budget %.3f (at most %d), a pool of %d "
                 "chunks a KV head%s\n",
                 zha_.C, zha_.R, zha_.budget, zha_.max_sel, zha_pool_,
                 (opt_.kv_grow && opt_.max_context > 65536) ? ", leaves grow from 64K" : "");
#endif
}

void Inkling2::zha_alloc(int slot, int il, int pass) {
    ZSlot& S = zs_[(size_t) slot];
    if (S.kc) return;
    if (zleaf_cap0_ == 0)
        zleaf_cap0_ = opt_.kv_grow ? std::min<int64_t>(opt_.max_context, (int64_t) 65536) : opt_.max_context;
    S.zleaf_cap = 0;
    S.zleaf_used = 0;
    zha_leaf_recap(S, zleaf_cap0_);
    const size_t codes = (size_t) n_kv_ * head_dim_, scales = codes / 32 * sizeof(uint16_t);
    S.rkc = (int8_t*) dmalloc((size_t) zring_ * codes, "the index's ring");
    S.rks = (uint16_t*) dmalloc((size_t) zring_ * scales, "the index's ring");
    S.rvc = (int8_t*) dmalloc((size_t) zring_ * codes, "the index's ring");
    S.rvs = (uint16_t*) dmalloc((size_t) zring_ * scales, "the index's ring");
    if (zha_pool_ > 0) {
        ink::ZhaCache& Z = S.zcache;
        const size_t keys = (size_t) n_kv_ * zha_pool_ * zha_.C, per_key = head_dim_ / 2;
        Z.P = zha_pool_;
        Z.pool.row = head_dim_;
        Z.pool.headed = 0;
        Z.pool.q4 = 1;
        Z.pool.kc = (int8_t*) dmalloc(keys * per_key, "the index's chunk cache");
        Z.pool.vc = (int8_t*) dmalloc(keys * per_key, "the index's chunk cache");
        Z.pool.ks = (uint16_t*) dmalloc(keys * head_dim_ / 32 * sizeof(uint16_t), "the index's chunk cache");
        Z.pool.vs = (uint16_t*) dmalloc(keys * head_dim_ / 32 * sizeof(uint16_t), "the index's chunk cache");
        Z.owner = (int*) dmalloc((size_t) n_kv_ * Z.P * sizeof(int), "the index's chunk cache");
        Z.stamp = (int*) dmalloc((size_t) n_kv_ * Z.P * sizeof(int), "the index's chunk cache");
        Z.pmap = (int*) dmalloc((size_t) n_kv_ * zha_max_chunks_ * sizeof(int), "the index's chunk cache");
        Z.aux = (int*) dmalloc((size_t) n_kv_ * (Z.P + 2) * sizeof(int), "the index's chunk cache");
        ink::zha_cache_clear(Z, n_kv_, zha_max_chunks_, st_);
    }
    size_t k = 0;
    while (k < zha_ids_.size() && zha_ids_[k] != std::make_pair(il, pass)) ++k;
    if (k == zha_ids_.size())
        throw std::runtime_error("STRATA_INK_ZHA: no references for layer " + std::to_string(il) + " pass " +
                                 std::to_string(pass));
    const size_t refs = (size_t) n_kv_ * zha_.R * head_dim_;
    S.zrefs = (float*) dmalloc(refs * sizeof(float), "the index");
    ck(cudaMemcpy(S.zrefs, zha_refs_.data() + k * refs, refs * sizeof(float), cudaMemcpyHostToDevice), "the index");
    if (zcap0_ == 0) {
        zgrow_ = vmm_ && !pool_on_ && !freeze_experts_;
        zcap0_ = zgrow_ ? std::min(zha_max_chunks_, 1024) : zha_max_chunks_;
    }
    S.zcap = 0;
    S.zbuilt = 0;
    S.zpos = -1;
    zha_recap(S, zcap0_);
}

void Inkling2::zha_recap(ZSlot& S, int cap) {
    if (cap == S.zcap) return;
    const size_t R = (size_t) zha_.R, kv = (size_t) n_kv_;
    const size_t code = head_dim_ / 2, scale = head_dim_ / 32 * sizeof(uint16_t);
    const size_t rows_new = kv * (size_t) cap * R, rows_old = kv * (size_t) S.zcap * R;
    const size_t bytes_new = 2 * rows_new * (code + scale) + rows_new * sizeof(float);
    const size_t bytes_old = 2 * rows_old * (code + scale) + rows_old * sizeof(float);
    ck(cudaStreamSynchronize(cs(st_)), "the index's summaries");
    if (S.zk && exp_ && bytes_new > bytes_old) {
        const size_t free_b = pool_on_ ? vpool_bytes() : vram_free_budget_bytes();
        const size_t keep_b = pool_on_ ? 0 : vram_keep_bytes();
        if (free_b < bytes_new + keep_b) {
            shrink_cache(bytes_new + keep_b - free_b);
            held_is_budget();
        }
    }
    void* p[3] = {};
    const size_t sizes[3] = {rows_new * (code + scale), rows_new * (code + scale), rows_new * sizeof(float)};
    for (int i = 0; i < 3; ++i)
        if (cudaMalloc(&p[i], std::max<size_t>(sizes[i], 256)) != cudaSuccess) {
            cudaGetLastError();
            for (int k = 0; k < i; ++k) cudaFree(p[k]);
            throw std::runtime_error("no VRAM for the index's summaries (" + std::to_string(cap) + " chunks a head)");
        }
    const int keep = std::min(S.zbuilt, std::min(cap, S.zcap));
    if (S.zk && keep > 0) {
        const size_t w = (size_t) keep * R;
        uint16_t* const olds[2] = {S.zk, S.zv};
        for (int i = 0; i < 2; ++i) {
            uint8_t* d = (uint8_t*) p[i];
            const uint8_t* s = (const uint8_t*) olds[i];
            ck(cudaMemcpy2DAsync(d, (size_t) cap * R * code, s, (size_t) S.zcap * R * code, w * code, kv,
                                 cudaMemcpyDeviceToDevice, cs(st_)),
               "the index's summaries");
            ck(cudaMemcpy2DAsync(d + rows_new * code, (size_t) cap * R * scale, s + rows_old * code,
                                 (size_t) S.zcap * R * scale, w * scale, kv, cudaMemcpyDeviceToDevice, cs(st_)),
               "the index's summaries");
        }
        ck(cudaMemcpy2DAsync(p[2], (size_t) cap * R * sizeof(float), S.zb, (size_t) S.zcap * R * sizeof(float),
                             w * sizeof(float), kv, cudaMemcpyDeviceToDevice, cs(st_)),
           "the index's summaries");
        ck(cudaStreamSynchronize(cs(st_)), "the index's summaries");
    }
    for (void* o : {(void*) S.zk, (void*) S.zv, (void*) S.zb})
        if (o) cudaFree(o);
    const bool had = S.zk != nullptr;
    S.zk = (uint16_t*) p[0], S.zv = (uint16_t*) p[1], S.zb = (float*) p[2];
    S.zbuilt = keep;
    S.zcap = cap;
    if (had && exp_ && bytes_old > bytes_new + kArenaPiece) {
        size_t give = bytes_old - bytes_new;
        if (!pool_on_) {
            const size_t free_b = vram_free_budget_bytes();
            const size_t floor_b = vram_keep_bytes();
            give = std::min(give, free_b > floor_b ? free_b - floor_b : 0);
        }
        if (give > 0) grow_cache(give, true);
        held_is_budget();
    }
}

void Inkling2::zha_room_all(int64_t end) {
    int need = (int) ((end + zha_.C - 1) / zha_.C) + 1;
    zread_full_ = end - pos_ >= read_rows_;
    if (zread_full_) need = zha_max_chunks_;
    const int64_t leaf_need = std::min<int64_t>(opt_.max_context, end + 256);
    for (ZSlot& S : zs_) {
        if (S.zk) zha_room(S, need);
        if (S.kc) zha_leaf_room(S, leaf_need);
    }
}

void Inkling2::zha_room(ZSlot& S, int need) {
    if (!zgrow_ || need <= S.zcap) return;
    int cap = S.zcap;
    while (cap < need && cap < zha_max_chunks_) cap = std::min(zha_max_chunks_, cap * 2);
    zha_recap(S, cap);
}

void Inkling2::zha_leaf_recap(ZSlot& S, int64_t cap) {
    cap = std::clamp<int64_t>(cap, 1, opt_.max_context);
    if (cap == S.zleaf_cap) return;
    if (S.kc) ck(cudaStreamSynchronize(cs(st_)), "the index's leaves");
    const int64_t keep = std::min<int64_t>(S.zleaf_used, std::min<int64_t>(cap, S.zleaf_cap));
    const size_t codes = zha_leaf_codes(), scales = zha_leaf_scales();
    auto realloc_pinned = [&](auto*& ptr, size_t row_bytes) {
        using T = std::remove_pointer_t<std::remove_reference_t<decltype(ptr)>>;
        void* np = nullptr;
        ck(cudaHostAlloc(&np, (size_t) cap * row_bytes, cudaHostAllocMapped | cudaHostAllocPortable),
           "the index's leaves (pinned RAM)");
        if (ptr && keep > 0) std::memcpy(np, ptr, (size_t) keep * row_bytes);
        if (ptr) cudaFreeHost(ptr);
        ptr = (T*) np;
    };
    realloc_pinned(S.kc, codes);
    realloc_pinned(S.ksc, scales);
    realloc_pinned(S.vc, codes);
    realloc_pinned(S.vsc, scales);
    S.zleaf_cap = cap;
    S.zleaf_used = keep;
}

void Inkling2::zha_leaf_room(ZSlot& S, int64_t need) {
    if (need <= S.zleaf_cap) return;
    int64_t cap = std::max<int64_t>(S.zleaf_cap, zleaf_cap0_);
    while (cap < need && cap < opt_.max_context) cap = std::min<int64_t>(opt_.max_context, cap * 2);
    zha_leaf_recap(S, cap);
}

void Inkling2::zha_scratch() {
    if (zw_.S) return;
    const int per = n_head_ / n_kv_;
    const size_t sub = (size_t) rows_, extent = (size_t) std::max(window_, 1024);
    zlend_ = vmm_;
    const size_t need = (size_t) per * sub * extent + 65536 + (sub / 64 + 1) * (size_t) zha_max_chunks_ +
                        sub * per * 9 * (head_dim_ + 2) + (size_t) n_head_ * 128 * (head_dim_ + 2);
    zw_.s_floats = std::max<size_t>(size_t(4) << 20, need + (size_t(1) << 20));
    zfull_[7] = zw_.s_floats * sizeof(float);
    if (zlend_) {
        vmm_reserve(zmap_[7], zfull_[7]);
        zw_.S = (float*) zmap_[7].base;
    } else {
        zw_.S = (float*) dmalloc(zfull_[7], "the index");
    }
    const size_t hs = sub * zha_max_chunks_;
    zw_.heads = zlend_ ? n_kv_ : 1;
    zw_.cmap = (int*) dmalloc((size_t) n_kv_ * zha_max_chunks_ * sizeof(int), "the index");
    zw_.count = (int*) dmalloc((size_t) n_kv_ * sizeof(int), "the index");
    zw_.glist = (int*) dmalloc((size_t) n_kv_ * (zha_max_chunks_ + 1) * sizeof(int), "the index");
    const size_t keys = (size_t) zw_.heads * zha_max_chunks_ * zha_.C, per_key = head_dim_ / 2;
    zw_.stage.row = head_dim_;
    zw_.stage.headed = 0;
    zw_.stage.q4 = 1;
    zfull_[0] = (size_t) (per + 1) * hs * sizeof(float);
    zfull_[1] = zfull_[2] = (size_t) per * hs;
    zfull_[3] = zfull_[4] = keys * per_key;
    zfull_[5] = zfull_[6] = keys * head_dim_ / 32 * sizeof(uint16_t);
    void* p[7];
    for (int i = 0; i < 7; ++i) {
        if (zlend_) {
            vmm_reserve(zmap_[i], zfull_[i]);
            p[i] = (void*) zmap_[i].base;
        } else {
            p[i] = dmalloc(zfull_[i], "the index");
        }
    }
    zw_.score = (float*) p[0];
    zw_.rstar = (uint8_t*) p[1];
    zw_.sel = (uint8_t*) p[2];
    zw_.stage.kc = (int8_t*) p[3];
    zw_.stage.vc = (int8_t*) p[4];
    zw_.stage.ks = (uint16_t*) p[5];
    zw_.stage.vs = (uint16_t*) p[6];
    if (zlend_) {
        const size_t hd = (size_t) 4 * zha_max_chunks_;
        zwd_ = zw_;
        zwd_.s_floats = size_t(2) << 20;
        zwd_.S = (float*) dmalloc(zwd_.s_floats * sizeof(float), "the index");
        zwd_.score = (float*) dmalloc((size_t) (n_head_ + n_kv_) * hd * sizeof(float), "the index");
        zwd_.rstar = (uint8_t*) dmalloc((size_t) n_head_ * hd, "the index");
        zwd_.sel = (uint8_t*) dmalloc((size_t) n_head_ * hd, "the index");
        zwd_.heads = 1;
    }
}

void Inkling2::zha_widen(bool keep) {
    const int heads = keep && zw_.heads == n_kv_ ? n_kv_ : 1;
    size_t want[8] = {};
    const int nm = zmap_[7].base ? 8 : 7;
    for (int i = 0; i < nm; ++i)
        want[i] = i >= 3 && i < 7 && zw_.heads > 1 ? zfull_[i] / zw_.heads * heads : zfull_[i];
    bool enough = zwide_;
    for (int i = 0; i < nm && enough; ++i) enough = zmap_[i].mapped >= want[i];
    if (enough) {
        zkeep_now_ = std::max(zkeep_now_, heads);
        return;
    }
    ck(cudaStreamSynchronize(cs(st_)), "the index's prompt buffers");
    const size_t grain = pool_on_ ? pg_ : grain_;
    const auto whole = [&](size_t x) { return (x + grain - 1) / grain * grain; };
    size_t more = 0;
    for (int i = 0; i < nm; ++i) more += whole(want[i]) > zmap_[i].mapped ? whole(want[i]) - zmap_[i].mapped : 0;
    const size_t free_b = pool_on_ ? vpool_bytes() : vram_free_budget_bytes();
    const size_t keep_b = pool_on_ ? 0 : vram_keep_bytes();
    if (exp_ && free_b < more + keep_b) {
        const size_t had = arena_.mapped;
        shrink_cache(more + keep_b - free_b);
        held_is_budget();
        zcut_ += had - arena_.mapped;
    }
    int mapped_heads = heads;
    try {
        for (int i = 0; i < nm; ++i) vmm_map(zmap_[i], want[i], SIZE_MAX, "the index's prompt buffers");
    } catch (const std::runtime_error&) {
        if (heads == 1) throw;
        mapped_heads = 1;
        for (int i = 0; i < nm; ++i)
            vmm_map(zmap_[i], i >= 3 && i < 7 ? zfull_[i] / zw_.heads : zfull_[i], SIZE_MAX,
                    "the index's prompt buffers");
    }
    zwide_ = true;
    zkeep_now_ = mapped_heads;
    zw_owner_ = nullptr;
}

void Inkling2::zha_narrow() {
    if (!zwide_) return;
    ck(cudaStreamSynchronize(cs(st_)), "the index's prompt buffers");
    for (int i = 0; i < (zmap_[7].base ? 8 : 7); ++i) vmm_unmap(zmap_[i], 0);
    zwide_ = false;
    zkeep_now_ = 1;
    zw_owner_ = nullptr;
    if (exp_ && zcut_ > 0) {
        size_t give = zcut_;
        if (!pool_on_) {
            const size_t free_b = vram_free_budget_bytes();
            const size_t floor_b = vram_keep_bytes();
            give = std::min(give, free_b > floor_b ? free_b - floor_b : 0);
        }
        if (give > 0) grow_cache(give, true);
        held_is_budget();
    }
    zcut_ = 0;
}

void Inkling2::zha_attention(int slot, int n, int64_t pos, const float* sink_p) {
    const Block& B = blocks_[(size_t) schedule_[(size_t) slot].phys];
    ZSlot& S = zs_[(size_t) slot];
    const int hk = n_kv_ * head_dim_, C = zha_.C;
    if (pos != S.zpos && opt_.kv_grow && S.zleaf_cap > zleaf_cap0_) {
        const int64_t want_rows = std::max<int64_t>(pos + n, read_end_);
        if (4 * want_rows <= S.zleaf_cap) {
            S.zleaf_used = std::min<int64_t>(S.zleaf_used, pos);
            int64_t cap = zleaf_cap0_;
            while (cap < want_rows && cap < opt_.max_context) cap = std::min<int64_t>(opt_.max_context, cap * 2);
            zha_leaf_recap(S, cap);
        }
    }
    zha_leaf_room(S, pos + n);
    S.zleaf_used = std::max<int64_t>(S.zleaf_used, pos + n);
    ink::store_q4(k_, S.kc, S.ksc, hk, pos, n, 0, st_);
    ink::store_q4(v_, S.vc, S.vsc, hk, pos, n, 0, st_);
    {
        const int64_t first = pos % zring_;
        const int head = (int) std::min<int64_t>(n, zring_ - first);
        store_q8(k_, S.rkc, S.rks, hk, first, head, st_);
        store_q8(v_, S.rvc, S.rvs, hk, first, head, st_);
        if (head < n) {
            store_q8(k_ + (size_t) head * hk, S.rkc, S.rks, hk, 0, n - head, st_);
            store_q8(v_ + (size_t) head * hk, S.rvc, S.rvs, hk, 0, n - head, st_);
        }
    }
    ink::ZhaKv leaves, ring;
    leaves.kc = S.kc, leaves.ks = S.ksc, leaves.vc = S.vc, leaves.vs = S.vsc;
    leaves.q4 = 1;
    ring.kc = S.rkc, ring.ks = S.rks, ring.vc = S.rvc, ring.vs = S.rvs;
    leaves.row = ring.row = (long long) hk;
    const int64_t n_ring = zring_;
    if (pos != S.zpos) {
        const int64_t refill = std::max<int64_t>(0, pos - (zring_ - rows_));
        if (refill < pos) {
            ink::q4_to_q8(S.kc, S.ksc, refill, S.rkc, S.rks, refill, zring_, hk, (int) (pos - refill), st_);
            ink::q4_to_q8(S.vc, S.vsc, refill, S.rvc, S.rvs, refill, zring_, hk, (int) (pos - refill), st_);
        }
        S.zbuilt = std::min(S.zbuilt, (int) (pos / C));
        ink::zha_cache_clear(S.zcache, n_kv_, zha_max_chunks_, st_);
        const int want = (int) (std::max<int64_t>(pos + n, read_end_) / C) + 1;
        if (zgrow_ && !zread_full_ && S.zcap > zcap0_ && 4 * want <= S.zcap) {
            int cap = zcap0_;
            while (cap < want) cap *= 2;
            zha_recap(S, std::min(cap, zha_max_chunks_));
        }
        zha_room(S, (int) (pos / C));
        ink::zha_build(S.zrefs, leaves, (int64_t) opt_.max_context + 1, S.zbuilt, (int) (pos / C), C, zha_.R, n_kv_,
                       zha_max_chunks_, S.zk, S.zv, S.zb, st_, zha_q8n(S.zcap));
        S.zbuilt = (int) (pos / C);
    }
    const int done = (int) ((pos + n) / C);
    zha_room(S, done);
    ink::zha_build(S.zrefs, ring, n_ring, S.zbuilt, done, C, zha_.R, n_kv_, zha_max_chunks_, S.zk, S.zv, S.zb, st_,
                   zha_q8n(S.zcap));
    S.zbuilt = std::max(S.zbuilt, done);
    S.zpos = pos + n;
    const bool small = zlend_ && n <= 4 && zha_pool_ > 0;
    if (zlend_ && !small && !zwide_) zha_widen(false);
    ink::ZhaWork wkp = zw_;
    if (zlend_) wkp.heads = zkeep_now_;
    const ink::ZhaWork& wk = small ? zwd_ : wkp;
    const bool keep = n >= kStageRows && zw_owner_ == &S && zw_next_ == pos;
    ink::zha_attend(blas_, zha_, q_, leaves, ring, n_ring, S.zk, S.zv, S.zb, S.zbuilt, zha_max_chunks_, pos, n, n_head_,
                    n_kv_, r_cur_, fp(B.attn_rel_bias), n_rel_extent_, n_rel_, sink_p, wk, att_, st_, &S.zcache, keep,
                    S.zcap);
    zw_owner_ = &S;
    zw_next_ = n >= kStageRows ? pos + n : -1;
}

}  // namespace strata::poly
