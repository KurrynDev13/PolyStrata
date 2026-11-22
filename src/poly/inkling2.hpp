// src/poly/inkling2.hpp - Thinking Machines Lab's Inkling 2 Mini (inkling2_loop and inkling2).
//
// 24 physical layers executed over 48 logical steps (prelude 0..5, looped core 6..17 x 3 passes, epilogue 18..23).
// 6 residual streams carry Block Attention Residuals and cross-pass loop messages across the schedule.
// 40 logical steps attend over a 1024-token sliding window; 8 steps attend globally with log-scale tau(pos).
// Causal 4-tap short convolutions run on K, V (with reversed-lag dynamic taps on global layers), attn_out, and ffn_out.
// MoE uses 128 routed experts (top-8 sigmoid routing, bias, w_scale=8.0) with SiTU activation.
#pragma once

#include "engine.hpp"

#include <vector>

namespace strata::poly {

class Inkling2 final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;
    bool sees() const override { return true; }

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    int exec_steps() const override { return (int) schedule_.size(); }
    ExecStep exec_step(int step) const override {
        const Step& s = schedule_[(size_t) step];
        return {s.phys, s.slot};
    }
    void norm_text_embed(float* dst) const override;
    size_t recurrent_floats() const override;
    bool snaps_spread() const override { return true; }
    void save_layer(int slot, int64_t pos, uint8_t* dst) override;
    size_t extra_bytes() const override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return dspark_ ? 9 : 6; }
    bool draft_at_once() const override { return dspark_; }
    bool exps_scale_any_type() const override { return true; }
    void load_drafter() override;
    void init_drafter() override;
    void reset_drafter() override;
    void draft_commit(int64_t pos, int n) override;
    void draft_block(int32_t token) override;
    void save_draft_extra(int64_t pos, uint8_t* dst) override;
    void layer_head(Layer& L, int slot, int n) override;
    void layer_tail(Layer& L, int slot, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void finish_logits(float* logits, int rows) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    void route_weights(int il, const float* logits, const int* ids, float* w) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    int head_key(int slot, int n) const override;
    const Names* tensor_alias(const char* name) const override;
    const Names* meta_alias(const char* key) const override;

private:
    struct Step {
        int phys = 0;
        int slot = 0;
        int pass = 0;
    };

    struct Block {
        bool swa = false;
        Dev wq, wk, wv, wo, wqkv_gate;
        Dev wq_fused, wkv_fused;
        int wq_fuse_flags = 0;   ///< bit 0: wq+wqkv_gate, bit 1: +attn_rel, bit 2: +attn_sink_proj
        int wq_rel_off = 0, wq_sink_off = 0;
        Dev attn_q_clip, attn_sink_proj, attn_rel, attn_rel_bias, attn_rel_clip;
        Dev attn_k_conv, attn_v_conv, attn_post_conv, ffn_post_conv;
        Dev q_inp_norm, k_inp_norm, v_inp_norm, k_conv_proj, v_conv_proj;
        Dev attn_res_query, ffn_res_query;
        // Loop controller weights (only for phys in [core_start_, core_end_))
        Dev loop_msg_query;
        Dev ctrl_down[4], ctrl_gain[4], ctrl_forget[4], ctrl_gf[4];
        std::vector<float> probs_b;
        Dev probs_b_dev;
    };

    struct SlotCache {
        bool swa = false;
        int swa_idx = -1;       ///< index among SWA slots for extra_bytes layout
        size_t state_off = 0;   ///< float offset into state_ / snap_
        uint16_t* keys = nullptr;
        uint16_t* vals = nullptr;
        int8_t* k_codes = nullptr;
        uint8_t* k_q4 = nullptr;
        uint16_t* k_scales = nullptr;
        int8_t* v_codes = nullptr;
        uint8_t* v_q4 = nullptr;
        uint16_t* v_scales = nullptr;
    };

    struct DsparkLayer {
        bool local = false;
        Dev input_norm, ff_norm, dflare_norm, q_norm, k_norm, k_norm_ctx;
        Dev q_proj, k_proj, v_proj, qkv_proj;
        Dev k_proj_ctx, v_proj_ctx, kv_proj_ctx;
        Dev o_proj, gate_proj, up_proj, gate_up_proj, down_proj;
        uint16_t *dk = nullptr, *dv = nullptr;
        int8_t *dk_codes = nullptr, *dv_codes = nullptr;
        uint8_t *dk_q4 = nullptr, *dv_q4 = nullptr;
        uint16_t *dk_scales = nullptr, *dv_scales = nullptr;
    };

    size_t kv_code_row_bytes(bool value = false) const;
    size_t kv_scale_row_bytes() const;
    size_t dspark_kv_code_row_bytes(bool value = false) const;
    size_t dspark_kv_scale_row_bytes() const;
    uint8_t* slot_k_codes(const SlotCache& sc) const;
    uint8_t* slot_v_codes(const SlotCache& sc) const;
    uint8_t* dspark_k_codes(const DsparkLayer& dl) const;
    uint8_t* dspark_v_codes(const DsparkLayer& dl) const;
    void apply_ctrl(const float* cur, const float* msg, const Block& B, int ctrl, float* out, int n);
    void store(const float* x, const SlotCache& sc, int64_t pos, int n, bool value);
    void attn(const Step& st, Block& B, SlotCache& sc, int64_t pos, int n);
    template <class F>
    void window_runs(int64_t pos, F&& f) const;

    bool root_layout_ = false;
    std::vector<Step> schedule_;
    std::vector<Block> blocks_;
    std::vector<SlotCache> slots_;

    // Global weights
    Dev tok_norm_dev_, output_res_query_;
    std::vector<float> tok_norm_host_;

    // Model parameters
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0;
    int n_rel_ = 0, n_rel_extent_ = 0, window_ = 0;
    int ring_ = 0, keep_ = 0;
    int core_start_ = 0, core_end_ = 0, n_passes_ = 0, gate_rank_ = 0;
    int block_size_ = 6, l_conv_ = 4, log_floor_ = 8192;
    float qk_clip_ = 0.f, rel_clip_ = 0.f, log_alpha_ = 0.f;
    float w_scale_ = 8.f, final_logit_softcapping_ = 0.f;
    int n_swa_slots_ = 0, n_global_slots_ = 0;
    size_t slot_conv_floats_ = 0, off_k_conv_ = 0, off_v_conv_ = 0, off_attn_post_ = 0, off_ffn_post_ = 0;
    int chunk_dec_ = 1, chunk_batch_ = 64;

    // Device scratch
    float *inputs_buf_ = nullptr, *msg_ = nullptr, *ctrl_cat_ = nullptr, *ctrl_r_ = nullptr, *ctrl_g_ = nullptr,
          *ctrl_f_ = nullptr;
    float *q_fused_buf_ = nullptr, *kv_fused_buf_ = nullptr;
    float *qraw_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *v_ = nullptr;
    float *r_raw_ = nullptr, *r_cur_ = nullptr, *dyn_ = nullptr, *sinks_ = nullptr, *gate_ = nullptr;
    float *att_ = nullptr, *table_ = nullptr, *score_ = nullptr;
    size_t score_n_ = 0;
    float* awork_ = nullptr;
    float* q8w_ = nullptr;
    size_t work_halves_ = 0;

    bool dspark_ = false;
    int dspark_layers_ = 0, dspark_ff_ = 0, dspark_heads_ = 0, dspark_kv_ = 0, dspark_head_dim_ = 0;
    int dspark_window_ = 0, dspark_ring_ = 0, dspark_block_ = 0, dspark_rank_ = 0, dspark_w1_vocab_ = 0;
    float dspark_eps_ = 1e-6f, dspark_rope_theta_ = 1000000.f, dspark_theta_scale_ = 0.f;
    int dspark_taps_[3] = {5, 23, 41};
    float dspark_fuse_[4][4] = {};
    std::vector<DsparkLayer> dspark_l_;
    Dev dspark_mask_, dspark_norm_, dspark_rnn_norm_, dspark_rnn_sp_, dspark_rnn_h_, dspark_rnn_f_, dspark_rnn_bias_;
    Dev dspark_w1_, dspark_fold_w_, dspark_fold_b_;
    std::vector<float> h_hidden_out_wt_, h_hidden_out_b_;
    mutable std::vector<float> h_dspark_anchor_;
    mutable Likeliest h_dspark_anchor_pick_{};
    float *d_taps_ = nullptr, *d_f_rnn_ = nullptr, *d_blk_x_ = nullptr, *d_blk_xn_ = nullptr, *d_blk_h_ = nullptr;
    float *d_blk_qkv_ = nullptr, *d_blk_qraw_ = nullptr, *d_blk_kraw_ = nullptr, *d_blk_attn_ = nullptr;
    float *d_blk_gate_ = nullptr, *dspark_mask_tiled_ = nullptr;
    float *d_base_logits_ = nullptr, *d_markov_bias_ = nullptr;
    float *d_rnn_in_ = nullptr, *d_rnn_gates_ = nullptr, *d_rnn_h_gates_ = nullptr, *d_rnn_f_gates_ = nullptr;
    float *d_rnn_state_ = nullptr, *d_rnn_latent_ = nullptr;
    void *d_blk_xq_ = nullptr, *d_blk_q8ff_ = nullptr;
    Likeliest* d_block_picks_ = nullptr;
    void* dspark_markov_graph_ = nullptr;
};

}  // namespace strata::poly
