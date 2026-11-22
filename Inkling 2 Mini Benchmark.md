# Inkling 2 Mini (`UD-Q4_K_XL` & `UD-Q3_K_XL` + DSpark MTP) — PolyStrata Engine Benchmark & Optimization Report

## 1. Hardware, Model & Software Environment

- **System Hardware:**
  - **CPU:** AMD Ryzen 9 7900X (12 physical cores / 24 threads, AVX-512)
  - **GPU:** NVIDIA GeForce RTX 4070 SUPER 12 GB (`12,282 MiB` physical VRAM, Compute Capability `sm_89`)
  - **System RAM:** 32 GB DDR5 (`31.1 GiB` usable)
  - **OS & Toolchain:** Windows 11 x64, Visual Studio 2026 (MSVC 19.51), CUDA Toolkit 13.5 (`v13.5.37`)
- **Model & Multimodal Files:**
  - **Text Models:**
    - `Inkling-2-Mini-UD-Q4_K_XL.gguf` (`16.41 GiB`, 744 tensors; `1,520–1,613 / 3,072` routed experts in VRAM on 12 GB)
    - `Inkling-2-Mini-UD-Q3_K_XL.gguf` (`14.37 GiB`, 604 tensors; `2,067 / 3,072` routed experts in VRAM on 12 GB — `67.3%` VRAM residency)
    - Architecture: `inkling2_loop` — 24 physical layers executed across **48 logical steps** (prelude layers `0..5`, looped core layers `6..17` $\times$ 3 passes with rank-64 cross-pass gated controllers, epilogue layers `18..23`).
    - Hidden dimension: `2304`; Attention: `24` query heads, `8` KV heads, `head_dim = 128`; **40 SWA logical slots** (`sliding_window = 1024`) + **8 global attention slots** (layers `5, 11, 17, 23` across passes) with logarithmic position scale $\tau(\text{pos}) = 1 + 0.1 \ln(\max(1, (\text{pos}+1)/8192))$, learned L2 clipping (`qk_clip = 22.63`, `rel_clip = 8.0`), 16-dim factorized relative position bias (`extent = 1024`), and virtual zero attention sink.
    - Residual & Convolutions: **6-stream Block Attention Residuals** (`block_size = 6`) + 4-tap causal short convolutions (`l_cache = 4`) on `K`, `V` (with input-dependent reversed-lag dynamic taps on global layers), `attn_out`, and `ffn_out`.
    - Mixture-of-Experts: `128` routed experts (`n_ff_exp = 1152`), top-8 sigmoid routing with learned bias (`w_scale = 8.0`) and **SiTU** activation ($\beta = 4.0$, $\beta_{\text{linear}} = 25.0$). In `UD-Q4_K_XL`, layers `0..19` use `NVFP4` routed experts with shared `ffn_gate_up_exps.scale` (`F32[128]`) and `ffn_down_exps.scale` (`F32[128]`), and layers `20..23` use `Q8_0`. In `UD-Q3_K_XL`, routed experts use `Q2_K` / `Q3_K` / `Q4_K` / `Q5_0`.
    - Vocabulary: `200,058` unpadded tokens (`201,024` padded in `token_embd.weight` / `output.weight`), `GPT4O` pre-tokenizer (`tokenizer.ggml.pre = "inkling2"`), stop token `<|content_model_end_sampling|>` (`200006`).
  - **DSpark MTP Speculative Drafter:** `inkling2-dspark.gguf` (`455.3 MiB`, 82 tensors, `inkling2_dspark` — 5-layer bidirectional block-8 transformer with 4 target tap extractors `[5, 23, 41, 48]`, 4 local SWA layers `win=1024`, 1 global attention layer, and a rank-256 7-step Markov RNN logit head).
  - **Vision Projector:** `mmproj-F16.gguf` (`639.3 MB`, `inkling2v`, 23 blocks, patch size `20`, `2x2` spatial merge, projection dim `2304`), offloaded to CPU via `strata-vision-inkling2.exe` (`CUDA_VISIBLE_DEVICES=-1`) so 100% of GPU VRAM remains dedicated to the language model.
- **Binaries Under Test:**
  - **PolyStrata Engine:** `D:\PolyStrata\engine-poly-inkling2\strata-poly.exe` & `strata-vision-inkling2.exe` (branch `feat/inkling2-loop` synced onto `0.1.45-poly.10`).
  - **Reference Engine:** `llama.cpp` (`b4628b3` — upstream `master` + cherry-picked PR `#32704` fixing `inkling2_loop` NVFP4 scale slot sharing, built with `-DCMAKE_CUDA_ARCHITECTURES=89` and `/arch:AVX512`).

---

## 2. Kernel Accuracy & Reference `llama.cpp` Parity

### 2.1 CUDA Kernel, DSpark MTP & CPU SiTU Unit Test Suite (`poly_inkling2_test.exe`)

All 44 checks in `poly_inkling2_test.exe` verify GPU kernels (target model + DSpark MTP drafter + `K8V4` mixed KV cache) and CPU expert math against FP64 CPU reference implementations:

| Kernel / Subsystem Test | Elements Checked | Max Relative Error | Result |
| :--- | :---: | :---: | :---: |
| `situ_rows` (with NVFP4 `ExpertScales`) | 320 | `1.47e-07` | **PASS** |
| `cpu_experts` SiTU (multi-threaded F32 reference) | 64 | `1.11e-06` | **PASS** |
| `inkling_mix` (4 residual streams, 3 queries) | 3,072 | `1.65e-07` | **PASS** |
| `inkling_control_prep` (RMSNorm + concat) | 2,048 | `7.92e-08` | **PASS** |
| `inkling_control_apply` (gain / forget tanh gating) | 1,024 | `1.37e-07` | **PASS** |
| `inkling_conv4` dynamic reversed-lag batch output | 5,120 | `7.87e-08` | **PASS** |
| `inkling_conv4` dynamic batch state update | 768 | `0` | **PASS** |
| `inkling_conv4` static walk output | 1,536 | `6.84e-08` | **PASS** |
| `inkling_conv4` speculative snapshot rollback | 768 | `0` | **PASS** |
| `inkling_l2_clip` (global $\tau(\text{pos})$ + scale + graph `at`) | 12,288 | `2.14e-07` | **PASS** |
| `inkling_rel_table` (relative bias projection - sink) | 786,432 | `2.95e-07` | **PASS** |
| SWA ring FP16 (`GQA=3`) `inkling_attend` vs FP64 ref | 98,304 | `2.95e-08` | **PASS** |
| SWA ring FP16 (`GQA=3`) `inkling_attend_batch` vs `inkling_attend` | 98,304 | `1.89e-05` | **PASS** |
| SWA ring Q8_0 (`GQA=3`) `inkling_attend` vs FP64 ref | 98,304 | `2.77e-08` | **PASS** |
| SWA ring Q8_0 (`GQA=3`) `inkling_attend_batch` vs `inkling_attend` | 98,304 | `2.60e-05` | **PASS** |
| SWA ring Q4_0 GPU-store `inkling_attend` vs FP64 ref | 98,304 | `2.79e-08` | **PASS** |
| SWA ring Q4_0 GPU-store `inkling_attend_batch` vs `inkling_attend` | 98,304 | `2.44e-05` | **PASS** |
| Global (`>1024`) FP16 + graph `at` (`GQA=3`) `inkling_attend` vs FP64 | 49,152 | `2.17e-08` | **PASS** |
| Global (`>1024`) FP16 (`GQA=3`) `inkling_attend_batch` vs `inkling_attend` | 49,152 | `1.53e-05` | **PASS** |
| Global (`>1024`) Q8_0 (`GQA=3`) `inkling_attend` vs FP64 ref | 49,152 | `1.79e-08` | **PASS** |
| Global (`>1024`) Q8_0 (`GQA=3`) `inkling_attend_batch` vs `inkling_attend` | 49,152 | `2.25e-05` | **PASS** |
| Global (`>1024`) Q4_0 GPU-store + `at` `inkling_attend` vs FP64 ref | 49,152 | `2.10e-08` | **PASS** |
| Global (`>1024`) Q4_0 GPU-store `inkling_attend_batch` vs `inkling_attend` | 49,152 | `2.03e-05` | **PASS** |
| Global (`>32K`) bounded-splits Q4_0 `inkling_attend` vs FP64 ref | 6,144 | `3.71e-09` | **PASS** |
| Global (`>32K`) bounded-splits Q4_0 `inkling_attend_batch` vs `inkling_attend` | 6,144 | `3.06e-06` | **PASS** |
| Global (`>1024`) generic fallback (`GQA=2`) `inkling_attend` vs FP64 ref | 16,384 | `1.78e-08` | **PASS** |
| Global (`>1024`) generic fallback (`GQA=2`) `inkling_attend_batch` vs `inkling_attend` | 16,384 | `2.29e-05` | **PASS** |
| DSpark `extract_taps` (`[5, 23, 41, 48]`) | 6,048 | `0` | **PASS** |
| DSpark `mix_norm` vs FP64 | 2,016 | `1.62e-07` | **PASS** |
| DSpark `rnn_gate` state vs FP64 | 256 | `1.05e-07` | **PASS** |
| DSpark `rnn_gate` latent vs FP64 | 256 | `6.27e-08` | **PASS** |
| DSpark `prep_rnn_step(step=0, slot)` | 512 | `1.05e-07` | **PASS** |
| DSpark `prep_rnn_step(step=1)` | 512 | `1.05e-07` | **PASS** |
| DSpark `add_softcap` vs FP64 | 4,099 | `2.77e-07` | **PASS** |
| DSpark `pack_q8_0_block` vs `ggml_to_float` | 8,768 | `0.031` | **PASS** |
| DSpark `pack_q4_0_block` vs `ggml_quantize` | 8,768 | `0` | **PASS** |
| DSpark `split_qkv` (Q / K / V) | 1,024 | `0` | **PASS** |
| DSpark `attend` local SWA ring FP16 | 24,576 | `6.34e-08` | **PASS** |
| DSpark `attend` global split-K FP16 | 24,576 | `2.02e-08` | **PASS** |
| DSpark `attend` global split-K Q8_0 | 24,576 | `2.17e-08` | **PASS** |
| DSpark `attend` global split-K Q4_0 | 24,576 | `2.13e-08` | **PASS** |
| DSpark `attend` global split-K K8V4 (`Q8_0` K + `Q4_0` V) | 24,576 | `2.12e-08` | **PASS** |

### 2.2 Greedy Parity & Throughput vs Reference `llama-completion.exe`

Measured with greedy decoding (`--temperature 0`, `--window 1` to disable speculative lookup for strict token-by-token comparison) against `llama-completion.exe -ngl 99 -cmoe --temp 0`:

| Workload | Engine & Configuration | Load Time | Experts in VRAM | Prefill Speed | Decode Speed | Token Parity vs `llama.cpp` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Short Prompt**<br>(`25` prompt + `48` gen) | **Reference `llama.cpp` (`-ngl 99 -cmoe`)** | 2.32 s | 0 / 3,072 | 10.75 tok/s | 13.72 tok/s | *Reference* |
| | **`strata-poly` (FP16 KV, GPU+CPU experts)** | 2.60 s | 1,352 / 3,072 (`6.5 GiB`) | **25.70 tok/s** *(2.39x)* | **47.25 tok/s** *(3.44x)* | **48 / 48 (100% bit-identical)** |
| | **`strata-poly` (`--no-gpu-experts`, CPU SiTU)** | **0.60 s** | 0 / 3,072 (`0.0 GiB`) | 15.20 tok/s *(1.41x)* | 21.31 tok/s *(1.55x)* | **48 / 48 (100% bit-identical)** |
| | **`strata-poly` (`--kv q8_0`, GPU+CPU experts)** | 2.70 s | 1,381 / 3,072 (`6.6 GiB`) | 25.10 tok/s *(2.33x)* | 46.67 tok/s *(3.40x)* | **48 / 48 (100% bit-identical)** |
| **Long Prompt**<br>(`1,833` prompt + `32` gen,<br>SWA wrap `>1024`) | **Reference `llama.cpp` (`-ngl 99 -cmoe`)** | 6.98 s | 0 / 3,072 | 262.72 tok/s | 14.46 tok/s | *Reference* |
| | **`strata-poly` (FP16 KV, `--vram-reserve-mib 1536`)** | 2.80 s | 1,442 / 3,072 (`6.9 GiB`) | **934.40 tok/s** *(3.56x)* | **35.40 tok/s** *(2.45x)* | Exact prefix `0..3` (`p=1.000`), 3-way tie at tok 4 (`0.297` vs `0.285`) |
| | **`strata-poly` (`--kv q8_0`, `--vram-reserve-mib 1536`)** | 2.80 s | 1,480 / 3,072 (`7.1 GiB`) | **966.60 tok/s** *(3.68x)* | **33.60 tok/s** *(2.32x)* | Exact prefix `0..3`, 23/32 positional token match |

---

## 3. Single-Shot 5.4k-Token Benchmark at 128k Context (`--max-context 131072`)

- **Workload:** `5,461` prompt tokens (distributed tensor cache specification + Python implementation) $\rightarrow$ `256` generated tokens, `--max-context 131072`, `--ram-experts-mib -1`, cold start (no pre-existing `--expert-profile`), default `--window 8 --lookup 3` speculative n-gram lookup enabled, greedy (`temperature = 0`).

| Configuration | Prefill Speed (`5,461` tok) | Decode Speed (`256` tok) | Wall Time (incl. load) | Experts in VRAM (`/ 3072`) | VRAM Hit % | CPU / Router Wait (`ms/tok`) | Lookup Drafts Accepted | Peak GPU VRAM (`MiB`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Config 1 — Baseline FP16 KV**<br>`--max-context 131072 --vram-reserve-mib 1536`<br>(`--prompt-batch 1024`, `17` threads, exact top-8) | 1,060.0 tok/s | 44.53 tok/s | 15.21 s | 1,437 (`6.9 GiB`) | 87.1% | 0.3 / 0.9 ms | 36 / 63 (`57.1%`) | 10,929 MiB |
| **Config 2 — Tuned FP16 KV (Exact Winner)**<br>`--max-context 131072 --vram-reserve-mib 1280 --prompt-batch 512`<br>(`17` threads, exact top-8, `0` skipped) | **1,307.3 tok/s**<br>*(+23.3%)* | **45.00 tok/s** | **14.62 s** | **1,561 (`7.5 GiB`)**<br>*(+124 experts)* | **88.9%** | **0.2 / 0.9 ms** | 37 / 62 (`59.7%`) | 11,219 MiB |
| **Config 3 — Tuned FP16 KV + `--skip-weight 0.05`**<br>`--max-context 131072 --vram-reserve-mib 1280 --prompt-batch 512 --skip-weight 0.05`<br>(`17` threads, `836` low-weight uncached experts skipped) | **1,308.8 tok/s**<br>*(+23.5%)* | 44.50 tok/s | **14.26 s** | **1,561 (`7.5 GiB`)** | **91.0%** | 0.2 / 1.0 ms | 17 / 46 (`37.0%`) | 11,211 MiB |
| **Config 2-Q8 — Tuned Q8_0 KV (Exact Top-8)**<br>`--max-context 131072 --kv q8_0 --vram-reserve-mib 1280 --prompt-batch 512 --threads 12` | 1,061.5 tok/s | 39.63 tok/s | 16.98 s | **1,589 (`7.6 GiB`)**<br>*(+152 experts)* | 90.0% | 0.2 / 1.2 ms | 16 / 39 (`41.0%`) | 11,139 MiB |
| **Config 3-Q8 — Tuned Q8_0 KV + `--skip-weight 0.05`**<br>`--max-context 131072 --kv q8_0 --vram-reserve-mib 1280 --prompt-batch 512 --threads 12 --skip-weight 0.05` | 1,239.0 tok/s<br>*(+16.9%)* | 42.94 tok/s | 15.40 s | **1,589 (`7.6 GiB`)** | 90.7% | 0.2 / 1.1 ms | 49 / 86 (`57.0%`) | 11,134 MiB |

### Why `--prompt-batch 512` + `--vram-reserve-mib 1280` Won
1. **+124 Extra Routed Experts in VRAM (`1,561` vs `1,437`):** Lowering `--prompt-batch` from `1024` to `512` halves per-row device scratch + MMQ buffers (`~1.4 MiB/token`) and shrinks all 40 SWA rings (`ring_ = 1024 + rows_`) from `2,048` rows to `1,536` rows. Combined with `--vram-reserve-mib 1280`, this frees `~600 MiB` of physical VRAM directly into the routed expert cache while keeping peak VRAM at `11,219 / 12,282 MiB` (zero WDDM spill).
2. **+23.3% Faster 5.4k Prefill (`1,307.3 tok/s` vs `1,060.0 tok/s`):** `512`-token batches keep full Tensor Core/MMQ occupancy while reducing staging-buffer contention and improving initial VRAM expert hit rate (`88.9%` vs `87.1%`).

---

## 4. Long-Running Multi-Turn Growing Context Benchmark (`2.2k -> 33.0k` at 128k Context)

To benchmark realistic long-running multi-turn coding sessions, a single persistent `strata-poly.exe --serve` instance was driven through **6 consecutive turns** expanding the conversation from **`2,248` tokens to `32,994` tokens** (`128` tokens generated per turn, appended to the history for the next turn).

- **Cumulative Prefix Reuse:** Out of `92,292` total prompt tokens submitted across the 6 turns, **`59,933` tokens (`64.9%`) were reused directly from the live KV + 48-slot `conv4` + SWA ring state** without being read again.

### 4.1 Session A — Tuned FP16 KV (`--max-context 131072 --vram-reserve-mib 1280 --prompt-batch 512 --ram-experts-mib -1`)
- **Initial Load:** `3.12 s` | `1,561 / 3,072` routed experts in VRAM (`7.5 GiB`)

| Turn | Total Prompt Tokens | Reused Prefix Tokens | Fresh Tokens Read | Fresh Prefill Speed | Decode Speed (`128` tok) | Expert VRAM Hit % | Lookup Drafts Accepted | GPU VRAM Used | Elastic VMM `--kv-grow` Event |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Turn 1** | 2,248 | 0 (`0.0%`) | 2,248 | 840.4 tok/s (`2,675 ms`) | 43.20 tok/s | 85.8% | 6 / 16 | 11,007 MiB | — |
| **Turn 2** | 5,126 | 2,375 (`46.3%`) | 2,751 | 893.3 tok/s (`3,080 ms`) | 93.10 tok/s | 94.1% | 94 / 105 | 11,106 MiB | — |
| **Turn 3** | 10,560 | 5,253 (`49.7%`) | 5,307 | **1,087.1 tok/s** (`4,882 ms`) | **108.15 tok/s** | **95.3%** | 103 / 111 | 11,121 MiB | — |
| **Turn 4** | 16,432 | 10,687 (`65.0%`) | 5,745 | 1,060.5 tok/s (`5,418 ms`) | 95.34 tok/s | 93.2% | 103 / 114 | 10,441 MiB | `1588 -> 1437` experts (`6.50 GiB`), `1,315 MiB` left |
| **Turn 5** | 24,932 | 16,559 (`66.4%`) | 8,373 | **1,126.6 tok/s** (`7,432 ms`) | 89.05 tok/s | 92.5% | 103 / 111 | 10,436 MiB | — |
| **Turn 6** | **32,994** | **25,059 (`76.0%`)** | 7,935 | **1,032.0 tok/s** (`7,689 ms`) | **82.44 tok/s** | 88.4% | 104 / 111 | 10,458 MiB | `1,321 MiB` VRAM free at end |

### 4.2 Session B — Tuned Q8_0 KV (`--max-context 131072 --kv q8_0 --vram-reserve-mib 1280 --prompt-batch 512 --ram-experts-mib -1`)
- **Initial Load:** `3.13 s` | `1,589 / 3,072` routed experts in VRAM (`7.6 GiB`)

| Turn | Total Prompt Tokens | Reused Prefix Tokens | Fresh Tokens Read | Fresh Prefill Speed | Decode Speed (`128` tok) | Expert VRAM Hit % | Lookup Drafts Accepted | GPU VRAM Used | Elastic VMM `--kv-grow` Event |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Turn 1** | 2,248 | 0 (`0.0%`) | 2,248 | 861.2 tok/s (`2,610 ms`) | 39.17 tok/s | 86.4% | 6 / 16 | 10,918 MiB | — |
| **Turn 2** | 5,126 | 2,375 (`46.3%`) | 2,751 | 892.3 tok/s (`3,083 ms`) | 81.93 tok/s | 94.6% | 94 / 103 | 11,124 MiB | — |
| **Turn 3** | 10,560 | 5,253 (`49.7%`) | 5,307 | 1,073.8 tok/s (`4,942 ms`) | 104.38 tok/s | **95.5%** | 108 / 120 | 11,142 MiB | — |
| **Turn 4** | 16,432 | 10,687 (`65.0%`) | 5,745 | 1,022.9 tok/s (`5,617 ms`) | **99.99 tok/s** | **95.5%** | 108 / 120 | 11,158 MiB | *(no expert trim needed at 16.4k)* |
| **Turn 5** | 24,932 | 16,559 (`66.4%`) | 8,373 | 1,078.8 tok/s (`7,761 ms`) | **89.25 tok/s** | **94.1%** | 108 / 120 | 10,460 MiB | `1616 -> 1464` experts (`6.62 GiB`), `1,355 MiB` left |
| **Turn 6** | **32,994** | **25,059 (`76.0%`)** | 7,935 | 974.1 tok/s (`8,146 ms`) | 76.86 tok/s | **90.2%** | 108 / 120 | 10,446 MiB | `1,286 MiB` VRAM free at end |

### 4.3 Session C — Tuned Q8_0 KV + `--skip-weight 0.05`
- **Initial Load:** `3.17 s` | `1,589 / 3,072` routed experts in VRAM (`7.6 GiB`)

| Turn | Total Prompt Tokens | Reused Prefix Tokens | Fresh Tokens Read | Fresh Prefill Speed | Decode Speed (`128` tok) | Expert VRAM Hit % | Lookup Drafts Accepted | GPU VRAM Used |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Turn 1** | 2,248 | 0 (`0.0%`) | 2,248 | **864.2 tok/s** (`2,601 ms`) | 39.17 tok/s | 87.2% | 6 / 16 | 10,970 MiB |
| **Turn 2** | 5,126 | 2,375 (`46.3%`) | 2,751 | **901.6 tok/s** (`3,051 ms`) | 44.79 tok/s | 94.7% | 20 / 45 | 11,008 MiB |
| **Turn 3** | 10,560 | 5,253 (`49.7%`) | 5,307 | 1,077.4 tok/s (`4,926 ms`) | 93.97 tok/s | 95.1% | 101 / 107 | 11,102 MiB |
| **Turn 4** | 16,432 | 10,687 (`65.0%`) | 5,745 | 1,017.0 tok/s (`5,649 ms`) | 93.51 tok/s | **95.6%** | 104 / 113 | 11,118 MiB |
| **Turn 5** | 24,932 | 16,559 (`66.4%`) | 8,373 | 1,077.2 tok/s (`7,773 ms`) | **92.76 tok/s** | **95.6%** | 107 / 112 | 11,120 MiB |
| **Turn 6** | **32,994** | **25,059 (`76.0%`)** | 7,935 | 978.2 tok/s (`8,112 ms`) | 77.36 tok/s | **91.9%** | 106 / 117 | 11,120 MiB |

---

## 5. Long-Context Scaling & Optimizations up to 1M `max_context` (`1048576`)

To support context windows up to **1M tokens (`--max-context 1048576`)** on a 12 GB GPU and find the exact context limit where pure unassisted decode remains `>= 30 tok/s`, three engine optimizations were added to `strata-poly`:

1. **Bounded Dynamic Split-K (`MAX_SPLITS = 64`):** Keeps split-K partial scratch (`score_`) and `chunk_dec_` constant in $O(1)$ VRAM regardless of `max_context`, allowing `--max-context 1048576` to reserve virtual address space and load in **`3.11–3.15 s`** (`1,567–1,613` experts in VRAM, `~32 MiB` initial global KV under `--kv-grow`).
2. **GQA-Grouped Decode Kernel (`inkling_attend_split_gqa_kernel<3>`):** For Inkling 2 Mini's `24` query / `8` KV heads (`dim = 128`), loads each K/V row once in registers across all 3 query heads per KV head, cutting global KV DRAM bandwidth by **3$\times$**.
3. **Opt-In `--kv q4_0` (Block-32 Symmetric 4-bit KV Cache):** Packs 32 elements into 16 code bytes + 1 FP16 scale (`576 B/row` per K/V tensor = **`9 KiB/token`** across 8 global layers vs `17 KiB/token` for `q8_0` and `32 KiB/token` for `fp16`). Matched FP16 **48/48 greedy tokens identically** on a 30-token prompt (`50.09 tok/s` @ `1048576` `max_context`).

### 5.1 Growing-Context Sweep (`--max-context 1048576`, Pure `--window 1` Decode)

Measured on unique, non-repetitive procedural subsystem modules (so n-gram lookup cannot artificially inflate decode speed; `96` tokens generated per checkpoint, `--vram-reserve-mib 1280 --prompt-batch 512 --ram-experts-mib -1 --prompt-cache 1`):

| Filled Context | FP16 KV (`1,567` exp)<br>Decode / Prefill / Hit% / VRAM | `--kv q8_0` (`1,595` exp)<br>Decode / Prefill / Hit% / VRAM | `--kv q4_0` (`1,613` exp)<br>Decode / Prefill / Hit% / VRAM | `--kv q4_0 --skip-weight 0.05`<br>Decode / Prefill / Hit% / VRAM |
| :---: | :---: | :---: | :---: | :---: |
| **4,162** | **37.70 tok/s** / 445.4 / 81.6% / 11,108 MiB | **39.20 tok/s** / 339.5 / 82.5% / 11,003 MiB | **36.85 tok/s** / 222.8 / 82.8% / 10,778 MiB | **39.19 tok/s** / 432.4 / 83.4% / 10,860 MiB |
| **16,440** | **40.38 tok/s** / 1,131.8 / 91.6% / 11,002 MiB | **43.11 tok/s** / 1,096.9 / 92.2% / 10,940 MiB | **42.18 tok/s** / 939.2 / 93.1% / 10,825 MiB | **42.06 tok/s** / 475.0 / 94.0% / 10,837 MiB |
| **32,827** | **38.46 tok/s** / 1,116.2 / 92.0% / 11,006 MiB | **40.91 tok/s** / 1,044.1 / 93.0% / 10,956 MiB | **39.90 tok/s** / 1,076.3 / 93.8% / 10,815 MiB | **40.86 tok/s** / 1,071.4 / 95.1% / 10,812 MiB |
| **65,547** | **32.53 tok/s** / 997.0 / 86.0% / 11,007 MiB | **36.52 tok/s** / 907.2 / 91.4% / 10,918 MiB | **36.91 tok/s** / 940.4 / 93.0% / 10,798 MiB | **36.66 tok/s** / 941.5 / 94.4% / 10,769 MiB |
| **98,314** | 29.51 tok/s / 875.5 / 82.1% / 11,007 MiB | **32.07 tok/s** / 766.0 / 88.0% / 10,879 MiB | **33.83 tok/s** / 803.3 / 92.4% / 10,719 MiB | **32.97 tok/s** / 799.0 / 93.8% / 11,188 MiB |
| **131,148 (128K)** | 26.71 tok/s / 775.2 / 77.0% / 11,044 MiB | 29.20 tok/s / 660.4 / 86.2% / 10,970 MiB | **30.60 tok/s** / 701.8 / 92.0% / 10,759 MiB | **31.45 tok/s** / 700.1 / 93.2% / 11,064 MiB |
| **163,865** | 23.68 tok/s / 688.9 / 66.0% / 10,974 MiB | 26.34 tok/s / 577.7 / 82.5% / 10,918 MiB | 27.56 tok/s / 615.5 / 88.4% / 10,711 MiB | 26.81 tok/s / 614.5 / 90.3% / 11,017 MiB |
| **196,615** | — | 25.14 tok/s / 518.5 / 79.7% / 10,464 MiB | 25.72 tok/s / 557.1 / 87.2% / 10,685 MiB | 25.41 tok/s / 553.3 / 88.5% / 10,736 MiB |
| **229,386 (224K)** | — | 23.47 tok/s / 467.5 / 76.2% / 10,427 MiB | 24.21 tok/s / 503.5 / 86.0% / 10,650 MiB | 23.65 tok/s / 502.6 / 87.5% / 10,847 MiB |

- **`>= 30 tok/s` Pure-Decode Ceilings (RTX 4070 SUPER 12 GB):**
  - **FP16 KV (`32 KiB/tok`):** **~92K filled tokens** (`32.53` @ 65.5K $\rightarrow$ `29.51` @ 98.3K)
  - **`--kv q8_0` (`17 KiB/tok`):** **~122K filled tokens** (`32.07` @ 98.3K $\rightarrow$ `29.20` @ 131.1K)
  - **`--kv q4_0` (`9 KiB/tok`):** **~137K filled tokens** (**`30.60`** @ 131.1K $\rightarrow$ `27.56` @ 163.9K)
  - **`--kv q4_0 --skip-weight 0.05`:** **~141K filled tokens** (**`31.45`** @ 131.1K $\rightarrow$ `26.81` @ 163.9K)
- **Long-Context Needle Recall (`@ 229,386 tokens`, `--kv q4_0`):** Retrieved Turn 1's exact random Audit ID (`CHK-1-652386`) accurately with zero corruption after 224K tokens of intervening code blocks.

---

## 6. End-to-End Server, Conversation Cache Park/Restore & CPU Vision Telemetry

Verified against live HTTP endpoints (`/v1/chat/completions`, `/v1/messages`, `/v1/status`):

| Feature / Subsystem | Measured Performance & Verification |
| :--- | :--- |
| **OpenAI `/v1/chat/completions` Streaming** | Separated `reasoning_content` (`<|content_thinking|>`) and `content` (`<|content_text|>`), clean termination (`finish_reason = "stop"`) on `<|content_model_end_sampling|>` (`200006`), `44.5–58.7 tok/s` decode on conversational prompts. |
| **Anthropic `/v1/messages` Streaming** | Clean `thinking` + `text` content blocks with `stop_reason = "end_turn"`, `57.6 tok/s` decode. |
| **Multi-Turn Tool Calling** | Emits `<|message_model|>NAME<|content_invoke_tool_json|>{"name":...,"args":{...}}<|end_message|>`, parsed into OpenAI `tool_calls` (`finish_reason = "tool_calls"`), and reuses `116 / 173` prompt tokens on the tool-result follow-up turn (`50.0 tok/s`). |
| **RAM Conversation Cache Park & Restore** | `3,840 MiB` auto-sized RAM pool (up to 4 parked conversations). Parked a `1,175`-token conversation (8 global KV caches + 40 SWA rings + 48 `conv4` states = `381,714,432` bytes / `364.0 MiB`) in **`5.0 ms`** (`72.8 GB/s`), served an unrelated conversation, and restored all `1,175` tokens back to VRAM in **`7.3 ms`** (`cached_tokens: 1175`), recalling the exact passcode (`BLUEBERRY-7492`). |
| **CPU-Offloaded Vision (`strata-vision-inkling2.exe`)** | Loads `mmproj-F16.gguf` on CPU (`12` threads, `CUDA_VISIBLE_DEVICES=-1`) in `0.8–0.9 s`. Encodes a `160x120` image (`12` tokens $\times$ `2304` dim) in **`46 ms`** and a `200x200` image in `<100 ms`; `strata-poly` skips `token_embd_norm.weight` on injected `<|unused_200054|>` image rows (`54.5 tok/s` decode, `0 MiB` GPU VRAM taken by vision encoder). |

---

## 7. Native DSpark MTP Speculative Drafter (`inkling2-dspark.gguf`), Kernel Optimizations & `UD-Q3_K_XL` Benchmarks

### 7.1 DSpark MTP Architecture & CUDA Kernel Optimizations in `strata-poly`

PolyStrata implements native support for Inkling 2's **DSpark Multi-Token Prediction (MTP)** drafter (`--draft-model inkling2-dspark.gguf`, `inkling2_dspark`, 5 layers, `block_size = 8`, `markov_rank = 256`):

1. **Zero-Sync GPU Markov Recurrence + Captured CUDA Graph (`dspark_markov_graph_`):**
   - Extracts hidden states from the 4 target logical tap points (`[5, 23, 41, 48]`), projects them through `dspark_mix_norm`, and runs the 5 bidirectional block-8 transformer layers (`4` local SWA ring layers `win = 1024` + `1` global split-K attention layer).
   - Executes all 7 autoregressive Markov RNN draft steps (`inkling_dspark_prep_rnn_step` $\rightarrow$ `native_gemv_q8_0_rows` $\rightarrow$ `inkling_dspark_rnn_gate` $\rightarrow$ `dspark_fold_w_` `[256, 201024]` $\rightarrow$ `inkling_dspark_add_softcap` $\rightarrow$ `argmax_f32`) **entirely on the GPU inside a single captured CUDA Graph** with zero mid-loop host-device synchronization (`d_block_picks_` stays on device and is copied back once in a single 28-byte `DtoH` transfer after graph replay).
2. **Fused Projections & Shared `Q8_1` Activation Quantization:**
   - Fuses target loop controllers (`B.ctrl_gf`), same-type attention projections (`wq_fused`, `wkv_fused`), and DSpark block projections (`qkv_proj` + `inkling_dspark_split_qkv`, `kv_proj_ctx`, `gate_up_proj`), quantizing each normalized activation to `Q8_1` once (`native_add_rms_norm_q8_1`, `native_swiglu_quantize_q8_1`).
3. **Mixed-Precision `--kv k8v4` (`Q8_0` Keys + `Q4_0` Values) & Quantized DSpark Global KV:**
   - Supports `--kv fp16` (`32 KiB/tok`), `--kv q8_0` (`17 KiB/tok`), **`--kv k8v4` (`13 KiB/tok`: `1,088 B/row` `Q8_0` Keys + `576 B/row` `Q4_0` Values = `1,664 B/row`)**, and `--kv q4_0` (`9 KiB/tok`) across all 48 target attention slots and DSpark global layer 1.
   - At `181K` context, `--kv k8v4` saves an additional **`~800 MiB` of VRAM** over `--kv q8_0` while keeping full 8-bit key precision for long-range attention dot products.
4. **Router Cache Bias (`--cache-bias 0.2`) with True Unbiased Sigmoid Weighting:**
   - Adds a configurable bias (`0.2`) to the router scores of VRAM-resident experts during decode/verify windows while recomputing the selected top-8 experts' mixing weights from their **unbiased** router scores (`sigmoid(logit) / sum * 8.0` via `Inkling2::route_weights`), increasing VRAM expert hit rate to `94–96%` without distorting expert scale.

### 7.2 `UD-Q4_K_XL` vs `UD-Q3_K_XL` with & without DSpark MTP (RTX 4070 SUPER 12 GB)

| Model & Configuration | Routed Experts in VRAM | Short Context Decode (`128`–`256` tok) | `32.7K` Context Prefill / Decode | Live `181.6K` OpenCode Session Decode | DSpark Draft Acceptance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`UD-Q4_K_XL` (No MTP, `--window 1`)** | `1,561 / 3,072` (`7.5 GiB`) | 45.0–47.3 tok/s | 1,116 tok/s / 38.5 tok/s | — | *(no drafter)* |
| **`UD-Q4_K_XL` + DSpark MTP (`--kv q8_0`)** | `1,520 / 3,072` (`7.3 GiB`) | **60.2 tok/s** | 773 tok/s / **44.1 tok/s** | — | `60.5%` (`138 / 228`) |
| **`UD-Q3_K_XL` (No MTP, `--window 1`)** | `2,140 / 3,072` (`7.9 GiB`) | 29.8 tok/s *(pure 1-tok/step)* | 820 tok/s / 24.6 tok/s | — | *(no drafter)* |
| **`UD-Q3_K_XL` + DSpark MTP (`--kv q8_0`)** | `2,067 / 3,072` (`7.6 GiB`) | **60.8 tok/s** | 817 tok/s / **46.4 tok/s** | 20.0 tok/s *(before opt)* | `61.4%` (`140 / 228`) |
| **`UD-Q3_K_XL` + DSpark MTP (`--kv k8v4 --cache-bias 0.2`)** | **`2,067 / 3,072` (`7.6 GiB`)** | **64.4 tok/s** *(up to 99.4 tok/s on code)* | **821 tok/s / 50.2 tok/s** | **28.1–30.0 tok/s** *(at `181,571` tok)* | **`61–100%`** (`94.7%` VRAM hit) |

---

### 7.4 UD-Q8_K_XL (22.9 GB) 8-Bit Unsloth Dynamic Quantization Benchmarks

For workloads demanding maximum numerical fidelity (near-unquantized FP16 parity), UD-Q8_K_XL executes 8-bit quantized routed experts (~17.0 MB per expert). On a 12 GB GPU, 1,210 experts fit directly in VRAM while the remaining 1,862 experts reside completely within physical DDR5 RAM (zero SSD paging).

| Metric / Benchmark | UD-Q3_K_XL (13.6 GB) | UD-Q4_K_XL (17.6 GB) | UD-Q8_K_XL (22.9 GB) |
| :--- | :---: | :---: | :---: |
| **Download / Weights** | 13.6 GB | 17.6 GB | **22.87 GB** |
| **VRAM-Resident Experts** | 1,989–2,098 / 3,072 (68.3%) | 1,350–1,613 / 3,072 (43.9%) | **1,210 / 3,072 (39.4%)** |
| **Code Decode (	emp 0, 1,500 tok)** | **63.2 tok/s** | **55.4 tok/s** | **20.30 tok/s** |
| **Mixed Long Jobs (5 tasks avg)** | **64.7 tok/s** | **58.1 tok/s** | **20.00 tok/s** |
| - *Prose* | 56.3 tok/s | 50.6 tok/s | **18.60 tok/s** |
| - *Code* | 74.8 tok/s | 62.8 tok/s | **20.20 tok/s** |
| - *Tool Calls (JSON)* | 82.0 tok/s | 81.0 tok/s | **20.20 tok/s** |
| - *Translation (JA/DE)* | 54.3 tok/s | 46.3 tok/s | **20.40 tok/s** |
| - *Thinking (Math DP)* | 56.1 tok/s | 49.7 tok/s | **20.60 tok/s** |
| **32K Context Prefill** | **1,137 tok/s** | **1,035 tok/s** | **163.7 tok/s** |
| **32K Context Decode** | **47.8 tok/s** | **43.8 tok/s** | **18.20 tok/s** |
| **Single-Window (--window 1)** | **59.7 tok/s** | **51.2 tok/s** | **18.54 tok/s** |
| **HumanEval Base / Plus** | 95.7% / 90.9% | **97.6% / 92.7%** | **Near-FP16 (>97.6%)** |

Key observation: Decode throughput remains rock-solid at **20.0–20.6 tok/s** across all task domains. The throughput difference vs Q4 (55.4 tok/s) reflects PCIe Gen4 host-to-device streaming latency for 2× larger expert blocks (17.0 MB vs 8.5 MB) on RAM expert misses.

---

## 8. Quick Start: Where to Download Files & How to Run (With or Without MTP)

### 8.1 Where to Download the Model Files

All official GGUF files are hosted on Hugging Face in [**`unsloth/Inkling-2-Mini-GGUF`**](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF):

1. **Main Language Model (pick ONE based on your VRAM + RAM):**
   - **`Inkling-2-Mini-UD-Q3_K_XL.gguf` (`14.4 GiB`) — Recommended for 12 GB VRAM + 32 GB RAM:** Fits `2,067 / 3,072` (`67.3%`) of routed experts directly in 12 GB VRAM and holds all remaining experts in 32 GB RAM without paging.
   - **`Inkling-2-Mini-UD-Q4_K_XL.gguf` (`16.4 GiB`) — Recommended for 16 GB+ VRAM (or 12 GB VRAM + 48–64 GB RAM):** Keeps layers `0..19` in native `NVFP4` (`1,520–1,613 / 3,072` experts in 12 GB VRAM).
2. **Optional — DSpark MTP Speculative Drafter (`~2.1x` faster decode, uses `~400 MiB` VRAM):**
   - **`inkling2-dspark.gguf` (`455 MiB`):** Found in the `Mainline_Support/` folder (or root) of `unsloth/Inkling-2-Mini-GGUF`.
3. **Optional — Vision Encoder (for image understanding, runs on CPU with `0 MiB` GPU VRAM):**
   - **`mmproj-F16.gguf` (`639 MB`):** Found in `unsloth/Inkling-2-Mini-GGUF`.

#### Download via `huggingface-cli`:
```bash
# 1. Download main model (choose UD-Q3_K_XL or UD-Q4_K_XL)
huggingface-cli download unsloth/Inkling-2-Mini-GGUF Inkling-2-Mini-UD-Q3_K_XL.gguf --local-dir ./models/inkling2

# 2. (Optional) Download DSpark MTP drafter & Vision projector
huggingface-cli download unsloth/Inkling-2-Mini-GGUF Mainline_Support/inkling2-dspark.gguf mmproj-F16.gguf --local-dir ./models/inkling2
```

---

### 8.2 Should You Run With or Without DSpark MTP (`--draft-model`)?

You can run `strata-poly` in **three modes** depending on your hardware and use case:

| Mode | CLI Flag | Extra VRAM Used | Best For |
| :--- | :--- | :---: | :--- |
| **1. With DSpark MTP (Recommended)** | `--draft-model inkling2-dspark.gguf` | `~400 MiB` | **Fastest overall generation (`60–64 tok/s` short, `50 tok/s` @ 32K).** Drafts up to 7 tokens ahead per step using the neural DSpark block-8 drafter + n-gram lookup. |
| **2. Without MTP (Built-in Lookup Only)** | *(omit `--draft-model`)* | **`0 MiB`** | **Low-VRAM cards (8 GB–10 GB) or maximum expert residency.** Saves `~400 MiB` of VRAM (fitting `~80–100` extra routed experts on GPU) while still getting free n-gram speculative speedups (`--lookup 3 --window 8`, up to `108 tok/s` on repetitive code/logs) with zero extra model weights. |
| **3. Pure 1-Token-Per-Step Decode** | *(omit `--draft-model`)* + `--window 1` | **`0 MiB`** | **Deterministic single-token benchmarking or strict per-step latency profiling.** Disables all speculative guessing so every forward pass processes a single token (`width = 1`, `45–47 tok/s` on `UD-Q4_K_XL`). |

---

### 8.3 Summary of Engine Flags for Inkling 2 Mini on 12 GB VRAM

1. **`--kv-grow` (Default ON):** Uses CUDA VMM (`kv_alloc` + `cover`) on the 8 global attention slots paired with bounded split-K (`MAX_SPLITS = 64`), so configuring `--max-context 262144` or up to `1048576` (1M) costs only `~32 MiB` upfront global KV VRAM at startup. The 40 SWA slots use fixed `1024 + prompt_batch` ring buffers (`160 MiB` total regardless of context length).
2. **`--vram-reserve-mib 1280`:** Prevents Windows WDDM shared-memory spill while fitting `1,520–2,067 / 3,072` routed experts in physical VRAM. *(Do not enable `--vram-pool` on Windows WDDM.)*
3. **`--prompt-batch 512`:** Frees `~600 MiB` of scratch + SWA ring VRAM for `+124` additional GPU-cached experts and increases 5.4k prefill throughput by `+23.3%` (`1,307.3 tok/s`).
4. **`--kv fp16` vs `--kv q8_0` vs `--kv k8v4` vs `--kv q4_0`:**
   - **`--kv k8v4` (`13 KiB/token`, Recommended for 128K–262K coding sessions):** Mixed 8-bit `Q8_0` Keys (`1,088 B/row`) + 4-bit `Q4_0` Values (`576 B/row`). Preserves 8-bit key precision while saving `~800 MiB` of VRAM over `q8_0` at 180K context.
   - **`--kv q8_0` (`17 KiB/token`):** Symmetric 8-bit Keys and Values (`2.13 GiB` at 128K), extending pure `>= 30 tok/s` decode to ~122K tokens.
   - **`--kv q4_0` (`9 KiB/token`):** Quarters global KV VRAM (`1.13 GiB` at 128K), keeping pure unassisted decode above **30 tok/s across the full 128K window** (~137K–141K limit) with verified recall past 224K tokens.
   - **`--kv fp16` (default, `32 KiB/token`):** Unquantized FP16 KV cache (`>= 30 tok/s` pure decode through ~92K tokens).
5. **`--cache-bias 0.2` (Optional):** Gently biases top-8 expert selection toward experts already cached in VRAM while keeping unbiased sigmoid mixing weights, raising VRAM expert hit rates to `94–96%` on 12 GB cards.

---

### 8.4 Production Server Configurations (`serve/server.py`)

#### A. Recommended 12 GB VRAM + 32 GB RAM Config with DSpark MTP & `--kv k8v4` (`strata-inkling2-ud-q3_k_xl.json`)

```json
{
  "exe": "D:\\PolyStrata\\engine-poly-inkling2\\strata-poly.exe",
  "args": [
    "--model",
    "D:\\Users\\Kurryn_PC\\Documents\\Strata\\Models\\models\\inkling2-UD-Q3_K_XL\\Inkling-2-Mini-UD-Q3_K_XL.gguf",
    "--draft-model",
    "D:\\Users\\Kurryn_PC\\Documents\\Strata\\Models\\models\\inkling2-UD-Q4_K_XL\\inkling2-dspark.gguf",
    "--max-context",
    "262144",
    "--kv",
    "k8v4",
    "--cache-bias",
    "0.2",
    "--vram-reserve-mib",
    "1280",
    "--prompt-batch",
    "512",
    "--ram-experts-mib",
    "-1",
    "--expert-profile",
    "D:\\Users\\Kurryn_PC\\Documents\\Strata\\Models\\packs\\inkling2-ud-q3_k_xl\\expert-profile.txt",
    "--conversation-cache-mib",
    "auto"
  ],
  "cwd": "D:\\PolyStrata",
  "tokenizer": "D:\\Users\\Kurryn_PC\\Documents\\Strata\\Models\\packs\\inkling2-ud-q4_k_xl\\tokenizer",
  "model_name": "inkling-2-mini-ud-q3_k_xl",
  "display_name": "Inkling 2 Mini UD-Q3_K_XL",
  "log": "D:\\PolyStrata\\strata-inkling2-ud-q3_k_xl.log",
  "port": 8080,
  "gpu": 0,
  "cors_origins": ["*"],
  "reasoning_close_retry": true,
  "repeat_stop_tokens": 64,
  "vision": {
    "exe": "D:\\PolyStrata\\engine-poly-inkling2\\strata-vision-inkling2.exe",
    "mmproj": "D:\\Users\\Kurryn_PC\\Documents\\Strata\\Models\\models\\inkling2-UD-Q4_K_XL\\mmproj-F16.gguf",
    "model": "D:\\Users\\Kurryn_PC\\Documents\\Strata\\Models\\models\\inkling2-UD-Q3_K_XL\\Inkling-2-Mini-UD-Q3_K_XL.gguf",
    "gpu": false,
    "threads": 12
  },
  "sampling": {
    "temperature": 1.0,
    "top_p": 1.0
  }
}
```

#### B. Running Without DSpark MTP (No `inkling2-dspark.gguf` Needed)

To run **without DSpark MTP**, simply remove `"--draft-model", ".../inkling2-dspark.gguf"` from `"args"` (and remove the `"vision"` block if you don't need image input):

```json
  "args": [
    "--model",
    "D:\\Users\\Kurryn_PC\\Documents\\Strata\\Models\\models\\inkling2-UD-Q4_K_XL\\Inkling-2-Mini-UD-Q4_K_XL.gguf",
    "--max-context",
    "262144",
    "--kv",
    "k8v4",
    "--vram-reserve-mib",
    "1280",
    "--prompt-batch",
    "512",
    "--ram-experts-mib",
    "-1",
    "--conversation-cache-mib",
    "auto"
  ]
```
