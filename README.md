# PolyStrata-Inkling2

**Inkling-2-Mini on one 12 GB GPU — an unofficial PolyStrata fork**

PolyStrata-Inkling2 is an unofficial fork of [PolyStrata](https://github.com/VecSzn/PolyStrata) (built on [Strata](https://github.com/Niko1221/Strata)) tuned specifically to run Thinking Machines Lab's **Inkling-2-Mini** (`26B-A3B` 48-virtual-layer looped MoE) locally on **one 12 GB GPU and 32 GB of RAM**, using CUDA (`strata-poly.exe`) with optional **DSpark MTP** speculative drafting, mixed-precision **`--kv k8v4`** / **`q4_0`** KV caching, pinned-memory doorbell asynchronous offload, whole-pass CUDA graph capture, and CPU-offloaded vision.

It is an independent fork measured carefully on one PC. It is not affiliated with Thinking Machines Lab, Unsloth, Strata, or upstream PolyStrata.

- **One-click setup:** run [START-HERE.bat](START-HERE.bat) (Windows) or `./setup.sh --family inkling2 --kv k8v4` to install Python/CUDA runtime packages and download verified GGUFs from [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF).
- **Upstream README** (the other models and general multi-model guide) is preserved at [README.upstream.md](README.upstream.md).
- **Deep-dive documentation:** [docs/INKLING2.md](docs/INKLING2.md) and [Inkling 2 Mini Benchmark.md](Inkling%202%20Mini%20Benchmark.md).

![PolyStrata Inkling 2 Benchmark Card](docs/img/inkling2-card.png)

---

## Performance Summary (RTX 4070 SUPER 12 GB + Ryzen 9 7900X + 32 GB DDR5)

Measured on Windows 11 with Visual Studio 2026 / CUDA 13.5 (`sm_89`), commit `67ebf1c`, comparing `UD-Q3_K_XL` (13.6 GB) and `UD-Q4_K_XL` (16.4 GiB, layers 0–19 NVFP4) against `llama.cpp` PR #32704:

| Benchmark / Workload | UD-Q3_K_XL (`k8v4` + DSpark) | UD-Q4_K_XL (NVFP4/Q8 + DSpark) | llama.cpp PR #32704 (`-ngl 99 -cmoe`) |
| :--- | :---: | :---: | :---: |
| **Model Size / Weights** | **13.6 GB** | 16.4 GiB | 16.4 GiB |
| **VRAM-resident experts** | **1,989 / 3,072 (65%)** | 1,520–1,613 / 3,072 (50%) | 0 / 3,072 (0 GiB) |
| **Code Generation** *(temp 0, 1500 tok, avg of 3)* | **67.5 tok/s** | **57.9 tok/s** | 13.7 tok/s *(25-tok decode)* |
| **Mixed Long Jobs** *(prose, code, tools, trans, think)* | **59.2 tok/s** | **57.1 tok/s** | *not measured* |
| **Prompt Read (32K tokens)** | **1,249 tok/s** | **1,152 tok/s** | 263 tok/s *(1.8K prompt)* |
| **Prompt Read (5.4K tokens, no MTP)** | — | **1,307 tok/s** | — |
| **Prompt Read (120K tokens)** | **847 tok/s** | — | — |
| **120K Context Decode** | **31.8 tok/s** | — | — |
| **HumanEval (Base pass@1)** | **95.7 %** (157 / 164) | **97.0 %** (159 / 164) | *not measured* |
| **HumanEval+ (EvalPlus pass@1)** | **91.5 %** (150 / 164) | **92.7 %** (152 / 164) | *not measured* |
| **Greedy Parity vs llama.cpp** | **48 / 48 (100%)** | **48 / 48 (100%)** | *reference* |
| **GPQA Diamond** *(198 questions)* | *not measured* | *not measured* | 86.6 % *(Thinking Machines' card)* |

*Speeds vary 3–8% run-to-run depending on desktop memory activity.*

---

## Memory Architecture

![Memory Layout & Generation Speed](docs/img/inkling2-summary.png)

1. **Layer Looping**: 24 physical layers executed across 48 logical steps (prelude layers 0–5, looped core layers 6–17 × 3 passes with rank-64 cross-pass gated residual controllers, epilogue layers 18–23).
2. **Asymmetric KV Cache (`--kv k8v4`)**:
   - 40 sliding-window attention (SWA) layers maintain a fixed 1,024-token ring buffer (`Q4_0` values, `Q8_0` keys).
   - Only 8 global layers attend across the full 128K context window.
   - At 128K context, the KV cache requires only **1.63 GiB** (down from 4.0 GiB in FP16), fitting comfortably in 12 GB VRAM with 1,280 MiB kept free for Windows display management.
3. **Router Cache Biasing (`--cache-bias 0.2`)**:
   - During generation, tokens routed a few at a time add `0.20` to the routing score of experts already resident in VRAM.
   - Eliminates PCIe thrashing while keeping router weights mathematically faithful to the model.
4. **Doorbell Asynchronous Offload & CUDA Graph Capture**:
   - Pinned-memory `ExpertBell` coordination allows host worker threads and GPU kernels to exchange expert fetch signals with 0.1–0.2 ms latency (down from 33.3 ms/token busy waits).
   - Single-pass CUDA graph captures all 48 execution layers without CPU-GPU synchronization bubbles.
5. **CPU Vision Offloading**:
   - The multimodal projector (`mmproj-F16.gguf`) runs via `strata-vision-inkling2.exe` on host CPU threads, keeping 100% of GPU VRAM free for language experts.

---

## Quick Start & Setup

### 1. Download Model & Drafter Weights

Weights are available on Hugging Face at [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF):
- **UD-Q3_K_XL** (recommended default for 12 GB VRAM): `Inkling-2-Mini-UD-Q3_K_XL.gguf` + `inkling2-dspark.gguf`
- **UD-Q4_K_XL** (recommended for maximum coding accuracy): `Inkling-2-Mini-UD-Q4_K_XL.gguf` + `inkling2-dspark.gguf`

### 2. Configuration Presets

#### Recommended Preset (`strata-inkling2-ud-q3_k_xl.json`):
```json
{
  "exe": "engine-poly-inkling2/strata-poly.exe",
  "args": [
    "--model", "models/inkling2-UD-Q3_K_XL/Inkling-2-Mini-UD-Q3_K_XL.gguf",
    "--draft-model", "models/inkling2-UD-Q4_K_XL/inkling2-dspark.gguf",
    "--max-context", "131072",
    "--kv", "k8v4",
    "--cache-bias", "0.2",
    "--vram-reserve-mib", "1280",
    "--prompt-batch", "512",
    "--ram-experts-mib", "-1",
    "--expert-profile", "packs/inkling2-ud-q3_k_xl/expert-profile.txt",
    "--conversation-cache-mib", "auto"
  ],
  "tokenizer": "packs/inkling2-ud-q4_k_xl/tokenizer",
  "model_name": "inkling-2-mini-ud-q3_k_xl",
  "port": 8080,
  "gpu": 0,
  "sampling": {
    "temperature": 0.0,
    "top_p": 1.0
  }
}
```

#### Starting the Server:
```bash
python -m serve.server --engine strata --config strata-inkling2-ud-q3_k_xl.json
```

Connect any OpenAI-compatible client (`127.0.0.1:8080/v1`) or Anthropic-compatible tool (`/v1/messages`).
