# PolyStrata-Inkling2

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Türkçe](README.tr.md)

**Inkling-2-Mini on one 12 GB GPU — an unofficial PolyStrata fork**

![PolyStrata-Inkling2: UD-Q3_K_XL + DSpark against llama.cpp PR #32704 — Code, Mixed long jobs, 32K Prompt read, HumanEval+, GPQA Diamond](docs/img/inkling2-card.png)

PolyStrata-Inkling2 is an unofficial fork of [PolyStrata](https://github.com/VecSzn/PolyStrata) (built on [Strata](https://github.com/Niko1221/Strata)) tuned specifically to run Thinking Machines Lab's **Inkling-2-Mini** (`26B-A3B` 48-virtual-layer looped MoE) locally on **one 12 GB GPU and 32 GB of RAM**, using CUDA (`strata-poly.exe`) with optional **DSpark MTP** speculative drafting, mixed-precision **`--kv k8v4`** / **`q4_0`** KV caching, and CPU-offloaded vision.

It is an independent fork measured carefully on one PC. It is not affiliated with Thinking Machines Lab, Unsloth, Strata, or upstream PolyStrata.

- **One-click setup:** run [START-HERE.bat](START-HERE.bat) (Windows) or `./setup.sh --family inkling2 --kv k8v4` to install Python/CUDA runtime packages and download verified GGUFs from [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF).
- **Upstream README** (the other 12 models and general multi-model guide) is preserved at [README.upstream.md](README.upstream.md).
- **Deep-dive measurements & kernel logs:** [Inkling 2 Mini Benchmark.md](Inkling%202%20Mini%20Benchmark.md).
- Speeds below are software-only improvements measured on one desktop PC and will vary on other hardware (`~3–8 %` run-to-run variance).

## Result on the reference PC

![Inkling-2-Mini memory layout, generation speed, and prompt read throughput on an RTX 4070 SUPER 12 GB](docs/img/inkling2-summary.png)

Ryzen 9 7900X (12C/24T, AVX-512) · RTX 4070 SUPER 12 GB (`sm_89`) · DDR5 32 GB (`31.1 GiB` usable) · Windows 11 · model: [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) **`UD-Q3_K_XL`** (`13.6 GB`) and **`UD-Q4_K_XL`** (`16.4 GiB`, NVFP4 layers 0–19 + Q8_0 layers 20–23), DSpark MTP on (`inkling2-dspark.gguf`), `--kv k8v4 --cache-bias 0.2`, `131K–262K` context.

| | `llama.cpp` PR `#32704` (`-ngl 99 -cmoe`) | `UD-Q4_K_XL` + DSpark (`k8v4`, NVFP4/Q8) | `UD-Q3_K_XL` + DSpark (`k8v4`, fork default) |
|---|---|---|---|
| VRAM-resident experts | 0 / 3,072 (`0 GiB`) | 1,520–1,613 / 3,072 (`7.3–7.5 GiB`) | **1,989 / 3,072 (`7.5 GiB`, 64.7 %)** |
| Code (`temperature 0`, 1,500 tokens, avg of 3) | 13.7 tok/s *(25-tok decode)* | 55.4 tok/s (`59.1 / 53.0 / 54.2`) | **63.2 tok/s (`61.4 / 65.4 / 62.8`)** |
| Mixed long jobs (`prose, code, tool calls, translation, thinking`) | — | 58.1 tok/s (`50.6 / 62.8 / 81.0 / 46.3 / 49.7`) | **64.7 tok/s (`56.3 / 74.8 / 82.0 / 54.3 / 56.1`)** |
| Pure decode (frozen `--window 1 --no-draft`) | 13.7 tok/s | 47.3 tok/s | **59.7 tok/s** (`2,098` experts) |
| Decode @ 32K context | — | 43.8 tok/s | **47.8 tok/s** |
| Decode @ 120K context | — | — | **31.8 tok/s** |
| Live `181.6K` OpenCode coding session | — | — | **28.1–30.0 tok/s** |
| Prompt read, 1.8K–5.4K tokens | 263 tok/s *(1.8K)* | **1,307 tok/s** *(5.4K, no MTP)* | — |
| Prompt read, 32K tokens | — | 1,035 tok/s | **1,137 tok/s** |
| Prompt read, 120K tokens | — | — | **847 tok/s** |
| Greedy token parity vs `llama.cpp` | *reference* | **48 / 48 (100 %)** | 48 / 48 *(FP16 & `q8_0` KV)* |
| HumanEval / HumanEval+ (EvalPlus 0.3.1, `temp 0`, effort 0) | — | **97.6 / 92.7 %** (`160 / 152` of 164) | **95.7 / 90.9 %** (`157 / 149` of 164) |
| GPQA Diamond (198 questions) | — *(Thinking Machines' card: 86.6 %)* | not measured | not measured |

On **HumanEval / HumanEval+** (all 164 tasks, `temperature = 0`, `reasoning_effort = "none"`, `max_tokens = 4096`), **`UD-Q4_K_XL`** solved **160/164 (97.6 %)** base tests and **152/164 (92.7 %)** HumanEval+ tests; **`UD-Q3_K_XL`** solved **157/164 (95.7 %)** base tests and **149/164 (90.9 %)** HumanEval+ tests (133/140 on tasks that fit in 1,024 tokens, recovered to 149/164 once 24 truncated thinking responses were given 4,096 tokens). **GPQA Diamond** is listed as `not measured` on this fork; Thinking Machines Lab reports `86.6 %` for the original unquantized checkpoint.

Without `--kv-grow` and KV quantization, 12 GB VRAM fills quickly on long contexts. With **40 SWA ring layers** (`window = 1024`) and only **8 global layers**, `--kv k8v4` (`13 KiB/token`: `Q8_0` keys + `Q4_0` values) and `--kv q4_0` (`9 KiB/token`) keep `--window 1` pure decode `>= 30 tok/s` out to **~137K–141K filled tokens** (`24.2 tok/s` at 229K, verified 224K needle recall).

## What changed

All Inkling-2-Mini engine work lives in `src/poly/inkling2.*`, `src/poly/kernels_inkling2.cu`, `src/poly/inkling2_test.cu` (**50 / 50 unit tests passing against FP64 references**), and `setup.py`:

| Change | What it does | Measured effect |
|---|---|---|
| Native `inkling2_loop` / `inkling2` engine (`src/poly/inkling2.cpp`) | 48-step looped execution, 6-stream Block AttnRes, dynamic `conv4` + snapshot rollback, SiTU CPU+CUDA experts (`NVFP4`, `Q2_K`–`Q8_0`), 48/48 parity vs `llama.cpp` | **2.4–3.7x faster** than `llama.cpp` PR `#32704` |
| Native **DSpark MTP** (`--draft-model`, `dspark_markov_graph_`) | 5-layer bidirectional block-8 drafter + 7-step Markov RNN captured inside one CUDA Graph (single 28-byte DtoH transfer) | **63.2 tok/s** code, **64.7 tok/s** mixed (`UD-Q3_K_XL`) |
| `--kv k8v4` & `--kv q4_0` + GQA-3 grouped kernel (`inkling_attend_split_gqa_kernel<3>`) | `Q8_0` keys + `Q4_0` values (`13 KiB/tok`) or 4-bit KV (`9 KiB/tok`), 3x global KV bandwidth cut across 24Q/8KV heads, CUDA VMM `--kv-grow` to `1M` `max_context` | Saves `~800 MiB` at 181K over `q8_0`; `+9.1 %` @ 32K |
| `--cache-bias 0.2` with unbiased sigmoid weights | Gently favors VRAM-resident experts during router top-8 selection while recomputing final weights from unbiased logits | `94–96 %` VRAM expert hit rate on 12 GB |
| Tuned `647ae64e` kernel fast path | In-place `inkling_add_to_stream`, 40 fewer SWA `cudaMemcpyAsync` calls, `ctrl_arena` aliasing, vectorized `dequant_q8/q4` + SWA pre-dequantization, DSpark `gate_up_proj` SwiGLU fusion | **+13.1 %** frozen `--window 1` (`52.8 -> 59.7 tok/s`), **+9.1 %** 32K decode, **+5.2 %** 120K prefill |

## Running with or without DSpark MTP

| Mode | Flag | Extra VRAM | Notes |
|---|---|---|---|
| **With DSpark MTP (default)** | `--draft-model inkling2-dspark.gguf` | `~400 MiB` | Fastest overall decode (**63.2 tok/s** code, **47.8 tok/s** @ 32K). Requires converting Unsloth's `dflash` draft file with `tools/convert_inkling2_dspark.py`. |
| **Without MTP (built-in n-gram lookup)** | *(omit `--draft-model`)* | **`0 MiB`** | Fits `~80–100` more experts in VRAM while keeping free n-gram drafts (`--lookup 3 --window 8`, up to `108 tok/s` on repetitive code/logs). |
| **Pure 1-token decode** | *(omit `--draft-model`)* `--window 1` | **`0 MiB`** | Deterministic single-token forward pass (`59.7 tok/s` `UD-Q3_K_XL`, `47.3 tok/s` `UD-Q4_K_XL`). |

## Quick start & building

### 1. One-click installer (`setup.py` / `START-HERE.bat`)

Run **`START-HERE.bat`** and select **`inkling2`**, or install non-interactively:

```bat
python setup.py --family inkling2 --model UD-Q3_K_XL --kv k8v4 --context 131072 --yes
```

Pinned models from [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF): `UD-Q3_K_XL` (`13.6 GB`, recommended for 12 GB VRAM + 32 GB RAM), `UD-IQ3_XXS` (`11.7 GB`), `UD-Q4_K_XL` (`16.4 GiB`, native NVFP4 + Q8_0), and `UD-Q8_K_XL` (`22.9 GB`). Prebuilt Windows releases resolve from `https://github.com/KurrynDev13/PolyStrata/releases/latest/download/`.

### 2. Optional DSpark MTP drafter conversion

Unsloth's hosted `dspark-Inkling-2-Mini-*.gguf` uses the `dflash` layout; convert it once to `inkling2_dspark` for `strata-poly`:

```bat
.venv\Scripts\python.exe tools\convert_inkling2_dspark.py --out models\inkling2\inkling2-dspark.gguf
```

### 3. Building from source (Windows, CUDA 13, Ninja)

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -B build-inkling2 -G Ninja -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-inkling2 --target strata-poly poly_inkling2_test -j 8
```

Run the GPU + CPU verification suite (`50 / 50 ALL PASSED`):

```bat
set PATH=D:\PolyStrata\.venv\Lib\site-packages\nvidia\cu13\bin\x86_64;%PATH%
build-inkling2\poly_inkling2_test.exe
```

## Settings on the reference PC

### `UD-Q3_K_XL` + DSpark MTP (fork default — fastest)

```json
{
  "exe": "D:\\PolyStrata\\engine-poly-inkling2\\strata-poly.exe",
  "args": [
    "--model", "path\\to\\Inkling-2-Mini-UD-Q3_K_XL.gguf",
    "--draft-model", "path\\to\\inkling2-dspark.gguf",
    "--max-context", "262144",
    "--kv", "k8v4",
    "--cache-bias", "0.2",
    "--vram-reserve-mib", "1280",
    "--prompt-batch", "512",
    "--ram-experts-mib", "-1",
    "--expert-profile", "path\\to\\expert-profile.txt",
    "--conversation-cache-mib", "auto"
  ],
  "reasoning_close_retry": true,
  "repeat_stop_tokens": 64
}
```

### `UD-Q4_K_XL` + DSpark MTP (highest coding accuracy — `97.6 % / 92.7 %` HumanEval+)

Swap `--model` to `Inkling-2-Mini-UD-Q4_K_XL.gguf` (all other flags identical). To run **without DSpark MTP** on either quant, omit `"--draft-model"` and `"path\\to\\inkling2-dspark.gguf"`.

## Credits

- [PolyStrata](https://github.com/VecSzn/PolyStrata) by [VecSzn](https://github.com/VecSzn) and [Strata](https://github.com/Niko1221/Strata) by [Niko1221](https://github.com/Niko1221) — the hybrid CPU/GPU MoE engine, server, and installer.
- Thinking Machines Lab — **Inkling-2-Mini**; [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) — dynamic GGUF quants and DSpark draft weights; [llama.cpp](https://github.com/ggml-org/llama.cpp) (`PR #32704`) — `ggml` and reference `inkling2_loop` implementation.
- README and benchmark card layout inspired by [midokino/kuu-Strata-fork](https://github.com/midokino/kuu-Strata-fork).
