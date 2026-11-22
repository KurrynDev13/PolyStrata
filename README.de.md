# PolyStrata-Inkling2

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **Deutsch** · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Türkçe](README.tr.md)

**Inkling-2-Mini auf einer 12-GB-GPU — ein inoffizieller PolyStrata-Fork**

![PolyStrata-Inkling2: UD-Q3_K_XL + DSpark vs. llama.cpp PR #32704 — Code, Mixed long jobs, 32K Prompt read, HumanEval+, GPQA Diamond](docs/img/inkling2-card.png)

PolyStrata-Inkling2 ist ein inoffizieller Fork von [PolyStrata](https://github.com/VecSzn/PolyStrata) (auf Basis von [Strata](https://github.com/Niko1221/Strata)), der speziell für **Inkling-2-Mini** von Thinking Machines Lab (`26B-A3B` Looped MoE mit 48 virtuellen Schritten) auf **einer 12-GB-GPU + 32 GB RAM** optimiert wurde — mit CUDA (`strata-poly.exe`), optionalem **DSpark-MTP**-Drafting, **`--kv k8v4`** / **`q4_0`** und CPU-Vision-Offload.

Alle Werte stammen von einem einzelnen Desktop-PC (`~3–8 %` Schwankung zwischen Läufen); keine Verbindung zu Thinking Machines Lab, Unsloth, Strata oder PolyStrata.

- **Ein-Klick-Setup:** [START-HERE.bat](START-HERE.bat) oder `./setup.sh --family inkling2 --kv k8v4` lädt verifizierte GGUFs aus [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF).
- **Ursprüngliches Multi-Modell-README:** [README.upstream.md](README.upstream.md) · **Ausführlicher Benchmark-Bericht:** [Inkling 2 Mini Benchmark.md](Inkling%202%20Mini%20Benchmark.md).

## Ergebnisse auf dem Referenz-PC

![Speicheraufteilung und Durchsatz von Inkling-2-Mini auf RTX 4070 SUPER 12 GB](docs/img/inkling2-summary.png)

Ryzen 9 7900X (12C/24T, AVX-512) · RTX 4070 SUPER 12 GB (`sm_89`) · DDR5 32 GB (`31,1 GiB` nutzbar) · Windows 11 · Modell: [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) **`UD-Q3_K_XL`** (`13,6 GB`) und **`UD-Q4_K_XL`** (`16,4 GiB`, NVFP4 in Schichten 0–19 + Q8_0 in 20–23), DSpark MTP aktiv (`inkling2-dspark.gguf`), `--kv k8v4 --cache-bias 0.2`, Kontext `131K–262K`.

| | `llama.cpp` PR `#32704` (`-ngl 99 -cmoe`) | `UD-Q8_K_XL` + DSpark (`k8v4`, 8-bit) | `UD-Q4_K_XL` + DSpark (`k8v4`, NVFP4/Q8) | `UD-Q3_K_XL` + DSpark (`k8v4`, fork default) |
|---|---|---|---|---|
| Download / File Size | 17.6 GB | **22.9 GB** | 17.6 GB | 13.6 GB |
| VRAM-resident experts | 0 / 3,072 (`0 GiB`) | 1,210 / 3,072 (`39.4 %`) | 1,350–1,613 / 3,072 (`43.9 %`) | **1,989–2,098 / 3,072 (`68.3 %`)** |
| Code (`temperature 0`, 1,500 tokens, avg of 3) | 13.7 tok/s *(25-tok decode)* | 20.3 tok/s (`20.3 / 20.3 / 20.3`) | 55.4 tok/s (`59.1 / 53.0 / 54.2`) | **63.2 tok/s (`61.4 / 65.4 / 62.8`)** |
| Mixed long jobs (`prose, code, tool calls, translation, thinking`) | — | 20.0 tok/s (`18.6 / 20.2 / 20.2 / 20.4 / 20.6`) | 58.1 tok/s (`50.6 / 62.8 / 81.0 / 46.3 / 49.7`) | **64.7 tok/s (`56.3 / 74.8 / 82.0 / 54.3 / 56.1`)** |
| Pure decode (frozen `--window 1 --no-draft`) | 13.7 tok/s | 18.5 tok/s | 51.2 tok/s | **59.7 tok/s** (`2,098` experts) |
| Decode @ 32K context | — | 18.2 tok/s | 43.8 tok/s | **47.8 tok/s** |
| Decode @ 120K context | — | — | — | **31.8 tok/s** |
| Live `181.6K` OpenCode coding session | — | — | — | **28.1–30.0 tok/s** |
| Prompt read, 1.8K–5.4K tokens | 263 tok/s *(1.8K)* | — | **1,307 tok/s** *(5.4K, no MTP)* | — |
| Prompt read, 32K tokens | — | 163.7 tok/s | 1,035 tok/s | **1,137 tok/s** |
| Prompt read, 120K tokens | — | — | — | **847 tok/s** |
| Greedy token parity vs `llama.cpp` | *reference* | **48 / 48 (100 %)** | **48 / 48 (100 %)** | 48 / 48 *(FP16 & `q8_0` KV)* |
| HumanEval / HumanEval+ (EvalPlus 0.3.1, `temp 0`, effort 0) | — | **Near-FP16 (>97.6 %)** | **97.6 / 92.7 %** (`160 / 152` of 164) | **95.7 / 90.9 %** (`157 / 149` of 164) |
| GPQA Diamond (198 questions) | — *(Thinking Machines' card: 86.6 %)* | not measured | not measured | not measured |

## Wichtigste Änderungen & Build

Alle Kernel-Optimierungen liegen in `src/poly/inkling2.*`, `src/poly/kernels_inkling2.cu` und `src/poly/inkling2_test.cu` (**50 / 50 Unit-Tests bestanden**): natives `inkling2_loop`/`inkling2`, CUDA-Graph-basiertes **DSpark MTP** (`--draft-model`), **`--kv k8v4`** / **`q4_0`**, GQA-3-Kernel, `--cache-bias 0.2` (mit unverfälschten Sigmoid-Gewichten) sowie In-Place-Residual-Addition und vektorisierte KV-Dequantisierung (`647ae64e`).

```bat
python setup.py --family inkling2 --model UD-Q3_K_XL --kv k8v4 --context 131072 --yes
.venv\Scripts\python.exe tools\convert_inkling2_dspark.py --out models\inkling2\inkling2-dspark.gguf
```

Quellcode-Build (Windows, CUDA 13, Ninja):

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -B build-inkling2 -G Ninja -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-inkling2 --target strata-poly poly_inkling2_test -j 8
```

## Konfiguration auf dem Referenz-PC (`UD-Q3_K_XL` + DSpark MTP)

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

## Danksagung

- [PolyStrata](https://github.com/VecSzn/PolyStrata) ([VecSzn](https://github.com/VecSzn)), [Strata](https://github.com/Niko1221/Strata) ([Niko1221](https://github.com/Niko1221)), Thinking Machines Lab (**Inkling-2-Mini**), [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF), [llama.cpp](https://github.com/ggml-org/llama.cpp) (`PR #32704`) sowie [midokino/kuu-Strata-fork](https://github.com/midokino/kuu-Strata-fork) als Design-Vorlage für README und Grafiken.