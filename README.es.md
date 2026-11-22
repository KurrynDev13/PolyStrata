# PolyStrata-Inkling2

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · **Español** · [Português](README.pt-BR.md) · [Türkçe](README.tr.md)

**Inkling-2-Mini en una sola GPU de 12 GB — fork no oficial de PolyStrata**

![PolyStrata-Inkling2: UD-Q3_K_XL + DSpark frente a llama.cpp PR #32704 — Code, Mixed long jobs, lectura 32K, HumanEval+, GPQA Diamond](docs/img/inkling2-card.png)

PolyStrata-Inkling2 es un fork no oficial de [PolyStrata](https://github.com/VecSzn/PolyStrata) (basado en [Strata](https://github.com/Niko1221/Strata)) ajustado específicamente para ejecutar **Inkling-2-Mini** de Thinking Machines Lab (MoE en bucle de 48 pasos virtuales, `26B-A3B`) en **una GPU de 12 GB y 32 GB de RAM** con CUDA (`strata-poly.exe`), decodificación especulativa **DSpark MTP**, caché KV mixta **`--kv k8v4`** / **`q4_0`** y visión en CPU.

Medido en un solo PC de escritorio (`~3–8 %` de variación entre ejecuciones); sin afiliación con Thinking Machines Lab, Unsloth, Strata ni PolyStrata.

- **Instalación en un clic:** ejecuta [START-HERE.bat](START-HERE.bat) (Windows) o `./setup.sh --family inkling2 --kv k8v4` para instalar dependencias y descargar los GGUF verificados desde [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF).
- **README original** (los otros 12 modelos): [README.upstream.md](README.upstream.md) · **Informe completo:** [Inkling 2 Mini Benchmark.md](Inkling%202%20Mini%20Benchmark.md).

## Resultados en el PC de referencia

![Distribución de memoria y rendimiento de Inkling-2-Mini en RTX 4070 SUPER 12 GB](docs/img/inkling2-summary.png)

Ryzen 9 7900X (12C/24T, AVX-512) · RTX 4070 SUPER 12 GB (`sm_89`) · DDR5 32 GB (`31,1 GiB` útiles) · Windows 11 · modelo: [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) **`UD-Q3_K_XL`** (`13,6 GB`) y **`UD-Q4_K_XL`** (`16,4 GiB`, NVFP4 en capas 0–19 + Q8_0 en 20–23), DSpark MTP activo (`inkling2-dspark.gguf`), `--kv k8v4 --cache-bias 0.2`, contexto `131K–262K`.

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

## Cambios principales y compilación

Todas las mejoras están en `src/poly/inkling2.*`, `src/poly/kernels_inkling2.cu` y `src/poly/inkling2_test.cu` (**50 / 50 pruebas unitarias superadas**): motor nativo `inkling2_loop`/`inkling2`, grafo CUDA **DSpark MTP** (`--draft-model`), caché **`--kv k8v4`** / **`q4_0`**, kernel agrupado GQA-3, `--cache-bias 0.2` (con pesos sigmoides sin sesgo) y acumulación residual in-place (`647ae64e`).

```bat
python setup.py --family inkling2 --model UD-Q3_K_XL --kv k8v4 --context 131072 --yes
.venv\Scripts\python.exe tools\convert_inkling2_dspark.py --out models\inkling2\inkling2-dspark.gguf
```

Compilación en Windows (CUDA 13, Ninja):

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -B build-inkling2 -G Ninja -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-inkling2 --target strata-poly poly_inkling2_test -j 8
```

## Configuración de referencia (`UD-Q3_K_XL` + DSpark MTP)

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

## Créditos

- [PolyStrata](https://github.com/VecSzn/PolyStrata) ([VecSzn](https://github.com/VecSzn)), [Strata](https://github.com/Niko1221/Strata) ([Niko1221](https://github.com/Niko1221)), Thinking Machines Lab (**Inkling-2-Mini**), [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF), [llama.cpp](https://github.com/ggml-org/llama.cpp) (`PR #32704`) y [midokino/kuu-Strata-fork](https://github.com/midokino/kuu-Strata-fork) por el diseño del README y las gráficas.