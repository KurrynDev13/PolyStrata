# PolyStrata-Inkling2

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · **Português** · [Türkçe](README.tr.md)

**Inkling-2-Mini em uma única GPU de 12 GB — fork não oficial do PolyStrata**

![PolyStrata-Inkling2: UD-Q3_K_XL + DSpark vs llama.cpp PR #32704 — Code, Mixed long jobs, leitura 32K, HumanEval+, GPQA Diamond](docs/img/inkling2-card.png)

O PolyStrata-Inkling2 é um fork não oficial do [PolyStrata](https://github.com/VecSzn/PolyStrata) (baseado no [Strata](https://github.com/Niko1221/Strata)) otimizado especificamente para rodar o **Inkling-2-Mini** da Thinking Machines Lab (MoE em loop de 48 passos virtuais, `26B-A3B`) em **uma GPU de 12 GB e 32 GB de RAM** via CUDA (`strata-poly.exe`), com decodificação especulativa **DSpark MTP**, cache KV **`--kv k8v4`** / **`q4_0`** e visão na CPU.

Medido em um único PC desktop (`~3–8 %` de variação entre execuções); sem vínculo com Thinking Machines Lab, Unsloth, Strata ou PolyStrata.

- **Instalação em um clique:** execute [START-HERE.bat](START-HERE.bat) (Windows) ou `./setup.sh --family inkling2 --kv k8v4` para configurar tudo e baixar os GGUFs verificados de [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF).
- **README original** (outros 12 modelos): [README.upstream.md](README.upstream.md) · **Relatório completo:** [Inkling 2 Mini Benchmark.md](Inkling%202%20Mini%20Benchmark.md).

## Resultados no PC de referência

![Layout de memória e desempenho do Inkling-2-Mini na RTX 4070 SUPER 12 GB](docs/img/inkling2-summary.png)

Ryzen 9 7900X (12C/24T, AVX-512) · RTX 4070 SUPER 12 GB (`sm_89`) · DDR5 32 GB (`31,1 GiB` úteis) · Windows 11 · modelo: [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) **`UD-Q3_K_XL`** (`13,6 GB`) e **`UD-Q4_K_XL`** (`16,4 GiB`, NVFP4 nas camadas 0–19 + Q8_0 nas 20–23), DSpark MTP ativo (`inkling2-dspark.gguf`), `--kv k8v4 --cache-bias 0.2`, contexto `131K–262K`.

| | `llama.cpp` PR `#32704` (`-ngl 99 -cmoe`) | `UD-Q4_K_XL` + DSpark (`k8v4`, NVFP4/Q8) | `UD-Q3_K_XL` + DSpark (`k8v4`, padrão) |
|---|---|---|---|
| Experts residentes na VRAM | 0 / 3.072 (`0 GiB`) | 1.520–1.613 / 3.072 (`7,3–7,5 GiB`) | **1.989 / 3.072 (`7,5 GiB`, 64,7 %)** |
| Code (`temperature 0`, 1.500 tokens, média de 3) | 13,7 tok/s *(decode 25 tok)* | 55,4 tok/s (`59,1 / 53,0 / 54,2`) | **63,2 tok/s (`61,4 / 65,4 / 62,8`)** |
| Mixed long jobs (`prosa, código, tool calls, tradução, thinking`) | — | 58,1 tok/s (`50,6 / 62,8 / 81,0 / 46,3 / 49,7`) | **64,7 tok/s (`56,3 / 74,8 / 82,0 / 54,3 / 56,1`)** |
| Decode puro (congelado `--window 1 --no-draft`) | 13,7 tok/s | 47,3 tok/s | **59,7 tok/s** (`2.098` experts) |
| Decode @ 32K de contexto | — | 43,8 tok/s | **47,8 tok/s** |
| Decode @ 120K de contexto | — | — | **31,8 tok/s** |
| Sessão real no OpenCode em `181,6K` tokens | — | — | **28,1–30,0 tok/s** |
| Leitura de prompt, 1,8K–5,4K tokens | 263 tok/s *(1,8K)* | **1.307 tok/s** *(5,4K, sem MTP)* | — |
| Leitura de prompt, 32K tokens | — | 1.035 tok/s | **1.137 tok/s** |
| Leitura de prompt, 120K tokens | — | — | **847 tok/s** |
| Paridade greedy vs `llama.cpp` | *referência* | **48 / 48 (100 %)** | 48 / 48 *(KV FP16 e `q8_0`)* |
| HumanEval / HumanEval+ (EvalPlus 0.3.1, `temp 0`, esforço 0) | — | **97,6 / 92,7 %** (`160 / 152` de 164) | **95,7 / 90,9 %** (`157 / 149` de 164) |
| GPQA Diamond (198 questões) | — *(card da Thinking Machines: 86,6 %)* | não medido | não medido |

## O que mudou & compilação

Todas as otimizações estão em `src/poly/inkling2.*`, `src/poly/kernels_inkling2.cu` e `src/poly/inkling2_test.cu` (**50 / 50 testes unitários aprovados**): suporte nativo `inkling2_loop`/`inkling2`, grafo CUDA **DSpark MTP** (`--draft-model`), cache **`--kv k8v4`** / **`q4_0`**, kernel agrupado GQA-3, `--cache-bias 0.2` (pesos sigmoides sem viés) e soma residual in-place (`647ae64e`).

```bat
python setup.py --family inkling2 --model UD-Q3_K_XL --kv k8v4 --context 131072 --yes
.venv\Scripts\python.exe tools\convert_inkling2_dspark.py --out models\inkling2\inkling2-dspark.gguf
```

Compilação no Windows (CUDA 13, Ninja):

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -B build-inkling2 -G Ninja -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-inkling2 --target strata-poly poly_inkling2_test -j 8
```

## Configuração no PC de referência (`UD-Q3_K_XL` + DSpark MTP)

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

- [PolyStrata](https://github.com/VecSzn/PolyStrata) ([VecSzn](https://github.com/VecSzn)), [Strata](https://github.com/Niko1221/Strata) ([Niko1221](https://github.com/Niko1221)), Thinking Machines Lab (**Inkling-2-Mini**), [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF), [llama.cpp](https://github.com/ggml-org/llama.cpp) (`PR #32704`) e [midokino/kuu-Strata-fork](https://github.com/midokino/kuu-Strata-fork) pelo visual do README e dos gráficos.
