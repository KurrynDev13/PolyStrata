# PolyStrata-Inkling2

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · **Türkçe**

**Tek bir 12 GB GPU üzerinde Inkling-2-Mini — resmi olmayan PolyStrata çatalı**

![PolyStrata-Inkling2: UD-Q3_K_XL + DSpark vs llama.cpp PR #32704 — Code, Mixed long jobs, 32K Prompt okuma, HumanEval+, GPQA Diamond](docs/img/inkling2-card.png)

PolyStrata-Inkling2, Thinking Machines Lab'in **Inkling-2-Mini** (48 sanal adımlı döngüsel MoE, `26B-A3B`) modelini **tek bir 12 GB GPU ve 32 GB RAM** ile CUDA (`strata-poly.exe`) üzerinde çalıştırmak için özelleştirilmiş resmi olmayan bir [PolyStrata](https://github.com/VecSzn/PolyStrata) ([Strata](https://github.com/Niko1221/Strata) tabanlı) çatalıdır; isteğe bağlı **DSpark MTP** spekülatif taslak üretimi, **`--kv k8v4`** / **`q4_0`** KV önbelleği ve CPU görüntü kodlayıcı içerir.

Tek bir masaüstü bilgisayarda ölçülmüştür (`~3–8 %` çalıştırmalar arası fark olabilir); Thinking Machines Lab, Unsloth, Strata veya PolyStrata ile bağlantısı yoktur.

- **Tek tıkla kurulum:** [START-HERE.bat](START-HERE.bat) (Windows) veya `./setup.sh --family inkling2 --kv k8v4` çalıştırarak [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) dosyalarını otomatik indirebilirsiniz.
- **Önceki çoklu model README'si:** [README.upstream.md](README.upstream.md) · **Ayrıntılı ölçüm raporu:** [Inkling 2 Mini Benchmark.md](Inkling%202%20Mini%20Benchmark.md).

## Referans PC Sonuçları

![RTX 4070 SUPER 12 GB üzerinde Inkling-2-Mini bellek yerleşimi ve hız sonuçları](docs/img/inkling2-summary.png)

Ryzen 9 7900X (12C/24T, AVX-512) · RTX 4070 SUPER 12 GB (`sm_89`) · DDR5 32 GB (`31,1 GiB` kullanılabilir) · Windows 11 · model: [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) **`UD-Q3_K_XL`** (`13,6 GB`) ve **`UD-Q4_K_XL`** (`16,4 GiB`, 0–19 katmanlarda NVFP4 + 20–23'te Q8_0), DSpark MTP açık (`inkling2-dspark.gguf`), `--kv k8v4 --cache-bias 0.2`, `131K–262K` bağlam.

| | `llama.cpp` PR `#32704` (`-ngl 99 -cmoe`) | `UD-Q4_K_XL` + DSpark (`k8v4`, NVFP4/Q8) | `UD-Q3_K_XL` + DSpark (`k8v4`, varsayılan) |
|---|---|---|---|
| VRAM'deki uzman sayısı | 0 / 3.072 (`0 GiB`) | 1.520–1.613 / 3.072 (`7,3–7,5 GiB`) | **1.989 / 3.072 (`7,5 GiB`, 64,7 %)** |
| Code (`temperature 0`, 1.500 token, 3 koşu ort.) | 13,7 tok/s *(25-tok üretimi)* | 55,4 tok/s (`59,1 / 53,0 / 54,2`) | **63,2 tok/s (`61,4 / 65,4 / 62,8`)** |
| Mixed long jobs (`düzyazı, kod, araç çağrısı, çeviri, düşünme`) | — | 58,1 tok/s (`50,6 / 62,8 / 81,0 / 46,3 / 49,7`) | **64,7 tok/s (`56,3 / 74,8 / 82,0 / 54,3 / 56,1`)** |
| Saf üretim (`--window 1 --no-draft` sabit) | 13,7 tok/s | 47,3 tok/s | **59,7 tok/s** (`2.098` uzman) |
| 32K bağlamda üretim | — | 43,8 tok/s | **47,8 tok/s** |
| 120K bağlamda üretim | — | — | **31,8 tok/s** |
| Canlı `181,6K` OpenCode oturumu | — | — | **28,1–30,0 tok/s** |
| İstemi okuma, 1,8K–5,4K token | 263 tok/s *(1,8K)* | **1.307 tok/s** *(5,4K, MTP'siz)* | — |
| İstemi okuma, 32K token | — | 1.035 tok/s | **1.137 tok/s** |
| İstemi okuma, 120K token | — | — | **847 tok/s** |
| `llama.cpp` ile greedy token uyumu | *referans* | **48 / 48 (100 %)** | 48 / 48 *(FP16 & `q8_0` KV)* |
| HumanEval / HumanEval+ (EvalPlus 0.3.1, `temp 0`, effort 0) | — | **97,6 / 92,7 %** (`160 / 152` / 164) | **95,7 / 90,9 %** (`157 / 149` / 164) |
| GPQA Diamond (198 soru) | — *(model kartı: 86,6 %)* | ölçülmedi | ölçülmedi |

## Değişiklikler ve Derleme

Tüm iyileştirmeler `src/poly/inkling2.*`, `src/poly/kernels_inkling2.cu` ve `src/poly/inkling2_test.cu` dosyalarındadır (**50 / 50 birim testi geçti**): yerel `inkling2_loop`/`inkling2` motoru, CUDA Graph tabanlı **DSpark MTP** (`--draft-model`), **`--kv k8v4`** / **`q4_0`**, gruplanmış GQA-3 çekirdeği, `--cache-bias 0.2` ve yerinde artık akış ekleme (`647ae64e`).

```bat
python setup.py --family inkling2 --model UD-Q3_K_XL --kv k8v4 --context 131072 --yes
.venv\Scripts\python.exe tools\convert_inkling2_dspark.py --out models\inkling2\inkling2-dspark.gguf
```

Kaynaktan derleme (Windows, CUDA 13, Ninja):

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -B build-inkling2 -G Ninja -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-inkling2 --target strata-poly poly_inkling2_test -j 8
```

## Referans PC Ayarları (`UD-Q3_K_XL` + DSpark MTP)

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

## Teşekkürler

- [PolyStrata](https://github.com/VecSzn/PolyStrata) ([VecSzn](https://github.com/VecSzn)), [Strata](https://github.com/Niko1221/Strata) ([Niko1221](https://github.com/Niko1221)), Thinking Machines Lab (**Inkling-2-Mini**), [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF), [llama.cpp](https://github.com/ggml-org/llama.cpp) (`PR #32704`) ve README / grafik düzeni ilhamı için [midokino/kuu-Strata-fork](https://github.com/midokino/kuu-Strata-fork).
