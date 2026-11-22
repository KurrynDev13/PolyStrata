# PolyStrata-Inkling2

[English](README.md) · [简体中文](README.zh-CN.md) · **日本語** · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Türkçe](README.tr.md)

**12 GB GPU 1 枚で動かす Inkling-2-Mini — 非公式 PolyStrata フォーク**

![PolyStrata-Inkling2: UD-Q3_K_XL + DSpark 対 llama.cpp PR #32704 — Code、Mixed long jobs、32K Prompt read、HumanEval+、GPQA Diamond](docs/img/inkling2-card.png)

PolyStrata-Inkling2 は、[PolyStrata](https://github.com/VecSzn/PolyStrata)（および上流の [Strata](https://github.com/Niko1221/Strata)）をベースに、Thinking Machines Lab の **Inkling-2-Mini**（`26B-A3B` 48 仮想レイヤー・ループ型 MoE）を **12 GB VRAM GPU + 32 GB RAM** の一般的なデスクトップ PC 向けに最適化した非公式フォークです。CUDA エンジン（`strata-poly.exe`）上で **DSpark MTP** 投機的デコード、混合精度 **`--kv k8v4`** / **`q4_0`** KV キャッシュ、CPU オフロード版ビジョンエンコーダに対応しています。

1 台の PC で丁寧に計測した個人フォークであり、Thinking Machines Lab・Unsloth・Strata・上流 PolyStrata とは提携していません。

- **ワンクリック導入:** [START-HERE.bat](START-HERE.bat)（Windows）または `./setup.sh --family inkling2 --kv k8v4` を実行すると、環境構築から [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) の検証済みダウンロードまで自動で行われます。
- **元のマルチモデル版 README** は [README.upstream.md](README.upstream.md) に保存しています。
- **詳細ベンチマーク & カーネル解説:** [Inkling 2 Mini Benchmark.md](Inkling%202%20Mini%20Benchmark.md)
- 以下の高速化はすべてソフトウェア実装によるもので、別環境では数値が前後します（試行間ばらつき約 `3–8 %`）。

## 検証 PC での計測結果

![RTX 4070 SUPER 12 GB 上での Inkling-2-Mini メモリ配置・生成速度・プロンプト読込速度](docs/img/inkling2-summary.png)

Ryzen 9 7900X (12C/24T, AVX-512) · RTX 4070 SUPER 12 GB (`sm_89`) · DDR5 32 GB (`31.1 GiB` 使用可能) · Windows 11 · モデル：[`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) **`UD-Q3_K_XL`** (`13.6 GB`) および **`UD-Q4_K_XL`** (`16.4 GiB`, 0–19 層 NVFP4 + 20–23 層 Q8_0)、DSpark MTP 有効 (`inkling2-dspark.gguf`)、`--kv k8v4 --cache-bias 0.2`、コンテキスト `131K–262K`。

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

**HumanEval / HumanEval+** 全 164 問（`temperature = 0`, `reasoning_effort = "none"`, `max_tokens = 4096`）において、**`UD-Q4_K_XL`** は **97.6 % (160/164)** / **92.7 % (152/164)**、**`UD-Q3_K_XL`** は **95.7 % (157/164)** / **90.9 % (149/164)** を達成しました。GPQA Diamond は本フォークでは未計測です（Thinking Machines Lab 公式値は `86.6 %`）。

## 主な変更点

変更はすべて `src/poly/inkling2.*`、`src/poly/kernels_inkling2.cu`、`src/poly/inkling2_test.cu`（FP64 基準に対して **50 / 50 テスト合格**）および `setup.py` に実装されています。

| 内容 | 仕組み | 実測効果 |
|---|---|---|
| ネイティブ `inkling2_loop` / `inkling2` エンジン | 48 ステップ・ループ推論、6 ストリーム Block AttnRes、動的 `conv4` + ロールバック、CPU+GPU SiTU エキスパート | `llama.cpp` PR `#32704` 比 **2.4–3.7 倍高速** |
| ネイティブ **DSpark MTP** (`--draft-model`) | 5 層 block-8 ドラフタ + 7 ステップ Markov RNN を単一 CUDA Graph 化 | Code **63.2 tok/s** / Mixed **64.7 tok/s** |
| `--kv k8v4` / `q4_0` + GQA-3 統合カーネル + `--kv-grow` | 8-bit K + 4-bit V（`13 KiB/tok`）、CUDA VMM による動的確保（最大 `1M`）、純粋デコード `>= 30 tok/s` を **137K–141K** まで維持 | 181K で `q8_0` 比 `~800 MiB` 削減、32K **+9.1 %** |
| `--cache-bias 0.2`（無バイアス重み再計算） | ルーティング選択時のみ VRAM エキスパートを優遇し、最終結合重みは元のロジットから算出 | 12 GB GPU ヒット率 **94–96 %** |
| `647ae64e` カーネル高速化パス | インプレース残差加算、SWA `cudaMemcpyAsync` 40 回削減、ベクトル化 KV 逆量子化キャッシュ、DSpark FFN 融合 | 固定 `--window 1` **+13.1 %** (`52.8 -> 59.7 tok/s`) |

## クイックスタート & ビルド

```bat
python setup.py --family inkling2 --model UD-Q3_K_XL --kv k8v4 --context 131072 --yes
```

DSpark MTP を使う場合（任意・追加 VRAM `~400 MiB`）は、Unsloth の `dspark-Inkling-2-Mini-*.gguf` を一度だけ変換します：

```bat
.venv\Scripts\python.exe tools\convert_inkling2_dspark.py --out models\inkling2\inkling2-dspark.gguf
```

ソースビルド（Windows / CUDA 13 / Ninja）：

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -B build-inkling2 -G Ninja -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-inkling2 --target strata-poly poly_inkling2_test -j 8
```

## 検証 PC の設定例 (`UD-Q3_K_XL` + DSpark)

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

`UD-Q4_K_XL` を使う場合は `--model` のパスを差し替えます。MTP なし（0 MiB 追加 VRAM）で動かす場合は `--draft-model` 行を削除してください。

## クレジット

- [PolyStrata](https://github.com/VecSzn/PolyStrata) ([VecSzn](https://github.com/VecSzn)) ・ [Strata](https://github.com/Niko1221/Strata) ([Niko1221](https://github.com/Niko1221))
- Thinking Machines Lab（**Inkling-2-Mini**）・ [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)・ [llama.cpp](https://github.com/ggml-org/llama.cpp) (`PR #32704`)
- README / カードレイアウト参考：[midokino/kuu-Strata-fork](https://github.com/midokino/kuu-Strata-fork)