# PolyStrata-Inkling2

[English](README.md) · **简体中文** · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Türkçe](README.tr.md)

**在单张 12 GB 显卡上本地运行 Inkling-2-Mini —— 非官方 PolyStrata 分支**

![PolyStrata-Inkling2：UD-Q3_K_XL + DSpark 对比 llama.cpp PR #32704 —— 代码生成、混合长任务、32K 提示词读取、HumanEval+、GPQA Diamond](docs/img/inkling2-card.png)

PolyStrata-Inkling2 是基于 [PolyStrata](https://github.com/VecSzn/PolyStrata)（及上游 [Strata](https://github.com/Niko1221/Strata)）的非官方分支，专为在**单张 12 GB 显存 + 32 GB 内存**的普通桌面电脑上运行 Thinking Machines Lab 的 **Inkling-2-Mini**（`26B-A3B` 48 虚拟层循环 MoE 模型）而优化，使用 CUDA 引擎（`strata-poly.exe`），支持原生 **DSpark MTP** 投机草稿解码、混合精度 **`--kv k8v4`** / **`q4_0`** KV 缓存以及零显存占用的 CPU 视觉编码器。

本分支仅在单台参考 PC 上实测调优，与 Thinking Machines Lab、Unsloth、Strata 或上游 PolyStrata 无任何从属关系。

- **一键安装：** 运行 [START-HERE.bat](START-HERE.bat)（Windows）或 `./setup.sh --family inkling2 --kv k8v4`，自动配置 Python/CUDA 运行环境并从 [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) 下载带哈希校验的 GGUF 模型。
- **上游 README**（其余 12 个模型及多模型总览）已归档至 [README.upstream.md](README.upstream.md)。
- **完整基准数据与内核优化细节：** [Inkling 2 Mini Benchmark.md](Inkling%202%20Mini%20Benchmark.md)。
- 下方加速均来自纯软件优化，数据测自单台桌面 PC，不同硬件上存在差异（多次运行波动约 `3–8 %`）。

## 参考 PC 实测结果

![Inkling-2-Mini 在 RTX 4070 SUPER 12 GB 上的显存/内存分布、生成速度与提示词读取吞吐量](docs/img/inkling2-summary.png)

Ryzen 9 7900X (12 核 24 线程, AVX-512) · RTX 4070 SUPER 12 GB (`sm_89`) · DDR5 32 GB (`31.1 GiB` 可用) · Windows 11 · 模型：[`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) **`UD-Q3_K_XL`** (`13.6 GB`) 与 **`UD-Q4_K_XL`** (`16.4 GiB`, 0–19 层为 NVFP4、20–23 层为 Q8_0)，开启 DSpark MTP (`inkling2-dspark.gguf`)，`--kv k8v4 --cache-bias 0.2`，`131K–262K` 上下文。

| | `llama.cpp` PR `#32704` (`-ngl 99 -cmoe`) | `UD-Q4_K_XL` + DSpark (`k8v4`, NVFP4/Q8) | `UD-Q3_K_XL` + DSpark (`k8v4`, 本分支默认) |
|---|---|---|---|
| 驻留显存专家数 | 0 / 3,072 (`0 GiB`) | 1,520–1,613 / 3,072 (`7.3–7.5 GiB`) | **1,989 / 3,072 (`7.5 GiB`, 64.7 %)** |
| 代码生成（`temperature 0`，1,500 tokens，3 次均值） | 13.7 tok/s *（25-tok 短解码）* | 55.4 tok/s (`59.1 / 53.0 / 54.2`) | **63.2 tok/s (`61.4 / 65.4 / 62.8`)** |
| 混合长任务（散文、代码、工具调用、翻译、思考） | — | 58.1 tok/s (`50.6 / 62.8 / 81.0 / 46.3 / 49.7`) | **64.7 tok/s (`56.3 / 74.8 / 82.0 / 54.3 / 56.1`)** |
| 纯单 token 解码（冻结 `--window 1 --no-draft`） | 13.7 tok/s | 47.3 tok/s | **59.7 tok/s** (`2,098` 专家) |
| 32K 上下文解码 | — | 43.8 tok/s | **47.8 tok/s** |
| 120K 上下文解码 | — | — | **31.8 tok/s** |
| 真实 `181.6K` OpenCode 长对话解码 | — | — | **28.1–30.0 tok/s** |
| 提示词读取（1.8K–5.4K tokens） | 263 tok/s *（1.8K）* | **1,307 tok/s** *（5.4K，无 MTP）* | — |
| 提示词读取（32K tokens） | — | 1,035 tok/s | **1,137 tok/s** |
| 提示词读取（120K tokens） | — | — | **847 tok/s** |
| 与 `llama.cpp` 贪心输出一致率 | *参考基线* | **48 / 48 (100 %)** | 48 / 48 *（FP16 与 `q8_0` KV）* |
| HumanEval / HumanEval+（EvalPlus 0.3.1，`temp 0`, effort 0） | — | **97.6 / 92.7 %** (`160 / 152` / 164) | **95.7 / 90.9 %** (`157 / 149` / 164) |
| GPQA Diamond（198 题） | — *（Thinking Machines 官方卡：86.6 %）* | 未测试 | 未测试 |

在完整的 **HumanEval / HumanEval+** 全部 164 题测试中（`temperature = 0`, `reasoning_effort = "none"`, `max_tokens = 4096`），**`UD-Q4_K_XL`** 拿下 **97.6 % (160/164)** 基础通过率与 **92.7 % (152/164)** HumanEval+ 通过率；**`UD-Q3_K_XL`** 拿下 **95.7 % (157/164)** 基础通过率与 **90.9 % (149/164)** HumanEval+ 通过率。**GPQA Diamond** 尚未在本分支实测，Thinking Machines Lab 官方发布的原始权重成绩为 `86.6 %`。

借助 Inkling-2-Mini 的 **40 层局部滑动窗口环形缓存**（`window = 1024`）+ **8 层全局注意力层**，配合 CUDA VMM 按需扩容（`--kv-grow`）以及 `--kv k8v4`（`13 KiB/token`）/ `--kv q4_0`（`9 KiB/token`），12 GB 显卡在 `--window 1` 纯解码下可将 `>= 30 tok/s` 维持到 **~137K–141K 已填充 tokens**（229K 下仍达 `24.2 tok/s`，224K 检索精准通过）。

## 引擎改动要点

所有 Inkling-2-Mini 改动集中在 `src/poly/inkling2.*`、`src/poly/kernels_inkling2.cu`、`src/poly/inkling2_test.cu`（**50 / 50 单元测试对照 FP64 参考实现全部通过**）以及 `setup.py`：

| 改动项 | 作用 | 实测收益 |
|---|---|---|
| 原生 `inkling2_loop` / `inkling2` 引擎（`src/poly/inkling2.cpp`） | 48 步循环执行、6 流 Block AttnRes、动态 `conv4` 与投机回滚、CPU+CUDA 混合 SiTU 专家（支持 NVFP4 与 Q2_K–Q8_0） | 对比 `llama.cpp` PR `#32704` **快 2.4–3.7 倍** |
| 原生 **DSpark MTP**（`--draft-model`、`dspark_markov_graph_`） | 5 层双向 block-8 草稿网络 + 7 步 Markov RNN 整体捕获进单个 CUDA Graph（每步仅回传 28 字节） | 代码生成 **63.2 tok/s**，混合长任务 **64.7 tok/s** |
| `--kv k8v4` / `q4_0` 与 GQA-3 分组注意力核 | 8-bit K + 4-bit V 混合 KV（`13 KiB/tok`），24Q/8KV 分组寄存器复用减少 3 倍全局 KV 带宽，支持最高 `1M` 虚拟上下文 | 181K 下比 `q8_0` 省 `~800 MiB` 显存，32K 解码 **+9.1 %** |
| 路由缓存偏置 `--cache-bias 0.2`（保持无偏 Sigmoid 权重） | Top-8 路由选专家时轻度偏向显存内专家，最终仍按未加偏置的原始 logits 计算 Sigmoid 权重 | 12 GB 显卡显存专家命中率提升至 `94–96 %` |
| `647ae64e` 精选内核快速路径 | 原地 `inkling_add_to_stream` 减少 88 次内核启动、移除 40 次 SWA `cudaMemcpyAsync`、向量化 KV 反量化预热及 DSpark FFN 融合 | 冻结 `--window 1` **+13.1 %** (`52.8 -> 59.7 tok/s`) |

## 启用与不启用 DSpark MTP 的运行模式

| 模式 | 启动参数 | 额外显存 | 说明 |
|---|---|---|---|
| **开启 DSpark MTP（推荐默认）** | `--draft-model inkling2-dspark.gguf` | 约 `400 MiB` | 整体生成最快（代码 **63.2 tok/s**，32K 解码 **47.8 tok/s**）。需使用 `tools/convert_inkling2_dspark.py` 转换 Unsloth 的草稿权重。 |
| **关闭 MTP（内置 n-gram 查找草稿）** | *（省略 `--draft-model`）* | **`0 MiB`** | 释放约 `400 MiB` 显存给更多专家（多驻留约 80–100 个专家），同时保留零开销 n-gram 草稿加速（重复代码/日志最高 `108 tok/s`）。 |
| **纯单 token 解码** | *（省略 `--draft-model`）* `--window 1` | **`0 MiB`** | 严格的逐 token 确定性生成（`UD-Q3_K_XL` 达 `59.7 tok/s`，`UD-Q4_K_XL` 达 `47.3 tok/s`）。 |

## 快速开始与编译

### 1. 一键安装脚本（`START-HERE.bat` / `setup.py`）

双击运行 **`START-HERE.bat`** 并选择 **`inkling2`**，或使用命令行自动部署：

```bat
python setup.py --family inkling2 --model UD-Q3_K_XL --kv k8v4 --context 131072 --yes
```

支持从 [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) 下载并校验 `UD-Q3_K_XL`（`13.6 GB`，推荐 12 GB 显存 + 32 GB 内存）、`UD-IQ3_XXS`（`11.7 GB`）、`UD-Q4_K_XL`（`16.4 GiB`）和 `UD-Q8_K_XL`（`22.9 GB`）。预编译发布包地址指向本仓库：`https://github.com/KurrynDev13/PolyStrata-Inkling2/releases/latest/download/`。

### 2. 可选：转换 DSpark MTP 草稿权重

```bat
.venv\Scripts\python.exe tools\convert_inkling2_dspark.py --out models\inkling2\inkling2-dspark.gguf
```

### 3. 从源码编译（Windows, CUDA 13, Ninja）

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -B build-inkling2 -G Ninja -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-inkling2 --target strata-poly poly_inkling2_test -j 8
```

运行 50 项单元测试：

```bat
set PATH=D:\PolyStrata\.venv\Lib\site-packages\nvidia\cu13\bin\x86_64;%PATH%
build-inkling2\poly_inkling2_test.exe
```

## 参考 PC 配置预设

### `UD-Q3_K_XL` + DSpark MTP（默认最快方案）

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

如切换至 **`UD-Q4_K_XL`**（HumanEval+ 最高 `92.7 %`），将 `--model` 替换为 `Inkling-2-Mini-UD-Q4_K_XL.gguf` 即可；如需关闭 MTP，删除 `"--draft-model"` 及对应路径参数。

## 致谢

- [PolyStrata](https://github.com/VecSzn/PolyStrata)（作者 [VecSzn](https://github.com/VecSzn)）与 [Strata](https://github.com/Niko1221/Strata)（作者 [Niko1221](https://github.com/Niko1221)）。
- Thinking Machines Lab（**Inkling-2-Mini**）、[Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)（GGUF 量化与 DSpark 草稿权重）、[llama.cpp](https://github.com/ggml-org/llama.cpp)（`PR #32704` 参考实现）。
- README 版式与基准卡片设计参考 [midokino/kuu-Strata-fork](https://github.com/midokino/kuu-Strata-fork)。
