![LOGO](http://opencompass.openxlab.space/utils/MMLB.jpg)

<b>大型视觉-语言模型评估工具包。 </b>

[![][github-contributors-shield]][github-contributors-link] • [![][github-forks-shield]][github-forks-link] • [![][github-stars-shield]][github-stars-link] • [![][github-issues-shield]][github-issues-link] • [![][github-license-shield]][github-license-link]

English | [简体中文](/docs/zh-CN/README_zh-CN.md) | [日本語](/docs/ja/README_ja.md)

<a href="https://rank.opencompass.org.cn/leaderboard-multimodal">🏆 OC 排行榜 </a> •
<a href="#%EF%B8%8F-quickstart">🏗️快速开始 </a> •
<a href="#-datasets-models-and-evaluation-results">📊数据集 & 模型 </a> •
<a href="#%EF%B8%8F-development-guide">🛠️开发指南 </a>

<a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">🤗 HF 排行榜</a> •
<a href="https://huggingface.co/datasets/VLMEval/OpenVLMRecords">🤗 评估记录</a> •
<a href="https://huggingface.co/spaces/opencompass/openvlm_video_leaderboard">🤗 HF 视频排行榜</a> •

<a href="https://discord.gg/evDT4GZmxN">🔊 Discord</a> •
<a href="https://www.arxiv.org/abs/2407.11691">📝 技术报告</a> •
<a href="#-the-goal-of-vlmevalkit">🎯目标 </a> •
<a href="#%EF%B8%8F-citation">🖊️引用 </a>
</div>

**VLMEvalKit**（Python 包名为 **vlmeval**）是一个用于评估**大型视觉-语言模型（LVLMs）**的**开源评估工具包**。它支持在各种基准测试上对 LVLMs 进行**一键式评估**，无需在多个代码库间切换即可完成数据准备。在 VLMEvalKit 中，我们对所有 LVLMs 采用**生成式评估**，并提供了基于**精确匹配**和**LLM 引导的答案提取**的评估结果。

## 🆕 最新动态

> 我们与[**MME 团队**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)和[**LMMs-Lab**](https://lmms-lab.github.io)联合发布了[**大型多模态模型评估综合调研**](https://arxiv.org/pdf/2411.15296) 🔥🔥🔥
- **[2025-02-20]** 支持模型：**InternVL2.5 系列、QwenVL2.5 系列、QVQ-72B、Doubao-VL、Janus-Pro-7B、MiniCPM-o-2.6、InternVL2-MPO、LLaVA-CoT、Hunyuan-Standard-Vision、Ovis2、Valley、SAIL-VL、Ross、Long-VITA、EMU3、SmolVLM**。支持基准测试：**MMMU-Pro、WeMath、3DSRBench、LogicVista、VL-RewardBench、CC-OCR、CG-Bench、CMMMU、WorldSense**。详情请参阅[**VLMEvalKit 功能**](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb)。感谢所有贡献者 🔥🔥🔥
- **[2024-12-11]** 支持[**NaturalBench**](https://huggingface.co/datasets/BaiqiL/NaturalBench)，一个挑战视觉-语言模型处理自然图像简单问题的视觉中心 VQA 基准测试（NeurIPS'24）。
- **[2024-12-02]** 支持[**VisOnlyQA**](https://github.com/psunlpgroup/VisOnlyQA/)，一个评估视觉感知能力的基准测试 🔥🔥🔥
- **[2024-11-26]** 支持[**Ovis1.6-Gemma2-27B**](https://huggingface.co/AIDC-AI/Ovis1.6-Gemma2-27B)，感谢[**runninglsy**](https://github.com/runninglsy) 🔥🔥🔥
- **[2024-11-25]** 新增 `VLMEVALKIT_USE_MODELSCOPE` 标志。通过设置此环境变量，您可以从[**ModelScope**](https://www.modelscope.cn)下载支持的视频基准测试 🔥🔥🔥
- **[2024-11-25]** 支持[**VizWiz**](https://vizwiz.org/tasks/vqa/)基准测试 🔥🔥🔥
- **[2024-11-22]** 支持[**MMGenBench**](https://mmgenbench.alsoai.com)的推理，感谢[**lerogo**](https://github.com/lerogo) 🔥🔥🔥
- **[2024-11-22]** 支持[**Dynamath**](https://huggingface.co/datasets/DynaMath/DynaMath_Sample)，一个包含501个SEED问题和基于随机种子生成的10个变体的多模态数学基准测试。该基准可用于衡量多模态大模型在多模态数学求解中的鲁棒性 🔥🔥🔥
- **[2024-11-21]** 集成了新的配置系统，支持更灵活的评估设置。请参阅[文档](/docs/en/ConfigSystem.md)或运行 `python run.py --help` 了解详情 🔥🔥🔥
- **[2024-11-21]** 支持[**QSpatial**](https://andrewliao11.github.io/spatial_prompt/)，一个用于定量空间推理（如确定大小/距离）的多模态基准测试，感谢[**andrewliao11**](https://github.com/andrewliao11)提供官方支持 🔥🔥🔥
- **[2024-11-21]** 支持[**MM-Math**](https://github.com/kge-sun/mm-math)，一个新的包含约6000道中学多模态推理数学题的多模态数学基准测试。GPT-4o-20240806在此基准上达到22.5%的准确率 🔥🔥🔥

## 🏗️ 快速开始

请参阅[[快速开始](/docs/zh-CN/Quickstart.md)]了解快速入门指南。

## 📊 数据集、模型和评估结果

### 评估结果

**我们官方多模态排行榜的性能数据可从此下载！**

[**OpenVLM 排行榜**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard): [**3 所有详细结果**](http://opencompass.openxlab.space/assets/OpenVLM.json)。

在[**VLMEvalKit 功能**](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb)的**支持的基准测试**标签页中查看所有支持的图像和视频基准测试（70+）。

在[**VLMEvalKit 功能**](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb)的**支持的LMMs**标签页中查看所有支持的LMMs，包括商业API、开源模型等（200+）。

**Transformers版本建议：**

请注意，某些VLMs在特定transformers版本下可能无法运行，我们建议按以下设置评估每个VLM：

- **请对以下模型使用** `transformers==4.33.0`：`Qwen系列`、`Monkey系列`、`InternLM-XComposer系列`、`mPLUG-Owl2`、`OpenFlamingo v2`、`IDEFICS系列`、`VisualGLM`、`MMAlaya`、`ShareCaptioner`、`MiniGPT-4系列`、`InstructBLIP系列`、`PandaGPT`、`VXVERSE`。
- **请对以下模型使用** `transformers==4.36.2`：`Moondream1`。
- **请对以下模型使用** `transformers==4.37.0`：`LLaVA系列`、`ShareGPT4V系列`、`TransCore-M`、`LLaVA (XTuner)`、`CogVLM系列`、`EMU2系列`、`Yi-VL系列`、`MiniCPM-[V1/V2]`、`OmniLMM-12B`、`DeepSeek-VL系列`、`InternVL系列`、`Cambrian系列`、`VILA系列`、`Llama-3-MixSenseV1_1`、`Parrot-7B`、`PLLaVA系列`。
- **请对以下模型使用** `transformers==4.40.0`：`IDEFICS2`、`Bunny-Llama3`、`MiniCPM-Llama3-V2.5`、`360VL-70B`、`Phi-3-Vision`、`WeMM`。
- **请对以下模型使用** `transformers==4.42.0`：`AKI`。
- **请对以下模型使用** `transformers==4.44.0`：`Moondream2`、`H2OVL系列`。
- **请对以下模型使用** `transformers==4.45.0`：`Aria`。
- **请对以下模型使用** `transformers==latest`：`LLaVA-Next系列`、`PaliGemma-3B`、`Chameleon系列`、`Video-LLaVA-7B-HF`、`Ovis系列`、`Mantis系列`、`MiniCPM-V2.6`、`OmChat-v2.0-13B-sinlge-beta`、`Idefics-3`、`GLM-4v-9B`、`VideoChat2-HD`、`RBDash_72b`、`Llama-3.2系列`、`Kosmos系列`。

**Torchvision版本建议：**

请注意，某些VLMs在特定torchvision版本下可能无法运行，我们建议按以下设置评估每个VLM：

- **请对以下模型使用** `torchvision>=0.16`：`Moondream系列`和`Aria`

**Flash-attn版本建议：**

请注意，某些VLMs在特定flash-attention版本下可能无法运行，我们建议按以下设置评估每个VLM：

- **请对以下模型使用** `pip install flash-attn --no-build-isolation`：`Aria`

```python
# 示例
from vlmeval.config import supported_VLM
model = supported_VLM['idefics_9b_instruct']()
# 单张图像前向推理
ret = model.generate(['assets/apple.jpg', '这张图片中有什么？'])
print(ret)  # 图片中有一个带有叶子的红苹果。
# 多张图像前向推理
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', '提供的图片中有多少个苹果？ '])
print(ret)  # 提供的图片中有两个苹果。
```

## 🛠️ 开发指南

要开发自定义基准测试、VLMs或为**VLMEvalKit**贡献其他代码，请参阅[[开发指南](/docs/zh-CN/Development.md)]。

**征集贡献**

为促进社区贡献并分享相应成果（在下次报告更新中）：

- 所有贡献将在报告中致谢。
- 贡献3次或以上主要贡献（实现MLLM、基准测试或主要功能）的贡献者可加入[VLMEvalKit技术报告](https://www.arxiv.org/abs/2407.11691)的作者名单。符合条件的贡献者可在[VLMEvalKit Discord频道](https://discord.com/invite/evDT4GZmxN)创建issue或私信kennyutc。

以下是根据记录整理的[贡献者名单](/docs/en/Contributors.md)。

## 🎯 VLMEvalKit的目标

**本代码库旨在：**

1. 提供一个**易用的**、**开源的评估工具包**，方便研究人员和开发者评估现有LVLMs，并使评估结果**易于复现**。
2. 让VLM开发者能轻松评估自己的模型。要在多个支持的基准测试上评估VLM，只需**实现一个单独的`generate_inner()`函数**，所有其他工作负载（数据下载、数据预处理、预测推理、指标计算）都由代码库处理。

**本代码库不旨在：**

1. 复现所有**第三方基准测试**原始论文中报告的精确准确率数字。原因可能有两方面：
   1. VLMEvalKit对所有VLMs采用**生成式评估**（并可选使用**LLM引导的答案提取**）。而某些基准测试可能使用不同方法（例如SEEDBench使用PPL-based评估）。对于这些基准测试，我们在相应结果中同时比较两种分数。我们鼓励开发者在代码库中支持其他评估范式。
   2. 默认情况下，我们对所有VLMs使用相同的提示模板来评估某个基准测试。然而，**某些VLMs可能有特定的提示模板**（有些可能目前未被代码库覆盖）。我们鼓励VLM开发者在VLMEvalKit中实现自己的提示模板，如果当前尚未覆盖的话。这将有助于提高复现性。

## 🖊️ 引用

如果您觉得本工作有帮助，请考虑**给这个仓库加星标🌟**。感谢您的支持！

[![Stargazers repo roster for @open-compass/VLMEvalKit](https://reporoster.com/stars/open-compass/VLMEvalKit)](https://github.com/open-compass/VLMEvalKit/stargazers)

如果您在研究中使用VLMEvalKit或希望引用已发布的开源评估结果，请使用以下BibTeX条目以及您使用的特定VLM/基准测试对应的BibTeX条目。

```bib
@inproceedings{duan2024vlmevalkit,
  title={Vlmevalkit: An open-source toolkit for evaluating large multi-modality models},
  author={Duan, Haodong and Yang, Junming and Qiao, Yuxuan and Fang, Xinyu and Chen, Lin and Liu, Yuan and Dong, Xiaoyi and Zang, Yuhang and Zhang, Pan and Wang, Jiaqi and others},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={11198--11201},
  year={2024}
}
```

<p align="right"><a href="#top">🔝返回顶部</a></p>

[github-contributors-link]: https://github.com/open-compass/VLMEvalKit/graphs/contributors
[github-contributors-shield]: https://img.shields.io/github/contributors/open-compass/VLMEvalKit?color=c4f042&labelColor=black&style=flat-square
[github-forks-link]: https://github.com/open-compass/VLMEvalKit/network/members
[github-forks-shield]: https://img.shields.io/github/forks/open-compass/VLMEvalKit?color=8ae8ff&labelColor=black&style=flat-square
[github-issues-link]: https://github.com/open-compass/VLMEvalKit/issues
[github-issues-shield]: https://img.shields.io/github/issues/open-compass/VLMEvalKit?color=ff80eb&labelColor=black&style=flat-square
[github-license-link]: https://github.com/open-compass/VLMEvalKit/blob/main/LICENSE
[github-license-shield]: https://img.shields.io/github/license/open-compass/VLMEvalKit?color=white&labelColor=black&style=flat-square
[github-stars-link]: https://github.com/open-compass/VLMEvalKit/stargazers
[github-stars-shield]: https://img.shields.io/github/stars/open-compass/VLMEvalKit?color=ffcb47&labelColor=black&style=flat-square
