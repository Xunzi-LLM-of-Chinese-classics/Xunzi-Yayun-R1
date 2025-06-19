<div align="center">
  <img src="https://github.com/ricardozhy/QPM-1K-32B-R1/blob/main/%E5%94%90%E8%AF%97logo.png?raw=true" width="20%" />
</div>

# Xunzi-Yayun-R1

<div align="center">

[![ModelScope](https://img.shields.io/badge/ModelScope-07ced1?style=flat&logo=modelscope&logoColor=white)](https://modelscope.cn/models/njauzwh/Xunzi-Yayun-R1/summary)
[![GitHub Stars](https://img.shields.io/github/stars/Xunzi-LLM-of-Chinese-classics/Xunzi-Yayun-R1?style=social)](https://github.com/Xunzi-LLM-of-Chinese-classics/Xunzi-Yayun-R1)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Yayun--R1-yellow)](https://huggingface.co/ricardozhy/Xunzi-Yayun-R1)
![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen)

</div>

## 简介

Xunzi-Yayun-R1 是一个基于GRPO强化学习的小样本唐诗生成推理模型。该模型致力于解决传统唐诗生成面临的两大核心挑战：一方面，避免对超大规模参数量模型的依赖，降低算力消耗；另一方面，克服“形神割裂”现象，使生成的诗歌既符合格律要求，又具备较高的艺术表现力。

Xunzi-Yayun-R1 通过“规则编码-知识蒸馏-动态强化-检索增强”的方法论体系，在仅有32B参数规模的情况下，成功实现了优于DeepSeek-R1-671B等超大模型的唐诗生成能力。

## 主要特点

- **低资源高效能**：仅使用1K数据，32B参数规模，显著降低了推理能耗，使文化遗产数字化更加经济可行
- **格律准确性卓越**：平仄、押韵、对仗、字数控制准确性显著，押韵准确率高达91.23%
- **艺术表现力优异**：通过知识蒸馏和RAG技术，解决了“形神割裂”问题，生成诗歌意境深远
- **技术创新性强**：首次将离散的诗歌格律规则转化为可微调的强化学习奖励信号
- **通用框架可迁移**：构建的技术框架可推广应用于其他古籍文本生成领域

## 使用方法

### 模型加载

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_id = "njauzwh/Xunzi-Yayun-R1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
```

您也可以从Hugging Face加载模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ricardozhy/Xunzi-Yayun-R1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
```

### 推理示例

```python
system_prompt = "Respond in the following format:<think>...</think><answer>...</answer>"


query = "请以'春风'为题创作一首五言绝句，押平水韵东韵"
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": query}
]
response = model.chat(tokenizer, messages)
print(response)

```

### 格律要求说明

Xunzi-Yayun-R1 支持以下格律要求的诗歌创作：

- **诗体**：绝句、律诗
- **字数**：五言、七言
- **平仄**：遵循唐诗的平仄规则
- **押韵**：支持平水韵，可指定韵部
- **题材/意象**：可指定创作主题、题材和意象词汇

## 技术细节

Xunzi-Yayun-R1 基于以下技术创新：

1. **GRPO强化学习**：使用Group Relative Policy Optimization对模型进行训练，将离散的诗歌格律转化为可微调奖励信号

2. **知识定向蒸馏**：通过DeepSeek-R1-671B对数据进行蒸馏，使用冷启动策略初始化参数

3. **RAG检索增强**：集成《平水韵》库驱动的实时检索机制，动态优化诗歌韵律

4. **规则编码机制**：建立规则连续化编码机制，将诗歌格律规则编码为模型可优化的形式

## 评估结果

### 详细评测

下表展示了Xunzi-Yayun-R1与其他模型在唐诗生成任务上的详细对比评测结果：

| 模型类型 | 是否冷启动 | 模型名称 | 平仄(tones) | 押韵(rhymes) | 对仗(antithesis) | 字数(length) | 总分(total) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 推理模型+RAG | 冷启动 | **Xunzi-Yayun-R1-32B** | 75.63 | **91.23** | 94.20 | 98.76 | **86.34** |
| 推理模型+RAG | 冷启动 | Qwen2.5-32B-Instruct-RAG | 76.81 | 87.86 | 94.69 | 99.77 | 86.00 |
| 推理模型+RAG | 未冷启动 | Qwen2.5-32B-Instruct-GRPO-RAG | 80.89 | 83.26 | 93.88 | 97.55 | 85.86 |
| 推理模型 | / | DeepSeek-R1-671B | 79.94 | 80.92 | 94.67 | 99.59 | 85.15 |
| 数据集 | / | 唐诗三百首 | 72.99 | 87.20 | 93.72 | 98.13 | 83.91 |
| 推理模型 | 冷启动 | Xunzi-Yayun-R1-32B | 77.74 | 77.36 | 94.85 | 99.80 | 83.25 |
| 数据集 | / | 全唐诗 | 71.57 | 85.96 | 93.18 | 97.62 | 82.81 |
| 推理模型 | 未冷启动 | Qwen2.5-32B-Instruct-GRPO | 79.74 | 72.38 | 94.38 | 99.22 | 82.41 |
| 推理模型+RAG | 冷启动 | Qwen2.5-14B-Instruct-RAG | 72.28 | 87.54 | 90.63 | 91.47 | 82.44 |



## 应用场景

- 古典诗词创作辅助
- 数字人文研究
- 文化遗产数字化
- 教育领域的古典文学教学
- 文化创意产业内容生成


## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 致谢

感谢所有为本项目做出贡献的研究人员和开发者。

## 联系方式

如有任何问题，请通过以下方式联系我们：

- GitHub Issues: [提交问题](https://github.com/Xunzi-LLM-of-Chinese-classics/Xunzi-Yayun-R1/issues)
- 邮箱：zhaowenhua@njau.edu.cn
