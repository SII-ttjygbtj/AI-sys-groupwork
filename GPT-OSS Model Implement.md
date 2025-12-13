# Model Implement

## Problem Statement
当前，TPU在LLM推理领域的软件生态和基础设施支持相对薄弱，这限制了其潜力的发挥。JAX作为TPU上的主流运行时环境，为构建高性能计算生态提供了基础。SGLang-JAX项目旨在将专为LLM推理优化的SGLang后端集成到JAX生态中，从而在TPU上建立高效的LLM推理能力。然而，一个关键的生态缺口在于，OpenAI发布的高性能开源模型GPT-OSS目前缺乏成熟的JAX框架实现，导致社区无法直接在TPU上高效评估和部署该模型。

针对这一问题，我们计划在SGLang-JAX上实现GPT-OSS模型，并依据TPU的体系结构探索深度的性能优化方案。我们的最终目标是交付一个相比baseline实现性能有显著提升、可投入生产环境使用的模型，从而充分挖掘TPU在LLM推理任务上的潜力。

## Proposed Method
我们最终要在sglang-jax中重写模型定义、加载开源weight，而后对接sglang-jax中的kvcache manager和flashattention，来初步达成现阶段推理引擎的成熟特性。

我们小组由两位专注于算法模型和两位专注于系统实现的成员构成；我们的方案将遵循一个循序渐进的路径（如图1），能充分发挥组员能力、并确保风险可控。

### 时间规划（Roadmap）
| 时间节点       | 核心任务                                                                 | 时长 |
|----------------|--------------------------------------------------------------------------|------|
| 前期准备       | GPU上跑通pytorch朴素定义[6]                                              | 1天  |
| 前期准备       | 摸索TPU运行时，随意挑一个模型跑通bonsai[4]                               | 1天  |
| 前期准备       | 不同TP、不同batch size下，测试TTFT、吞吐量                               | 1天  |
| 2025-12-08前   | 重写模型定义                                                             | 2天  |
| 2025-12-08前   | 端到端精度Benchmark（如infiniteBench[3]）                                | 2天  |
| 2025-12-08前   | Weight加载，在bonsai[4]上跑通GPT-OSS[1,2]                                | 3天  |
| 2025-12-08前   | 对齐精度                                                                 | 1天  |
| 2025-12-08前   | 调通TP                                                                   | 2天  |
| 2025-12-08     | 模型定义转SGLang-JAX                                                     | 1天  |
| 2025-12-15前   | FlashAttention对接[7]                                                    | 4天  |
| 2025-12-15前   | KVCache Manager对接                                                      | 4天  |
| 2025-12-15前   | 移植SGLang-JAX[5] GPT-OSS[1,2]                                           | -    |
| 2025-12-15后   | 模型规模从20B扩展至120B                                                  | 2天  |
| 2025-12-15后   | 性能调优                                                                 | 4天  |
| 选做（拔高）   | 对比不同Batch Size下的模型并行策略与算子运行效率、Roofline差异，完成第一版系统优化 | -    |
| 选做（拔高）   | 利用Pallas手写kernel进一步提升整体性能                                    | -    |

### 实施步骤
1. 首先，在GPU上运行并验证开源的PyTorch版GPT-OSS，建立性能与准确性的baseline。
2. 借助Bonsai项目，在JAX上实现GPT-OSS的朴素定义，熟悉TPU运行时环境和模型定义；重点解决weight加载、掌握TPU上的distributed tensor技术点，调通TP，形成初步JAX版本。
3. 在初步JAX版本上，与baseline进行精度对齐，确保端到端准确性达到baseline水平；并使用attention的朴素实现完成性能测试。
4. 将初步JAX版本模型集成到SGLang-JAX框架中，接入KVCache manager、FlashAttention，形成初版优化系统。
5. 将模型规模从20B参数扩展至120B，验证方法在大规模模型上的有效性。
6. （选做拔高）对比不同Batch Size下的模型并行策略与算子的运行效率与Roofline之间的差异，完成第一版系统优化。
7. （选做拔高）利用Pallas手写kernel进一步提升整体性能。

## Evaluation Plan
### 功能正确性
以GPU-PyTorch朴素定义为baseline，在标准评测数据集上对比TPU版的端到端任务准确性，确保优化不引入功能误差。

### 性能评估
模拟实际的长文本交互场景（输入8K Token，输出1K Token），测试TTFT（首次token输出时间）和throughput（吞吐量）；将TPU系统性能与同代际GPU上的PyTorch朴素定义进行对比。

## 参考文献
[1] https://huggingface.co/openai/gpt-oss-20b  
[2] https://github.com/openai/gpt-oss/tree/main  
[3] https://github.com/OpenBMB/InfiniteBench  
[4] https://github.com/jax-ml/bonsai/tree/main  
[5] https://github.com/sgl-project/sglang-jax/  
[6] https://github.com/openai/gpt-oss/tree/main/gpt_oss/torch  
[7] https://github.com/sgl-project/sglang-jax/blob/main/python/sgl_jax/srt/layers/attention/base_attn_backend.py  

> 注：本项目同时作为冯老师课程作业提交