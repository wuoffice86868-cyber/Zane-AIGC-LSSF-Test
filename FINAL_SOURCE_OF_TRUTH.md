# Prompt Evaluation & Optimization System
## Final Source of Truth

> 编写者：Zane | 最后更新：2026-03-14
> 本文档为项目唯一权威参考，整合了 Leo 的研究文档、生产 Prompt 模板、实验数据及阶段性结论。

---

## 一、项目背景与动机

### 为什么做这个

在 AIGC 视频生产流水线中，Prompt 是连接输入与输出的核心环节。好的 Prompt → 稳定、自然的视频；差的 Prompt → 抖动、动作循环、背景扭曲。全自动化流水线中，这个问题尤其关键。

**当前面临的四个核心问题：**

1. **没有标准** — "好视频"的定义因人而异，缺乏可量化的评判标准
2. **没有数据** — 修改 Prompt 后，效果是变好还是变差？只能靠人工抽查几条，凭直觉判断
3. **没有积累** — 今天的好 Prompt 不一定适用明天不同的场景，经验无法沉淀
4. **没有闭环** — 没有数据就无法系统性改进，只能反复试错

### 核心理念（来自 Anthropic Evaluation Harness）

评测一个系统，评的不是单个模型，而是整个 pipeline（结构化输入 + 模型 + 评分器）的协同效果。

**实际意义**：即使 Seedance（生成模型）固定不变，通过优化 Prompt 结构和 QC 标准，也能持续提升系统表现。评测系统本身是改进工具，不只是检验工具。

### 策略

1. **先建基础设施**：结构化 Prompt + 自动 QC + 数据积累
2. **再用数据驱动优化**：DSPy 自动优化、Reward Model 训练
3. **核心原则**：边生产边评测（build while producing）——评测是持续基础设施，不是一次性检验。流水线运行 → 数据积累 → 数量足够后 → 分析和训练

---

## 二、生产流水线架构

### 主流程

```
原始图片
  ↓
CLIP + YOLO + Aesthetic Scoring（~2.5秒/张）
  ↓
图片分析 JSON（场景、主体、瑕疵、人脸、文字…）
  ↓
Gemini + gemini_image_analysis.txt → Seedream 4.5
（修复瑕疵 / 16:9→9:16 纵向 / 色彩校正 / 人脸处理）
  ↓
增强后的图片
  ↓
Gemini + gemini_cinematography_prompt.txt
（结构化 JSON → 30-45 词 Seedance Prompt）
  ↓
Seedance 1.5 Pro（图生视频，8秒，1080p，9:16）
  ↓
视频
  ↓
QC 旁路（异步，不阻塞生产）
  ↓
Lark Base 写入（全量记录）
```

**当前处理规模：** 45个文件夹，1004张图片，约2.5秒/张

### QC 旁路逻辑

```
视频 → Gemini QC 评分 → 是否通过？
                         ├── 通过 → 保留
                         └── 不通过 → 删除 + 恢复原图
                       → Lark Base（全量结果入库）
```

---

## 三、三份核心 Prompt 模板（生产级）

### 3.1 gemini_image_analysis.txt — 图片分析模板

**作用：** 接收原始图片，输出结构化 JSON，驱动 Seedream 增强

**输出字段：**

| 字段 | 说明 |
|------|------|
| scene | 场景描述（一句话） |
| main_subject | 核心主体（必须保留） |
| subject_position | 主体位置（center / left third 等） |
| current_aspect / target_aspect | 当前/目标宽高比（目标始终为 9:16） |
| extend_top/bottom/left/right | 各边延伸纹理描述（视觉描述，非分类名） |
| preserve | 3-5 个必须保留的元素 |
| lighting_current / lighting_fix | 当前光照 / 需要调整的内容 |
| color_style | 色彩风格（保持自然，不做 HDR） |
| defects | 需要修复的瑕疵列表 |
| faces | 人脸处理：quality=natural → keep；morphed/distorted/blurry → remove person（整人 inpaint 掉） |
| text | 文字处理：signage → preserve；watermark → remove |
| remove | 需要移除/inpaint 的元素 |

**关键规则：** 增强后的图片必须看起来像一张曝光良好的照片，不能像 AI 生成图或 HDR 效果。

---

### 3.2 gemini_enhancement_prompt_9x16.txt — Seedream 增强模板

**作用：** 将图片分析 JSON 转成 Seedream 自然语言增强指令

**6步流程：**

1. **Reframing** — 宽高比转换（16:9→9:16），用纹理描述填充延伸区域
2. **Removals** — 瑕疵/变形人脸 → 用周围纹理填充（fill with surrounding texture）
3. **Lighting & Color** — 仅做细微调整（KEEP IT SUBTLE）
4. （保留步骤）
5. **Preservation** — 明确哪些元素不能改变
6. **Negative Constraints** — 固定结尾："No new text or signs. Enhance subtly, natural photograph look. No fog, haze, lens flares, new people. Extended areas text-free."

**10条规则（重点）：**
- 只用自然语言，不输出 JSON
- 直接指令式表达（"Extend the top with..." 而非 "The top should be extended..."）
- 纹理描述词（"white plaster ceiling" 不是 "ceiling"）
- 始终以 "9:16 vertical format" 开头
- 始终以 "No new text or signs" 结尾
- 每个任务单独一段
- Subtle is better — 目标是自然照片，不是 AI 艺术风

---

### 3.3 gemini_cinematography_prompt.txt — 视频 Prompt 生成模板

**作用：** 接收场景分析 JSON，输出 30-45 词的 Seedance 视频生成 Prompt

**输入字段：**
scene_description, main_subject, foreground, background, camera_move, camera_direction, shot_size, lighting, subtle_motion, stable_element, sky_instruction

**8段式 Prompt 结构：**

| 段 | 内容 |
|----|------|
| #0 | Shot Size（close-up / medium / wide） |
| #1 | Scene Context（一句话场景描述） |
| #2 | Camera Movement（使用官方词汇） |
| #3 | What's Revealed（前景/背景关系） |
| #4 | Lighting（融入自然光照描述） |
| #5 | Subtle Motion（仅一个动态元素） |
| #6 | Stability Anchor（固定元素描述） |
| #7 | Sky/Time Realism（室外场景必加） |

**Seedance 官方确认可用的镜头词汇：**
push / pull / pan left·right / move left·right / orbit / follow / rise / lower / zoom

---

## 四、prompt_evaluator Python 模块

### 模块结构

```
prompt_evaluator/
├── models.py               # Pydantic 数据模型（全量）
├── reward_calculator.py    # 多维度 Reward 计算
├── prompt_analyzer.py      # Prompt 特征提取 + 相关性分析
├── optimizer.py            # OPRO 优化器（已废弃，TextGrad 替代中）
├── dspy_optimizer.py       # DSPy 优化器（新，已接通 Gemini 实测）
├── calibration.py          # QC 校准（Accuracy/Precision/Recall/F1）
├── pipeline.py             # EvalPipeline 主编排器
├── clients/
│   ├── kie_client.py       # Kie.ai API 封装（图片+视频生成）
│   └── gemini_client.py    # Gemini QC + LLM 客户端
└── tests/                  # 142 个单元测试，全部通过
```

### 核心类

**RewardCalculator** — 多维度打分，公式：
```
reward = 50 × pass_flag
       + 20 × (aesthetic_score / 10)
       + 15 × (motion_score / 10)
       + 15 × (adherence_score / 10)
       − minor_issues_count × 2
```

**GeminiVideoQC** — 视频上传到 Gemini File API → 多模态评测
- Auto-Fail 规则（任一触发即 FAIL）：人脸变形（五官错位）/ 手指错误（6根/融合）/ 循环 artifact / 穿透问题 / 结构性塌陷 / 严重物体形变
- Minor Issues（累计 ≥2 则 FAIL）：轻微闪烁 / 颜色不一致 / 边缘细节丢失 / 纹理颤动
- 四维打分（1-10）：aesthetic / motion / prompt_adherence / scroll_stop

**KieClient** — Kie.ai 封装
- 图片生成：Seedream 4.5
- 视频生成：Seedance 1.5 Pro（默认 1080p / 9:16 / 8秒 / generate_audio=false）
- 内置 Budget Tracker + 指数退避轮询

**DSPy Optimizer（新）** — 3个模块，均已用真实 Gemini API 测试通过：
1. 生成模块：场景 JSON → 30-48 词 Seedance Prompt
2. Critique 模块：分析 Prompt 失败原因（已正确识别吊灯形变问题）
3. Template 改进模块：基于失败案例自动重写 System Prompt

---

## 五、System Prompt 演进

| 版本 | 结构 | 平均 Reward | 关键变化 |
|------|------|-------------|---------|
| hotel_v1 | 5段式：[镜头]+[场景]+[运动]+[动态]+[稳定锚点] | ~71 | 基线。"Single continuous shot" 开头，含负面约束 |
| hotel_v2_diverse | 同 v1，增加镜头多样性 | ~71 | 几乎无提升，验证换镜头类型本身没用 |
| hotel_v3 | 主体优先结构，无负面约束，丰富描述 | ~82 | 基于官方指南彻底重写，去掉稳定锚点，加程度副词 |

**v3 对比 v1（12个样本，6个场景）：**

| 场景 | v1 Reward | v3 Reward | Delta |
|------|-----------|-----------|-------|
| pool | 45.2 | 58.4 | +13.2 |
| room | 94.0 | 94.0 | 0 |
| lobby | 92.0 | 69.6 | **−22.4** |
| spa | 69.6 | 90.0 | +20.4 |
| restaurant | 57.2 | 92.0 | **+34.8** |
| beach | 69.6 | 86.0 | +16.4 |
| **平均** | **71.3** | **81.7** | **+10.4** |

v3 在 4/6 场景胜出，Pass Rate 从 2/6 → 4/6。lobby 回退原因：v3 描述了更多复杂细节（吊灯水晶、倒影等），这类复杂场景反而更容易触发 object_morphing。

---

## 六、实验数据总结

### API 消耗（截至 2026-03-12）
- Kie.ai：**$20.31**（约 20 次图片生成 + 39 次视频生成）
- Gemini：约 70 次调用（AI Studio 免费额度内，约 $0）

### 三轮实验结果

| 轮次 | 样本数 | 场景数 | 平均 Reward | 结论 |
|------|--------|--------|-------------|------|
| Round 1（v1 baseline） | 19 | 10 | 70.8 | 建立基线 |
| Round 2（OPRO 改进） | 16 | 6 | 62.9 | OPRO 反而变差 −7.9 |
| Round 3（v3 vs v1） | 12 | 6 | v1=71.3 / v3=81.7 | v3 +10.4，正向改进 |

### 高频失败原因（所有轮次汇总）

| 失败类型 | 出现次数 | 场景关联 |
|----------|----------|---------|
| object_morphing | 12 | 大堂（吊灯）、海滩（植物）、餐厅（建筑结构） |
| action_loop | 7 | 泳池（水面）、水疗（蒸汽） |
| structural_collapse | 5 | 餐厅、大堂 |
| sky_anomaly | 3 | 所有室外场景 |

### 场景难度分级（基于实验数据）

| 难度 | 场景 | 典型 Reward |
|------|------|-------------|
| ✅ 低风险 | 泳池（简单）、客房（单主体）、外观（静态） | 88–94 |
| ⚠️ 中等 | 水疗、餐厅（无复杂玻璃反射） | 70–85 |
| ❌ 高风险 | 大堂（吊灯/倒影）、海滩（多动态元素）、浴室（反射） | 42–65 |

---

## 七、技术研究：优化方法对比

| 方法 | 来源 | 评价 |
|------|------|------|
| **OPRO** | DeepMind, ICLR 2024 | 将 (prompt, score) 对排序后让 LLM 猜规律。理论直觉，实践有限。我们测试：改了 Prompt 结构反而变差 −7.9 |
| **DSPy** ⭐ | Stanford, 2024-2025 | 把 Prompt 工程变成"编程"。模块化、可测试、可追踪。MIPROv2 Bayesian 优化比 OPRO 随机搜索效率高。**主选** |
| **TextGrad** | Nature, 2024 | 用 LLM 文字反馈做"梯度"：输出差 → 分析原因 → "反向传播"改 Prompt。方向性强但每轮成本高（3次 LLM 调用）。**备选** |
| **VPO** | ICCV 2025 | 唯一专为视频生成 Prompt 优化设计的方法，核心是收集人类偏好数据训练优化器。**最相关学术研究** |

**我们测的结论：** OPRO 在 "约束词优化" 方向完全走不通（模型忽略负面约束）。DSPy 的 Critique 模块已验证能正确识别失败原因，Template 改进模块也实测通过。下一步：把 generate → score → critique → improve → loop 全链路跑通。

---

## 八、技术研究：视频评分方法对比

| 方法 | 评价 |
|------|------|
| **当前：Gemini QC** | 自定义 Prompt + Gemini 2.5 Pro 多模态评测。优点：快速、灵活、API 可用。局限：打分标准手写未经 benchmark 校准，跨轮一致性未验证 |
| **VideoScore / VideoScore2** | TIGER-AI-Lab, EMNLP 2024。专为 AI 视频质量评测训练，5维度与人工判断相关性 77%。HF Space 可用（无需 GPU），VideoScore2（2025）带 CoT 解释。**下一步集成验证** |
| **VBench 2.0** | CVPR 2025 标准 benchmark。Human Fidelity / Controllability / Creativity / Physics / Aesthetics 5大维度。适合作为长期对齐标准 |

---

## 九、关键发现（Seedance Prompt 实战规律）

**✅ 有效：**
- 程度副词强烈影响输出："slowly", "gently", "gradually"（酒店内容必用）
- 每个镜头只用一个动词（多动词 = 几何畸变）
- 丰富视觉描述（材质、颜色、光质）显著提升质量
- 风格锚定词有效："cinematic", "editorial", "film grain"

**❌ 无效：**
- 负面 Prompt 完全不生效（官方确认，所有版本）
- 稳定锚点（"stays still", "remains fixed"）被模型忽略
- 器械名称（gimbal, steadicam）在 1.5 Pro 上效果不稳定

**⚠️ 未验证：**
- 正面约束（"Maintain face consistency"）— 2.0 有效，1.5 Pro 未测
- 品质后缀（"4K Ultra HD"）— 可能有效，待 A/B 测试

---

## 十、闭环设计：长期路线图

### 完整闭环

```
生成 Seedance → QC 自动评测 → 数据积累（Lark Base）
      ↑                                    ↓
 更好的视频 ← RLHF/DPO ← Reward Model 训练
```

### 三个具体方向

| 方向 | 内容 | 时间线 |
|------|------|--------|
| 方向1：QC 数据训练 Reward Model | 参考 VideoScore 架构 + VBench-2.0 维度，训练酒店场景专用评分模型 | 中期（数据量依赖） |
| 方向2：Reward 信号优化生成模型 | 参考 T2V-Turbo-v2，RLHF/DPO 让模型学会"生成高分视频" | 长期（需计算资源） |
| 方向3：Fine-tune Prompt Generator | 参考 VPO，用 QC 高分视频对应 Prompt 微调生成模块 | 短期（最快见效） |

### 验证案例：T2V-Turbo-v2

T2V-Turbo-v2（2024）VBench 85.13，学术方法超越 Gen-3/Kling 工业产品。

**关键启示：** Reward 信号设计 > 原始训练数据量。10× 数据可能只提升 10%，设计良好的 Reward 信号可以提升 50%。

---

## 十一、当前进度与下一步

### 已完成

| 组件 | 状态 |
|------|------|
| prompt_evaluator 完整模块 | ✅ 142 测试通过，pip installable |
| 三份生产 Prompt 模板 | ✅ 完整原文存入 prompts/ |
| KieClient（图片+视频生成） | ✅ 真实 API 验证，1080p 默认 |
| GeminiVideoQC（自动评测） | ✅ Gemini 2.5 Pro，多模态 |
| System Prompt v3 | ✅ 基于官方指南重写，+10.4 avg reward |
| DSPy Optimizer（3个模块） | ✅ 实测通过 Gemini 调用 |
| 35个视频样本 + 数据 | ✅ eval_results/ 存档 |
| 两份 TikTok 研究文档（中文） | ✅ Creator Rewards + 去重机制 |

### 立即要做

1. **DSPy 完整优化循环** — 把 generate → score → critique → improve → repeat 跑通，这是当前最大 Gap
2. **QC 一致性验证** — 同一视频跑 3 次取中位数，确认评分稳定
3. **VideoScore2 HF Space 测试** — 验证专业视频评分器能否替代/补充 Gemini QC

### API 约束

- Kie.ai key ✅（已存 .credentials/）
- Gemini key ✅（已存 .credentials/）
- generate_audio 始终 false
- 默认：1080p / 9:16 / 8秒
- Seedance 2.0 API 尚未开放（BytePlus 版权争议延期）

---

## 附录：项目文件索引

| 文件 | 位置 | 说明 |
|------|------|------|
| gemini_image_analysis.txt | prompts/ | 图片分析 Prompt 完整原文 |
| gemini_cinematography_prompt.txt | prompts/ | 视频 Prompt 生成模板（含5个完整示例） |
| gemini_enhancement_prompt_9x16.txt | prompts/ | Seedream 增强 Prompt（含4个示例） |
| integration_architecture.md | prompts/ | Pipeline 接口定义 + 数据格式 |
| hotel_v1/v2/v3.txt | system_prompts/ | System Prompt 各版本 |
| seedance_vocabulary_research.md | research/ | Seedance Prompt 词汇实战研究 |
| v3_comparison_20260313_002051.json | eval_results/ | v3 vs v1 完整对比数据（12 samples） |
| PROJECT.md | 根目录 | 技术细节扩展版（英文，460行） |

