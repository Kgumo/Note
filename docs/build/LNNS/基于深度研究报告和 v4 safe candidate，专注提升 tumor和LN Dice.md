# 提示词：基于深度研究报告和 v4 safe candidate，专注提升 tumor和LN Dice

我们现在重新聚焦，不讨论 Q2 论文包装，不讨论 HD95，不讨论下游诊断/预测模型。  
当前唯一目标是：**基于已有 v4 safe candidate，建立可复现、可回滚、可追踪的 Dice rescue pipeline，尽可能提升 tumor Dice 和 LN Dice。**

我接下来会上传：

```text
1. 深度文献研究报告；
2. 当前历史记录 / 代码 / 日志 / CSV / v4 文件；
3. 必要的 evaluation、fusion、inference、postprocess 脚本。
```

请你先阅读这些材料，再继续工作。

---

## 一、当前唯一正式基线

当前正式版本名：

```text
ln_thr065_combo_joint_refine_v4_tumor_airhu
```

当前正式 full-case 指标：

```text
tumor Dice = 0.532261
LN Dice    = 0.289387
```

当前判断：

```text
v4 是目前唯一正式 safe candidate。
v4 不是最终结果，只是后续 Dice rescue 的起点。
当前核心目标是提升 Dice，不是写论文叙事。
```

---

## 二、已经作废 / 暂停的内容

### 1. v5_ln_dice_safe 已作废

原因：

```text
v5 错误假设 local_component_id == pred_component_id，
导致误删 LN true positive，
使 LN Dice 从 0.289387 掉到 0.119937。
```

禁止：

```text
不要沿用 v5；
不要复用 v5 的 component 删除逻辑；
不要假设 local_component_id == pred_component_id；
不要在没有 reliable overlap matching 的情况下删除 LN component。
```

---

### 2. tumorB_zlong_s15_besttumor_raw 暂停

原因：

```text
samegrid audit 显示：
total_components = 137
total_pure_fp_components = 137
total_gt_overlap_vox_after_resample = 0
```

这更像是：

```text
pred_dir 错；
case_id 不匹配；
label 映射错；
shape 不一致；
spacing 不一致；
GT 对齐错误；
resample / paste-back 错误。
```

禁止：

```text
不要直接把 tumorB_zlong_s15_besttumor_raw 当有效结果；
不要直接拿它 fusion；
不要基于它判断 tumorB 无效；
必须先查路径、case_id、label、shape、spacing、resample、paste-back。
```

---

## 三、当前总原则

请严格遵守：

```text
1. 不要重新开题；
2. 不要换方向；
3. 不要用 Q2 论文话术掩盖 Dice 问题；
4. 不要讨论 HD95，除非我后面明确要求；
5. 不要讨论下游诊断/预测模型，先把分割 Dice 救起来；
6. 不要说“Dice 不重要”；
7. 不要用 lesion-level evaluation 替代 Dice；
8. lesion-level、size-bin、component analysis 只能作为诊断工具，最终必须回到 Dice 提升；
9. 不要盲目训练；
10. 不要乱开 v6/v7；
11. 不要大改 backbone；
12. 不要虚构路径、实验、指标、结论；
13. 不要把 audit CSV 当 official metrics；
14. 不要在未复现 v4 前做任何优化。
```

---

## 四、项目主线保留，但当前只服务 Dice

项目长期主线仍然是：

```text
CT tumor/LN 自动勾画
→ Stage-A high-recall proposal generator
→ Stage-B real-Δt / liquid sequence ROI refinement
→ size-aware LN refinement
→ tumor/LN label-wise fusion
→ 后续双流液态诊断/预测模型
```

但是当前阶段只做一件事：

```text
提升 tumor Dice 和 LN Dice。
```

请保留但不要展开这些创新点：

```text
1. Stage-A high-recall proposal；
2. Stage-B ROI refinement；
3. real-Δt / 真实物理层间距；
4. Conv-CfC / liquid sequence modeling；
5. size-aware LN refinement；
6. tumor/LN 分标签、分分支或 label-wise fusion；
7. 后续 tumor flow + LN flow + clinical branch。
```

普通 2D U-Net、3D U-Net、ConvLSTM、nnU-Net 等只能作为 baseline 或 ablation，不要建议替换主线。

---

# 当前任务：Dice rescue

请把所有工作围绕下面两个目标展开：

```text
目标 1：提升 tumor Dice，当前 baseline = 0.532261；
目标 2：提升 LN Dice，当前 baseline = 0.289387。
```

不要用其他指标转移重点。  
所有诊断、表格、实验、代码修改，都必须回答：

```text
它为什么可能提升 Dice？
它先验证什么假设？
它的风险是什么？
失败后如何回滚？
成功标准是什么？
```

---

## 第一步：锁路径、锁 baseline、复现 v4

请首先确认以下路径和文件：

```text
1. v4 pred_dir；
2. GT dir；
3. official decomp dir；
4. metrics dir；
5. fullcase_metrics.csv；
6. sizebin_metrics.csv；
7. pred_component_matches.csv；
8. fp_components.csv；
9. gt_lesion_components.csv；
10. stageA_coverage.csv；
11. shape_resampling_log.csv；
12. evaluation script；
13. inference script；
14. fusion script；
15. postprocess script；
16. case list；
17. label definition。
```

要求：

```text
路径不能猜；
文件不存在就明确说不存在；
路径不确定就让我提供；
不允许根据记忆补路径；
不允许用 audit CSV 代替 official metrics；
必须先复现 v4 的 tumor Dice = 0.532261，LN Dice = 0.289387。
```

如果复现不了，停止优化，先查：

```text
case list 是否一致；
pred_dir 是否正确；
GT dir 是否正确；
label 1 / label 2 是否映射正确；
spacing 是否一致；
shape 是否一致；
resample 是否正确；
paste-back 是否正确；
evaluation script 是否一致；
postprocess/fusion 是否与 v4 一致。
```

---

## 第二步：只为提升 Dice 做错误分解

请不要泛泛分析。  
请把 tumor Dice 和 LN Dice 低的原因拆成可验证假设。

---

# A. Tumor Dice rescue 诊断

当前：

```text
tumor Dice = 0.532261
```

请逐项判断 tumor Dice 低来自哪里：

```text
1. Stage-A tumor proposal 没覆盖；
2. ROI crop 不完整；
3. z 方向上下文不足；
4. Stage-B ROI 内 tumor 分割差；
5. paste-back 坐标错误；
6. resize/resample 错误；
7. pred 与 GT case_id 不匹配；
8. label 映射错误；
9. fusion 时 LN 抢占 tumor；
10. postprocess 删除 tumor TP；
11. tumor FP components 太多；
12. tumor FN / under-seg 太明显；
13. tumor probability threshold 不合适；
14. crop 太小导致边界断；
15. seq_len 太短导致 z 连续性差。
```

请输出 tumor 相关诊断表或脚本需求：

```text
per-case tumor Dice；
tumor TP / FP / FN volume；
tumor pred component matches；
tumor FP components；
tumor FN regions；
tumor ROI crop coverage；
tumor paste-back audit；
tumor pre-fusion vs post-fusion Dice；
tumor pre-postprocess vs postprocess Dice；
tumor threshold sweep；
tumor case-level gain/loss table。
```

每个结论必须说明：

```text
证据来自哪个文件；
是否能解释 Dice 低；
是否值得改；
预期怎么提升 Dice。
```

---

# B. LN Dice rescue 诊断

当前：

```text
LN Dice = 0.289387
```

请逐项判断 LN Dice 低来自哪里：

```text
1. Stage-A LN recall 不足；
2. ROI crop 没覆盖 LN；
3. Stage-B ROI 内 LN 分割差；
4. LN FP components 太多；
5. LN FN lesions 太多；
6. threshold 不合适；
7. postprocess 误删 LN TP；
8. component ID 映射错误；
9. fusion 时 tumor 抢占 LN；
10. label 2 映射或读写错误；
11. spacing/resample/paste-back 错误；
12. 小 LN 导致 voxel Dice 天然不稳定；
13. LN oversampling 或 loss 权重不合适；
14. LN positive/negative batch 构成不合理；
15. LN presence gate 过松或过严。
```

请输出 LN 相关诊断表或脚本需求：

```text
per-case LN Dice；
LN TP / FP / FN volume；
LN pred_component_matches；
LN fp_components；
LN gt_lesion_components；
LN FN lesion table；
LN size-bin Dice；
LN size-bin recall；
LN threshold sweep；
LN pre-fusion vs post-fusion Dice；
LN pre-postprocess vs postprocess Dice；
LN component deletion safety audit；
LN case-level gain/loss table。
```

注意：

```text
size-bin、lesion-level、component analysis 不是最终成绩；
它们只是为了定位 LN Dice 为什么低；
最终目标仍然是 LN Dice 提升。
```

---

## 第三步：给出最小 Dice rescue 路线

请按低风险优先级给出下一步路线。

不要一次性建议很多大实验。  
每一步必须是：

```text
可复现；
可回滚；
改动小；
能解释；
能判断成败。
```

建议优先级如下：

---

### 1. Evaluation / path / resample 修正

优先级最高。  
因为 tumorB_zlong raw 出现 0 overlap，说明可能存在 pipeline 错误。

请先查：

```text
case_id；
pred_dir；
GT dir；
label mapping；
shape；
spacing；
origin / direction；
resample；
paste-back；
保存 dtype；
NIfTI metadata；
evaluation 读取逻辑。
```

如果发现错误，请先修 pipeline，不要训练。

---

### 2. Threshold / postprocess / fusion 小范围修正

在 baseline 复现后，先做无需训练的 Dice rescue：

```text
tumor threshold sweep；
LN threshold sweep；
tumor component size filter；
LN component size filter，但不能误删 TP；
presence gate 调整；
tumor/LN label priority 调整；
probability-based fusion；
case-level best threshold diagnostic；
postprocess before/after delta。
```

要求：

```text
必须报告修改前后 tumor Dice 和 LN Dice；
必须报告哪些 case gain、哪些 case loss；
不能只报平均值；
不能为了提升 tumor Dice 大幅牺牲 LN Dice，反之亦然；
不能复用 v5 的错误 component 删除逻辑。
```

---

### 3. Tumor-specific Dice rescue

只有当确认 pipeline 无误后，才考虑 Tumor-B zlong。

Tumor-B zlong 的目的：

```text
在保留 Stage-B / real-Δt / liquid sequence 主线下，
给 tumor 更大的 crop 和更长 z-context，
专门提升 tumor Dice。
```

建议搜索空间：

```text
crop = 128 或 160；
seq_len = 11 / 15 / 17；
tumor loss 权重提高；
LN loss 降低或不参与；
只用 tumor_only + both；
固定 LN-B，不动；
只更新 tumor branch 或 tumor-specific refinement。
```

每个实验必须说明：

```text
验证什么假设；
输入 checkpoint 是什么；
输出 pred_dir 是什么；
是否 real-Δt；
是否 Conv-CfC；
是否和 v4 同 case list；
是否用同一 evaluation script；
tumor Dice 是否提升；
LN Dice 是否保持；
失败时如何回滚。
```

---

### 4. LN-specific Dice rescue

LN 不能盲训。  
请先根据诊断结果决定是 recall 问题还是 FP 问题。

如果是 FN / recall 问题，优先考虑：

```text
提高 LN positive sampling；
增加 small LN / tiny LN 采样；
降低 LN threshold；
调整 LN loss 权重；
增加 LN crop coverage；
增加 LN z-context；
检查 Stage-A LN proposal recall；
检查 LN ROI 是否覆盖 GT；
```

如果是 FP / precision 问题，优先考虑：

```text
hard negative mining；
LN absent loss；
LN top-k FP loss；
component-level FP rejector；
presence gate；
threshold 提高；
但必须保证不误删 LN TP。
```

LN component-level FP rejector 必须满足：

```text
不能假设 local_component_id == pred_component_id；
必须基于 overlap matching；
必须区分 TP component、FP component、ambiguous component；
必须报告删除前后 LN Dice；
必须报告删除前后 LN TP 是否损失；
必须报告删除前后 LN FP volume；
如果 LN Dice 下降，立即回滚。
```

候选特征：

```text
component volume；
bbox size；
short axis；
long axis；
z-span；
compactness；
elongation；
max probability；
mean probability；
CT intensity mean/std；
distance to tumor；
slice continuity；
near tumor boundary；
near vessel-like structure。
```

---

### 5. Joint fusion / label-wise fusion

最后才做 fusion。  
fusion 前必须已有：

```text
tumor 分支可靠；
LN 分支可靠；
component matching 可靠；
postprocess 不误删 TP。
```

fusion 必须输出：

```text
fusion 前 tumor Dice；
fusion 后 tumor Dice；
fusion 前 LN Dice；
fusion 后 LN Dice；
tumor 被 LN 抢占 voxel；
LN 被 tumor 抢占 voxel；
conflict components；
gain cases；
loss cases；
失败 case 可视化列表。
```

如果 fusion 只提升一个标签、明显伤另一个标签，需要说明是否接受。  
默认原则：

```text
不能为了微小提升 tumor Dice，把 LN Dice 打崩；
不能为了微小提升 LN Dice，把 tumor Dice 打崩。
```

---

## 第四步：实验命名和版本纪律

不要乱开版本。  
每个新实验必须命名清楚：

```text
dice_rescue_v4_evalfix_xxx
dice_rescue_v4_thr_xxx
dice_rescue_v4_tumorB_zlong_xxx
dice_rescue_v4_lnFPsafe_xxx
dice_rescue_v4_fusion_xxx
```

每个实验必须记录：

```text
实验名；
基于哪个版本；
输入路径；
输出路径；
修改内容；
预期提升哪个 Dice；
实际 tumor Dice；
实际 LN Dice；
是否保留；
是否回滚；
失败原因。
```

如果实验失败，不要继续叠加失败实验。  
先回到 v4 safe candidate。

---

## 第五步：深度研究报告怎么用

我会上传深度研究报告。  
你要从里面提取对 Dice rescue 有直接帮助的内容，不要泛泛复述文献。

报告只用于支持这些方向：

```text
1. two-stage pipeline 的常见失效点；
2. Stage-A recall / crop coverage / Stage-B ROI Dice / paste-back error 的诊断；
3. LN 小目标 Dice 低的原因；
4. LN FP/FN 分解；
5. candidate + FP rejection 的安全做法；
6. size-aware LN 训练或评价；
7. tumor 和 LN 是否需要分标签 refinement；
8. real-Δt / z-context 对 tumor Dice 的潜在帮助。
```

不要把报告用于：

```text
绕开 Dice；
证明当前结果已经足够；
包装 Q2 论文；
讨论 HD95；
讨论诊断/预测模型；
把 LN Dice 低合理化为“无所谓”。
```

---

## 第六步：你接下来输出的格式

请严格按下面格式回答。

---

### 第一部分：当前状态确认

请确认：

```text
v4 是唯一正式 safe candidate；
tumor Dice = 0.532261；
LN Dice = 0.289387；
v5 已作废；
tumorB_zlong raw 暂停；
当前唯一目标是 Dice rescue。
```

---

### 第二部分：先不能做什么

请列出当前禁止事项：

```text
不能盲训；
不能乱开版本；
不能大改网络；
不能用 Q2 话术掩盖 Dice；
不能讨论 HD95；
不能先谈诊断/预测；
不能把 lesion-level evaluation 当成 Dice 替代；
不能猜路径；
不能用 v5 删除逻辑；
不能在未复现 v4 前优化。
```

---

### 第三部分：最小文件需求

不要让我一次性上传所有东西。  
先列出第一批最小必要文件：

```text
1. v4 README；
2. v4 fullcase_metrics.csv；
3. v4 pred_component_matches.csv；
4. v4 fp_components.csv；
5. v4 gt_lesion_components.csv；
6. v4 stageA_coverage.csv；
7. v4 shape_resampling_log.csv；
8. evaluation script；
9. fusion/postprocess script；
10. v4 pred_dir 路径说明；
11. GT dir 路径说明。
```

如果缺文件，明确说缺什么，不要猜。

---

### 第四部分：复现 v4 的检查清单

请给出复现：

```text
tumor Dice = 0.532261
LN Dice = 0.289387
```

所需检查项：

```text
case list；
label mapping；
pred_dir；
GT dir；
shape；
spacing；
resample；
evaluation；
postprocess；
fusion。
```

---

### 第五部分：Dice 低原因诊断表

请分别给出 tumor 和 LN 的诊断表，格式为：

```text
可能原因 | 如何验证 | 需要文件 | 如果成立怎么改 | 预期影响 Dice | 风险
```

---

### 第六部分：第一轮 Dice rescue 方案

请只给第一轮低风险方案，不要一次给十几个大实验。

第一轮优先：

```text
1. evaluation / path / resample / paste-back 修正；
2. threshold sweep；
3. postprocess before/after delta；
4. fusion conflict audit；
5. tumor/LN component gain-loss analysis。
```

每个方案必须说明：

```text
目标提升 tumor Dice 还是 LN Dice；
是否需要训练；
预计风险；
如何回滚；
成功标准。
```

---

### 第七部分：如果第一轮不能提升，再进入训练方案

只有在复现和诊断完成后，才允许给训练方案：

```text
Tumor-B zlong；
LN size-aware refinement；
LN hard negative mining；
safe component-level FP rejector；
label-wise fusion。
```

每个训练方案必须说明：

```text
为什么现在该做；
解决哪个已证实问题；
预期提升哪个 Dice；
需要哪些文件；
需要改哪些代码；
不能做什么；
失败如何回滚。
```

---

## 最后强调

请记住：

```text
这次任务不是写论文，不是做包装，不是讨论 Q2，不是讨论 HD95，不是讨论诊断/预测。
这次任务就是把 tumor Dice 和 LN Dice 提上去。
所有额外评价只能作为诊断工具，不能替代 Dice。
```

现在请从我上传的深度研究报告和 v4 材料开始，帮我建立最小、可复现、低风险的 Dice rescue pipeline。