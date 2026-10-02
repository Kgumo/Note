# Stage A Manuscript Section  
# Two-Stage ROI-Guided Framework for Head-and-Neck Tumor and Lymph Node Segmentation: Stage-A Proposal Generation and Coverage Validation

---

## 1. One-Sentence Core Contribution

**English**

Stage A is a coverage-oriented, class-aware ROI proposal generator that converts head-and-neck tumor and lymph node auto-contouring from a full-volume search problem into a high-recall spatial screening problem, providing reliable lesion-level entry points for subsequent fine segmentation.

**中文翻译**

Stage A 是一个以覆盖率为核心、类别感知的 ROI 候选区域生成器，将头颈部肿瘤与淋巴结自动勾画从全图搜索问题转化为高召回率的空间筛选问题，为后续精细分割提供可靠的病灶级空间入口。

---

## 2. Research Question of Stage A

**English**

Stage A does not aim to directly produce the final segmentation boundary. Instead, it addresses a prerequisite methodological question: can a class-aware proposal generation module reliably cover true primary tumor and lymph node lesions in head-and-neck CT images, so that the downstream fine segmentation model receives sufficiently complete and clinically meaningful regions of interest?

This question is fundamentally different from voxel-level boundary optimization. If a lesion is not included in any candidate ROI, the downstream segmentation model has no opportunity to recover it, regardless of its local segmentation capacity. Therefore, Stage A is evaluated primarily as a high-recall spatial access mechanism rather than as a final segmentation model.

**中文翻译**

Stage A 的目标不是直接输出最终分割边界，而是回答一个更前置的方法学问题：在头颈部 CT 图像中，能否通过一个类别感知的候选区域生成模块，稳定覆盖真实原发肿瘤和淋巴结病灶，使后续精细分割模型获得足够完整且具有临床意义的感兴趣区域？

这个问题与体素级边界优化不同。如果某个病灶没有被任何候选 ROI 覆盖，那么无论后续局部分割模型能力多强，都没有机会恢复该病灶。因此，Stage A 的评价重点不是最终分割 Dice，而是其作为高召回率空间入口的可靠性。

---

## 3. Narrative Logic of Stage A

**English**

Head-and-neck tumor and lymph node auto-contouring is challenged by complex anatomy, large inter-patient variation, heterogeneous tumor morphology, and the dispersed distribution of metastatic lymph nodes. Primary tumors and lymph nodes differ substantially in size, shape, location, and imaging appearance. Meanwhile, true lesions occupy only a small fraction of the whole CT volume, whereas most voxels correspond to normal tissue or background. A direct full-volume fine segmentation strategy therefore has to simultaneously solve lesion localization, class discrimination, boundary delineation, and false-positive suppression under severe foreground-background imbalance.

To reduce this complexity, the proposed framework separates spatial screening from fine segmentation. Stage A serves as the first step of this framework. It deliberately prioritizes lesion coverage over boundary precision. Its purpose is not to delineate the exact tumor or lymph node contour, but to ensure that true lesions are included in a manageable set of candidate ROIs. In this way, the subsequent segmentation stage is no longer required to search blindly across the entire CT volume; instead, it can focus on local discrimination within candidate regions.

The key design choice in Stage A is class awareness. A single generic proposal strategy is suboptimal because primary tumors and lymph nodes have different spatial and morphological characteristics. Primary tumors are often larger and more spatially continuous, whereas lymph nodes can be small, multiple, low-contrast, and anatomically dispersed. Therefore, Stage A separately models tumor-oriented and lymph-node-oriented candidate generation, and allocates proposal capacity according to the different risks of missed detection.

The validity of Stage A is consequently demonstrated by coverage analysis. The central questions are whether the generated ROIs cover the majority of true tumor and lymph node components, whether small lymph nodes are adequately captured, whether improved coverage is achieved without excessive ROI redundancy, and whether the final proposal policy provides a practical balance between recall and computational burden.

**中文翻译**

头颈部肿瘤与淋巴结自动勾画面临复杂解剖结构、显著个体差异、肿瘤形态异质性以及转移淋巴结分布分散等挑战。原发肿瘤和淋巴结在大小、形态、位置和影像表现上均存在明显差异。同时，真实病灶只占整个 CT 体数据中的很小一部分，而绝大多数体素属于正常组织或背景。因此，直接在全图范围内进行精细分割，需要模型在严重前景-背景不平衡的条件下，同时完成病灶定位、类别区分、边界勾画和假阳性抑制。

为降低这一复杂度，本研究将空间筛选和精细分割解耦。Stage A 是该框架的第一步。它有意将病灶覆盖率置于边界精度之前。其目的不是直接勾画精确的肿瘤或淋巴结边界，而是确保真实病灶被纳入一组数量可控的候选 ROI 中。这样，后续分割阶段不再需要在整个 CT 体数据中盲目搜索，而可以集中在候选区域内进行局部判别。

Stage A 的关键设计是类别感知。单一通用候选策略并不适合同时处理原发肿瘤和淋巴结，因为两者具有不同的空间和形态特征。原发肿瘤通常体积更大、空间连续性更强，而淋巴结可能更小、多发、低对比且解剖分布分散。因此，Stage A 分别建模肿瘤导向和淋巴结导向的候选区域生成，并根据不同类别的漏检风险分配候选容量。

因此，Stage A 的有效性应通过覆盖率分析来证明。核心问题包括：候选 ROI 是否覆盖绝大多数真实肿瘤和淋巴结 component，小体积淋巴结是否得到充分捕获，覆盖率提升是否没有带来过度 ROI 冗余，以及最终候选策略是否在召回率和计算负担之间取得了可操作的平衡。

---

## 4. Methods: Stage-A Proposal Generation and Coverage Validation

### 4.1 Input Data

**English**

The input to Stage A consists of preprocessed head-and-neck CT volumes. Each volume is represented in a unified cache space and retains its axial organization. The corresponding manual annotations are used only for offline validation of proposal coverage. For lesion-level evaluation, the ground-truth masks are decomposed into connected components. Each primary tumor component and each lymph node component is treated as an independent lesion-level target.

This component-level formulation is important because patient-level positivity alone cannot reveal whether multiple lymph nodes, small lesions, or spatially separate lesions are missed by the proposal generator.

**中文翻译**

Stage A 的输入为预处理后的头颈部 CT 体数据。每例 CT 在统一缓存空间中表示，并保留其轴向结构。对应的人工标注仅用于离线验证候选 ROI 对真实病灶的覆盖情况。为了进行病灶级评价，真实标注被分解为 connected components。每个原发肿瘤 component 和每个淋巴结 component 均被视为一个独立的病灶级目标。

这种 component-level 评价方式非常重要，因为仅依赖病例级阳性判断无法揭示多发淋巴结、小病灶或空间分离病灶是否被候选区域生成模块遗漏。

### 4.2 Class-Aware ROI Proposal Generation

**English**

Stage A is formulated as a class-aware ROI proposal generation framework. The detector produces tumor-related and lymph-node-related responses, which are processed separately to generate candidate ROIs. This design reflects the clinical and imaging differences between primary tumors and lymph nodes. Tumor proposals emphasize coverage of relatively larger and more continuous abnormal regions, whereas lymph node proposals emphasize sensitivity to small, multiple, and spatially dispersed lesions.

The final proposal policy is a lymph-node-prioritized policy. Compared with a balanced baseline policy, it allocates more proposal capacity to lymph nodes while maintaining sufficient tumor coverage. This reflects the observation that lymph nodes are more likely to be missed due to their smaller size, multiplicity, and variable anatomical distribution.

**中文翻译**

Stage A 被构建为一个类别感知的 ROI 候选区域生成框架。检测器分别产生肿瘤相关响应和淋巴结相关响应，并对两类响应进行独立处理以生成候选 ROI。这一设计反映了原发肿瘤与淋巴结之间的临床和影像差异。肿瘤候选区域更强调覆盖相对较大且更连续的异常区域，而淋巴结候选区域更强调对小体积、多发和空间分散病灶的敏感性。

最终候选策略采用淋巴结优先策略。与基线均衡策略相比，该策略将更多候选容量分配给淋巴结，同时保持足够的肿瘤覆盖能力。这一设计基于淋巴结更容易因体积小、多发和解剖分布变化大而漏检的特点。

### 4.3 Tumor ROI Proposals

**English**

Tumor ROI proposals are generated from high-confidence tumor response regions. Candidate tumor regions are identified, spatially expanded into ROIs, and ranked according to detection confidence and component characteristics. The role of the tumor proposal branch is to ensure that the primary tumor body and its surrounding anatomical context are included in the candidate regions.

At this stage, exact boundary agreement is not required. Mild over-coverage or imperfect ROI boundaries are acceptable as long as the true tumor component is spatially accessible to the downstream fine segmentation model.

**中文翻译**

肿瘤 ROI 候选区域由高置信度肿瘤响应区域生成。候选肿瘤区域经过识别、空间扩展和排序后形成 ROI，排序依据包括检测置信度和 component 特征。肿瘤候选分支的作用是确保原发肿瘤主体及其周围解剖上下文被纳入候选区域。

在这一阶段，并不要求 ROI 与真实肿瘤边界完全一致。只要真实肿瘤 component 能够被后续精细分割模型在空间上访问，轻度过覆盖或候选边界不精确是可以接受的。

### 4.4 Lymph Node ROI Proposals

**English**

Lymph node ROI proposal generation is the most critical component of Stage A. Lymph nodes are often small, multiple, and distributed across variable anatomical regions. They may also appear with low contrast and may be confused with vessels, muscles, or normal lymphatic structures. To reduce missed detections, the final proposal policy assigns a larger proposal budget to lymph nodes than to primary tumors.

This lymph-node-prioritized design aims to increase the probability that small and dispersed lymph node components are included in the candidate ROI set, without simply expanding the total number of ROIs.

**中文翻译**

淋巴结 ROI 候选区域生成是 Stage A 中最关键的部分。淋巴结通常体积小、多发，并分布于变化较大的解剖区域。它们还可能表现为低对比，并容易与血管、肌肉或正常淋巴结构混淆。为降低漏检风险，最终候选策略为淋巴结分配了比原发肿瘤更多的候选预算。

这种淋巴结优先设计的目标是在不简单增加总 ROI 数量的情况下，提高小体积和分散淋巴结 component 被纳入候选 ROI 集合的概率。

### 4.5 ROI Post-processing and Selection

**English**

After candidate generation, Stage A applies post-processing and selection to control proposal redundancy. Candidate components are filtered, ranked, and retained according to class-specific budgets. A maximum ROI number is imposed to keep the downstream computational burden manageable.

This step is essential because an excessively small candidate set may miss true lesions, whereas an excessively large candidate set would increase the computational cost and false-positive burden of the downstream segmentation stage. The selected proposal policy therefore aims to maximize lesion coverage under a controlled candidate burden.

**中文翻译**

候选区域生成后，Stage A 通过后处理和筛选来控制候选冗余。候选 components 经过过滤、排序，并按照类别特异的候选预算进行保留。同时设置最大 ROI 数量，以保证后续分割阶段的计算负担可控。

这一步非常关键，因为候选集过小可能导致真实病灶漏检，而候选集过大则会增加后续分割阶段的计算成本和假阳性负担。因此，最终候选策略的目标是在候选负担受控的前提下最大化病灶覆盖率。

### 4.6 Definition of ROI Coverage

**English**

The primary endpoint of Stage A is lesion-level ROI coverage. For a ground-truth lesion component \(G_i\), if at least one predicted ROI \(R_j\) spatially overlaps it, the component is considered covered. ROI-any coverage is defined as:

\[
	ext{ROI-any coverage} =
rac{\# 	ext{covered ground-truth components}}
{\# 	ext{all ground-truth components}}.
\]

This metric answers whether a true lesion has entered the spatial search range of the downstream segmentation model, regardless of the proposal class label.

A stricter metric, same-class coverage, requires that a primary tumor component be covered by a tumor proposal and a lymph node component be covered by a lymph node proposal. This metric evaluates whether class-aware proposal generation is successful rather than relying on incidental cross-class overlap.

**中文翻译**

Stage A 的主要终点是病灶级 ROI 覆盖率。对于一个真实病灶 component \(G_i\)，如果至少存在一个预测 ROI \(R_j\) 与其发生空间重叠，则认为该 component 被覆盖。ROI-any coverage 定义为：

\[
	ext{ROI-any coverage} =
rac{\# 	ext{被覆盖的真实 components}}
{\# 	ext{全部真实 components}}.
\]

该指标回答的问题是：真实病灶是否进入了后续分割模型的空间搜索范围，而不考虑候选 ROI 的类别标签是否完全一致。

更严格的指标是 same-class coverage，即原发肿瘤 component 必须被肿瘤候选 ROI 覆盖，淋巴结 component 必须被淋巴结候选 ROI 覆盖。该指标用于评估类别感知候选区域生成是否真正有效，而不是依赖跨类别 ROI 的偶然覆盖。

### 4.7 Evaluation Metrics

**English**

Stage A is evaluated using coverage-oriented and burden-oriented metrics, including lesion-level ROI-any coverage, same-class coverage, tumor coverage, lymph node coverage, size-stratified lymph node coverage, candidate burden per case, label-negative candidate burden, newly covered lesions compared with the balanced baseline policy, and residual missed lesion analysis.

Voxel-level Dice is not used as the primary Stage-A metric because Stage A does not generate final segmentation masks. Dice, precision, recall, and boundary metrics are reserved for the fine segmentation stage.

**中文翻译**

Stage A 采用覆盖率导向和负担导向的指标进行评价，包括病灶级 ROI-any coverage、same-class coverage、肿瘤覆盖率、淋巴结覆盖率、按大小分层的淋巴结覆盖率、每例候选数量、label-negative candidate burden、相较于基线均衡策略新增覆盖的病灶数量，以及 residual missed lesion analysis。

Stage A 不以 voxel-level Dice 作为主要指标，因为 Stage A 并不生成最终分割 mask。Dice、precision、recall 和边界指标应保留给后续精细分割阶段。

### 4.8 Failure Case Analysis

**English**

Residual missed components are further analyzed to determine whether failure arises from detector non-response, localization error, class confusion, or candidate ranking and truncation. This analysis is important because the interpretation of a missed lesion differs across mechanisms. If missed lesions are mainly caused by detector non-response, Stage A requires further improvement. If they are mainly caused by marginal localization errors, cross-class coverage, or top-k selection, the proposal generator may already be approaching a practical coverage ceiling, and downstream improvement should focus on the fine segmentation stage.

**中文翻译**

对于仍未被覆盖的 residual missed components，需要进一步分析其失败原因，包括检测器无响应、定位偏移、类别混淆，以及候选排序或截断问题。这一分析非常重要，因为不同漏检机制对应不同的改进方向。如果漏检主要来自检测器无响应，则 Stage A 仍需进一步改进；如果漏检主要来自边缘定位误差、跨类别覆盖或 top-k 筛选，则说明候选区域生成器可能已经接近实际可用的覆盖率上限，后续优化应更多集中在精细分割阶段。

---

## 5. Results Narrative Chain

### Question 1: Can the candidate ROIs cover true lesions?

**English**

The question addressed in this section is whether the candidate ROIs generated by Stage A can cover the majority of true tumor and lymph node lesions.

Under the final lymph-node-prioritized proposal policy, Stage A achieved high lesion-level ROI-any coverage. In the training cohort, tumor coverage reached 442/447 components, corresponding to 98.88%, and lymph node coverage reached 567/575 components, corresponding to 98.61%. In the validation cohort, tumor coverage reached 122/127 components, corresponding to 96.06%, and lymph node coverage reached 146/148 components, corresponding to 98.65%.

These findings indicate that Stage A provides a high-recall spatial access mechanism for downstream fine segmentation. In particular, lymph node coverage remained close to 99% in the validation cohort, suggesting that the proposal generator is sufficiently sensitive to the more challenging lymph node targets.

**中文翻译**

本节要回答的问题是：Stage A 生成的候选 ROI 能否覆盖绝大多数真实肿瘤和淋巴结病灶。

在最终采用的淋巴结优先候选策略下，Stage A 实现了较高的病灶级 ROI-any coverage。训练集中，肿瘤覆盖率为 442/447，即 98.88%；淋巴结覆盖率为 567/575，即 98.61%。验证集中，肿瘤覆盖率为 122/127，即 96.06%；淋巴结覆盖率为 146/148，即 98.65%。

这些结果表明，Stage A 能够为后续精细分割提供高召回率的空间入口。尤其是验证集中淋巴结覆盖率接近 99%，说明该候选区域生成模块对更具挑战性的淋巴结目标具有较高敏感性。

### Question 2: Is Stage A effective for both primary tumors and lymph nodes?

**English**

The question addressed in this section is whether the class-aware proposal strategy benefits both primary tumors and lymph nodes, rather than only one lesion type.

The final proposal policy maintained high coverage for both categories. Tumor ROI-any coverage reached 98.88% in the training cohort and 96.06% in the validation cohort. Lymph node ROI-any coverage reached 98.61% in the training cohort and 98.65% in the validation cohort. Although lymph nodes are smaller and more spatially dispersed than primary tumors, their coverage was not inferior to tumor coverage.

Same-class analysis further confirmed the effectiveness of lymph-node-oriented proposal generation. Lymph node same-class coverage reached 97.57% in the training cohort and 97.97% in the validation cohort, indicating that most lymph node components were covered by lymph-node proposals rather than by incidental cross-class tumor proposals.

**中文翻译**

本节要回答的问题是：类别感知候选策略是否同时适用于原发肿瘤和淋巴结，而不是只对某一类病灶有效。

最终候选策略在两类目标上均保持了较高覆盖率。肿瘤 ROI-any coverage 在训练集为 98.88%，在验证集为 96.06%；淋巴结 ROI-any coverage 在训练集为 98.61%，在验证集为 98.65%。尽管淋巴结通常比原发肿瘤更小、分布更分散，其覆盖率并不低于肿瘤覆盖率。

进一步的 same-class 分析也验证了淋巴结导向候选区域生成的有效性。淋巴结 same-class coverage 在训练集达到 97.57%，在验证集达到 97.97%，说明大多数淋巴结 component 是由淋巴结候选 ROI 覆盖，而非依赖肿瘤候选 ROI 的偶然跨类别覆盖。

### Question 3: Which lesions remain difficult to cover?

**English**

The question addressed in this section is which lesion types remain vulnerable to missed coverage even after applying the final proposal policy.

Residual missed lesion analysis showed that the number of uncovered components was small. Remaining misses included a limited number of small lymph nodes, several lymph nodes with spatial localization mismatch, and a few tumor components. Importantly, these misses were not dominated by complete detector non-response. Instead, they were more often associated with localization deviation, class confusion, or candidate ranking and truncation.

This pattern suggests that Stage A has achieved a high practical coverage ceiling. The remaining errors represent difficult boundary cases rather than a systematic failure of the proposal generator.

**中文翻译**

本节要回答的问题是：在应用最终候选策略后，哪些病灶仍然容易未被覆盖。

Residual missed lesion analysis 显示，未覆盖 components 的数量较少。剩余漏检包括少量小体积淋巴结、若干存在空间定位偏移的淋巴结，以及少数肿瘤 components。重要的是，这些漏检并不主要由检测器完全无响应造成，而更多与定位偏移、类别混淆或候选排序和截断有关。

这一模式说明 Stage A 已经达到较高的实际覆盖率上限。剩余错误更多代表困难边界情况，而不是候选区域生成器的系统性失败。

### Question 4: Does increasing ROI coverage necessarily increase ROI redundancy?

**English**

The question addressed in this section is whether improved coverage is achieved only by increasing the number of candidate ROIs.

The comparison between the balanced baseline policy and the final lymph-node-prioritized policy showed that improved coverage did not require uncontrolled ROI expansion. In the training cohort, the average number of candidates decreased from 22.98 to 20.28 per case. In the validation cohort, it decreased from 27.03 to 24.19 per case. At the same time, the final policy newly covered several ground-truth components that were missed by the baseline policy, most of which were lymph nodes.

Thus, the improvement was not simply caused by generating more candidates. Instead, it resulted from a more effective class-aware allocation of proposal capacity.

**中文翻译**

本节要回答的问题是：覆盖率提高是否必须依赖候选 ROI 数量的无控制增加。

基线均衡策略与最终淋巴结优先策略的对比显示，覆盖率提升并不需要无限制扩大 ROI 数量。训练集中，平均候选数从每例 22.98 个下降到 20.28 个；验证集中，平均候选数从每例 27.03 个下降到 24.19 个。同时，最终策略新增覆盖了若干基线策略未覆盖的真实 components，其中多数为淋巴结。

因此，覆盖率提升并不是简单由生成更多候选区域造成的，而是来自更有效的类别感知候选容量分配。

### Question 5: Does the final proposal policy balance coverage and efficiency?

**English**

The question addressed in this section is whether the final proposal policy provides a practical balance between high lesion coverage and manageable computational burden.

The final lymph-node-prioritized policy achieved high ROI-any coverage for both tumor and lymph node components, maintained high same-class lymph node coverage, and reduced the average candidate burden compared with the balanced baseline policy. These results indicate that Stage A provides a reliable and computationally controlled spatial screening step.

Based on this evidence, Stage A can be fixed as the front-end proposal generator of the two-stage auto-contouring framework, while subsequent work should focus on voxel-level segmentation quality within the selected ROIs.

**中文翻译**

本节要回答的问题是：最终候选策略是否在高病灶覆盖率和可控计算负担之间取得了实际平衡。

最终淋巴结优先策略在肿瘤和淋巴结 components 上均获得了较高 ROI-any coverage，同时保持了较高的淋巴结 same-class coverage，并且相比基线均衡策略降低了平均候选负担。这些结果说明 Stage A 能够作为一个可靠且计算成本可控的空间筛选步骤。

基于这些证据，Stage A 可以被固定为两阶段自动勾画框架的前置候选区域生成模块。后续工作应集中于候选 ROI 内的体素级精细分割质量。

---

## 6. Discussion Outline

### 6.1 Why Stage A is suitable as the first step of a two-stage framework

**English**

Stage A is suitable as the first step because it addresses the spatial search problem before fine segmentation. In sparse head-and-neck lesions, full-volume segmentation exposes the model to overwhelming background and increases the risk of missed small targets and false positives. A high-recall proposal generator reduces the search space and allows the downstream model to focus on local segmentation.

**中文翻译**

Stage A 适合作为两阶段框架的第一步，因为它在精细分割之前解决了空间搜索问题。对于稀疏的头颈部病灶，全图分割会使模型暴露于大量背景中，并增加小目标漏检和假阳性风险。高召回率候选区域生成器可以缩小搜索空间，使后续模型专注于局部分割。

### 6.2 Why coverage is more important than boundary precision in Stage A

**English**

The errors of Stage A are asymmetric. An imprecise ROI boundary can still be corrected by the downstream segmentation model, but a completely missed lesion cannot be recovered if it is absent from all candidate ROIs. Therefore, coverage is the safety criterion of Stage A, whereas boundary precision belongs to the downstream segmentation stage.

**中文翻译**

Stage A 的错误具有不对称性。ROI 边界不够精确仍可由后续分割模型修正，但如果某个病灶没有出现在任何候选 ROI 中，则后续模型无法恢复。因此，覆盖率是 Stage A 的安全性标准，而边界精度应属于后续分割阶段的任务。

### 6.3 Advantage of class-aware proposal generation

**English**

Class-aware proposal generation is advantageous because primary tumors and lymph nodes have different failure modes. A unified proposal strategy may over-allocate capacity to larger tumor-like regions while missing small or dispersed lymph nodes. By assigning class-specific proposal budgets and selection criteria, the final policy improves lymph node coverage without substantially compromising tumor coverage or increasing ROI redundancy.

**中文翻译**

类别感知候选区域生成的优势在于，原发肿瘤和淋巴结具有不同的失败模式。统一候选策略可能会将过多容量分配给较大的肿瘤样区域，而遗漏小体积或分散淋巴结。通过设置类别特异的候选预算和筛选标准，最终策略在不明显牺牲肿瘤覆盖率、不增加 ROI 冗余的前提下，提高了淋巴结覆盖能力。

### 6.4 Limitations of Stage A

**English**

Stage A still has limitations. A small number of residual lesions remain uncovered, especially lesions with small volume, low contrast, atypical location, or localization mismatch. ROI-any coverage may also include cross-class overlap, which does not fully represent class-specific proposal accuracy. Although same-class analysis confirmed strong lymph node coverage, class confusion remains a potential issue that should be considered in the downstream stage.

**中文翻译**

Stage A 仍存在一定局限性。少数 residual lesions 仍未被覆盖，尤其可能包括小体积、低对比、位置不典型或存在定位偏移的病灶。ROI-any coverage 也可能包含跨类别覆盖，因此不能完全代表类别特异的候选准确性。尽管 same-class 分析证实淋巴结覆盖较稳定，类别混淆仍是后续阶段需要关注的问题。

### 6.5 How remaining limitations are handled downstream

**English**

The limitations of Stage A do not invalidate its role as a proposal generator. Its purpose is to provide spatial access, not to solve all segmentation challenges. Once a lesion is included in an ROI, the downstream fine segmentation model can refine boundaries, suppress false positives, apply threshold calibration, and perform component-level post-processing. Future work may further improve proposal ranking and hard-case mining, but the current coverage results support fixing Stage A as the front-end module.

**中文翻译**

Stage A 的局限性并不否定其作为候选区域生成器的作用。它的目的在于提供空间入口，而不是解决所有分割问题。一旦病灶进入 ROI，后续精细分割模型可以进一步修正边界、抑制假阳性、进行阈值校准并执行 component-level 后处理。未来工作可以继续优化候选排序和难例挖掘，但当前覆盖率结果已经支持将 Stage A 固定为前置模块。

---

## 7. Manuscript-Ready Introduction Paragraph for Stage A

**English**

Automatic contouring of head-and-neck tumors and lymph nodes is challenged by complex anatomy, heterogeneous lesion morphology, and highly imbalanced foreground-background distributions. Primary tumors and metastatic lymph nodes differ substantially in size, spatial distribution, and imaging appearance, while true lesion regions occupy only a small fraction of the entire CT volume. Direct full-volume fine segmentation therefore requires a model to simultaneously perform lesion localization, class discrimination, boundary delineation, and false-positive suppression across a large anatomical search space. To reduce this complexity, we introduce Stage A as a coverage-oriented, class-aware ROI proposal generation module within a two-stage auto-contouring framework. Rather than producing final segmentation masks, Stage A aims to provide reliable spatial access to true tumor and lymph node lesions by generating candidate ROIs that prioritize lesion-level coverage. Tumor and lymph node proposals are generated and selected with class-specific considerations, reflecting their distinct anatomical distributions and missed-detection risks. The effectiveness of Stage A is therefore evaluated by lesion-level coverage, same-class coverage, candidate burden, and residual missed lesion analysis, rather than by voxel-level Dice. Under the final lymph-node-prioritized proposal policy, Stage A achieved high coverage for both tumor and lymph node components while maintaining a manageable number of candidate ROIs. These results support its role as a safe spatial screening step that converts full-volume lesion search into ROI-based local segmentation for the downstream stage.

**中文翻译**

头颈部肿瘤与淋巴结自动勾画受到复杂解剖结构、病灶形态异质性以及显著前景-背景不平衡的影响。原发肿瘤和转移淋巴结在大小、空间分布和影像表现上存在明显差异，而真实病灶区域仅占整个 CT 体数据中的很小一部分。因此，直接进行全图精细分割需要模型在大范围解剖搜索空间内同时完成病灶定位、类别区分、边界勾画和假阳性抑制。为降低这一复杂度，本研究在两阶段自动勾画框架中引入 Stage A，将其设计为一个以覆盖率为核心、类别感知的 ROI 候选区域生成模块。Stage A 不输出最终分割 mask，而是通过生成候选 ROI，为真实肿瘤和淋巴结病灶提供可靠的空间入口，并优先保证病灶级覆盖。肿瘤和淋巴结候选区域根据类别特异的解剖分布和漏检风险进行生成与筛选。因此，Stage A 的有效性通过病灶级覆盖率、same-class coverage、候选负担和 residual missed lesion analysis 进行评价，而不是通过 voxel-level Dice 评价。在最终淋巴结优先候选策略下，Stage A 在保持候选 ROI 数量可控的同时，对肿瘤和淋巴结 components 均实现了较高覆盖率。这些结果支持 Stage A 作为一个安全的空间筛选步骤，将全图病灶搜索转化为后续阶段的 ROI 内局部分割问题。

---

## 8. Concise Results Paragraph

**English**

Stage A was evaluated as a proposal generation module rather than as a final segmentation model. Therefore, the primary endpoint was lesion-level ROI coverage instead of voxel-level Dice. Under the final lymph-node-prioritized proposal policy, Stage A achieved high ROI-any coverage for both tumor and lymph node components. In the training cohort, tumor and lymph node coverage reached 98.88% and 98.61%, respectively. In the validation cohort, tumor and lymph node coverage reached 96.06% and 98.65%, respectively. Same-class analysis further showed that lymph node coverage was largely class-consistent, with same-class lymph node coverage of 97.57% in the training cohort and 97.97% in the validation cohort. Compared with the balanced baseline proposal policy, the final policy reduced the average number of candidates per case while newly covering several previously missed components, most of which were lymph nodes. These findings indicate that Stage A provides a high-recall and computationally controllable ROI proposal mechanism for subsequent fine segmentation.

**中文翻译**

Stage A 被评价为候选区域生成模块，而不是最终分割模型。因此，其主要终点是病灶级 ROI coverage，而不是 voxel-level Dice。在最终淋巴结优先候选策略下，Stage A 对肿瘤和淋巴结 components 均取得了较高 ROI-any coverage。训练集中，肿瘤和淋巴结覆盖率分别为 98.88% 和 98.61%；验证集中，肿瘤和淋巴结覆盖率分别为 96.06% 和 98.65%。进一步的 same-class 分析显示，淋巴结覆盖主要具有类别一致性，训练集和验证集的淋巴结 same-class coverage 分别为 97.57% 和 97.97%。与基线均衡候选策略相比，最终策略在降低每例平均候选数量的同时，新增覆盖了若干此前遗漏的 components，其中多数为淋巴结。上述结果表明，Stage A 能够为后续精细分割提供高召回率且计算负担可控的 ROI 候选区域生成机制。

---

## 9. Concise Conclusion for Stage A

**English**

Stage A establishes a high-recall spatial screening mechanism for head-and-neck tumor and lymph node auto-contouring. By using class-aware and lymph-node-prioritized ROI proposal generation, it achieves strong lesion-level coverage while controlling candidate redundancy. Because its objective is to provide reliable spatial access rather than final boundaries, Stage A should be interpreted as a proposal ceiling module. The high coverage achieved in both training and validation cohorts supports freezing Stage A and shifting subsequent optimization to ROI-level fine segmentation.

**中文翻译**

Stage A 为头颈部肿瘤与淋巴结自动勾画建立了一个高召回率的空间筛选机制。通过类别感知且淋巴结优先的 ROI 候选区域生成，它在控制候选冗余的同时实现了较强的病灶级覆盖能力。由于其目标是提供可靠空间入口，而不是输出最终边界，因此 Stage A 应被理解为一个 proposal ceiling 模块。训练集和验证集上的高覆盖率支持将 Stage A 固定，并将后续优化重点转向 ROI 内精细分割。
