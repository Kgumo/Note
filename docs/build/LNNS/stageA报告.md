题目建议

**A Two-Stage ROI-Guided Framework for Head-and-Neck Tumor and Lymph Node Segmentation: Stage-A Proposal Generation and Coverage Validation**

中文可写为：

**面向头颈部肿瘤与淋巴结分割的两阶段 ROI 引导框架：A 阶段候选区域生成与覆盖率验证**

---

## Abstract 草稿

Accurate segmentation of head-and-neck tumors and metastatic lymph nodes in CT images remains challenging because target lesions are often small, sparse, and highly imbalanced against the background volume. To reduce the burden of full-volume dense segmentation, we designed a two-stage framework in which Stage A first generates class-aware tumor and lymph-node region proposals, and Stage B subsequently performs fine segmentation within selected regions of interest. This section reports the design and validation of Stage A.

Stage A uses a detector-based proposal generator to identify candidate tumor and lymph-node regions from CT volumes. Instead of evaluating Stage A with voxel-level Dice, which is reserved for the final segmentation stage, we evaluate it by ground-truth component coverage, including ROI-any coverage, same-class coverage, centroid coverage, and candidate burden. A q05 proposal policy was selected as the frozen Stage-A configuration, using a longer axial sequence length and a lymph-node-favored proposal budget. On the training cohort of 412 cases, q05 achieved ROI-any coverage of 98.88% for tumor components and 98.61% for lymph-node components. On the validation cohort of 103 cases, it achieved 96.06% tumor coverage and 98.65% lymph-node coverage. Same-class lymph-node coverage remained high, reaching 97.57% on the training cohort and 97.97% on the validation cohort. These results indicate that the Stage-A proposal generator provides a sufficiently high recall ceiling for the downstream segmentation stage, while reducing the number of candidate regions compared with the baseline q00 configuration.

---

## 1. Introduction / Motivation

Head-and-neck CT segmentation is difficult because tumor and lymph-node targets occupy only a small fraction of the full image volume. Direct dense segmentation over the entire CT volume may suffer from severe foreground-background imbalance, high computational cost, and excessive false positives. Therefore, a two-stage strategy is clinically and computationally attractive: the first stage identifies suspicious candidate regions, and the second stage performs fine segmentation only within those regions.

This design is consistent with ROI-guided medical image analysis, where local candidate regions are first detected and then processed by a more focused segmentation model. The reference material also emphasizes that for sparse lesions, processing only ROI-centered slices or cropped regions can reduce redundant computation and mitigate class imbalance. In addition, axial slice ordering is clinically meaningful for head-and-neck CT because adjacent slices preserve anatomical continuity, which can help the model maintain consistent lesion localization across the superior-inferior direction.

In this work, Stage A is not designed to produce the final segmentation mask. Instead, its goal is to provide high-recall tumor and lymph-node proposals for Stage B. Therefore, the key question for Stage A is not “what is the Dice score?”, but rather “whether the ground-truth lesion components are covered by at least one proposal.”

---

## 2. Stage-A Proposal Generation

### 2.1 Overview

Stage A takes a CT volume as input and produces a set of candidate ROIs for tumor and lymph-node segmentation. Each candidate proposal contains a spatial region, a predicted class label, and a detector confidence score. The downstream Stage-B model then receives these candidate ROIs for fine segmentation.

The Stage-A detector was fixed as:

```text
detector checkpoint:
runs/stageA_mined_hardneg_v2_conservative_20260629_070436/best_detector.pt

cache:
cache_320_fp16_realspacing_v2

detector base channels:
32

stack depth:
5

crop size:
160
```

Two proposal policies were compared:

|Policy|seq_len|tumor budget|LN budget|tumor keep_topk|LN keep_topk|tumor det thr|LN det thr|
|---|--:|--:|--:|--:|--:|--:|--:|
|q00|15|9|9|9|9|0.80|0.80|
|q05|19|6|12|6|12|0.85|0.85|

The final frozen Stage-A setting is **q05**.

The q05 policy increases the lymph-node proposal budget while reducing the tumor proposal budget. This design reflects the clinical and technical observation that lymph nodes are usually smaller, more numerous, and more easily missed than primary tumor regions. Therefore, Stage A should allocate more candidate capacity to lymph nodes while still maintaining high tumor proposal coverage.

---

## 3. Evaluation Metrics for Stage A

Because Stage A does not output the final segmentation mask, voxel-level Dice is not an appropriate primary metric. Instead, we evaluate whether each ground-truth component is covered by the proposal set.

Let (G_i) denote a ground-truth connected component and (R_j) denote a predicted proposal ROI. A component is considered covered if at least one proposal overlaps it.

### 3.1 ROI-any coverage

A ground-truth component is counted as covered if any proposal ROI overlaps it, regardless of the predicted proposal class.

[  
\text{ROI-any recall} =  
\frac{# \text{covered GT components}}{# \text{all GT components}}  
]

This is the main proposal ceiling metric. It measures whether Stage B has a chance to segment the lesion, assuming it can correct the class inside the ROI.

### 3.2 Same-class coverage

A stricter metric requires that a tumor component is covered by a tumor proposal and a lymph-node component is covered by a lymph-node proposal.

[  
\text{Same-class recall} =  
\frac{# \text{GT components covered by same-class proposals}}{# \text{all GT components}}  
]

This metric is important because cross-class ROI coverage may still allow Stage B to see the lesion, but it indicates class confusion in Stage A.

### 3.3 Centroid coverage

Centroid coverage measures whether the centroid of a GT component lies inside a proposal. It is stricter than simple overlap and reflects localization quality.

### 3.4 Candidate burden

Candidate burden measures the average number of proposals per case and the number of label-negative or GT-negative candidates. This reflects downstream computational cost and potential false-positive burden for Stage B.

---

## 4. Experimental Setup

Stage A was evaluated on:

```text
Train cohort: 412 cases
Validation cohort: 103 cases
```

Ground-truth components were grouped into tumor and lymph-node classes. Lymph-node components were further stratified by size:

```text
<5 mm
5–10 mm
>=10 mm
all
```

The final Stage-A report was saved under:

```text
diagnostics/stageA_pure_proposal_coverage_phase4_20260709_021207/final_stageA_pure_report
```

The frozen reproducible Stage-A package was saved under:

```text
/root/LNNs/bin/stageA_q05_proposal_freeze_20260710
```

The package can reproduce the Stage-A audit using Python only:

```bash
python -u /root/LNNs/bin/stageA_q05_proposal_freeze_20260710/run_stageA_audit.py --mode full
```

---

## 5. Results

### 5.1 Main Stage-A coverage results

The q05 policy achieved high proposal coverage on both training and validation cohorts.

|Split|Policy|Class|Covered / Total|ROI-any recall|
|---|---|---|--:|--:|
|Train412|q05|Tumor|442 / 447|98.88%|
|Train412|q05|LN|567 / 575|98.61%|
|Val103|q05|Tumor|122 / 127|96.06%|
|Val103|q05|LN|146 / 148|98.65%|

These results show that Stage A provides a high-recall proposal ceiling, especially for lymph-node components on the validation cohort.

### 5.2 Lymph-node size-stratified coverage

|Split|Policy|LN size bin|Covered / Total|ROI-any recall|
|---|---|---|--:|--:|
|Train412|q05|<5 mm|107 / 111|96.40%|
|Train412|q05|5–10 mm|279 / 281|99.29%|
|Train412|q05|≥10 mm|181 / 183|98.91%|
|Val103|q05|<5 mm|23 / 24|95.83%|
|Val103|q05|5–10 mm|78 / 78|100.00%|
|Val103|q05|≥10 mm|45 / 46|97.83%|

The q05 configuration maintained strong coverage across lymph-node size bins. Importantly, the validation recall for small lymph nodes below 5 mm improved to 95.83%, suggesting that the lymph-node-favored proposal budget effectively improves sensitivity to small lesions.

### 5.3 Comparison with q00

Compared with q00, q05 improved lymph-node coverage while reducing candidate burden.

|Split|Policy|Candidates / case|Label-negative candidates / case|Any-GT-negative candidates / case|
|---|---|--:|--:|--:|
|Train412|q00|22.98|18.84|16.70|
|Train412|q05|20.28|16.00|13.96|
|Val103|q00|27.03|19.91|16.59|
|Val103|q05|24.19|17.25|14.15|

Thus, q05 improved or maintained proposal recall while reducing the number of candidate regions passed to Stage B.

### 5.4 q05 unique coverage gains

The q05 policy newly covered several GT components that were missed by q00.

|Split|Class|Size bin|Newly covered components|
|---|---|---|--:|
|Train412|Tumor|all|1|
|Train412|LN|<5 mm|2|
|Train412|LN|5–10 mm|1|
|Val103|LN|<5 mm|2|
|Val103|LN|5–10 mm|1|

In total, q05 introduced 7 newly covered GT components compared with q00. Most of these gains occurred in lymph-node components, supporting the choice of a lymph-node-favored q05 policy.

### 5.5 Same-class coverage

Same-class analysis was performed to confirm that the high ROI-any coverage was not solely due to cross-class proposal overlap.

|Split|Class|q05 ROI-any recall|q05 same-class recall|Drop|
|---|---|--:|--:|--:|
|Train412|LN|98.61%|97.57%|1.04 pp|
|Val103|LN|98.65%|97.97%|0.68 pp|

The small gap between ROI-any and same-class lymph-node recall indicates that most lymph-node GT components were covered by lymph-node proposals rather than by tumor proposals. Therefore, the q05 proposal generator is not merely relying on cross-class coverage.

For tumor, the validation same-class drop was larger, indicating that some tumor components were covered by proposals of the opposite class. This issue should be monitored in Stage B, but it does not invalidate the lymph-node proposal ceiling.

---

## 6. Residual Miss Analysis

After applying q05, only 20 GT components remained uncovered at the proposal level:

```text
Train412:
  LN 5–10 mm: 2
  LN <5 mm: 4
  LN ≥10 mm: 2
  Tumor all: 5

Val103:
  LN <5 mm: 1
  LN ≥10 mm: 1
  Tumor all: 5
```

The residual miss audit showed that the misses were not dominated by complete detector non-response. Instead, the remaining errors were primarily associated with localization miss, class confusion, or ranking/top-k truncation. This suggests that Stage-A performance has reached a sufficiently high proposal ceiling and that further improvement should focus on Stage-B segmentation, thresholding, and post-processing rather than continuing to modify Stage A.

---

## 7. Discussion

The Stage-A q05 proposal generator achieved high ground-truth component coverage on both the training and validation cohorts. In particular, lymph-node coverage reached 98.65% ROI-any recall and 97.97% same-class recall on the validation cohort. This is important because lymph nodes are sparse and small targets, and failure to propose them would create an unrecoverable upper bound for downstream segmentation.

The q05 policy improved upon q00 by increasing lymph-node coverage while reducing the average candidate burden per case. This indicates that the proposal policy did not simply increase recall by generating more candidates. Instead, it achieved a better allocation of proposal capacity, prioritizing lymph nodes while maintaining tumor coverage.

The remaining gap between Stage-A coverage and perfect recall is small. Therefore, Stage A should be considered frozen for the current pipeline. Subsequent errors in the final segmentation stage should be attributed primarily to Stage-B segmentation quality, checkpoint selection, thresholding, presence gating, or post-processing, rather than to Stage-A proposal failure.

---

## 8. Limitations

Stage A has several limitations.

First, Stage A does not produce a final segmentation mask. Therefore, Dice, precision, recall, and boundary metrics such as HD95 must be evaluated in Stage B after voxel-level predictions are generated. Dice and related metrics are standard for final medical segmentation evaluation, but they are not the primary metrics for proposal generation.

Second, ROI-any coverage allows cross-class proposal coverage. Although same-class analysis showed only a small drop for lymph nodes, tumor same-class coverage showed a larger gap on the validation cohort. This suggests that class confusion may still affect downstream segmentation and should be analyzed in Stage B.

Third, high proposal coverage does not guarantee high final segmentation quality. Even if a GT component is covered by a proposal, Stage B may still fail due to poor mask prediction, excessive false positives, threshold miscalibration, or insufficient checkpoint generalization.

---

## 9. Conclusion

Stage A was designed as a high-recall proposal generation module for a two-stage tumor and lymph-node segmentation pipeline. The frozen q05 configuration achieved strong proposal coverage on both train and validation cohorts, with validation ROI-any recall of 96.06% for tumor and 98.65% for lymph nodes. Same-class lymph-node recall remained high at 97.97%, confirming that lymph-node coverage was largely class-consistent.

These results support freezing Stage A and moving the pipeline focus to Stage B, where voxel-level segmentation performance, checkpoint quality, threshold calibration, and post-processing should be systematically evaluated.

---

## 可直接放论文里的精简版结论

```text
Stage A was not evaluated by Dice because it does not generate the final segmentation mask. Instead, it was evaluated by ground-truth component coverage. The frozen q05 proposal policy achieved 98.61% lymph-node ROI-any coverage on the training cohort and 98.65% on the validation cohort. Same-class lymph-node coverage remained high, reaching 97.57% and 97.97% on the training and validation cohorts, respectively. The q05 policy also reduced the average number of candidates per case compared with q00, indicating improved proposal efficiency. Therefore, Stage A provides a sufficiently high proposal ceiling and was frozen before proceeding to Stage B voxel-level segmentation.
```

如果你要写中文论文，可以把上面整体翻译成中文正式版；现在这版更接近英文论文 Methods + Results + Discussion 的结构。