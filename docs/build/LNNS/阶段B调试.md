请重点分析 9 个问题：

1. 当前 B 阶段是否已经摆脱塌缩型失败？请依据 ln_prob_max、ln_gt_mean、ln_recall、ln_fp、threshold sweep 和 epoch 变化判断。

2. 为什么 LN 端明显好转，但 tumor_pos_dice 长期只有约 0.27–0.31？请结合 crop_size=96、loss 权重、采样比例、presence gate、tumor/both 样本、score 设计分析。

3. 当前是否存在早期最好、后续退化或过拟合？请重点比较 epoch 2 FAST、epoch 4 FAST、epoch 5 FULL、epoch 10 FULL。

4. FAST validation 与 FULL validation 是否一致？FAST samples=512、FULL val_samples=4096，这会不会导致 best epoch 判断不稳定？

5. 当前 checkpoint 保存策略是否会漏掉好模型？请检查是否需要保存 best_fast、best_full、best_ln_dice、best_tumor、top-k checkpoints。

6. threshold 0.35–0.75 指标几乎不变，说明什么？请判断是 probability hardening、presence gate 主导、还是输出过稀疏。请建议加入 presence gate ON/OFF 对照。

7. 只基于日志中的 ROI buckets 分析训练/验证分布是否偏斜。特别注意 hardneg 数量、tumor/both 数量、ln_large 数量过少、FAST 抽样波动。

8. 当前 validation 还缺哪些 error analysis 指标？请列出 per-ROI CSV 应输出的字段，包括 gt_voxels、pred_voxels、intersection、voxel_recall、voxel_precision、volume_ratio、pred_max、presence_prob、kind、size_bin、case_id、error_type。

9. 最后给出明确下一步：当前是否继续训练？是否先做 debug/error analysis？如果只能改三个地方，优先改哪三个？

输出格式：
第一部分：日志事实摘要
第二部分：当前结果的正面信号
第三部分：当前结果的主要风险
第四部分：9 个问题逐项分析
第五部分：最小代码修改建议
第六部分：下一轮实验计划
第七部分：最终判断