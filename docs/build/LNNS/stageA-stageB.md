# LN Stage-B 转化失败：文献证据、机制诊断与可执行实验方案

## 1. Executive summary

### 1.1 当前最可能的三个原因

**第一：Stage-B 丢弃了 Stage-A 的预测空间，只把 Stage-A 当作裁剪器。**

这是目前证据最强的原因，对应 H1。RSTN 曾直接观察到：粗到细模型只保留粗阶段产生的裁剪区域、丢弃粗概率图和上下文后，细阶段可能比粗阶段更差；其解决方案正是把前一阶段概率图持续传递给后续阶段。nnU-Net cascade、MedSAM 的 U-Net 对照、DeepIGeoS、M-SAM 和近期 label-refinement 工作也都把前一阶段预测作为显式输入，而不是仅提取 ROI。([开放获取计算机视觉论坛](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Recurrent_Saliency_Transformation_CVPR_2018_paper.html "https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Recurrent_Saliency_Transformation_CVPR_2018_paper.html"))

你当前的 C0/E2 现象与这一机制高度一致：Stage-A 中已有的弱阳性和粗形状没有 identity path，一旦 Stage-B 对 CT 外观不确信，就只能重新决定“这里是不是 LN”，最容易得到的低风险解是整体降低 LN logit。

---

**第二：大量 hard negative 与正 LN 共用 segmentation 输出，导致 presence 判断压倒 delineation，形成 conservative collapse。**

E2 的结果不是“更会区分假阳性”，而是：

- Dice：0.1398 → 0.1236；
    
- recall：0.0973 → 0.0815；
    
- FP components/case：3.71 → 2.98。
    

这正是小前景在类别不平衡和域偏移下发生 logit 向背景侧整体移动的表现。TMI 的类别不平衡研究发现，小结构的未见样本容易发生 foreground logit shift，表现为系统性欠分割和 sensitivity 下降；多任务文献也表明 classification 与 segmentation 目标可能产生冲突梯度。([arXiv](https://arxiv.org/abs/2102.10365 "https://arxiv.org/abs/2102.10365"))

因此，**负 proposal 不应继续用全背景 segmentation loss 大规模更新 residual decoder**。hard negative 的主要职责应是训练独立的 candidate-presence head；只有正 proposal 才承担像素级边界学习。

---

**第三：proposal 几何误差与最终 full-volume 转化没有进入训练目标。**

普通 proposal-matched retraining 并不等于匹配了真实 proposal-quality 分布。中心偏移、各轴尺度误差、GT containment、crop 边界截断和 proposal score 之间通常相关，独立均匀 jitter 很难模拟这种联合分布。Cascade R-CNN 的核心结论就是：某一级模型必须在前一级实际输出分布上训练，否则训练时 proposal 质量与推理时输入质量不匹配；RoBox-SAM 则直接学习低质量 box 到高质量 box 的偏移并迭代修正。两者对本项目属于强设计启发，但前者是自然图像，后者主要是 2D 医学提示分割，尚未在 <5 mm CT LN 上得到证明。([开放获取计算机视觉论坛](https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html "https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html"))

此外，当前训练主要优化 ROI mask，而部署结果取决于 crop、inverse transform、pasteback、proposal fusion、component filtering 的整条链。只要该链不在训练和 checkpoint 选择中出现，ROI 指标可以改善而 full-volume conversion 继续恶化。

### 1.2 最终方向选择

首选顺序明确如下：

1. **A. coarse-mask-guided residual refinement——主方向**
    
2. **B. proposal correction + iterative recrop——第二优先级**
    
3. **C. multi-proposal aggregation/consistency——第三优先级**
    
4. **D. Stage-A/Stage-B 有限联合优化——仅在前述路径成立后**
    
5. **E. 放弃两阶段、重做联合检测分割——当前不建议**
    

当前不应放弃 Stage-A。Stage-A 已经显示出候选和粗分割信息，真正缺失的是一个**保留已有 TP、只修改有证据错误区域的 Stage-B**。

---

## 2. 首先执行的 A0 审计门控

在任何 R1–R7 训练前，先核实“Stage-A Dice 约 0.4–0.5”是否真正成立。当前不能把这个数字视为已验证事实。

### 2.1 必须完全一致的条件

Stage-A 与 C0/E2 必须同时满足：

- 相同 Val103 患者；
    
- 相同 LN 标签定义，不含 tumor；
    
- 相同原始 full-volume 空间；
    
- 相同 spacing、orientation 和 resize inverse；
    
- 相同 evaluator；
    
- 相同 case-macro / voxel-micro 定义；
    
- 相同空病例处理；
    
- 相同 component matching 规则；
    
- 使用冻结的 Stage-A operating point，不重新做 threshold sweep；
    
- 不得使用 Train411/412、ROI Dice 或 proposal-conditioned Dice替代。
    

### 2.2 A0 输出表

每个病例、每个 GT component 输出：

```text
case_id
gt_component_id
gt_volume_mm3
equivalent_diameter_mm
short_axis_mm
stageA_component_hit
stageA_component_dice
stageA_voxel_recall
best_proposal_id
proposal_score
proposal_iou_gt
gt_containment_in_crop
normalized_center_offset_xyz
crop_boundary_touch_xyz
stageB_component_hit
stageB_component_dice
pasteback_component_dice
```

### 2.3 A0 的判定意义

- 若 Stage-A full-volume Dice 确实约 0.4–0.5，而 Stage-B 约 0.14：H1/H7 获得极强项目内证据，优先做 residual identity path。
    
- 若 Stage-A full-volume 本身也约 0.1–0.2：Stage-A coverage 仍是主要上限，但 residual refinement 仍有价值，只是预期目标应改为“保护 Stage-A 命中的 component”，而不是期望 Stage-B修复 Stage-A 完全漏检。
    
- 若 Stage-A ROI Dice 高、full-volume Dice低：问题集中在 proposal/fusion/evaluator，不应先改 segmentation backbone。
    

---

# 3. Evidence table

说明：

- **直接证据**：医学影像中显式使用前一级 mask/probability 或修正不完美 ROI。
    
- **邻近证据**：医学任务相关，但不是微小 LN CT。
    
- **设计启发**：自然图像 instance segmentation，不得视为 LN CT 的实证。
    
- **Preprint**：未确认正式同行评议版本。
    

|论文|年份 / 渠道|同行评议|任务与核心方法|解决的具体失败模式|与本项目的对应关系|代码|
|---|---|--:|---|---|---|---|
|RSTN|2018, CVPR|是|小器官 CT；前一迭代概率图作为空间权重，跨阶段联合优化|只用粗 mask 定位 crop 会丢失上下文，细阶段甚至可能劣于粗阶段|H1 最直接证据；支持 probability/logit 持续传递和 curriculum|未核实稳定官方仓库 ([开放获取计算机视觉论坛](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Recurrent_Saliency_Transformation_CVPR_2018_paper.html "https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Recurrent_Saliency_Transformation_CVPR_2018_paper.html"))|
|DeepIGeoS|2019, IEEE TPAMI|是|初始自动分割 + 初始 mask + geodesic interaction 输入第二 CNN|refinement 网络若看不到初始结果，无法针对已有错误修正|支持 CT 与 Stage-A mask共同输入；交互提示可替换为自动误差/不确定性图|有公开实现生态，官方状态未逐项核实 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/29993532/ "https://pubmed.ncbi.nlm.nih.gov/29993532/"))|
|nnU-Net cascade|2021, Nature Methods|是|低分辨率输出进入高分辨率网络继续精修|大体积任务中单独重分割会丢失粗位置先验|支持把 Stage-A segmentation 作为额外通道；不支持“只裁剪不用 mask”|是 ([Nature](https://www.nature.com/articles/s41592-020-01008-z "https://www.nature.com/articles/s41592-020-01008-z"))|
|Label Refinement Network from Synthetic Error Augmentation|2025, Medical Image Analysis|是|图像 + 含结构错误的初始 segmentation；按真实错误外观生成训练错误|泛化失败往往来自训练错误分布与基础模型真实错误不一致|强支持“经验 error/proposal distribution”，而非普通随机 augmentation|未核实 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/39368280/ "https://pubmed.ncbi.nlm.nih.gov/39368280/"))|
|M-SAM|2024, MICCAI|是|3D CT/MRI 肿瘤；coarse mask位置特征进入 adapter，并迭代精修|SAM/3D segmenter 未充分利用 coarse mask 位置语义|支持 mask-enhanced迭代 refinement；但依赖交互误差点，不可直接照搬|是 ([MICCAI 2025 - Open Access](https://papers.miccai.org/miccai-2024/491-Paper0762.html "https://papers.miccai.org/miccai-2024/491-Paper0762.html"))|
|MedSAM|2024, Nature Communications|是|多模态 2D 医学分割；box prompt；训练 box 有0–20 px扰动|理想 box 与实际 box 的轻微差异|支持 prompt mask作为额外通道和 jitter；其固定像素 jitter不适合直接用于不同尺寸 LN|是 ([Nature](https://www.nature.com/articles/s41467-024-44824-z "https://www.nature.com/articles/s41467-024-44824-z"))|
|CPC-SAM|2024, MICCAI|是|不同提示位置下的 cross-prompting 与一致性学习|输出对 prompt 位置过度敏感|直接启发 full-coordinate multi-proposal consistency；论文也指出对小或稀疏目标验证不足|是 ([MICCAI 2025 - Open Access](https://papers.miccai.org/miccai-2024/170-Paper0321.html "https://papers.miccai.org/miccai-2024/170-Paper0321.html"))|
|RoBox-SAM|2024, MLMI@MICCAI workshop|是，workshop|学习低质量 box 的 offset，在线迭代提升 box质量|不准确 box 导致注意力漂移和 mask失败|H2/H5直接启发：center/scale head + second recrop；主要为2D，需3D验证|未核实 ([Springer](https://link.springer.com/book/10.1007/978-3-031-73290-4 "https://link.springer.com/book/10.1007/978-3-031-73290-4"))|
|SegVol|2024, NeurIPS|是|3D CT foundation model；zoom-out/zoom-in，空间与语义 prompt|全体积尺度与局部细节难以兼得|启发粗定位后重新放大局部，但其90K无标注、6K有标注规模与本项目不可比|是 ([NeurIPS 会议论文集](https://papers.nips.cc/paper_files/paper/2024/hash/c7c7cf10082e454b9662a686ce6f1b6f-Abstract-Conference.html "https://papers.nips.cc/paper_files/paper/2024/hash/c7c7cf10082e454b9662a686ce6f1b6f-Abstract-Conference.html"))|
|Cascade R-CNN|2018, CVPR|是|多级检测器，每级在前一级输出质量分布上训练|train/inference proposal IoU分布错配|**自然图像设计启发**；支持经验 proposal-quality curriculum|是 ([开放获取计算机视觉论坛](https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html "https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html"))|
|Hybrid Task Cascade|2019, CVPR|是|detection与mask交错精修，并加入全卷积上下文分支|detection与segmentation独立 cascade 不能相互纠错|**自然图像设计启发**；支持 mask反向修正 box，而不是单向错误传播|是 ([开放获取计算机视觉论坛](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Hybrid_Task_Cascade_for_Instance_Segmentation_CVPR_2019_paper.html "https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Hybrid_Task_Cascade_for_Instance_Segmentation_CVPR_2019_paper.html"))|
|Mask R-CNN|2017, ICCV|是|class/box 与 mask 平行分支，RoIAlign|把实例存在性与像素 mask混成一个输出|**自然图像设计启发**；支持 presence、bbox、mask解耦|是 ([开放获取计算机视觉论坛](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html "https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html"))|
|Mask Scoring R-CNN|2019, CVPR|是|单独预测 mask IoU，校准 classification score与mask质量不一致|candidate confidence 不能代表 mask质量|支持独立 segmentation-quality head，避免用proposal score代替mask可靠度|是 ([开放获取计算机视觉论坛](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html "https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html"))|
|Retina U-Net|2020, ML4H/NeurIPS workshop|是，workshop|检测与语义分割联合，保留像素级监督|纯 segmentation需靠启发式恢复对象分数；纯检测浪费mask监督|支持 detection/presence与segmentation并行，而非用背景mask承担分类|是 ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v116/jaeger20a "https://proceedings.mlr.press/v116/jaeger20a"))|
|FocusNet|2019, MICCAI|是|小器官先预测概率位置，再高分辨率ROI pooling与专用分支|小目标被大结构和背景主导|支持高分辨率局部特征和显式位置先验；固定小器官不同于散在 LN|未核实 ([ResearchGate](https://www.researchgate.net/publication/336380278_FocusNet_Imbalanced_Large_and_Small_Organ_Segmentation_with_an_End-to-End_Deep_Neural_Network_for_Head_and_Neck_CT_Images "https://www.researchgate.net/publication/336380278_FocusNet_Imbalanced_Large_and_Small_Organ_Segmentation_with_an_End-to-End_Deep_Neural_Network_for_Head_and_Neck_CT_Images"))|
|FocusNetv2|2021, Medical Image Analysis|是|两阶段定位/ROI/分割，加小器官形状约束|小器官类别不平衡和形状不稳定|支持 size-specific branch；但不能证明单纯扩大 crop或shape prior可解决本项目|未核实 ([The Chinese University of Hong Kong](https://research.cuhk.edu.hk/en/publications/focusnetv2-imbalanced-large-and-small-organ-segmentation-with-adv-2/ "https://research.cuhk.edu.hk/en/publications/focusnetv2-imbalanced-large-and-small-organ-segmentation-with-adv-2/"))|
|Thoracic LN station stratification and size encoding|2022, MICCAI|是|LN station先验、多encoder、small/large decoder|LN低对比、位置与尺寸异质性导致低recall/precision|证明解剖位置与size branch有价值；研究对象限定可见LN短轴≥5 mm，不能外推到<5 mm|否/未发布 ([MICCAI](https://conferences.miccai.org/2022/papers/509-Paper0158.html "https://conferences.miccai.org/2022/papers/509-Paper0158.html"))|
|CT LN Segmentation Foundation Model / DGST|2025, MICCAI|是|3,346例、36,106个可见头颈LN，nnU-Netv2基础模型与few-shot适配|LN领域先验不足和小样本适配不稳定|说明LN专用预训练很有价值；但不直接解决Stage-A→B conversion|是 ([MICCAI 2025 - Open Access](https://papers.miccai.org/miccai-2025/0268-Paper0605.html "https://papers.miccai.org/miccai-2025/0268-Paper0605.html"))|
|Medical segmentation with imperfect 3D boxes|2021, preprint|否，preprint|学习修正不紧致3D box，再进行弱监督分割|假设box紧致时，真实不完美box会显著破坏分割|支持先修正 box；证据等级低，且为弱监督而非本项目|未核实 ([arXiv](https://arxiv.org/abs/2108.03300 "https://arxiv.org/abs/2108.03300"))|

---

# 4. Failure-mechanism map

|失败位置|当前可能机制|文献映射|项目内可见信号|首选干预|
|---|---|---|---|---|
|Stage-A mask/logit信息丢失|Stage-B只看CT crop，重新决定前景；弱阳性与粗形状被清空|RSTN、nnU-Net cascade、DeepIGeoS、M-SAM、label refinement|Stage-B显著低于疑似Stage-A结果；E2继续收缩|Stage-A logit anchor + zero-init residual|
|Proposal geometry mismatch|训练 jitter与真实偏移、尺度、containment、score联合分布不一致|Cascade R-CNN、MedSAM、RoBox-SAM|proposal-matched仍失败；tiny/边缘目标更差|Train Stage-A经验分布bootstrap + curriculum|
|Presence–segmentation冲突|“有没有LN”与“每个voxel边界”共用背景mask目标|Mask R-CNN、Retina U-Net、PCGrad|hard negative后FP与recall同步下降|presence head解耦；负样本不更新seg decoder|
|Hard-negative conservative collapse|背景监督数量和体素占比远大于正目标|类别不平衡logit-shift研究|4958 hard negatives 对1050 positives；E2 recall下降|negative主要训练presence；监控positive logit mass|
|Fixed-crop error|偏心/截断后单次ROI内根本没有完整目标|RoBox-SAM、HTC、SegVol|crop边界接触或低containment时失败|center/scale correction + shared-weight second pass|
|Pasteback/fusion loss|ROI结果经inverse mapping、重叠融合和过滤后被损坏|CPC-SAM的prompt一致性、HTC的交错精修是间接证据|ROI指标与full-volume指标背离|full-coordinate consistency与differentiable pasteback loss|

---

# 5. 七个核心假设的证据排序

## 总体排序

$$  
\boxed{H1 > H3 > H2 \approx H7 > H5 > H4 > H6}  
$$

H4不是不重要，而是当前没有直接的梯度相似度记录；H6也可能存在，但通常应在单proposal保真建立后再处理。

|排名|假设|支持证据|限制或反证|与现有实验的一致性|最小验证实验|成功判据|失败判据|
|--:|---|---|---|---|---|---|---|
|1|**H1：丢弃Stage-A probability/mask破坏信息**|RSTN直接报告fine劣于coarse；nnU-Net/MedSAM/M-SAM/DeepIGeoS均显式传递先前预测 ([开放获取计算机视觉论坛](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Recurrent_Saliency_Transformation_CVPR_2018_paper.html "https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Recurrent_Saliency_Transformation_CVPR_2018_paper.html"))|前一级mask错误也可能造成confirmation bias；简单拼接不一定足够|极一致：E2没有identity path，只学会减少输出|R1：CT+prob完整预测；R2：logit residual，其他不变|R2在internal full-volume上提高usable-proposal conversion ≥5个百分点；Stage-A-covered recall非劣，损坏component比例下降；paired bootstrap 95% CI下界>0|ROI Dice上升但full-volume conversion不变，或Stage-A已命中component仍大量退化|
|2|**H3：hard negative使整体LN概率下降**|小结构类别不平衡可造成前景logit向背景侧移动和欠分割 ([arXiv](https://arxiv.org/abs/2102.10365 "https://arxiv.org/abs/2102.10365"))|不一定由negative数量单独引起；域偏移也会收缩|极一致：E2 FP与recall同步下降|R5：引入presence head，负样本只更新presence专用参数；seg训练正样本不变|negative proposal specificity提高，同时正proposal平均foreground logit、recall和conversion不下降超过1个百分点|FP下降仍伴随正proposal voxel mass与recall明显下降|
|3|**H2：proposal-quality分布仍失配**|Cascade R-CNN强调每级应在前一级真实输出分布训练；MedSAM只做有限随机扰动；RoBox-SAM显示低质量prompt需显式修正 ([开放获取计算机视觉论坛](https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html "https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html"))|E2已用真实proposal，说明“是否真实proposal”本身不是充分条件|一致：E2可能匹配proposal身份，却未匹配多质量条件分布与curriculum|R4：按Train Stage-A实测联合分布生成同LN多proposal并做full-coordinate consistency|低containment/高offset bin conversion提高；Dice-vs-offset下降斜率绝对值降低≥25%|只提升高质量proposal，对实际低质量bin无改善|
|4|**H7：只优化ROI loss，没有优化pasteback/fusion后的转换**|CPC-SAM说明不同prompt坐标下输出需在共同空间一致；HTC表明各任务完全分离的cascade收益有限 ([MICCAI 2025 - Open Access](https://papers.miccai.org/miccai-2024/paper/0321_paper.pdf "https://papers.miccai.org/miccai-2024/paper/0321_paper.pdf"))|缺乏针对CT LN pasteback的直接论文；属于强项目内推断|很一致：内部ROI表现与Val103 full-volume严重背离|R7：仅新增differentiable pasteback/fusion loss，不联合微调Stage-A|pasteback前后Dice差缩小；full-volume conversion改善且ROI指标不必明显变化|ROI不变、pasteback loss下降，但final component指标仍不变，说明后处理之外还有失败|
|5|**H5：固定ROI无法修正偏心/截断，需要二次recrop**|RoBox-SAM直接预测box offset并迭代；HTC交错box/mask；SegVol zoom-in ([arXiv](https://arxiv.org/abs/2407.21284 "https://arxiv.org/abs/2407.21284"))|proposal无重叠或离目标过远时无法救回；二次裁剪增加计算和FP风险|tiny LN与边界proposal退化支持，但尚无recrop rescue统计|R6：center/log-scale head + 单次shared-weight recrop|可恢复低containment组中，≥10%的first-pass miss被second pass救回；新增FP≤0.3/case|correction误差大、反复漂移，或救回量接近零|
|6|**H4：presence与mask梯度冲突**|Mask R-CNN并行分离class/box/mask；多任务研究证实冲突梯度可损害单任务 ([开放获取计算机视觉论坛](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html "https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html"))|共享特征也可能有正迁移；未测梯度cosine不能直接断言|与conservative collapse一致，但可与H3混杂|在R5前记录positive seg loss与presence loss对共享层梯度cosine；比较detach-negative|冲突频率高且detach后positive conversion提高|梯度大多同向，detach不改善，则H4降级|
|7|**H6：同LN多proposal独立处理，缺少聚合与一致性**|CPC-SAM用多prompt预测ensemble及consistency；Mask Scoring表明需要独立质量估计 ([MICCAI 2025 - Open Access](https://papers.miccai.org/miccai-2024/paper/0321_paper.pdf "https://papers.miccai.org/miccai-2024/paper/0321_paper.pdf"))|多proposal高度相关，错误也可能被共同放大；简单union会增加FP|尚缺同一GT对应proposal数、分歧度统计|R4同时评估K个proposal的full-coordinate variance与quality-weighted fusion|proposal间方差下降；fusion优于best fixed proposal policy且不增FP|多proposal无实际多样性，或融合劣于单best proposal|

---

# 6. Ranked solution list

|排名|方案|预期收益|实现难度|训练成本|主要风险|科学创新性|
|--:|---|---|---|---|---|---|
|1|**Stage-A anchored residual refinement + core protection**|高：直接针对已命中component被破坏|中|中|coarse mask错误被过度保留|高；与CfC、real-Δt、tiny LN转化结合具有明确项目创新|
|2|**Presence/segmentation解耦，negative不更新seg decoder**|中高：最直接阻止conservative collapse|低至中|低|presence阈值仍可能造成candidate级漏检|中；机制清晰、可归因|
|3|**经验proposal扰动 + full-coordinate consistency**|中高：提升偏移与crop鲁棒性|中|中高|一致性可能把共同错误固化|高；特别适合真实Stage-A proposal链|
|4|**Center/scale correction + second-pass recrop**|中：主要救低containment但可恢复目标|中高|约增加一部分候选的第二次推理|correction漂移、计算增大|高；适合固定ROI失败机制|
|5|**Differentiable pasteback/fusion loss**|中：解决ROI→full-volume接口损失|高|中高|内存和坐标实现错误|高；对两阶段LN系统很有研究价值|
|6|**Stage-A/Stage-B有限联合微调**|不确定|高|高|破坏Stage-A recall、训练不稳定|中|
|7|**完全改成联合检测分割或单阶段**|不确定；不应预设更好|很高|很高|丢失现有资产，515例不足以支撑大规模重构|低至中，除非新建端到端实例级目标|

---

# 7. Recommended architecture：Stage-A 保真残差精修模型

建议命名为：

## **AFRR-Net：Anchor-Faithful Residual Refinement Network**

核心原则不是“再预测一个LN mask”，而是：

> **以 Stage-A logit 为默认答案；Stage-B 只学习有证据的修改。**

## 7.1 输入

保持现有：

- 320 cache；
    
- crop160；
    
- seq_len=19；
    
- LN branch；
    
- Conv-CfC/liquid sequence；
    
- real physical Δt；
    
- tumor branch冻结。
    

每个序列切片输入：

$$  
X_t =  
[  
CT_t,,  
P^A_t,,  
L^A_t,,  
M^A_t,,  
H(P^A_t),,  
SDT(M^A_t),,  
M^{prop}_t  
]  
$$

其中：

- (P^A)：Stage-A LN probability；
    
- (L^A=\operatorname{logit}(P^A))，从未阈值化的缓存读取；
    
- (M^A)：冻结 operating point 得到的binary mask；
    
- (H(P^A)=-p\log p-(1-p)\log(1-p))：entropy；
    
- (SDT)：以毫米归一化的signed distance transform；
    
- (M^{prop})：proposal内部、边缘和crop边界的空间编码。
    

proposal score、bbox尺寸、physical spacing、中心坐标和序列real-Δt作为标量输入，经MLP后用于FiLM调制CfC特征，不需要粗暴broadcast到每个voxel。

## 7.2 模块结构

```text
CT channels ─────────────── CT shallow encoder ──────┐
                                                     ├─ multi-scale fusion
Stage-A prior channels ─── prior shallow encoder ───┘
                                                            │
                                                    LN encoder/decoder
                                                            │
                                           Conv-CfC over 19 slices
                                           with real physical Δt
                                                            │
        ┌───────────────────────┬───────────────────┬────────┴─────────┐
        │                       │                   │                  │
 residual-logit head      presence head      geometry head     quality/error head
 ΔL(x)                    p_present          Δcenter, Δlogscale q_mask, U_B, FP/FN map
```

prior信息必须在高分辨率浅层和decoder skip层进入，不能只在最深bottleneck拼接；小LN经过多次下采样后，其先验会近乎消失。

## 7.3 残差输出

基础形式：

$$  
L_B(x)=L_A(x)+g(x),\Delta L(x)  
$$

其中：

$$  
\Delta L(x)=\alpha\tanh(r(x))  
$$

- residual head最后一层权重和bias初始化为0；
    
- 初始时 (\Delta L=0)，因此网络严格等于Stage-A；
    
- (\alpha)限制单次修改幅度，防止训练早期把logit整体推向背景；
    
- Stage-A logit使用 `detach()`，Stage-A保持冻结。
    

门控：

$$  
g(x)=\sigma\left(  
h_g[  
H(P_A), |SDT_A|, M_{prop},  
q_{prop}, \hat U_B, e_{FP}, e_{FN}  
]\right)  
$$

建议进一步区分：

## $$  
L_B=L_A+  
g_{\mathrm{add}}\Delta L_{+}

g_{\mathrm{del}}\Delta L_{-}  
$$

- `add gate`主要在高entropy、Stage-A边界外附近或预测FN区域开放；
    
- `delete gate`主要在预测FP区域、低presence或低mask-quality区域开放；
    
- 不使用无条件 `union(StageA, StageB)`。
    

## 7.4 为什么 residual 比重新预测完整mask安全

完整mask预测的最优解可以完全忽略 Stage-A。即使把 probability 拼成额外通道，网络仍可能把它当成弱特征。

residual formulation带来四个具体安全性：

1. **identity是参数空间中的显式解**：zero residual即恢复Stage-A。
    
2. **优化目标变成错误修正**：网络不必重新学习LN全部外观。
    
3. **可设置修改幅度与空间信任区间**：防止整体logit收缩。
    
4. **可以直接测量每个component的修改方向**：新增、删除和不变均可审计。
    

RSTN和label-refinement工作共同支持“针对基础预测错误进行训练”比从头重分割更符合cascade精修的性质。([开放获取计算机视觉论坛](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Recurrent_Saliency_Transformation_CVPR_2018_paper.html "https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Recurrent_Saliency_Transformation_CVPR_2018_paper.html"))

## 7.5 高置信度正核心保护

仅在训练中定义：

$$  
C^+ = {x:Y(x)=1 \land P_A(x)\ge \tau_{core}}  
$$

核心保护损失：

# $$  
\mathcal L_{core}

\frac{1}{|C^+|}  
\sum_{x\in C^+}  
\operatorname{ReLU}\big(L_A(x)-L_B(x)-m\big)  
$$

它保护的是“Stage-A高置信且GT确认正确”的核心，因此：

- 不会保护Stage-A高置信FP；
    
- 不会阻止边界修正；
    
- 不需要在推理时知道GT。
    

另加 trust-region：

# $$  
\mathcal L_{trust}

\sum_x w_{\mathrm{stable}}(x)  
|L_B(x)-L_A(x)|  
$$

`w_stable`在Stage-A低entropy且与GT一致区域较高，在错误区和边界区较低。

## 7.6 如何允许删除Stage-A FP

分两种情况处理。

### 正proposal内部的溢出FP

正proposal完整mask监督会使 residual 在GT背景处产生负修正，可删除血管、腺体或肌肉上的粗mask外溢。

### 整个proposal是hard negative

不应使用大量全背景segmentation loss训练residual decoder。采用：

- `presence head`判断该candidate是否存在LN；
    
- 高置信negative在candidate级丢弃；
    
- residual segmentation head对绝大多数hard negative不反向传播；
    
- 可保留一个很小、固定比例的“segmentation-negative calibration subset”，但不是主训练信号，且必须单独消融证明。
    

presence score不得直接逐voxel乘到mask上，否则会重现整体收缩。它只用于：

- candidate reject；
    
- proposal fusion权重；
    
- 是否触发second pass；
    
- 质量审计。
    

## 7.7 输出头

1. **Residual logit head**：(\Delta L)
    
2. **Presence head**：(p_{\mathrm{LN}})
    
3. **Center correction**：
    

$$  
\Delta c =  
\left[  
\frac{c^{GT}_x-c^P_x}{s^P_x},  
\frac{c^{GT}_y-c^P_y}{s^P_y},  
\frac{c^{GT}_z-c^P_z}{s^P_z}  
\right]  
$$

4. **Scale correction**：
    

$$  
\Delta s =  
[  
\log(s^{GT}_x/s^P_x),  
\log(s^{GT}_y/s^P_y),  
\log(s^{GT}_z/s^P_z)  
]  
$$

5. **Mask-quality head**：预测当前mask与GT的soft Dice或IoU。classification score与mask质量不应假定一致，这一点由Mask Scoring R-CNN直接展示。([开放获取计算机视觉论坛](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html "https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html"))
    
6. **Error/uncertainty head**：
    
    - `FP error map = 1[M_A=1,Y=0]`
        
    - `FN error map = 1[M_A=0,Y=1]`
        
    - 可同时输出heteroscedastic log variance。
        

---

# 8. Loss设计

总损失：

$$  
\begin{aligned}  
\mathcal L =;&  
I_{pos}  
(  
\mathcal L_{seg}  
+\lambda_{core}\mathcal L_{core}  
+\lambda_{trust}\mathcal L_{trust}  
+\lambda_{err}\mathcal L_{err}  
+\lambda_q\mathcal L_q  
)\  
&+\lambda_{pres}\mathcal L_{presence}\  
&+I_{geom}\lambda_{box}\mathcal L_{box}\  
&+I_{group}  
(  
\lambda_{cons}\mathcal L_{cons}  
+\lambda_{full}\mathcal L_{full}  
)  
\end{aligned}  
$$

## 8.1 正proposal segmentation loss

# $$  
\mathcal L_{seg}

\mathcal L_{Dice}(P_B,Y)  
+  
\mathcal L_{BCE}(P_B,Y)  
$$

这里不是重新做loss sweep。保持固定的基础组合，变量只来自架构和监督路由。

## 8.2 Presence loss

proposal级 focal BCE或普通BCE：

# $$  
\mathcal L_{presence}

BCE(p_{\mathrm{LN}}, y_{\mathrm{presence}})  
$$

关键不在具体BCE权重，而在**梯度路由**：

```text
positive proposal:
    presence loss → presence head + shared positive representation
    segmentation loss → residual encoder/decoder

hard negative:
    presence loss → presence-specific adapter/head
    segmentation decoder → no gradient
```

最安全的实现是对hard negative：

```python
presence_feature = shared_feature.detach()
```

这样负proposal不能通过共享backbone把LN表征整体推向背景。

## 8.3 Box correction loss

# $$  
\mathcal L_{box}

SmoothL1(\Delta c,\Delta c^_)  
+  
SmoothL1(\Delta s,\Delta s^_)  
+  
\lambda_{giou}\mathcal L_{GIoU}  
$$

只对存在目标且几何可恢复的proposal计算。

## 8.4 Quality loss

$$  
q^* = Dice(\operatorname{stopgrad}(P_B),Y)  
$$

$$  
\mathcal L_q = |q_{\mathrm{pred}}-q^*|  
$$

质量head必须看mask本身和ROI特征，而不只是proposal score。

## 8.5 Multi-proposal full-coordinate consistency

同一LN的 (K) 个proposal输出通过可微inverse crop映射到共同坐标：

$$  
P_k^F=T_k^{-1}(P_k)  
$$

质量加权ensemble：

$$  
\bar P^F(x)=  
\frac{\sum_k w_k(x)P_k^F(x)}  
{\sum_k w_k(x)+\epsilon}  
$$

$$  
w_k=  
p_{pres,k},  
q_k,  
(1-U_k),  
W_{center,k}(x)  
$$

一致性损失：

# $$  
\mathcal L_{cons}

\frac{1}{K}  
\sum_k  
JS(P_k^F,\operatorname{stopgrad}(\bar P^F))  
$$

只在各proposal共同可见区域、GT局部canvas或Stage-A支持邻域计算。不能强迫截断proposal在其不可见区域与完整proposal一致。

## 8.6 Full-coordinate fusion loss

不必一开始构造整个320³ volume。对同一GT LN及其proposal建立一个能容纳全部crop的local full-coordinate canvas：

# $$  
\mathcal L_{full}

DiceBCE(\bar P^F,Y^F)  
$$

R7再扩展到病例级稀疏full-volume canvas。

---

# 9. Proposal-quality建模与curriculum

## 9.1 不采用纯均匀随机jitter作为主策略

MedSAM的0–20像素随机扰动说明prompt augmentation有用，但固定像素扰动不考虑：

- LN真实尺寸；
    
- spacing；  
    -各轴各向异性；
    
- proposal score；
    
- 中心误差与尺度误差相关性；
    
- crop截断概率。([Nature](https://www.nature.com/articles/s41467-024-44824-z "https://www.nature.com/articles/s41467-024-44824-z"))
    

本项目首选：

## **经验分布 bootstrap + 少量support-smoothing jitter**

从冻结Stage-A在Train411/412的实际输出中统计：

$$  
o_i=\frac{c^P_i-c^{GT}_i}{s^{GT}_i}  
$$

$$  
r_i=\log\frac{s^P_i}{s^{GT}_i}  
$$

以及：

- proposal IoU；
    
- GT containment；
    
- crop-boundary touch；
    
- proposal score；
    
- LN size bin；
    
- anatomical region；
    
- 是否多个proposal对应同一GT。
    

按 `size bin × proposal-score bin` 条件bootstrap联合误差向量，而不是分别独立采样xyz。

## 9.2 四类正proposal

每个正LN构造3–4个proposal，不必每个epoch全用：

|类别|初始定义建议|监督|
|---|---|---|
|高质量|containment ≥0.9，offset低，无遮挡|full mask + presence + quality|
|中质量|containment 0.7–0.9或中等offset|full mask + correction + consistency|
|边缘containment|GT接近crop边界但仍完整可见|full mask + correction + boundary consistency|
|部分截断|containment约0.3–0.7且通过允许的center/scale correction可恢复|first-pass partial/ignore + correction；second-pass full mask|

具体分界应在A0统计后按训练分布分位数冻结，不应在Val103上选择。

## 9.3 Curriculum

- **Phase 1：identity与高质量精修**
    
    - 高containment正proposal；
        
    - 少量hard negative只训练presence；
        
    - residual zero-init；
        
    - 目标是Stage-B不再破坏Stage-A。
        
- **Phase 2：中等几何误差**
    
    - 加入经验中质量proposal；
        
    - 启用full-coordinate consistency；
        
    - 训练quality head。
        
- **Phase 3：可恢复低质量proposal**
    
    - 加入边缘和部分截断proposal；
        
    - 启用center/scale head；
        
    - second-pass shared-weight recrop。
        

Cascade R-CNN和RSTN都支持“后续阶段使用前一级真实输出分布，并从容易/可靠状态逐渐转向模型预测”的训练逻辑。([开放获取计算机视觉论坛](https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html "https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html"))

## 9.4 Geometry-insufficient proposal

定义三个状态，而不是简单positive/negative：

1. **Segmentation-sufficient**
    
    - 完整GT在crop内；
        
    - full mask supervision。
        
2. **Correction-recoverable**
    
    - 当前crop不完整；
        
    - GT仍在允许的recrop搜索半径内；
        
    - first pass以bbox correction为主，visible部分可用partial target；
        
    - second pass才计算full segmentation loss。
        
3. **Unrecoverable**
    
    - 零重叠且离目标超过训练期最大纠正范围；
        
    - 不强迫segmentation；
        
    - 若proposal确实指向其他结构，训练presence negative；
        
    - 若是Stage-A漏检，则属于Stage-A coverage问题，不能归罪Stage-B。
        

---

# 10. Minimal ablation plan

官方Val103始终冻结。所有R1–R7先在Train内部患者级validation上选模，最后仅对满足go criteria的最终模型做一次Val103评估。

|实验|唯一核心变化|验证假设|
|---|---|---|
|**R0**|当前C0，增加完整conversion instrumentation，不改模型|建立同口径基线；测H7表象|
|**R1**|CT之外增加Stage-A probability，仍预测完整mask|probability作为普通输入是否已有价值|
|**R2**|Stage-A logit anchor + zero-init residual logit|H1及identity path|
|**R3**|R2 + entropy/SDT/error gate + core/trust保护|是否能减少Stage-A已命中component损坏|
|**R4**|R3 + 经验proposal扰动 + 同LN full-coordinate consistency|H2/H6|
|**R5**|R4 + 独立presence head；hard negative不更新seg decoder|H3/H4|
|**R6**|R5 + center/scale correction + shared-weight second recrop|H5|
|**R7**|R6 + differentiable pasteback/fusion full-coordinate loss|H7|

Stage-A有限联合微调不放进这8组。只有R7已经证明Stage-B能稳定非退化后，才建立单独后续实验：

```text
freeze Stage-A encoder
仅解冻Stage-A最后一层LN logit head
学习率 ≤ Stage-B的1/10
保留Stage-A recall non-inferiority约束
```

---

# 11. 新指标与定义

## 11.1 Component matching

对每个GT LN component (G_j)，分别匹配Stage-A和Stage-B预测：

- primary hit：任意正体素重叠；
    
- secondary hit：overlap/GT volume达到固定比例；
    
- 不建议只用IoU≥0.5，tiny LN对一两个voxel误差过于敏感。
    

同一预测与多个GT冲突时采用最大overlap的一对一匹配。

## 11.2 核心指标

### Stage-A与Stage-B full-volume Dice

$$  
Dice_A=Dice(P_A^F,Y^F)  
$$

$$  
Dice_B=Dice(P_B^F,Y^F)  
$$

### Component-wise Delta Dice

$$  
\Delta Dice_j=Dice_{B,j}-Dice_{A,j}  
$$

报告：

- mean；
    
- median；
    
- 25/75分位；
    
- paired bootstrap 95% CI。
    

### Improve / damage比例

建议冻结一个临床可解释的改变阈值：

$$  
improved_j=1[\Delta Dice_j>0.05]  
$$

$$  
damaged_j=1[\Delta Dice_j<-0.05]  
$$

同时报告严格非退化率：

$$  
P(\Delta Dice_j\ge0)  
$$

### Usable-proposal recall

# $$  
R_{usable}

\frac{  
#{GT_j:\exists proposal;可由StageB理论恢复}  
}{  
#GT  
}  
$$

“可恢复”需满足：

- proposal与GT有重叠，或
    
- GT center位于允许的second-recrop纠正半径内；
    
- 不能用Stage-B结果反向定义usable。
    

### Conversion rate

$$  
Conversion=  
\frac{  
#{GT_j: usable_j=1\land StageB\ hit}  
}{  
#{GT_j: usable_j=1}  
}  
$$

进一步报告：

$$  
Preservation=  
P(StageB\ hit\mid StageA\ component\ hit)  
$$

这是本项目最关键的主指标之一。

## 11.3 几何条件指标

按以下变量绘制曲线及bootstrap CI：

- Dice vs normalized center offset；
    
- Dice vs proposal IoU；
    
- Dice vs GT containment；
    
- Dice vs proposal score；
    
- conversion vs crop boundary touch；
    
- x/y/z轴分别的offset与截断；
    
- 纠正前后proposal containment。
    

normalized offset建议按GT bbox尺寸归一化，而不是按crop160归一化：

$$  
o_x=\frac{|c_x^P-c_x^{GT}|}{s_x^{GT}+\epsilon}  
$$

## 11.4 Size-bin指标

至少：

- `<5 mm`；
    
- `5–10 mm`；
    
- `10–20 mm`；
    
- `>20 mm`。
    

每组报告：

- Stage-A component recall；
    
- usable-proposal recall；
    
- first-pass conversion；
    
- second-pass conversion；
    
- final Dice；
    
- damage rate；
    
- FP关联。
    

已有MICCAI thoracic LN研究只评估短轴≥5 mm可见LN，因此其高Dice不能用来推断本项目<5 mm表现。([MICCAI](https://conferences.miccai.org/2022/papers/509-Paper0158.html "https://conferences.miccai.org/2022/papers/509-Paper0158.html"))

## 11.5 Presence和负病例指标

- proposal-level sensitivity；
    
- proposal-level specificity；
    
- AUROC和AUPRC；
    
- negative-case FP；
    
- FP components/case；
    
- positive proposal被presence错误拒绝的数量；
    
- Stage-A命中但presence拒绝的component数。
    

## 11.6 Pasteback指标

每个proposal同时计算：

```text
ROI-native Dice
inverse-transformed local-full-coordinate Dice
after-overlap-fusion Dice
after-component-filter Dice
```

报告四步差值：

$$  
\Delta_{paste}=Dice_{after\ paste}-Dice_{ROI}  
$$

$$  
\Delta_{fusion}=Dice_{after\ fusion}-Dice_{after\ paste}  
$$

若主要损失发生在pasteback或fusion，R7才有明确依据。

---

# 12. Training pseudocode

```python
# Stage-A is frozen.
stage_a.eval()
for p in stage_a.parameters():
    p.requires_grad = False

for batch in train_loader:
    # A batch is grouped by lesion/candidate, not independent random ROIs.
    # batch.groups[g] contains multiple proposals for the same GT LN when positive.
    total_loss = 0.0

    for group in batch.groups:
        full_coord_predictions = []
        fusion_weights = []

        for sample in group.proposals:
            ct_volume = load_cache320(sample.case_id)

            with torch.no_grad():
                # Prefer cached raw logits from the frozen Stage-A operating point.
                stage_a_logit_full = load_stage_a_ln_logit(sample.case_id)
                stage_a_prob_full = torch.sigmoid(stage_a_logit_full)
                stage_a_mask_full = fixed_stage_a_binarize(stage_a_prob_full)

            # Crop exactly the same physical region from every aligned channel.
            ct_roi = crop_sequence(
                ct_volume,
                proposal=sample.proposal,
                crop_size=160,
                seq_len=19,
            )
            logit_a_roi = crop_sequence(
                stage_a_logit_full,
                proposal=sample.proposal,
                crop_size=160,
                seq_len=19,
            )
            prob_a_roi = torch.sigmoid(logit_a_roi)
            mask_a_roi = crop_sequence(
                stage_a_mask_full,
                proposal=sample.proposal,
                crop_size=160,
                seq_len=19,
            )

            entropy_a = binary_entropy(prob_a_roi)
            sdt_a = signed_distance_mm(mask_a_roi, sample.spacing)
            proposal_mask = make_proposal_spatial_encoding(
                proposal=sample.proposal,
                roi_geometry=ct_roi.geometry,
            )

            outputs = afrr_net(
                ct=ct_roi.data,
                stage_a_logit=logit_a_roi,
                stage_a_prob=prob_a_roi,
                stage_a_mask=mask_a_roi,
                entropy=entropy_a,
                signed_distance=sdt_a,
                proposal_mask=proposal_mask,
                proposal_score=sample.proposal.score,
                real_delta_t=sample.real_delta_t,
            )

            # Identity-preserving residual update.
            delta_logit = ALPHA * torch.tanh(outputs.raw_residual)
            final_logit_roi = logit_a_roi + outputs.edit_gate * delta_logit
            final_prob_roi = torch.sigmoid(final_logit_roi)

            # Presence is supervised for all proposals.
            presence_loss = bce(
                outputs.presence_logit,
                sample.presence_target,
            )

            if sample.is_hard_negative:
                # Critical anti-collapse rule:
                # no all-background segmentation loss on the residual decoder.
                total_loss += LAMBDA_PRES * presence_loss
                continue

            # Positive/recoverable proposal.
            if sample.segmentation_sufficient:
                target_roi = crop_gt_mask(sample)

                seg_loss = dice_bce(final_prob_roi, target_roi)
                core_loss = high_confidence_core_loss(
                    final_logit_roi,
                    logit_a_roi,
                    target_roi,
                    prob_a_roi,
                )
                trust_loss = stage_a_trust_region_loss(
                    final_logit_roi,
                    logit_a_roi,
                    target_roi,
                    prob_a_roi,
                )

                fp_target = (mask_a_roi.bool() & ~target_roi.bool()).float()
                fn_target = (~mask_a_roi.bool() & target_roi.bool()).float()
                error_loss = (
                    bce(outputs.fp_error_logit, fp_target)
                    + bce(outputs.fn_error_logit, fn_target)
                )

                true_quality = soft_dice(
                    final_prob_roi.detach(),
                    target_roi,
                )
                quality_loss = smooth_l1(
                    outputs.mask_quality,
                    true_quality,
                )

                total_loss += (
                    seg_loss
                    + LAMBDA_CORE * core_loss
                    + LAMBDA_TRUST * trust_loss
                    + LAMBDA_ERROR * error_loss
                    + LAMBDA_QUALITY * quality_loss
                    + LAMBDA_PRES * presence_loss
                )
            else:
                # Geometry insufficient but correction-recoverable:
                # do not force impossible full-mask segmentation.
                total_loss += LAMBDA_PRES * presence_loss

            if sample.geometry_recoverable:
                center_target, scale_target = geometry_targets(
                    sample.proposal,
                    sample.gt_bbox,
                )
                box_loss = (
                    smooth_l1(outputs.center_offset, center_target)
                    + smooth_l1(outputs.log_scale_offset, scale_target)
                )
                total_loss += LAMBDA_BOX * box_loss

            # Differentiable inverse crop for consistency/fusion.
            prob_full_local = pasteback_differentiable(
                final_prob_roi,
                sample.roi_to_full_transform,
                group.local_full_canvas,
            )
            uncertainty_full = pasteback_differentiable(
                torch.sigmoid(outputs.uncertainty_logit),
                sample.roi_to_full_transform,
                group.local_full_canvas,
            )

            weight = (
                torch.sigmoid(outputs.presence_logit)
                * outputs.mask_quality.clamp(0, 1)
                * (1.0 - uncertainty_full)
                * spatial_center_window(prob_full_local.shape)
            )

            full_coord_predictions.append(prob_full_local)
            fusion_weights.append(weight)

        if group.is_positive and len(full_coord_predictions) >= 2:
            fused = weighted_logit_fusion(
                full_coord_predictions,
                fusion_weights,
                stage_a_anchor=group.stage_a_prob_local_canvas,
            )

            consistency_loss = full_coordinate_js_consistency(
                full_coord_predictions,
                fused.detach(),
                valid_common_support=group.common_visible_region,
            )

            full_loss = dice_bce(
                fused,
                group.gt_local_canvas,
            )

            total_loss += (
                LAMBDA_CONS * consistency_loss
                + LAMBDA_FULL * full_loss
            )

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    clip_grad_norm_(afrr_net.parameters(), MAX_GRAD_NORM)
    optimizer.step()
```

---

# 13. Inference pseudocode

```python
def infer_case(case_id):
    ct_full = load_cache320(case_id)

    # Frozen Stage-A.
    stage_a_logit_full, proposals = frozen_stage_a_inference(ct_full)
    stage_a_prob_full = torch.sigmoid(stage_a_logit_full)
    stage_a_mask_full = fixed_stage_a_binarize(stage_a_prob_full)

    candidate_predictions = []

    for proposal in proposals:
        first = run_afrr_pass(
            ct_full=ct_full,
            stage_a_logit_full=stage_a_logit_full,
            stage_a_mask_full=stage_a_mask_full,
            proposal=proposal,
            crop_size=160,
            seq_len=19,
            real_delta_t=True,
        )

        # Presence does not multiply the voxel mask.
        # Reject only a high-confidence candidate-level negative.
        if (
            first.presence_prob < FIXED_NEG_REJECT
            and first.mask_quality < FIXED_LOW_QUALITY
        ):
            candidate_predictions.append({
                "proposal": proposal,
                "prediction": None,
                "reason": "high_confidence_absent",
            })
            continue

        selected = first

        # Trigger recrop only when geometry evidence says it is useful.
        need_second_pass = (
            first.predicted_boundary_touch
            or first.predicted_containment < FIXED_CONTAINMENT_TRIGGER
            or first.correction_confidence > FIXED_CORRECTION_CONFIDENCE
        )

        if need_second_pass:
            corrected_proposal = apply_center_scale_correction(
                proposal,
                center_offset=first.center_offset,
                log_scale_offset=first.log_scale_offset,
                max_center_shift=TRAIN_SUPPORT_MAX_SHIFT,
                max_log_scale=TRAIN_SUPPORT_MAX_LOG_SCALE,
            )

            second = run_afrr_pass(
                ct_full=ct_full,
                stage_a_logit_full=stage_a_logit_full,
                stage_a_mask_full=stage_a_mask_full,
                proposal=corrected_proposal,
                crop_size=160,
                seq_len=19,
                real_delta_t=True,
            )

            # Choose/fuse by predicted mask quality, not proposal score alone.
            selected = quality_weighted_pass_fusion(first, second)

        pred_full = pasteback_to_full_volume(
            selected.final_logit_roi,
            selected.roi_to_full_transform,
            output_shape=ct_full.shape,
        )

        candidate_predictions.append({
            "proposal": proposal,
            "prediction": pred_full,
            "presence": selected.presence_prob,
            "quality": selected.mask_quality,
            "uncertainty": selected.uncertainty_full,
        })

    # Logit fusion, never binary union.
    final_logit = quality_weighted_multi_proposal_fusion(
        candidate_predictions,
        stage_a_logit_anchor=stage_a_logit_full,
    )

    final_prob = torch.sigmoid(final_logit)
    final_mask = fixed_final_operating_point(final_prob)

    # Keep the existing fixed component policy during ablation.
    return final_mask, candidate_predictions
```

---

# 14. Checkpoint selection

不能用单一ROI Dice排序checkpoint。采用约束式选择，而不是随意加权总分。

## 14.1 硬约束

候选checkpoint必须同时满足：

1. Stage-A-covered component recall不低于R0超过1个百分点；
    
2. `<5 mm` conversion不出现明确退化；
    
3. negative-case FP与FP components/case不突破预设non-inferiority margin；
    
4. Stage-B damages Stage-A的component比例不高于R0；
    
5. presence不得错误拒绝大量Stage-A已命中component。
    

## 14.2 满足约束后排序

依次比较：

1. usable-proposal conversion；
    
2. Stage-A-covered preservation；
    
3. component-wise median (\Delta Dice)；
    
4. `<5 mm` conversion；
    
5. full-volume internal Dice；
    
6. FP components/case；
    
7. pasteback/fusion损失。
    

---

# 15. Stop/go criteria

以下阈值是项目决策门槛，不是文献承诺的预期提升。所有改进均需paired bootstrap验证。

|实验|Go|Stop|
|---|---|---|
|A0|Stage-A、C0、E2全部获得同evaluator full-volume结果；组件链可追踪|任一结果来自ROI、Train或标签口径不一致|
|R1|full-volume conversion提高且recall非劣|只有ROI Dice改善；或FP下降伴随recall继续下降|
|R2|相比R1，Stage-A-covered damage rate下降≥20%相对值；conversion提高≥5个百分点或CI明确为正|residual仍整体负偏；平均foreground logit显著下降|
|R3|高置信真核心保留率提高，FP不过度增加；边界区有正负双向修正|residual几乎恒为0，或gate只学会全部关闭|
|R4|低containment/high-offset bins提高；Dice-vs-offset斜率绝对值下降≥25%；multi-proposal variance降低|只改善理想proposal；fusion增加FP或固化错误|
|R5|FP下降而Stage-A-covered recall下降≤1个百分点；positive foreground logit mass稳定|再次出现FP与recall同步下降|
|R6|至少10%的“correction-recoverable first-pass miss”被second pass救回；FP增加≤0.3/case|correction偏移超出真实支持、反复漂移或救回近零|
|R7|pasteback/fusion损失显著缩小；full-volume conversion改善，即使ROI Dice基本不变|只降低训练full loss，实际final component指标不动|

额外全局淘汰条件：

- Stage-B在多数Stage-A已命中component上仍退化；
    
- `<5 mm` recall出现超过1个百分点的稳定下降；
    
- 模型主要通过降低positive voxel volume减少FP；
    
- 改善只存在于proposal-conditioned ROI指标；
    
- 结果依赖重新选threshold才能成立。
    

---

# 16. 对小于5 mm LN的具体判断

对于 `<5 mm` LN，当前不应把“更高像素分辨率”列为首要变量，因为320→448配对实验已经没有支持该方向。

四类失败必须分开：

## 16.1 Stage-A完全漏检

- 无proposal；
    
- 无Stage-A probability support；
    
- Stage-B无法解决。
    

这部分需要未来改Stage-A或增加全卷积rescue branch，不应计入Stage-B conversion分母。

## 16.2 存在重叠但几何不可用

- proposal与GT有少量重叠；
    
- containment低；
    
- crop截断。
    

首选center/scale correction和second recrop，而不是要求第一pass凭不可见信息完成mask。

## 16.3 Stage-B raw response失败

- GT完整可见；
    
- Stage-A logit在LN区域有响应；
    
- Stage-B final logit反而降低。
    

这是residual identity、core protection和negative解耦的直接目标。

## 16.4 threshold/filter/pasteback丢失

- ROI raw prediction命中；
    
- inverse transform或component policy后消失。
    

这是R7和conversion instrumentation的目标，不应继续通过threshold sweep掩盖。

对tiny LN而言，当前优先级应为：

$$  
\boxed{  
目标表示与保真路径

proposal geometry

presence/segmentation解耦

full-coordinate转换

进一步提高feature resolution  
}  
$$

FocusNet和thoracic LN size-encoding工作说明小目标需要专用位置和size表征，但前者是固定小器官，后者排除了<5 mm LN，因此只能作为结构启发。([arXiv](https://arxiv.org/abs/1907.12056 "https://arxiv.org/abs/1907.12056"))

---

# 17. Final recommendation

## 主方向：A. coarse-mask-guided residual refinement

立即执行A0，然后按R1→R5推进。模型的核心不是更复杂的backbone，而是：

- Stage-A raw logit进入Stage-B；
    
- final logit以Stage-A为anchor；
    
- residual head zero-init；
    
- 高置信真核心保护；
    
- 错误/不确定性门控；
    
- 正proposal训练segmentation；
    
- hard negative主要训练presence；
    
- checkpoint以Stage-A-covered conversion和non-degradation选取。
    

这是最有可能解释并修复“Stage-A有信息、Stage-B反而丢失”的方案，也有RSTN、nnU-Net cascade、DeepIGeoS、M-SAM和label-refinement家族的连续证据链。([开放获取计算机视觉论坛](https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Recurrent_Saliency_Transformation_CVPR_2018_paper.html "https://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Recurrent_Saliency_Transformation_CVPR_2018_paper.html"))

## 第二方向：B. proposal correction + iterative recrop

只有R2–R5证明Stage-B已经能够保护高质量proposal后，再加入R6。否则二次recrop只是让一个仍会破坏Stage-A的模型运行两次。

## 第三方向：C. multi-proposal aggregation

与R4结合，但必须：

- 在full-volume坐标一致；
    
- quality/uncertainty加权；
    
- 使用logit fusion；
    
- 保留Stage-A anchor；
    
- 禁止binary union。
    

## 暂缓：D. Stage-A/Stage-B joint optimization

只有在R7证明完整conversion链可训练后，才有限解冻Stage-A最后LN logit层。过早联合微调可能把E2的保守偏置反向传播到Stage-A，进一步损害proposal recall。

## 不选择：E. 放弃两阶段

当前证据不支持推倒重来。Stage-A仍是有价值的高召回候选和空间先验来源；515例也不足以保证一个全新联合实例分割框架能稳定超过现有系统。

**项目下一阶段的科学问题应正式改写为：**

> 在冻结Stage-A的条件下，能否构造一个以Stage-A logit为identity anchor、只在可证明错误区域进行有限残差修改的Stage-B，使其在大多数Stage-A已命中的LN component上非退化，并在保持LN recall的同时减少真实FP？

这比继续调loss、threshold、crop或分辨率，更直接、更可证伪，也更有机会形成一项完整的医学图像方法学贡献。