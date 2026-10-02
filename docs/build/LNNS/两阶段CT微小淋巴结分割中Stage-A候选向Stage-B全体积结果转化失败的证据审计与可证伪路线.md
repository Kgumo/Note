# 两阶段CT微小淋巴结分割中Stage-A候选向Stage-B全体积结果转化失败的证据审计与可证伪路线

## 执行摘要与路线结论

截至 **2026年8月3日**，现有公开文献可以较有力地支持三项一般结论。

第一，成功的医学影像级联并不只是让第一阶段“告诉第二阶段去哪里裁剪”。nnU-Net级联把低分辨率阶段的分割结果作为高分辨率阶段的额外输入；RSTN反复传递显著性或概率信息；多发性硬化病灶级联则明确把高敏感候选生成与假阳性削减分成不同职责。也就是说，**显式保留第一阶段的像素级证据、对象身份或候选状态，是成熟级联设计中的常见机制；仅传ROI位置会使Stage-B实际上重新完成一次检测与分割，而不是严格意义上的refinement。** citeturn1search1turn1search13turn1search0turn8search0

第二，小病灶场景中的“降低FP同时删除弱TP”不是本项目独有现象。肝病灶假阳性过滤研究明确报告，过滤提高precision和F1时会因将真病灶误判为假阳性而降低recall；其尺寸相关过滤尤其容易删除小真病灶，而且在不同数据集间学习到的阈值不能直接泛化。2026年的低剂量CT溶骨性病灶研究同样显示，增强负样本抑制可以控制假阳性，但伴随召回和病灶检测率下降。citeturn15view1turn15view2turn13search2

第三，当前项目给出的 **145/148、146/148以及103与146等分母冲突**，使任何关于Stage-A覆盖率、Stage-B丢失病灶数和模型增益的结论都暂时不具备充分可核验性。这些数字是项目陈述，不是已经由本报告审计的数据事实。医学分割研究中，置信区间宽度常明显大于模型间微小Dice差距；对221篇MICCAI分割论文的分析发现，超过一半未报告性能变异，仅0.5%报告置信区间，重建后的中位95%置信区间宽度为0.03，而第一、第二名的中位差异仅0.01。citeturn15view3

因此，本报告的主结论是：

> **选择路线G：多个机制同时存在，但必须按明确顺序处理。当前操作层面的第一优先级是A——评价、组件身份和实现证据链；通过后，机制层面最值得检验的是C——Stage-A→Stage-B接口信息不足，以及D——真实proposal与训练ROI分布不匹配。尚无证据支持把主要资源继续投入新Stage-B结构、CfC变体、额外损失或更多后处理模块。**

当前最可能的三个真实机制，按“应先验证的顺序”而不是未经审计的概率排序如下。

| 顺序 | 候选机制 | 当前项目线索 | 文献支持强度 | 立即判定方法 |
|---|---|---|---|---|
| 首位 | 评价、组件身份、checkpoint与pasteback链不统一 | 多个覆盖分母和组件分母冲突；ROI与full-volume结果口径可能不同 | 强：医学影像排名、指标和置信区间研究表明，评价定义及小样本变异足以改变模型排序 citeturn11view4turn14view5turn15view3 | A0固定合同；同一checkpoint重复产生相同prediction hash和逐组件结果 |
| 次位 | Stage-A只提供位置，Stage-B丢失第一阶段对象证据，同时受到真实proposal分布偏移 | GT-centered ROI高，真实proposal和full-volume下降；稠密prior是否真实存在尚未核实 | 中到强：成熟级联常显式传递粗分割；proposal质量不匹配是通用级联风险，但本项目因果关系尚未证明 citeturn1search1turn1search0turn2search2 | A1、A2、A5；比较CT-only、CT+真实prior，其他条件完全不变 |
| 再次 | 弱真阳性与困难假阳性在现有输入下高度重叠，抑制模块只能沿同一score轴交换recall和FP | presence gate、hard-negative和FP抑制使模型更保守；无prior偶尔救回病灶 | 强邻近证据，但不是淋巴结特异证明：FP过滤和负样本抑制会删除小真病灶 citeturn15view1turn13search2 | A3绘制病灶级recall–FP曲线；A4计算完美proposal分类器的oracle上限 |

在A0–A5完成之前，应冻结：CfC结构优化、分辨率继续提升、序列长度搜索、新loss、更多presence gate、通用残差修正、second-pass recrop、复杂proposal fusion，以及任何根据最终验证集为各模型单独选择阈值的实验。

## 项目证据状态与两阶段转化机制

**项目历史路线与当前问题必须分开。** CfC、真实层间距、2D编码器和轴向序列融合是最初的表征方案；当前待解释的则是一个端到端系统转化问题。即使CfC的ROI分割能力良好，Stage-A未提出候选、ROI截断、prior缺失、坐标逆变换错误、阈值删除或component filter删除，都不会因改进轴向序列单元而自动消失。

**当前可以接受为“项目陈述、待审计”的内容：**

项目曾观察到宽松定义下Stage-A覆盖大多数GT病灶；GT-centered或内部ROI上的Stage-B指标可能很高；真实Stage-A proposals与完整体积部署指标明显较差；提高分辨率、增加序列长度、small-object loss、proposal-matched微调、presence gate和残差修正未带来稳定的全体积改善；FP抑制通常伴随recall下降；真实prior与零prior实验出现相互矛盾的行为；Stage-A可能主要传递裁剪位置，而非严格配准的dense logit。

这些现象在未获得逐病例输出、数据划分、代码版本、checkpoint、proposal快照和evaluator之前，不能升级为“已确定的实验事实”。

**当前已经确定的逻辑事实：**

1. 145/148与146/148不能在同一个固定定义下同时作为唯一Stage-A覆盖结果；103与146也不能在不解释组件合并、排除规则或数据子集的情况下共享同一召回分母。
2. GT-centered ROI表现不能估计Stage-A遗漏病灶，也不能包含真实proposal中心偏差、截断、重复候选和负候选分布。
3. ROI内Dice不能检测pasteback错位、重复融合、全体积component filter或候选缺失。
4. threshold只能重新切分已有raw probability，不能恢复Stage-B根本没有产生的病灶信号。
5. 没有真实dense prior时，所谓“Stage-B refiner”实际上更接近“在proposal crop内重新检测并分割”。

**两阶段转化链应重写为可追踪的条件链，而不是一个总Dice：**

```text
固定GT component身份
        ↓
Stage-A是否产生覆盖该GT的proposal
        ↓
proposal几何是否足以包含病灶
        ↓
crop / resampling是否保留病灶并保持坐标一致
        ↓
Stage-B是否产生raw foreground probability
        ↓
threshold后是否仍有连通信号
        ↓
重复proposal fusion后是否仍命中
        ↓
component filtering后是否仍命中
        ↓
inverse transform与pasteback位置是否正确
        ↓
full-volume匹配器是否将其计为TP
```

对第 \(i\) 个GT组件，可把最终命中概率按条件概率链写成：

\[
P(F_i)=P(A_i)\,
P(C_i\mid A_i)\,
P(R_i\mid C_i,A_i)\,
P(T_i\mid R_i,\ldots)\,
P(U_i\mid T_i,\ldots)\,
P(P_i\mid U_i,\ldots)
\]

其中 \(A\) 是Stage-A覆盖，\(C\) 是有效crop，\(R\) 是raw signal，\(T\) 是阈值生存，\(U\) 是fusion/filter生存，\(P\) 是正确pasteback与最终匹配。这是条件概率恒等分解，不要求各环节相互独立。它直接说明：**即使条件于“进入Stage-B且裁剪良好”的ROI Dice很高，只要任一前后环节生存率低，full-volume召回仍会很低。**

逐病灶谱系必须给每个病灶一个唯一且互斥的“首次死亡位置”：

| 首次死亡标签 | 判定定义 | 指向的问题 |
|---|---|---|
| `NO_PROPOSAL` | Stage-A无任何满足预注册覆盖规则的proposal | Stage-A召回 |
| `GEOMETRY_FAIL` | 有proposal但GT containment不足或被crop截断 | proposal几何或ROI设计 |
| `RESAMPLE_FAIL` | 原crop含病灶，重采样后GT/图像对应异常 | 预处理实现 |
| `NO_RAW_SIGNAL` | ROI有效，但GT邻域raw probability低于最低审计水平 | Stage-B输入或表征 |
| `THRESHOLD_KILL` | raw signal存在，固定阈值后二值信号消失 | calibration/threshold |
| `FUSION_KILL` | 单proposal可命中，融合后消失或被覆盖 | proposal融合 |
| `FILTER_KILL` | fusion后存在，component filter后删除 | 后处理 |
| `PASTEBACK_MISALIGN` | ROI内位置正确，回填后与GT错位 | 坐标变换 |
| `MATCHING_DISAGREEMENT` | 可视上相交，但evaluator未匹配或重复计数 | 评估器规则 |
| `SURVIVED` | 最终命中 | 完整转化成功 |

该标签体系比“Stage-B漏了43个病灶”更有解释力，因为后者可能把模型无信号、阈值删除和pasteback错位混为一个数字。

## 文献证据分级与可迁移结论

本次检索优先采用期刊或会议原文、PubMed、出版商页面、CVF/MICCAI正式页面和作者代码仓库。A级不意味着论文与本项目完全相同，而是其对象、级联机制或失败链与本项目直接相近；D级只能作为设计启发。

| 论文 | 年份 | 渠道及出版状态 | 任务 | 等级 | 研究机制及与本项目对应 | 主要结果 | 负面结果或局限 | 代码 |
|---|---:|---|---|:---:|---|---|---|---|
| Iuga等，*Automated detection and segmentation of thoracic lymph nodes from CT using 3D foveal fully convolutional neural networks* | 2021 | BMC Medical Imaging；同行评议；DOI 10.1186/s12880-021-00599-z | 胸部增强CT淋巴结检测与分割 | A | 直接展示从全CT发现到LN分割的完整任务，且按淋巴结大小分析检测 | 89例训练、4275个LN；未见15例测试中的检测率低于交叉验证；较大LN明显更易检测 | 5–10 mm LN检测率明显低于≥20 mm，说明“微小病灶候选机会”本身就是独立瓶颈；外部数据下降 citeturn10view0turn13search0 | 未见与现行主流框架同等完整的官方训练代码证据 |
| Mathai等，*Segmentation of mediastinal lymph nodes in CT with anatomical priors* | 2024 | IJCARS；同行评议；DOI 10.1007/s11548-024-03165-4 | 纵隔LN全体积分割 | A | 显式引入28个解剖结构先验，说明LN与血管、食管等困难组织区分需要超出局部CT crop的信息 | 使用89例训练，并在另一机构15例上测试；报告不同LN尺寸下Dice | 摘要不足以证明每一种prior相对no-prior的独立因果增益；总体性能仍受小LN和外域影响 citeturn10view1turn13search1 | 基于公开TotalSegmentator与nnU-Net组件；完整项目代码公开状态需逐版本核实 |
| Bouget等，*Mediastinal lymph nodes segmentation using 3D convolutional neural network ensembles and anatomical priors guiding* | 2023 | 正式期刊；同行评议；DOI 10.1080/21681163.2022.2043778 | 纵隔LN检测/分割 | A | 比较完整体积、slab及ensemble，评估解剖prior；与困难FP和上下文直接相关 | 最优组合报告约92%召回、约5 FP/患者及约80.5%重叠 | 研究对象以≥10 mm LN为主；作者认为仅四个解剖prior仍不足；ensemble增加计算量 citeturn9search1turn9search5 | 有作者GitHub及训练模型 citeturn9search21 |
| Xie等，*Recurrent Saliency Transformation Network for Tiny Target Segmentation in Abdominal CT Scans* | 2020 | IEEE TMI；同行评议；DOI 10.1109/TMI.2019.2930679 | 腹部CT微小器官/目标分割 | A | 多阶段间显式传递saliency/probability并迭代修正，不是只传裁剪位置 | 证明多阶段视觉线索对微小目标有用 | 不是胸部LN；并未证明任意二阶段都会改善；复杂迭代可能累积第一阶段偏差 citeturn1search0turn9search3 | 作者页面报告代码或预训练模型 |
| Setio等，*Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in CT images: The LUNA16 challenge* | 2017 | Medical Image Analysis；同行评议；DOI 10.1016/j.media.2017.06.015 | 肺结节候选检测与FP削减 | A | 把高召回候选生成与FP reduction分开，并以FROC评价；与Stage-A→Stage-B职责划分高度相似 | 统一候选和FROC评价使算法可比较，组合方法提高总体检测表现 | 检测而非像素分割；不能证明某种Stage-B接口一定最佳 citeturn8search3turn3search26 | 挑战数据和评价工具公开 |
| van Leeuwen等，*The negative sigmoid loss for controlling false positive rate in osteolytic lesion segmentation* | 2026 | Computers in Biology and Medicine；同行评议；DOI 10.1016/j.compbiomed.2026.111713 | 低剂量CT溶骨性小病灶分割 | A | 直接研究通过负样本/损失控制FP，与项目“保守化后召回下降”对应 | 可降低FP并提高precision | 控制FP伴随recall和病灶检测下降，证明FP–TP不是可免费分离的 citeturn13search2turn12search2 | 作者代码公开 citeturn12search6 |
| Valverde等，*Improving automated multiple sclerosis lesion segmentation with a cascaded 3D convolutional neural network approach* | 2017 | NeuroImage；同行评议；DOI 10.1016/j.neuroimage.2017.04.034 | MRI多发性硬化小病灶 | B | 第一网络高敏感筛候选，第二网络降低误分类；支持职责分解 | 级联改善小病灶分割和FP控制 | MRI脑病灶而非LN；第二阶段仍可能删除TP；不能说明位置-only接口足够 citeturn8search0 | 官方代码公开 citeturn8search4 |
| Isensee等，*nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation* | 2021 | Nature Methods；同行评议；DOI 10.1038/s41592-020-01008-z | 多任务医学分割 | B | 低分辨率级联输出被作为全分辨率阶段的额外分割输入；不是仅用于crop | 强基线在多数据集表现稳健；级联按数据特征选择 | 级联并非默认总优于单阶段，且任务多为器官或肿瘤；不能直接解释微小LN citeturn1search1turn1search13 | 官方代码公开 |
| Roth等，*An application of cascaded 3D fully convolutional networks for medical image segmentation* | 2018 | Computerized Medical Imaging and Graphics；同行评议；DOI 10.1016/j.compmedimag.2018.03.001 | 腹部CT器官分割 | B | 粗网络定位候选区，细网络处理缩小的体素空间 | 粗到细在未见CT上改善胰腺等器官分割，并减少需分类体素 | 目标远大于微小LN；性能提高可能来自更高有效分辨率而非接口本身 citeturn8search1turn8search5 | 有作者代码 |
| Bhat等，*Influence of uncertainty estimation techniques on false-positive reduction in liver lesion detection* | 2022 | MELBA；同行评议；DOI 10.59275/j.melba.2022-5937 | CT/MRI肝病灶FP后处理 | B | 使用第二阶段分类器过滤分割组件，直接对应presence gate/component filter | precision和F1可提高 | recall下降；小真病灶易被尺寸相关规则删除；跨数据集阈值失效；不确定性特征贡献有限 citeturn14view4turn15view1turn15view2 | 论文提供实现细节；代码公开程度需核实 |
| Reinke等，*Metrics Reloaded: recommendations for image analysis validation* | 2024 | Nature Methods；同行评议；DOI 10.1038/s41592-023-02151-z | 医学图像指标选择 | C | 强调指标必须对应任务、对象级与像素级问题，支持Dice、病灶召回和FROC并报 | 提供结构化指标选择框架 | 不解决模型本身；只能防止错误结论 citeturn14view5 | 有配套资源 |
| Reinke等，*Understanding metric-related pitfalls in image analysis validation* | 2024 | Nature Methods；同行评议；DOI 10.1038/s41592-023-02150-0 | 评价陷阱 | C | 解释空预测、小对象、对象匹配和聚合方式如何改变结论 | 系统归纳评价错误 | 属方法学证据，不是LN实证 citeturn10view8 | 有配套材料 |
| Maier-Hein等，*Why rankings of biomedical image analysis competitions should be interpreted with care* | 2018 | Nature Communications；同行评议；DOI 10.1038/s41467-018-07619-7 | 生物医学挑战排名稳定性 | C | 证明测试集、聚合和排名规则可改变算法次序 | 排名对评价选择和样本变化敏感 | 不提供具体二阶段修复方法 citeturn11view4 | 分析材料公开 |
| Kofler等，*Confidence intervals uncovered: Are we ready for real-world medical imaging AI?* | 2024 | MICCAI相关工作/公开全文；发表状态应以正式论文集版本核实 | 分割性能不确定性 | C | 支持患者级bootstrap、多种子和不以0.01左右变化草率决策 | 221篇论文中仅0.5%报CI；中位CI宽0.03，模型差异中位0.01 | 主要是回顾性方法学分析 citeturn14view6turn15view3 | 分析代码公开 |
| He等，*Mask R-CNN* | 2017 | ICCV；同行评议 | 自然图像实例分割 | D | 平行分类、框和mask职责；RoIAlign用于保持像素对齐 | 说明分类与边界任务可分头处理，精确ROI对齐重要 | 自然图像证据；不能直接证明医学Stage-B应这样设计 citeturn2search14turn2search6 | 官方代码生态丰富 |
| Cai与Vasconcelos，*Cascade R-CNN: Delving into High Quality Object Detection* | 2018 | CVPR；同行评议 | 自然图像检测 | D | 指出训练proposal质量与推理proposal质量不匹配；按IoU质量逐级匹配 | 支持“随机jitter不一定模拟真实detector误差”的机制推断 | 不是医学分割；只属设计启发 citeturn2search2turn2search29 | 有官方实现 |
| Hasani等，*Closed-form continuous-time neural networks* | 2022 | Nature Machine Intelligence；同行评议；DOI 10.1038/s42256-022-00556-7 | 连续时间序列建模 | D | CfC可用于Stage-B切片上下文，但与proposal身份、pasteback和filter无直接机制联系 | 闭式近似避免数值ODE solver，并在论文任务中加速连续时间计算 | 不是医学分割或二阶段接口证据；不能解释当前全体积转化失败 citeturn14view7turn15view5 | 作者官方实现公开 |

文献中存在一个重要的不对称：**发表工作大多报告级联改善，而很少公开“第二阶段使结果变差”的完整逐病灶谱系。** 因此，不能把“已有很多成功级联”解释为第二阶段必然有益。较直接的负面证据主要来自FP过滤、负样本损失、跨域阈值失效和强基线复核，而不是专门针对胸部微小LN的二阶段失败研究。

nnU-Net Revisited进一步表明，在公平训练预算、强基线和充分调参下，不少复杂架构的优势会消失，经典U-Net类模型仍可能更优；这支持当前项目在审计完成前避免继续用结构复杂度解释系统性失败。citeturn11view2

## 分层瓶颈分析：候选、接口、raw probability与后处理

**Stage-A候选层。** Stage-A“覆盖”必须至少拆成四种不同口径：

| 口径 | 示例定义 | 能回答的问题 |
|---|---|---|
| 任意交叠覆盖 | proposal与GT任一体素重叠 | Stage-A是否提供过最宽松机会 |
| 中心覆盖 | GT中心或proposal中心落入对方区域 | proposal定位是否基本正确 |
| containment覆盖 | GT体素有预注册比例位于ROI内，例如≥90% | Stage-B是否有可能恢复完整边界 |
| 可分割覆盖 | 经crop、resample后，GT仍保留足够体素且不触边 | 实际Stage-B输入是否有效 |

145/148可能是“任意交叠”，146/148可能来自不同checkpoint、不同GT版本或不同匹配器，也可能包括一对多proposal；在逐组件身份锁定前，无法判断哪一个数字更接近部署能力。

胸部LN文献显示，小LN的检测明显难于大LN；因此即使总体宽松coverage很高，也必须按短轴、体积、对比度和解剖区分层。citeturn10view0 Stage-A仍有保留价值，但前提是其高召回能在固定定义下复现。若严格containment覆盖远低于任意交叠覆盖，首要问题不是Stage-B网络，而是proposal几何和ROI裁剪。

**Stage-A→Stage-B接口层。** 接口至少有四种强度：

1. **位置-only：** 仅传中心、框或crop。
2. **稀疏提示：** 传box mask、中心高斯、proposal score等人工构造信息。
3. **模型证据：** 传Stage-A真实dense probability、logit、不确定性或特征。
4. **identity-preserving refinement：** Stage-B输出是对Stage-A概率的残差或受约束修改，默认保留第一阶段已有信号。

这四者不能统称为“使用prior”。人工中心高斯不是Stage-A概率图，二值box也不是coarse mask，标量score更不是像素级objectness。

nnU-Net与RSTN表明，成熟级联可以把第一阶段分割证据直接提供给后续阶段；这为“位置-only接口可能不足”提供了机制支持，但不能单独证明本项目因此失败。citeturn1search1turn1search0

位置-only接口会迫使Stage-B从局部CT重新推断三件事：proposal是否真实、病灶在哪里、边界在哪里。对小LN而言，血管截面、软组织结节和噪声可能在有限ROI内外观相似；若Stage-A的置信度、形状、上下文和不确定性被丢弃，Stage-B没有理由保留Stage-A曾经依赖的证据。Mathai与Bouget的工作至少说明，解剖先验和更广上下文对纵隔LN识别有现实价值。citeturn10view1turn9search1

但prior也可能带来**依赖性捷径**：模型直接复制Stage-A提示，无法修复Stage-A漏提示或错提示。项目陈述中“真实prior总体指标较好，但无prior偶尔救回未明确提示的病灶”与这一风险相符，不过尚未形成经审计的因果证据。真正的测试不是“有prior模型对零prior模型”，而是：

- 正确prior；
- 空间错位prior；
- lesion identity随机置换prior；
- prior强度校准后随机化；
- prior-only、不看CT；
- CT-only；
- CT+prior。

若模型在prior错位或身份置换后仍几乎不变，prior未被有效利用；若prior-only接近CT+prior，则模型可能主要复制Stage-A；若CT+prior显著提高Stage-A-covered弱病灶生存率，同时对错prior敏感且仍能修复边界，才说明接口信息提供了独立价值。

**训练—部署proposal分布偏移。** GT-centered ROI、独立随机jitter和真实Stage-A proposal通常不共享同一联合分布。真实误差可能同时表现为：

- 中心偏差与框尺寸相关；
- 低score proposal更偏、更小或更易截断；
- 小病灶有更高相对中心误差；
- 正proposal和hard negative的解剖位置不同；
- 一个GT对应多个高度相关proposal；
- detector升级后，score、框尺寸和负候选构成都改变；
- proposal数量在患者间高度不均衡。

简单地独立采样 \(x/y/z\) 平移和尺度扰动，只匹配边际误差，不一定匹配上述联合关系。Cascade R-CNN把proposal质量与训练目标匹配视为关键问题，但这是自然图像D级证据；截至检索截止日，本次未找到一篇同行评议论文证明“独立随机jitter能够忠实模拟胸部LN Stage-A proposal联合误差分布”。citeturn2search2

proposal-matched微调若没有改善，不能立即否定分布偏移，除非确认使用的是**当前同一Stage-A版本、同一threshold、同一fusion前状态、同一正负采样权重**产生的训练proposal。旧proposal或过滤后的proposal仍可能与部署分布不匹配。

**Stage-B raw probability层。** Stage-B表征不足只有在以下条件同时成立时才得到支持：

- Stage-A proposal严格包含GT；
- crop和resample正确；
- 在任何合理阈值之前，GT邻域raw probability均低；
- pasteback前ROI坐标中的预测已无信号；
- 该现象在多个随机种子和真实proposal上重复；
- 简单2.5D、GRU/ConvLSTM和当前CfC都出现同样失败，或CfC明显弱于公平基线。

仅看二值mask不能区分“模型无信号”和“阈值过高”。每个GT至少应记录：

\[
p_{\max}^{GT},\quad
p_{95}^{GT},\quad
\sum_{x\in GT}p(x),\quad
\max_{x\in N_r(GT)}p(x),\quad
\text{rank of GT component among ROI peaks}
\]

同时记录最相似hard-negative区域的相同量。若弱TP与hard FP的score分布大幅重叠，任何单一presence threshold都只会沿同一ROC/FROC曲线交换recall与FP，而不会创造可分信息。

**threshold、fusion、filter与pasteback层。** Bhat等的研究说明，FP过滤器可以显著提高precision，却将部分真病灶过滤掉；小预测组件尤其危险，因为尺寸特征既标识大量FP，也标识真实小病灶。跨数据集时，形状特征阈值分布变化还会导致性能下降。citeturn15view1turn15view2

因此，应把后处理看成有损决策，而不是免费清理。必须分别输出：

| 状态 | 输出 |
|---|---|
| 每个proposal的raw probability ROI | 不阈值、不融合 |
| 固定阈值后的proposal mask | 检查threshold损失 |
| fusion前的全体积概率 | 检查单proposal是否可命中 |
| fusion后的全体积概率 | 检查max/mean/weighted策略 |
| component filter前二值体积 | 检查阈值和融合联合效果 |
| component filter后体积 | 检查尺寸、形状和位置过滤损失 |
| 最终坐标空间预测 | 检查inverse transform和pasteback |
| evaluator匹配表 | 检查一对多、多对一及重复计数 |

重复proposal融合必须明确区分：

- **max probability：** 较能保留任一强信号，但可能保留FP；
- **mean probability：** 多个低质量proposal可稀释弱TP；
- **binary union：** 召回友好但可能增大组件并连接相邻FP；
- **weighted fusion：** 依赖proposal score校准，若score跨患者或版本失配会引入新偏差；
- **NMS后单proposal：** 简单，但可能选择框更准、mask更差的proposal。

这些策略没有脱离数据的普遍最优者，必须在固定raw outputs上进行无训练比较。

## 评价合同、逐病灶审计与oracle容量

**A0：权威评价合同。** 在任何新训练前，建立不可变manifest：

| 类别 | 必须固定的字段 |
|---|---|
| 数据 | patient_id、series_uid、原始图像hash、GT版本hash、spacing、orientation |
| GT | component_id、体素数、体积、短轴、质心、bounding box、是否纳入评价 |
| Stage-A | checkpoint hash、代码commit、proposal生成配置、proposal_id、score、box、coarse mask/logit文件hash |
| Stage-B | checkpoint hash、配置hash、encoder/decoder版本、输入ROI manifest |
| 几何 | crop定义、padding、重采样坐标约定、插值方式、affine、inverse transform |
| 后处理 | threshold、fusion、NMS、component connectivity、最小尺寸及其他filter |
| 评价 | matcher版本、命中规则、分母、空病例规则、重复proposal处理 |
| 输出 | 每例raw prediction hash、最终prediction hash、运行环境和随机性设置 |

同一checkpoint、同一输入manifest和同一容器环境至少重复运行两次。成功标准应是：

- proposal列表完全一致；
- prediction文件hash一致，或在明确记录的非确定性容差内逐体素一致；
- 每个GT的first-death标签完全一致；
- 所有汇总指标完全一致。

A0通过前，**小于约0.01的Dice变化不得作为路线依据**。这一阈值不是通用统计定律，而是项目的保守决策规则；其合理性受到医学分割文献中0.03中位CI宽度和0.01中位模型差距的支持。citeturn15view3

**评价集用途必须分开：**

- training：参数学习；
- development：损失、采样和结构选择；
- calibration：选择唯一threshold和后处理参数；
- validation/test：只应用冻结配置，不允许为每个模型单独寻找最优最终阈值。

若数据量不足以单独划分calibration集，可以在训练/开发患者内部做嵌套交叉验证，但不能利用最终验证患者选择threshold。

统计不确定性应以**患者为聚类单位**。同一患者的多个病灶、proposal和切片不独立，因此不能简单把每个病灶当独立样本计算过窄CI。主要模型差异用患者级配对bootstrap，病灶级结果作为分层描述；训练实验至少三个随机种子，或在固定预测下提供配对患者bootstrap 95% CI。Metrics Reloaded和metric-pitfall研究均强调评价单位与临床任务必须一致。citeturn14view5turn10view8

**A1：逐病灶转化谱系。** 输出主表每行一个固定GT component：

| 字段组 | 关键字段 |
|---|---|
| 身份 | patient_id、component_id、GT版本、体积、短轴、解剖区 |
| Stage-A | covered、最佳proposal_id、proposal数量、score、中心误差 |
| 几何 | containment、crop边界距离、是否截断、resample后GT体素数 |
| Stage-B raw | GT内max/mean/分位数、邻域max、预测峰坐标 |
| 决策链 | threshold前后、fusion前后、filter前后、pasteback前后 |
| 最终 | 是否命中、匹配prediction_id、Dice、first-death |
| 追溯 | checkpoint hash、prediction hash、evaluator版本 |

A1通过的成功标准不是某个模型指标，而是 **100%的纳入GT拥有唯一、稳定、可人工抽查的first-death标签**。无法完成这一表，说明系统还不能回答“病灶在哪里丢失”。

**A2：proposal分布审计。** 对GT-centered、人工jitter、旧Stage-A、当前Stage-A和真实validation proposals，比较联合分布，而不只是均值：

- 中心误差的毫米值及相对病灶直径；
- 各轴尺度误差；
- containment；
- 病灶是否触及crop边界；
- ROI内病灶体素比例；
- proposal score；
- HU与解剖位置；
- 每病例proposal数及重复度；
- 正、负、近邻hard-negative比例；
- score与中心误差、尺寸、containment的相关性。

应用患者级bootstrap或分布距离，并绘制二维关系，例如“score–center error”“lesion size–containment”，以检验人工jitter是否匹配真实联合误差。

**A3：raw probability—后处理分解。** 无需训练。对同一raw输出扫描一组预注册阈值，分别报告：

- lesion recall–FP/case曲线；
- FROC；
- voxel Dice；
- 小于5 mm病灶召回；
- Stage-A-covered病灶生存率；
- 每个first-death阶段的病灶数；
- fusion和filter造成的配对变化。

该实验可以最快区分三种情况：

1. **raw signal已经不存在：** threshold优化无意义，问题位于输入、proposal或Stage-B。
2. **raw signal存在但被阈值删除：** calibration或score shift是主要问题。
3. **阈值后存在但最终丢失：** fusion、filter、坐标或evaluator是主要问题。

**A4：oracle容量审计。** Oracle不是可部署模型，而是判断某条路线理论上是否有足够收益。

| Oracle | 做法 | 回答的问题 | 何时停止该路线 |
|---|---|---|---|
| 完美proposal分类器 | 用GT匹配保留所有正proposal、删除所有负proposal，再沿现有fusion/pasteback评价 | presence head最多能减少多少FP | 在保持当前病灶命中时，改善仍低于预注册临床/统计阈值 |
| 完美component filter | 保留所有匹配GT的预测组件，删除未匹配组件 | 后处理FP清理的上限 | oracle仍不能明显改善full-volume指标 |
| proposal几何oracle | 用GT中心/包围框形成理想ROI，使用同一Stage-B checkpoint推理 | 框修正或recrop的乐观上限 | 理想框也不能恢复大量病灶 |
| threshold oracle | 仅作上限分析，为每病灶检查是否存在任何阈值可命中 | 丢失是否主要来自阈值 | 即使病灶特异阈值也无raw信号 |
| Stage-A TP保留oracle | 若有稠密coarse mask，强制保留其已命中的病灶，再加入Stage-B修改 | identity-preserving refiner的上限 | Stage-A稠密输出本身无可利用TP或边界太差 |
| 完美pasteback | 直接将ROI预测按GT验证过的变换回填 | 坐标实现错误的上限 | 与现有pasteback相同 |

项目应预注册“值得训练”的最低oracle headroom，例如full-volume Dice提高至少0.02，或在固定FP/case下病灶召回提高至少5个百分点。数值应根据临床需求和现有CI确定，而不是实验后调整。

**A5：Stage-A信息可用性审计。** 必须在磁盘和代码中确认，而非根据概念图假定：

- 是否保存原始dense logits；
- softmax/sigmoid前后定义；
- 是否经过阈值或连通组件化；
- coarse mask与原CT的affine、spacing、origin和方向；
- proposal score的含义及校准；
- 是否有可追踪的proposal-level特征；
- 是否有不确定性或ensemble方差；
- 是否存在从Stage-A component到Stage-B proposal的稳定ID。

审计结论只能属于以下一种：

1. **有真实、配准正确的dense prior；**
2. **只有二值coarse mask；**
3. **只有proposal score和box geometry；**
4. **只有ROI位置；**
5. **使用的是人工构造prior，而非Stage-A输出。**

没有第1或第2项时，不应在论文中声称Stage-B“refines Stage-A segmentation”。

## 最小可证伪实验矩阵与CfC定位

实验必须按门控顺序执行。A0–A5均不需要训练；任何一项失败，都应先修复该项。

| 实验 | 单一假设 | 是否训练 | 输入与固定条件 | 核心输出 | 成功标准 | 失败后停止的路线 |
|---|---|:---:|---|---|---|---|
| A0 评价合同 | 历史数字差异主要来自版本、身份或实现不一致 | 否 | 固定数据、checkpoint、proposal、后处理、evaluator | provenance表、重复运行hash、固定分母 | 同一输入完全复现；所有组件身份唯一 | 停止全部模型比较 |
| A1 逐病灶谱系 | 可定位每个病灶第一次消失的环节 | 否 | A0固定预测 | 每GT一行first-death表 | 100%组件稳定归类并可抽查 | 停止用汇总Dice解释瓶颈 |
| A2 proposal审计 | 训练ROI与部署proposal存在可量化分布偏移 | 否 | GT-centered、jitter、旧/新Stage-A proposals | 几何、score、正负与重复分布表 | 明确偏移方向、效应量和CI | 若无偏移，停止泛化的proposal-matched训练 |
| A3 raw/后处理分解 | 主要病灶损失发生于raw模型后，而非模型内部 | 否 | 同一raw predictions | threshold–FROC、fusion/filter/pasteback配对表 | 主要first-death环节明确 | 若raw无信号，停止继续调threshold；若后处理不丢失，停止后处理路线 |
| A4 oracle容量 | presence、几何、filter或refiner有足够理论上限 | 否 | A1/A3输出及GT oracle | 各路线乐观上限 | 超过预注册最低有意义改善 | oracle不足即停止相应模型 |
| A5 信息可用性 | Stage-A存在可配准、可利用的真实prior | 否 | Stage-A原始输出和几何元数据 | 信息资产及配准审计表 | prior来源、数值和坐标均可验证 | 无dense prior则停止“prior refiner”表述 |
| A6 最小接口对照 | 显式Stage-A证据能提高弱TP生存，而非仅复制prior | 是，有限 | 相同proposals、CT、backbone、训练和后处理 | CT-only、CT+box/mask、CT+dense logit；必要时residual | full-volume配对增益；错位/置换prior反事实成立 | 无稳定增益则停止接口复杂化 |
| A7 简单基线与CfC | CfC或复杂序列模块提供独立价值 | 是，最后 | 完全相同ROI、encoder、decoder、预算、选模、阈值 | 2D、2.5D、GRU/ConvLSTM、CfC、可行时简单3D | 多种子/配对CI下full-volume和小病灶FROC稳定改善 | CfC不优则降为比较模型或移除 |

**A6必须分阶段，而不是一次增加多个分支。**

第一阶段只比较：

\[
\text{CT-only}
\quad\text{vs}\quad
\text{CT + Stage-A真实dense logit}
\]

编码器、decoder、采样、loss、参数预算和后处理完全相同。若没有dense logit，则比较CT-only与CT+真实coarse mask，但必须明确其信息弱于logit。

只有prior实验显示明确headroom后，才比较identity-preserving residual refinement。其输出可形式化为：

\[
p_B(x)=\operatorname{clip}\bigl(p_A(x)+\Delta p_B(x),0,1\bigr)
\]

或在logit空间相加。关键不是公式本身，而是设置一个“零残差即复制Stage-A”的identity path，使Stage-B不必先重新发现Stage-A已有的病灶。是否值得使用取决于Stage-A coarse map的真实质量和A4 oracle，而不是级联文献的一般成功。

**presence判断与positive mask学习解耦**只应在完美proposal分类器oracle显示充足空间后测试。最小实现原则是：

- 所有proposal参与真实性分类；
- 只有与GT匹配的正proposal承担主要像素边界损失；
- hard negative主要训练proposal分类/objectness，而不是以巨大纯背景mask梯度压制所有foreground logit；
- 总损失权重预注册并匹配训练预算。

Mask R-CNN的平行分类与mask职责提供D级设计启发，但不是医学LN的直接证明。citeturn2search14

**CfC的正确定位如下：**

CfC是一类从Liquid Time-Constant dynamics推导的闭式连续时间网络。其连续变量显式进入状态更新，避免了每一步调用通用数值ODE solver；原论文报告，在与微分方程式连续模型比较的特定任务和精度设置下，训练或推理可快一个到五个数量级。该速度结论相对于ODE solver模型成立，不意味着它必然比高度优化的GRU、2.5D CNN或ConvLSTM更快。citeturn15view5

项目最初沿CT轴向使用CfC的动机是：把切片特征视为按物理 \(z\) 位置采样的序列，并将层间距作为连续间隔。这个动机只涉及Stage-B内部的切片间上下文。

| 当前问题 | CfC关系 | 判断 |
|---|---|---|
| Stage-A完全没有proposal | CfC只在已有ROI内运行 | 与CfC基本无关 |
| proposal框截断病灶 | 序列模型无法看到crop外信息 | 与CfC基本无关 |
| Stage-A未传dense prior | CfC不能重建被接口丢弃的Stage-A证据 | 与CfC基本无关 |
| GT-centered与真实proposal分布偏移 | CfC也会遭受相同输入偏移 | 与CfC基本无关 |
| ROI内切片上下文不足 | CfC可能改善z向特征融合 | 可能直接帮助Stage-B raw probability |
| 弱TP与hard FP外观相近 | 轴向上下文可能提供额外判别，但无保证 | 可能间接帮助，证据不足 |
| threshold/component filter删除TP | CfC无法直接改变已固定的后处理逻辑 | 基本无关；仅可能通过更高raw score间接影响 |
| inverse transform/pasteback错误 | 模型结构无关 | 与CfC基本无关 |
| 重复proposal融合 | CfC不决定跨proposal合并 | 与CfC基本无关 |
| evaluator分母或匹配错误 | 评价实现问题 | 完全无关 |

截至本报告截止日，没有文献证据证明当前系统失败由CfC造成，也没有本项目经审计的公平结果证明CfC优于2.5D、GRU、ConvLSTM或简单3D基线。因此，**CfC应暂时冻结为A7中的比较模型，而非当前核心优化对象。**

最快否定“CfC或Stage-B结构是主要瓶颈”的实验不是再训练GRU，而是A1+A3：若大多数遗漏病灶在Stage-B raw probability中已有信号，却在threshold、fusion、filter或pasteback后死亡，则更换CfC几乎不可能解决主要损失。

## 弱真阳性、困难假阳性与ROI到全体积的不可分问题

“降低FP会删除TP”至少有三种不同原因，必须分开。

**信息不可分。** 在当前ROI输入下，弱LN与血管、肌肉或噪声的条件分布高度重叠。此时模型只能改变决策阈值，不能同时任意提高sensitivity和specificity。其表现是TP与FP的proposal score或max probability分布重叠，所有presence gate都落在相同FROC包络附近。

**训练目标冲突。** 当大量负proposal以全背景mask参与segmentation loss时，最容易降低损失的方式可能是整体降低foreground logit。对微小病灶而言，正体素少，负区域多，这种保守化尤其危险。现有公开证据能支持负样本和FP抑制存在recall代价，但尚不能仅凭文献断言本项目的具体梯度冲突；需要比较正proposal与负proposal的gradient norm、foreground logit分布和学习曲线。citeturn14view4turn13search2

**过滤器利用了病灶大小捷径。** 小组件大量是FP，但真实微小病灶同样小。Bhat等发现，log-sum uncertainty与组件大小强相关，因而会将许多小预测判为FP；小真病灶的存在可使过滤性能变差。citeturn15view1 这与本项目中component size filter或presence gate降低FP却损害小病灶召回的现象高度相符，但仍属近直接证据。

应在固定Stage-A-covered病灶集合上绘制下列分布：

- 真阳性proposal的Stage-A score与Stage-B presence score；
- hard FP的对应score；
- 小于5 mm与≥5 mm病灶；
- 低containment与高containment；
- 有dense prior与无prior；
- GT-centered与真实proposal；
- first-death为`NO_RAW_SIGNAL`与`THRESHOLD_KILL`的病例。

真正有价值的新信息，应使TP与FP的**排序**改变，而不是只使所有logit整体上移或下移。校准可以修正整体score偏移，却不能改善排序；threshold只沿已有排序选择工作点。

ROI到full-volume的转化失败还受到选择条件影响。ROI Dice常在下列子集计算：

\[
\{\text{有proposal}\}
\cap
\{\text{被认定为正ROI}\}
\cap
\{\text{GT未被截断}\}
\]

而部署指标作用于全部GT和全部proposal。ROI Dice因此可能排除了：

- Stage-A漏检；
- 真实proposal负样本；
- 被截断的GT；
- 重复proposal；
- 无病灶病例；
- pasteback后的错位；
- component filter删除；
- 阈值校准失败。

一个模型可以在剩余的“容易、已对准正ROI”上得到高Dice，同时全体积病灶召回很低。Metrics Reloaded所强调的核心原则正是：像素级、对象级和病例级指标回答不同问题，不能用一个高Dice替代病灶级FROC或部署评价。citeturn14view5

正式报告至少需要同时提供：

| 层级 | 指标 |
|---|---|
| Stage-A机会 | 任意交叠coverage、严格containment coverage、proposal/GT、FP/case |
| ROI条件性能 | positive ROI Dice、ROI lesion recall/precision、截断分层 |
| Stage-B原始能力 | raw score AUROC/PR仅作描述；固定工作点下病灶召回 |
| 完整体积 | full-volume Dice、lesion recall、precision、FP/case、FROC |
| 小病灶 | <5 mm或预注册尺寸分组的召回与FROC |
| 转化效率 | Stage-A-covered GT中最终保留比例 |
| 组件行为 | split、merge、重复命中、首次死亡位置 |
| 不确定性 | 三个种子及患者级配对bootstrap 95% CI |
| 资源 | 参数量、显存、训练时间、端到端推理时间 |

Stage-A covered lesion preservation应定义为：

\[
\text{Preservation}
=
\frac{\#\{GT:\text{Stage-A covered且最终命中}\}}
{\#\{GT:\text{Stage-A covered}\}}
\]

但其分母必须来自固定组件清单，不能从每个模型的可见ROI重新计算。

## 路线决策、停止条件与前三项行动

**当前哪些结论可以保留：**

- 两阶段系统存在明显的“候选机会到最终命中”转化损失，是需要审计的核心问题。
- GT-centered或内部ROI结果不能代表真实proposal部署结果。
- FP抑制与病灶召回存在现实权衡，不能默认presence gate或component filter会免费改善系统。
- Stage-B结构不是完整系统的唯一决定因素。
- ROI和full-volume必须使用独立而一致的评价层次。

**哪些历史结论应降级：**

- “Stage-A覆盖145/148”或“146/148”：降级为不同历史口径，直到重建固定组件表。
- “Stage-B漏掉43个”：降级，直到明确分母和first-death阶段。
- “真实prior有效”：降级为初步观察，直到确认输入确为Stage-A真实prior并完成错位、置换和prior-only反事实。
- “proposal-matched微调无效”：降级，直到证明训练proposal与当前部署Stage-A版本严格匹配。
- “presence gate无效”或“residual refinement无效”：降级，直到A4证明其理论上限，并确认阈值和后处理固定。
- “CfC导致或解决当前失败”：应撤回；当前无直接证据。

**Stage-A是否保留。** 应保留，但必须先重新确定其严格coverage、containment和FP/case。若Stage-A确实以可接受FP代价覆盖绝大多数病灶，它提供了重要计算门控和搜索空间缩小。Stage-A不是因为Stage-B转化失败就应被删除；反之，宽松任意交叠coverage也不能证明其proposal足以支持精细分割。

**Stage-B应如何定义。**

当前不应把Stage-B定义为一个职责不清的“独立检测+真假判断+精细分割+几何修复+去重+后处理”模型。更合理的操作定义取决于A5：

- 有可靠dense prior时：定义为 **proposal-conditioned、identity-preserving的保守refiner**，允许修边界和有限纠错，但必须监控Stage-A TP preservation。
- 没有dense prior、只有位置时：诚实定义为 **ROI内重新检测与分割模型**；此时不能声称其在refine Stage-A mask。
- 完美proposal分类器oracle显示较大空间时：将真实性判断与正proposal边界学习做最小解耦。
- 几何oracle显示明显空间时：才考虑box correction或second-pass recrop。

**是否需要获取或重新生成Stage-A dense prior。** 值得优先核实和保存。建立可靠prior并不意味着它一定改善结果，而是因为没有它就无法公平回答“信息接口不足”这一关键假设。应保存logit而不只是threshold后的mask，并保存原始CT坐标变换和proposal身份。

**应立即停止的路线：**

- 在A0失败时继续比较模型；
- 为每个模型在最终验证集单独寻找最优threshold；
- 同时改变backbone、loss、sampling、prior和后处理；
- 用GT-centered Dice作为主要选模指标；
- 在未做oracle前训练新的presence gate、geometry head或recrop网络；
- 在没有真实dense Stage-A输出时构造高斯中心图并称为“Stage-A prior”；
- 因为CfC新颖而增加CfC变体、注意力或额外分支；
- 仅根据单次训练或小于0.01的Dice变化决定路线；
- 以减少FP为唯一目标，而不固定病灶召回或FROC工作点。

**应冻结的路线：**

CfC、长序列、高分辨率、small-object loss、通用residual refinement、复杂fusion及proposal correction全部冻结，直到A0–A5完成并出现对应oracle headroom。

**应保留的路线：**

Stage-A高召回候选框架、当前最稳定Stage-B checkpoint、所有raw probability、现有2D/2.5D或序列基线，以及完整体积评价管线。保留的目的不是确认它们正确，而是作为不可移动的审计对象。

**允许进入下一轮训练的门槛：**

只有同时满足以下条件，才允许执行A6：

1. 固定GT组件数及所有分母，145/148、146/148和103/146等差异已有书面解释；
2. 同一checkpoint重复推理得到一致prediction hash和组件级结果；
3. 所有GT均有稳定first-death标签；
4. proposal分布偏移已经量化；
5. Stage-A prior的真实可用性及配准已经核实；
6. 某一接口、presence或几何oracle达到预注册最低有意义headroom；
7. threshold和后处理已冻结；
8. 新实验只改变一个因素。

**最可能推翻当前主假设的最小实验。**

最小实验是 **A1+A3，不训练模型**。使用当前固定checkpoint，对所有GT建立first-death表，并在raw probability、threshold、fusion、filter和pasteback各阶段评价。

- 若大多数遗漏病灶为`THRESHOLD_KILL`、`FILTER_KILL`或`PASTEBACK_MISALIGN`，则“Stage-B表征能力或CfC是主要瓶颈”被快速否定。
- 若大多数为`NO_RAW_SIGNAL`且集中在低containment真实proposal，后处理主假设被否定，proposal几何或训练—部署偏移成为主线。
- 若高containment真实proposal仍大量`NO_RAW_SIGNAL`，而GT-centered同病灶有强信号，则接口或proposal-conditioned分布偏移获得最强支持。
- 若GT-centered和真实proposal均无raw signal，才有理由进入A7公平表征比较。
- 若大量失败属于`MATCHING_DISAGREEMENT`或运行不复现，则所有历史模型结论暂停。

**真正证明Stage-A→Stage-B接口有价值的结果**必须同时满足：

- 使用同一真实Stage-A proposals；
- 只增加真实、配准的Stage-A dense logit或coarse mask；
- encoder、decoder、训练预算、采样、threshold和后处理不变；
- 在至少三个种子或患者级配对bootstrap下，full-volume病灶召回或FROC稳定改善；
- Stage-A-covered弱病灶的preservation提高；
- FP/case不发生不可接受增加；
- prior错位或component identity置换会消除增益；
- prior-only不能达到CT+prior的全部性能；
- 增益不仅存在于ROI Dice，还存在于完整体积结果。

**最终主路线：G，多机制但有严格顺序。**

前三项行动为：

1. **立即执行A0，重建评价和数据合同。** 在145/148、146/148、103与146等冲突消除前，不训练任何新模块。
2. **执行A1+A3，生成逐病灶转化谱系和raw-to-final消融。** 这是决定Stage-A、Stage-B、后处理与pasteback责任的核心证据。
3. **执行A2+A4+A5，量化真实proposal偏移、各候选路线oracle上限，并核实Stage-A dense prior。** 只有某一路线显示充分理论空间后，才允许一个单变量A6训练实验。

现阶段没有足够证据支持继续增加Stage-B复杂度。最可信的研究方向不是提出另一个网络，而是首先证明：

> 对一个固定GT病灶，Stage-A是否真正提供了可分割的机会；该机会是否在crop和接口中保留；Stage-B是否产生了原始信号；以及该信号是否被后续决策链删除。

在这条证据链建立之前，CfC应作为冻结的比较模型，Stage-A应作为待严格审计的候选生成器，Stage-B不应被默认视为失败根源，而任何小幅ROI Dice提升都不应被解释为完整系统改善。