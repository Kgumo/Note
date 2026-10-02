已打包好了，直接下载这个：

[stageA_q05_proposal_freeze_20260710.tgz](sandbox:/mnt/data/stageA_q05_proposal_freeze_20260710.tgz)

备用 zip：

[stageA_q05_proposal_freeze_20260710.zip](sandbox:/mnt/data/stageA_q05_proposal_freeze_20260710.zip)

放到服务器这个位置解压：

```bash
cd /root/LNNs/bin
tar -xzf /path/to/stageA_q05_proposal_freeze_20260710.tgz
```

解压后目录就是：

```text
/root/LNNs/bin/stageA_q05_proposal_freeze_20260710
```

运行方式，只用 Python，不跑 sh：

```bash
python -u /root/LNNs/bin/stageA_q05_proposal_freeze_20260710/run_stageA_audit.py --mode smoke
```

完整复跑：

```bash
python -u /root/LNNs/bin/stageA_q05_proposal_freeze_20260710/run_stageA_audit.py --mode full
```

只复跑 q05：

```bash
python -u /root/LNNs/bin/stageA_q05_proposal_freeze_20260710/run_stageA_audit.py --mode q05-full
```

same-class v3 后处理：

```bash
python -u /root/LNNs/bin/stageA_q05_proposal_freeze_20260710/run_stageA_audit.py --mode sameclass-v3
```

包里包含：

```text
run_stageA_audit.py
stageA_params.json
scripts/audit_stageA_pure_tumor_ln_proposals_v2.py
scripts/audit_stageA_proposals_policy_explicit.py
scripts/same_class_postprocess_v3.py
cases/train412_cases.txt
cases/val103_cases.txt
cases/train_smoke1_with_tumor_and_ln.txt
reference_reports/*.csv
INSTALL_AND_RUN.txt
MANIFEST.txt
SHA256SUMS.txt
```

注意：这个包仍依赖服务器原来的 repo 和数据路径：

```text
/root/LNNs/bin/cache_store.py
/root/LNNs/bin/common.py
/root/LNNs/bin/models.py
/root/LNNs/bin/cache_320_fp16_realspacing_v2
/root/LNNs/bin/data/masks
/root/LNNs/bin/runs/stageA_mined_hardneg_v2_conservative_20260629_070436/best_detector.pt
```