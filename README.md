# ECG 冠脉相关风险研究

完整规划见 [项目规划](docs/ECG_CAD项目规划.md)，本地数据统计见 [核验记录](research/local_data_audit.md)。当前定位为医院疑似冠心病或胸痛患者的研究原型；目前没有本地临床配对资料，超声属于可选后续阶段。

最新模型选型与组合设计见 [模型调研与架构方案](docs/模型调研与架构方案.md)，包括 ECG-FM、ECG-CPC/S4、ST-MEM、ECGFounder、JEPA、Mamba 等比较、每层张量尺寸及消融顺序。[模型配置规格](configs/model_architecture_v1.json) 是待实现的设计文件，当前脚本不会加载它；下文 NumPy 基线与推荐架构是不同模型。

已实现一个无需新装深度学习依赖的原始波形统计特征加逻辑回归基线。它输出 OMI、AMI、ACS、CTO、UA、STEMI、NSTEMI 的历史数据标签概率，不输出 CAD 总概率。波形特征没有使用诊断、PCI、造影结果或 Time_Interval 作为输入。训练、校准、内部评估按患者哈希分开，结果是探索性内部评估，不是外部验证或临床性能证明。

运行：

```powershell
$cadPython = 'C:\Users\jguo121\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $cadPython D:\CAD_ML\src\check_baseline.py
& $cadPython D:\CAD_ML\src\baseline_ecg.py
& $cadPython D:\CAD_ML\src\predict_ecg.py 'D:\新建文件夹\row_data\00001.dat'
```

训练默认读取 `D:\新建文件夹\CSV\train.csv` 和 `D:\新建文件夹\row_data`，结果保存到 `D:\CAD_ML\artifacts\baseline_raw`。脚本只支持本次核验的 WFDB format 16、12 导联、500 Hz、10 秒及对应增益；其他格式直接报错，格式或缺失值不合格记录会计数并保存排除日志。原始数据未修改，未提交到外部服务。

五个无效波形记录已排除，详情见 `artifacts/baseline_raw/excluded_records.json`。代码输出 CAD 概率为空，并说明缺少 CAD 金标准。七个独立输出尚未约束临床层级，不可相加。

## 已执行的第一轮深度模型

按 `docs/ECG_CAD项目规划.md` 和 `docs/模型调研与架构方案.md`，当前已完成数据锁定、患者级划分、工程质量审计、常数/人口学/波形统计控制和 E0 小型一维 ResNet OMI 基线。训练没有使用诊断、PCI、造影字段或 `Time_Interval` 作为输入，也没有读取公开隐藏标签。

工作区虚拟环境使用 RTX 5060 可用的 `torch==2.9.1+cu128`。首次运行前安装依赖：

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

执行检查和数据锁定：

```powershell
& .venv\Scripts\python.exe src\check_pipeline.py
& .venv\Scripts\python.exe src\ecg_data.py
```

E0 配置、训练和独立评估：

```powershell
& .venv\Scripts\python.exe src\train_ecg.py
& .venv\Scripts\python.exe src\train_ecg.py --evaluate
```

结果位于 `artifacts/data_v1`、`artifacts/e0_resnet_omi_v1_run` 和 `artifacts/e0_resnet_omi_v1/controls`。`predict_deep_ecg.py` 只对完整合格 12 导联输入输出研究性 OMI 概率；CAD 总概率明确返回不可估计。当前 E0 仍是同源内部探索，下一阶段需先锁定并核验 ECG-FM/ECG-CPC 权重及其原生预处理，再做冻结表征对照。
