# 智能投顾演示项目（中文）

本项目是一套端到端的“模拟数据 → 特征工程 → 风险预测（含时序/GNN增强与联邦平均）→ 资产配置推荐 → 可解释性分析 → GUI演示”流水线，便于在本地快速体验从数据到应用的完整流程。

- 数据来源：M1 生成的合成数据（用户画像 + 市场 + 交易 + 问答情绪）
- 模块划分：M1~M6 对应数据准备、特征与拆分、联邦风险模型、资产配置、可解释性、GUI
- 额外组件：时序建模模块 temporal_model.py（支持 n_days 窗口体现市场变化）、单元测试 test_temporal_model.py

## 操作指南（简单完整）

前提：Windows + Python 3.10 及以上（cmd.exe 环境）。以下命令可直接复制粘贴到命令行。

1) 可选：创建并激活虚拟环境，然后安装依赖

```bat
# 在项目根目录下执行
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) 一键跑通全流程（推荐）

```bat
python run_experiment.py
```

运行完成后，主要产物：
- 数据：`data/`
- 结果：`results/`（多种中文图表与指标 JSON、Markdown 报告）
- 投顾设计：`advisor_design/`（配色/仪表盘/横幅等素材，投顾建议 HTML 与文本：`advice_all.html`、`advice_all.txt`、`advice_user_*.txt`，以及示例小图标）

3) 启动 GUI 演示（本地交互生成建议）

```bat
python ui/M6_ui.py
```

4) 可选：按模块逐步运行（便于理解流水线）

```bat
python data_process/M1_data_prep.py
python data_process/M2_features_and_split.py
python models/M3_fedavg_risk_model.py
python advisor/M4_portfolio_recommender.py
python advisor/M5_explainability_report.py
```

说明：
- M3 会在 `work/predictions.csv` 输出风险预测，M4 会在 `work/portfolio_recs.csv` 输出资产配置；M5 会在 `m5_results/` 输出分析图表，并将模板风格的投顾建议文本写入 `advisor_design/`。
- 仅想生成投顾建议文档时，按 1→2（或手动执行 1→M1→M2→M3→M4→M5）即可在 `advisor_design/` 查看 `advice_all.html`、`advice_all.txt` 与 `advice_user_*.txt`。

---

## 目录结构与产物

为了更好地组织代码，项目文件已按功能分类到以下文件夹：

- **data_process/**: 数据准备与特征工程
  - `M1_data_prep.py`: 生成合成数据
  - `M2_features_and_split.py`: 特征提取与拆分
  - `kb_*.py`, `update_*.py`: 知识库相关
- **models/**: 模型训练与定义
  - `M3_fedavg_risk_model.py`: 联邦风险模型
  - `temporal_model.py`: 时序模型
  - `train_user_model.py`, `hf_risk_baseline.py`: 其他训练脚本
- **advisor/**: 投顾建议与报告生成
  - `M4_portfolio_recommender.py`: 资产配置
  - `M5_explainability_report.py`: 可解释性报告
  - `advisor_llm.py`: LLM 生成建议核心逻辑
  - `make_report.py`: 报告生成脚本
- **ui/**: 用户界面
  - `M6_ui.py`: GUI 演示程序
- **utils/**: 工具类与客户端
  - `*_client.py`: API 客户端 (SiliconFlow, Qwen, Tushare)
  - `retriever_*.py`: 检索器
  - `rag_*.py`: RAG 相关

**根目录脚本**:
- `run_experiment.py`: 一键运行全流程（CPU/混合）
- `run_gpu.py`: GPU 加速版全流程

### 关于 advice_all 生成

`advice_all.html` 和 `advice_all.txt` 是投顾建议的汇总文件，通常由 `run_experiment.py` 或 `run_gpu.py` 在流程最后调用 `advisor/advisor_llm.py` 生成。

**生成依赖文件：**
1. **用户画像数据**: `data/profiles.csv` (由 M1 生成)
2. **风险预测结果**: `work/predictions.csv` (由 M3/M4 生成) 或 `results/` 中的预测结果
3. **API 客户端**: 依赖 `utils/siliconflow_client.py` 及 API Key (若无 Key 则回退到本地模板)
4. **设计素材**: `advisor_design/` 目录下的模板和图片 (运行时自动生成)

无需手动运行其他文件，只要按顺序执行 M1->M3 或直接运行 `run_experiment.py` 即可生成。

## 环境依赖

建议 Python 3.10+（项目中存在 CPython 3.13 的字节码文件，3.13 亦可）。

必需依赖：

- numpy, pandas, seaborn, matplotlib
- scikit-learn, joblib
- tkinter（Python 内置，Windows 通常可用）

可选依赖：

- tensorflow（使用 LSTM 时需要；未安装将自动回退至 RandomForest）
- pytest（运行单元测试）
- yagmail、requests、ddddocr（仅 main.py 抢课脚本使用，与投顾流水线无关）

安装示例（Windows cmd）：

```
pip install numpy pandas seaborn matplotlib scikit-learn joblib
pip install tensorflow  # 可选，用于 LSTM
pip install pytest      # 可选，运行单测
```

## 快速开始（推荐顺序）

1) 生成合成数据（M1）

```
python c:\Users\13412\Desktop\M1_data_prep.py
```

2) 特征工程与数据拆分（M2）

```
python c:\Users\13412\Desktop\M2_features_and_split.py
```

3) 训练风险模型（M3，默认融合时序与 GNN 特征，自动保存到 work/）

```
python c:\Users\13412\Desktop\M3_fedavg_risk_model.py
```

可选开关（在代码中已有默认启用）：use_temporal/use_gnn。

4) 资产配置推荐（M4）

```
python c:\Users\13412\Desktop\M4_portfolio_recommender.py
```

5) 可解释性分析与投资建议（M5）

```
python c:\Users\13412\Desktop\M5_explainability_report.py
```

6) GUI 演示（M6）

```
python c:\Users\13412\Desktop\M6_ui.py
```

## 时序模型 temporal_model.py（体现 n_days 窗口内市场变化）

- build_sequences：
  - 将用户逐日交易 action 与市场 mkt_ret/mkt_vol（含 3/7/14 日滚动均值，可关）对齐，
  - 以滑动窗口（seq_len = n_days）构建样本，标签为“下一个交易日的动作”。
  - 对缺失日期进行鲁棒重索引（缺失动作填 0=观望），保证每个用户序列连续。
- 训练：
  - --model auto：优先 LSTM（若安装 TF），否则回退 RandomForest
  - --seq_len 30：窗口大小即 n_days，直接刻画“近 n_days 市场变化”
  - --scale：可选对特征做 StandardScaler

示例（Windows cmd）：

```
# 使用随机森林 + 二分类（买入 vs 非买入）
python c:\Users\13412\Desktop\temporal_model.py --seq_len 30 --model rf --binary --max_samples 10000

# 自动选择模型（若 TF 可用则用 LSTM），保存 scaler
python c:\Users\13412\Desktop\temporal_model.py --seq_len 30 --model auto --early_stop --save_scaler
```

单元测试：

```
pytest c:\Users\13412\Desktop\test_temporal_model.py -q
```

## 数据流与典型文件

- 数据目录：`data/`（通过 `run_experiment.py` 生成）或 `outputs/`（单独运行 `M1_data_prep.py`）
- 工作目录 `work/`：
  - M2：train_features.csv、test_features.csv
  - M3：risk_model.pkl、predictions.csv、used_features_m3.csv
  - M4：portfolio_recs.csv
  - temporal：temporal_model_rf.joblib 或 temporal_model_lstm.h5、scaler.joblib（可选）
- 结果目录 `m5_results/`：feature_importance.csv/png、user_advice.txt

M3 在 load_data 中可选融合：

- 时序特征：近窗 EMA/波动/趋势/与市场相关性/买入占比
- GNN 特征：构建用户画像 kNN 图并做 2 步传播，得到 _gnn1/_gnn2 扩散特征

## 常见问题（FAQ）

1. 运行 temporal_model 提示“没有足够的数据构建序列”？
   
   - 先运行 M1 生成数据；或检查 seq_len 是否过大。

2. LSTM 报错 tensorflow 未安装？
   
   - 可安装 tensorflow，或加参数 --model rf 使用随机森林。

3. GUI 无法加载模型？
   
   - GUI 会在缺少模型时回退规则推断；若需加载训练模型，先运行 M3。

4. 字体导致中文图例乱码？
   
   - 已在 M1/M5 中设置中文字体；如仍有问题可安装 SimHei/微软雅黑/宋体。

## 免责声明

本项目仅用于课程/学习演示，不构成任何投资建议。请勿用于真实投顾或生产环境。

## 运行 run_experiment.py 的快速指南

该脚本一键完成：生成合成数据 → 构建时间序列 → 训练多分类与二分类模型 → 导出多种中文图表与指标报告，并额外在项目根目录创建 `advisor_design/` 文件夹存放投顾设计美化素材与最终建议文档。

- 准备依赖（Windows cmd）：

```bat
python -m pip install -r C:\Users\13412\Desktop\gxb\requirements.txt
```

- 运行脚本：

```bat
python C:\Users\13412\Desktop\gxb\run_experiment.py
```

- 产物说明：
  - 数据：`gxb/data/`（profiles.csv、market.csv、transactions.csv、dialogs.csv、user_data_vis.png）
  - 结果：`gxb/results/`（多分类/二分类的混淆矩阵、特征重要性、PCA、ROC/PR、类别分布、市场概览、相关性热力图、概率校准曲线等中文图表，以及 metrics_*.json、report_*.md）
  - 投顾设计：`gxb/advisor_design/`（生成的配色方案图、仪表盘样例图、设计说明文档、示例图标，以及最终投顾建议：`advice_all.html`、`advice_all.txt`、`advice_user_*.txt`）

- 参数调整：如需控制模拟规模与性能，可在 `run_experiment.py` 的 `get_simulation_params(...)` 中调整 `n_users`、`n_days`、`max_samples` 等；脚本默认写入 `data/` 目录。

- 预览 HTML 建议：运行完毕后，直接双击打开 `advisor_design/advice_all.html` 即可在浏览器查看彩色可视化版汇总建议。
