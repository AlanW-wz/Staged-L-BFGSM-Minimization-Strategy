# 分阶段 L-BFGSM：数值实验代码

本仓库对应修订稿的两项解析函数实验：[英文说明](README.md)、[复现与来源](docs/reproducibility.md)、[整理记录](docs/migration.md)。

| 实验 | 论文位置 | 规模 | 代码目录 |
|---|---|---|---|
| 谨慎更新参数研究 | 3.1.1，表 2 | 七函数、四维度、四组参数，共 112 次 | `experiments/parameter_study/` |
| 局部淬火切换比较 | 3.1.2，图 1 | 十函数、十维度、30 个配对起点、六方法，每套数据 18,000 次 | `experiments/local_quench/` |

两项实验分别保存函数定义、预算、停止条件和计时方式。局部淬火使用阈值切换，不包含分子模拟的五步前瞻比较。本仓库未提供 GROMACS 接口、分子输入、MD 轨迹或 SAXS 数据。

## 安装与快速检查

推荐 Python 3.12。当前验证环境为 Python 3.12.14、NumPy 2.3.5、SciPy 1.16.3、Matplotlib 3.11.2。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python verify.py
python -m unittest discover -s tests -v
```

原参数实验使用 Python 3.12.7、NumPy 1.26.4、SciPy 1.13.1；如需研究其历史数值行为，请另建环境并安装 `requirements-parameter-archive.txt`。不同环境下不承诺优化轨迹或耗时完全一致。

## 从归档数据恢复表格和图

```powershell
python -m experiments.parameter_study.summary --out results/parameter_summary.csv
python -m analysis.rebuild --out results/rebuilt_data
python replot.py --out results/quench_figure
```

这些命令不重新执行完整优化实验。核验涵盖文件哈希、十个函数的完整采样、起点配对和停止条件，并逐条核对补充起点的六条测量。程序还核对参数表的 12 个数值和三联图的 60 个维度与方法组合。绘图导出 PDF、PNG、SVG、统计 CSV、图注及生成清单，图例使用方法名，图注不列具体种子编号。

`data/` 保留原测量；`results/` 用于生成结果并被 Git 忽略。重建和绘图命令要求新的或空的输出目录，避免覆盖已有文件。

## 运行代码

先查看实验计划：

```powershell
python reproduce.py --study parameters --plan-only
python reproduce.py --study quench --plan-only
```

小规模运行检查：

```powershell
python reproduce.py --study parameters --smoke --out results/parameter_smoke
python reproduce.py --study quench --smoke --out results/quench_smoke --workers 2
```

参数 smoke 为 28 次（10 维）；淬火 smoke 为 120 次（16、32 维）。这些检查不计入论文数据。

完整实验：

```powershell
python reproduce.py --study parameters --out results/parameter_full
python reproduce.py --study quench --out results/quench_full --workers 4
```

参数实验可加 `--dims 10000` 只运行一个论文维度。淬火入口可按函数、维度、种子、方法选取子集；完整选项见 `python -m experiments.local_quench.run --help`。六方法的补充起点可通过以下命令重跑：

```powershell
python -m experiments.local_quench.run --functions block8_corrugated_rosenbrock --dims 1000 --seeds 2026110731 --out results/supplementary_rerun --workers 1
```

完整实验可能耗时较长。淬火结果及接受迭代轨迹逐次写入 SQLite；相同命令可以续跑，源码、环境或配置变化会拒绝继续旧检查点。参数实验逐次保存结果，但需新目录启动，不自动续跑。

## 数据与解释边界

图 1 比较六种方法在十个测试函数上的表现，属于探索性比较。在查看终点残差后，八维分块波纹 Rosenbrock 函数在维度 1,000 处改用一个补充起点，六种方法均从该点运行。复跑重现了原起点的高残差。

`data/local_quench/paired_starts.csv` 保存这十个函数在原配对起点上的 18,000 条测量。`supplementary_runs.csv` 和 `excluded_original_rows.csv` 分别列出补充起点和对应原起点的六条测量。`analysis.rebuild` 根据这些数据重建绘图用的 `manuscript.csv`。

每个图点包含所有 300 次终点，包括预算内未达标结果；阴影为 30 个十函数批次均值的实际最小—最大范围，不是置信区间。淬火采用进程 CPU 时间，参数研究采用实际经过的 wall-clock 时间。参数表中的内存利用率、曲率对接受率仍作为历史报告值保留，其完整逐次来源未恢复，程序不把它们冒充重新统计得到的结果。

正文说明与图例使用方法名和函数名；归档 CSV 的内部编号与生成随机起点所需的数字索引保持稳定。旧准牛顿家族比较、旧十一函数实验、重复绘图脚本和一次性报告入口已移出当前目录，可由 Git 历史恢复。原本地 `quench_study` 目录未改动。
