# 实验代码使用说明

本文件夹整理的是终稿第 3.1 节的数值优化实验，覆盖 A、B、C 三组实验；文件夹名 `code_of_experiment1` 不代表只包含实验 A。原始代码与实验数据保持原样，另提供统一运行入口。

**先看 [审查结果](AUDIT_zh.md)。当前不能把这个包描述为整篇论文的完整复现包。** 五步预比较的论文描述与归档算法存在差异，表 3 两个内存统计指标的来源尚不完整，分子模拟及 SAXS 资料也不在所检查的实验包内。

## 最简单的使用方式

在本文件夹打开终端，使用 Python 3.12.7：

```sh
python -m pip install -r requirements.txt
python verify.py
python replot.py
```

这会检查代码来源、函数梯度、数据完整性，随后利用真实归档数据重绘数值实验图，输出到 `results/reference_replot/`。

重新计算而非读取历史结果：

```sh
python reproduce.py --experiment all --smoke --out results/smoke
python reproduce.py --experiment A --out results/run_A
python reproduce.py --experiment B --out results/run_B
python reproduce.py --experiment C --out results/run_C
```

第一条仅为小规模运行检查，不是论文结果。后面三条分别运行完整的 A、B、C。B/C 包括 50 万维测试，需要较多时间和内存。输出目录必须为空；再次执行时换一个目录名，避免覆盖既有结果。

如果要把新计算的结果一起绘图：

```sh
python reproduce.py --experiment all --out results/full
python replot.py --data results/full --out results/full_plots
```

## 论文与代码的对应关系

| 论文内容 | 实际依赖 | 整理后的入口 |
|---|---|---|
| 3.1.1 / 图 1(a)：9 函数、4 方法、4 维度 | 原始拟牛顿函数、`lbfgsm_algorithm.py`、测试函数 | `reproduce.py --experiment A ...` |
| 3.1.2 / 表 3：7 函数、4 参数、4 维度 | 原始 L-BFGSM、Levy 变体与 trig_chain | `reproduce.py --experiment B ...` |
| 3.1.3 / 图 1(b,c)：11 函数、6 方法、4 维度 | `hybrid_methods.py`、ELS-NCG、L-BFGSM、5 个补充函数 | `reproduce.py --experiment C ...` |
| 原始图表数据和统计 | `reference/` 内归档记录 | `replot.py` |

原始文件位于 `legacy/`；每个复制文件的出处和校验值位于 `source_manifest.json`。`source_selection.csv` 列出实验目录内各 Python 文件是否入选。早期二维 Rosenbrock、Wood、LJ/FENE 演示、替代参数探索和旧 Word 报告脚本不作为最终实验入口。

## 上传 GitHub 的范围

上传本文件夹中的源码、`reference/`、`legacy/` 所选数据、说明及验证记录即可。`.gitignore` 已排除新运行结果、缓存、虚拟环境和日志。`legacy/` 中重复保留的归档数据用于核对原始绘图输入，不应误当成新生成的测试结果。

本次仅做本地整理，没有创建仓库或执行推送。作者仍需确定仓库许可证与引用信息。不要把 `AFM_article` 整个目录上传：本代码包已经与论文 PDF、投稿源文件和参考文献分开。
