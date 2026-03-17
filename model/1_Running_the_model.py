"""
批量运行时序超图模型模拟

主要功能：
- 参数空间准备: 创建包含所有参数组合的 DataFrame
  - 固定参数：N=700, t_max=2000, beta=0.8, L=1
  - 可变参数：alpha (6 个值), n0 (10 个值), epsilon (7 个值)
  - 总计：6 × 10 × 7 = 420 次模拟
- 保存参数: 将参数 DataFrame 保存为 CSV 文件
- 并行运行模型: 使用多进程池并行运行所有模拟
- 进度跟踪: 实时显示任务提交进度

目录结构：
model/
├── results/                      # 模型运行结果 (由本脚本生成)
│   ├── parameters.csv
│   ├── run_pars_id0/
│   ├── run_pars_id1/
│   └── ...
├── results-gsize-dist/           # 群组大小分布
│   ├── Pk_pars_id0.csv
│   ├── Pk_pars_id1.csv
│   └── ...
└── results-gtrans-mat/           # 节点转移矩阵
    ├── T_pars_id0.csv
    ├── T_pars_id1.csv
    └── ...
"""

import multiprocessing as mp
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# 添加 code 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from code.model import TemporalHypergraphModel  # noqa: E402
from code.model import run_from_df_and_save_edgelists, read_edgelists_from_df  # noqa: E402

def setup_matplotlib_fonts():
    """设置 Matplotlib 中文字体支持"""
    try:
        plt.rcParams['font.sans-serif'] = [
            'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans'
        ]
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f'警告：字体设置出现问题：{e}')
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

def create_parameter_space():
    """创建参数空间 DataFrame"""
    # 固定参数
    N = 300
    t_max = 1000
    beta = 0.8
    L = 1

    # 测试模式 - 参数组合
    alphas = [0.2]
    n0s = [5]
    epsilons = [5,20]

    # 生产模式 - 完整参数组合
    # alphas = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    # n0s = [3, 5, 7, 8, 9, 10, 11, 12, 13, 14]
    # epsilons = [1, 5, 10, 15, 20, 25, 30]

    n_combinations = len(alphas) * len(n0s) * len(epsilons)
    print(f'\n参数组合总数：{n_combinations}')
    print(f'Alpha 取值：{len(alphas)} 个')
    print(f'n0 取值：{len(n0s)} 个')
    print(f'Epsilon 取值：{len(epsilons)} 个')

    # 创建参数 DataFrame
    parnames = ['N', 't_max', 'beta', 'alpha', 'n0', 'L', 'epsilon']
    pars_df = pd.DataFrame(columns=parnames, index=range(n_combinations))
    pars_df.index.name = 'pars_id'

    parsindex = 0
    for alpha in alphas:
        for n0 in n0s:
            for epsilon in epsilons:
                pars_df.loc[parsindex] = pd.Series({
                    'N': N,
                    't_max': t_max,
                    'beta': beta,
                    'alpha': alpha,
                    'n0': n0,
                    'L': L,
                    'epsilon': epsilon
                })
                parsindex += 1

    # 转换数据类型
    pars_df['N'] = pars_df['N'].astype(int)
    pars_df['t_max'] = pars_df['t_max'].astype(int)
    pars_df['epsilon'] = pars_df['epsilon'].astype(int)

    return pars_df

def run_simulations(pars_df, output_path):
    """并行运行所有模拟"""
    n_processes = mp.cpu_count()
    print(f'\n使用 CPU 核心数：{n_processes}')
    print('开始批量运行模型...')

    pool = mp.Pool(n_processes)
    total_runs = len(pars_df.index)

    print(f'\n总共需要运行 {total_runs} 次模拟...')

    for pars_id in pars_df.index:
        pool.apply_async(
            run_from_df_and_save_edgelists,
            args=(pars_id, pars_df, output_path)
        )
        if (pars_id + 1) % 50 == 0 or pars_id == total_runs - 1:
            print(f'已提交 {min(pars_id + 1, total_runs)}/{total_runs} 个任务到进程池,等待完成...')

    pool.close()
    pool.join()

def print_results_summary(output_path):
    """打印结果摘要"""
    result_dirs = [d for d in os.listdir(output_path) if d.startswith('run_pars_id')]
    print(f'\n生成的结果目录数量：{len(result_dirs)}')

    if len(result_dirs) > 0:
        print('\n前 5 个结果目录:')
        for dir_name in sorted(result_dirs)[:5]:
            dir_path = os.path.join(output_path, dir_name)
            file_count = len([f for f in os.listdir(dir_path) if f.endswith('.csv')])
            print(f'  {dir_name}: {file_count} 个时间步文件')

        print("\n提示：可以使用 '2_Processing_model_results.ipynb' 或相应的.py 脚本处理结果")

def main():
    """主函数"""
    print('=' * 60)
    print('批量运行模型模拟')
    print('=' * 60)

    # 设置字体
    setup_matplotlib_fonts()

    # 创建结果目录
    output_dir = 'results/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'创建结果目录：{output_dir}')

    # 创建参数空间
    pars_df = create_parameter_space()
    print('\n参数 DataFrame 前 5 行:')
    print(pars_df.head())

    # 保存参数
    pars_df.to_csv(output_dir + 'parameters.csv')
    print(f'\n参数已保存到：{output_dir}parameters.csv')

    # 运行模拟
    run_simulations(pars_df, output_dir)

    # 打印结果摘要
    print('\n' + '=' * 60)
    print('所有模拟运行完成!')
    print('=' * 60)
    print_results_summary(output_dir)

if __name__ == '__main__':
    main()