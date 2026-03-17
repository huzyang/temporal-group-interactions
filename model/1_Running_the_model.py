"""
时序超图模型批量运行脚本

功能：
- 创建参数空间 DataFrame
- 使用多进程并行运行多个模拟
    - 参数空间准备: 创建包含所有参数组合的 DataFrame
      - 固定参数：N=700, t_max=2000, beta=0.8, L=1
      - 可变参数：alpha (6 个值), n0 (10 个值), epsilon (7 个值)
      - 总计：6 × 10 × 7 = 420 次模拟
    - 保存参数: 将参数 DataFrame 保存为 CSV 文件
    - 并行运行模型: 使用多进程池并行运行所有模拟
    - 进度跟踪: 实时显示任务提交进度
- 保存所有模拟结果到指定目录
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

import os
import sys
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append("../code/")

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
    """
    创建参数空间 DataFrame

    返回:
        tuple: (pars_df, parnames) 参数 DataFrame 和参数名称列表
    """
    # 固定模型参数
    N = 300  # 节点数量
    t_max = 1000  # 最大模拟时间步数
    beta = 0.8  # 停留时间的指数参数
    L = 1  # Logistic 函数的分子参数

    # 测试模式 - 参数组合
    alphas = [0.2]
    n0s = [5]
    epsilons = [5,20]

    # 可变模型参数
    # alphas = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]  # Logistic 增长率参数
    # n0s = [3, 5, 7, 8, 9, 10, 11, 12, 13, 14]  # Logistic 函数中点参数
    # epsilons = [1, 5, 10, 15, 20, 25, 30]  # 空组数量参数

    # 参数名称列表
    parnames = ["N", "t_max", "beta", "alpha", "n0", "L", "epsilon"]

    # 计算参数组合总数
    n_combinations = len(alphas) * len(n0s) * len(epsilons)
    parsindexes = range(n_combinations)

    # 创建参数 DataFrame
    pars_df = pd.DataFrame(columns=parnames, index=parsindexes)
    pars_df.index.name = 'pars_id'

    # 填充参数值
    parsindex = 0
    for alpha in alphas:
        for n0 in n0s:
            for epsilon in epsilons:
                pars_df.loc[parsindex] = pd.Series({
                    "N": N,
                    "t_max": t_max,
                    "beta": beta,
                    "alpha": alpha,
                    "n0": n0,
                    "L": L,
                    "epsilon": epsilon
                })
                parsindex += 1

    # 确保整数类型
    pars_df['N'] = pars_df['N'].astype(int)
    pars_df['t_max'] = pars_df['t_max'].astype(int)
    pars_df['epsilon'] = pars_df['epsilon'].astype(int)

    return pars_df, parnames


def save_parameters(pars_df, out_dir):
    """
    保存参数 DataFrame 到 CSV 文件

    参数:
        pars_df: 参数 DataFrame
        out_dir: 输出目录路径
    """
    output_path = os.path.join(out_dir, 'parameters.csv')
    pars_df.to_csv(output_path, index=True)
    print(f'参数已保存到：{output_path}')


def load_parameters(path):
    """
    从 CSV 文件加载参数 DataFrame

    参数:
        path: CSV 文件路径

    返回:
        DataFrame: 参数 DataFrame
    """
    pars_df = pd.read_csv(path)
    pars_df.set_index('pars_id', inplace=True)
    return pars_df


def run_simulations(pars_df, out_dir):
    """
    使用多进程并行运行所有模拟

    参数:
        pars_df: 参数 DataFrame
        out_dir: 结果输出目录
    """
    # 获取 CPU 核心数
    n_processes = min(mp.cpu_count() / 2, len(pars_df))
    print(f'\n使用 {n_processes} 个进程并行运行模拟...')

    # 创建进程池
    pool = mp.Pool(n_processes)

    # 提交所有任务
    total_runs = len(pars_df)
    for pars_id in pars_df.index:
        pool.apply_async(
            run_from_df_and_save_edgelists,
            args=(pars_id, pars_df, out_dir)
        )

    # 关闭进程池并等待完成
    pool.close()
    pool.join()

    print(f'\n所有 {total_runs} 个模拟运行完成!')


def print_parameter_summary(pars_df):
    """
    打印参数摘要信息

    参数:
        pars_df: 参数 DataFrame
    """
    print('\n' + '=' * 60)
    print('参数空间摘要')
    print('=' * 60)
    print(f'总模拟次数：{len(pars_df)}')
    print(f'\n固定参数:')
    print(f'  N (节点数): {pars_df["N"].iloc[0]}')
    print(f'  t_max (最大时间步): {pars_df["t_max"].iloc[0]}')
    print(f'  beta: {pars_df["beta"].iloc[0]}')
    print(f'  L: {pars_df["L"].iloc[0]}')

    print(f'\n可变参数:')
    print(f'  alpha: {sorted(pars_df["alpha"].unique())}')
    print(f'  n0: {sorted(pars_df["n0"].unique())}')
    print(f'  epsilon: {sorted(pars_df["epsilon"].unique())}')
    print('=' * 60)


def main():
    """主函数"""
    # 结果目录
    OUT_DIR = 'results/'


    # 设置字体
    setup_matplotlib_fonts()

    # 创建结果目录（如果不存在）
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
        print(f'创建结果目录：{OUT_DIR}')

    # 创建参数空间
    print('=' * 60)
    print('创建参数空间...')
    print('=' * 60)
    pars_df, parnames = create_parameter_space()

    # 显示前几行参数
    print('\n参数 DataFrame (前 5 行):')
    print(pars_df.head())

    # 打印参数摘要
    print_parameter_summary(pars_df)

    # 保存参数
    print('\n保存参数配置...')
    save_parameters(pars_df, OUT_DIR)

    # 运行模拟
    print('\n开始运行模拟...')
    run_simulations(pars_df, OUT_DIR)

    print('\n' + '=' * 60)
    print('所有任务完成！')
    print('=' * 60)


if __name__ == '__main__':
    main()
