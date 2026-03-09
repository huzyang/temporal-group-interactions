"""
模型拟合脚本 - 计算模型与经验数据的距离

主要功能：
1️⃣ 读取参数和经验数据
   - 从 results/parameters.csv 读取模型参数
   - 从 ../data-analysis/results/CNS/ 加载经验数据：
     * Pk_out-of-class.csv - 经验群组大小分布
     * T_out-of-class.csv - 经验群组转移矩阵

2️⃣ 计算群组大小分布的 JSD
   - 线性尺度: JSD_gsize
   - 对数尺度: JSD_gsize_log

3️⃣ 计算群组转移矩阵的 JSD
   - 输出：JSD_T

4️⃣ 保存拟合结果
   - results/parameters_fit.csv（基本）
   - results/parameters_fit_with_combined.csv（含联合 JSD）

5️⃣ 联合最小化分析
   - 最佳群组大小分布拟合
   - 最佳转移矩阵拟合
   - 最佳联合拟合（两者平均）

6️⃣ 统计摘要
   - 提供所有 JSD 指标的统计描述

前置要求：
    ⚠️ 重要: 在运行此脚本之前，必须先运行：
    - process_model_results.py - 生成 results-gsize-dist/ 和 results-gtrans-mat/ 目录
    - 确保 ../data-analysis/results/CNS/ 目录下有经验数据
"""

import os
import sys

import numpy as np
import pandas as pd

# 添加 code 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from fitting import fit_gsize_dist, fit_gtrans_mat  # noqa: E402
from utils import change_width, centered_np_hist  # noqa: E402

# ----------------------------------------------------------------
# 常量配置
# ----------------------------------------------------------------
RESULTS_PATH = 'results/'
DATA_ANALYSIS_PATH = '../data-analysis/results/'
DATASET = 'CNS'
CONTEXT = 'out-of-class'

GSIZE_DIST_PATH = 'results-gsize-dist/'
GTRANS_MAT_PATH = 'results-gtrans-mat/'

MAX_K = 20


def load_parameters():
    """加载参数 DataFrame"""
    pars_df = pd.read_csv(RESULTS_PATH + 'parameters.csv')
    pars_df.set_index('pars_id', inplace=True)
    return pars_df


def load_empirical_data():
    """加载经验数据"""
    in_path = os.path.join(DATA_ANALYSIS_PATH, DATASET, '')

    # 群组大小分布
    pk_fname = f'Pk_{CONTEXT}.csv'
    pk_emp = pd.read_csv(in_path + pk_fname)
    print(f'已加载经验群组大小分布：{len(pk_emp)} 个群组大小')

    # 群组转移矩阵
    t_fname = f'T_{CONTEXT}.csv'
    t_emp = pd.read_csv(in_path + t_fname)
    print(f'已加载经验群组转移矩阵：{len(t_emp)} 个转移条目')

    return pk_emp, t_emp


def compute_group_size_jsd(pars_df, pk_emp):
    """计算群组大小分布的 JSD"""
    print('\n' + '=' * 60)
    print('第二部分：计算群组大小分布的 JSD')
    print('=' * 60)

    print('\n计算非对数尺度的 JSD...')
    fit_df = fit_gsize_dist(pk_emp, pars_df, GSIZE_DIST_PATH, log=False)
    print('JSD 计算完成!')
    print(fit_df[['N', 't_max', 'beta', 'alpha', 'n0', 'L', 'epsilon', 'JSD_gsize']].head())

    print('\n计算对数尺度的 JSD...')
    fit_df = fit_gsize_dist(pk_emp, fit_df, GSIZE_DIST_PATH, log=True)
    print('对数 JSD 计算完成!')
    print(fit_df[['N', 't_max', 'beta', 'alpha', 'n0', 'L', 'epsilon', 'JSD_gsize', 'JSD_gsize_log']].head())

    return fit_df


def compute_transition_matrix_jsd(fit_df, t_emp):
    """计算群组转移矩阵的 JSD"""
    print('\n' + '=' * 60)
    print('第三部分：计算群组转移矩阵的 JSD')
    print('=' * 60)

    print('\n计算群组转移矩阵的 JSD (未加权)...')
    fit_df = fit_gtrans_mat(t_emp, fit_df, GTRANS_MAT_PATH, weighted=False, k_cut='max')
    print('转移矩阵 JSD 计算完成!')
    print(fit_df[
              ['N', 't_max', 'beta', 'alpha', 'n0', 'L', 'epsilon', 'JSD_gsize',
               'JSD_gsize_log', 'JSD_T']
          ].head())

    return fit_df


def save_fitting_results(fit_df):
    """保存拟合结果"""
    print('\n' + '=' * 60)
    print('第四部分：保存拟合结果')
    print('=' * 60)

    out_path = RESULTS_PATH
    fname = 'parameters_fit.csv'
    fit_df.to_csv(out_path + fname, header=True, index=False)
    print(f'\n拟合结果已保存到：{out_path}{fname}')

    return fit_df


def perform_joint_optimization(fit_df):
    """执行联合最小化分析"""
    print('\n' + '=' * 60)
    print('第五部分：联合最小化分析')
    print('=' * 60)

    # 找到最佳参数组合
    print('\n--- 最佳参数组合 ---')

    # 按群组大小分布 JSD (对数尺度)
    best_gsize_idx = fit_df['JSD_gsize_log'].idxmin()
    best_gsize_params = fit_df.loc[best_gsize_idx]
    print(f'\n最佳群组大小分布拟合 (JSD_log={best_gsize_params["JSD_gsize_log"]:.4f}):')
    print(f'  alpha={best_gsize_params["alpha"]}, n0={best_gsize_params["n0"]}, epsilon={best_gsize_params["epsilon"]}')

    # 按转移矩阵 JSD
    best_trans_idx = fit_df['JSD_T'].idxmin()
    best_trans_params = fit_df.loc[best_trans_idx]
    print(f'\n最佳转移矩阵拟合 (JSD_T={best_trans_params["JSD_T"]:.4f}):')
    print(f'  alpha={best_trans_params["alpha"]}, n0={best_trans_params["n0"]}, epsilon={best_trans_params["epsilon"]}')

    # 联合优化（简单平均）
    fit_df['JSD_combined'] = (fit_df['JSD_gsize_log'] + fit_df['JSD_T']) / 2.0
    best_combined_idx = fit_df['JSD_combined'].idxmin()
    best_combined_params = fit_df.loc[best_combined_idx]
    print(f'\n最佳联合拟合 (JSD_combined={best_combined_params["JSD_combined"]:.4f}):')
    print(f'  alpha={best_combined_params["alpha"]}, n0={best_combined_params["n0"]}, epsilon={best_combined_params["epsilon"]}')
    print(f'  JSD_gsize_log={best_combined_params["JSD_gsize_log"]:.4f}, JSD_T={best_combined_params["JSD_T"]:.4f}')

    return fit_df


def generate_statistics_summary(fit_df):
    """生成统计摘要"""
    print('\n' + '=' * 60)
    print('第六部分：统计摘要')
    print('=' * 60)

    print('\nJSD 统计信息:')
    print(fit_df[['JSD_gsize', 'JSD_gsize_log', 'JSD_T', 'JSD_combined']].describe())

    # 保存更新后的 DataFrame（包含联合 JSD）
    out_path = RESULTS_PATH
    fname = 'parameters_fit_with_combined.csv'
    fit_df.to_csv(out_path + fname, header=True, index=False)
    print(f'\n包含联合 JSD 的结果已保存到：{out_path}{fname}')


def main():
    """主函数"""
    print('=' * 60)
    print('模型拟合 - 计算与经验数据的距离')
    print('=' * 60)

    # 第一部分：读取参数和经验数据
    pars_df = load_parameters()
    print('\n读取的参数 DataFrame:')
    print(pars_df.head())
    print(f'总共有 {len(pars_df)} 个参数组合')

    pk_emp, t_emp = load_empirical_data()

    # 第二部分：计算群组大小分布的 JSD
    fit_df = compute_group_size_jsd(pars_df, pk_emp)

    # 第三部分：计算群组转移矩阵的 JSD
    fit_df = compute_transition_matrix_jsd(fit_df, t_emp)

    # 第四部分：保存拟合结果
    fit_df = save_fitting_results(fit_df)

    # 第五部分：联合最小化分析
    fit_df = perform_joint_optimization(fit_df)

    # 第六部分：统计摘要
    generate_statistics_summary(fit_df)

    print('\n' + '=' * 60)
    print('模型拟合完成！')
    print('=' * 60)

    print('\n后续步骤:')
    print('  1. 查看 results/parameters_fit.csv 获取所有参数组合的 JSD 值')
    print('  2. 使用最佳参数组合重新运行模型进行验证')
    print('  3. 可以运行 "4_Processing_model_results_post_fit.ipynb" 进行后拟合分析')


if __name__ == '__main__':
    main()