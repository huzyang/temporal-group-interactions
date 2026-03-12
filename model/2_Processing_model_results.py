"""
处理模型结果

功能说明：

1️⃣ 群组大小分布 (Group Size Distributions)
   - 输入: results/run_pars_id{ID}/yel_t*.csv 文件
   - 处理: 计算每个时间步的群组大小及其概率分布
   - 输出: results-gsize-dist/Pk_pars_id{ID}.csv
   - 包含两列：k (群组大小) 和 Pk (概率)

2️⃣ 节点转移矩阵 (Node Transition Matrices)
   - 输入: results/run_pars_id{ID}/yel_t*.csv 文件
   - 处理: 计算节点在不同群组大小之间的转移概率
   - 输出: results-gtrans-mat/T_pars_id{ID}.csv
   - 包含三列：k(t) (当前群组大小), k(t+1) (下一时刻群组大小), Prob. (转移概率)
   - 矩阵大小为 20×20 (max_k=20)

前置要求：
    ⚠️ 重要: 在运行此脚本之前，必须先运行：
    - 1_Running_the_model.ipynb 或
    - run_model_batch.py
    确保 results/ 目录下已经有模型运行的结果文件。
"""

import os
import sys

import pandas as pd

# 添加 code 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from code.model import read_edgelists_from_df  # noqa: E402
from code.model_analysis import group_size_dist, get_transition_matrix, transition_matrix_to_df  # noqa: E402

# ----------------------------------------------------------------
# 常量配置
# ----------------------------------------------------------------
IN_PATH = 'results/'
OUT_PATH_GSIZE = 'results-gsize-dist/'
OUT_PATH_TRANS = 'results-gtrans-mat/'
MAX_K = 20

def load_parameters(path):
    """读取并返回参数 DataFrame"""
    pars_df = pd.read_csv(path + 'parameters.csv')
    pars_df.set_index('pars_id', inplace=True)
    return pars_df

def ensure_dir(path):
    """若目录不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'创建输出目录：{path}')

def compute_group_size_distributions(pars_df):
    """
    第一部分：计算并保存群组大小分布

    Parameters
    ----------
    pars_df : pd.DataFrame
        参数 DataFrame
    """
    print('\n' + '=' * 60)
    print('第一部分：计算群组大小分布')
    print('=' * 60)

    ensure_dir(OUT_PATH_GSIZE)

    total = len(pars_df.index)
    success = 0

    for idx, run_id in enumerate(pars_df.index):
        print(f'\n处理 ID: {run_id} ({idx + 1}/{total})')
        try:
            Hs = read_edgelists_from_df(run_id, pars_df, IN_PATH)
            print(f'  读取了 {len(Hs)} 个时间步的超图数据')

            ks, Pks = group_size_dist(Hs)
            print(f'  计算出 {len(ks)} 个不同的群组大小')

            out_fname = f'Pk_pars_id{run_id}.csv'
            pd.DataFrame({'k': ks, 'Pk': Pks}).to_csv(
                OUT_PATH_GSIZE + out_fname, header=True, index=False
            )
            print(f'  已保存到：{OUT_PATH_GSIZE}{out_fname}')
            success += 1

        except FileNotFoundError as e:
            print(f'  错误：找不到文件 - {e}')
            print("  提示：请确保已经运行了 '1_Running_the_model.ipynb' 或 'run_model_batch.py'")
        except Exception as e:
            print(f'  错误：处理 ID {run_id} 时发生异常 - {e}')

    print(f'\n群组大小分布处理完成！')
    print(f'成功处理：{success}/{total}')
    print(f'结果保存在：{OUT_PATH_GSIZE}')

    return success

def compute_transition_matrices(pars_df):
    """
    第二部分：计算并保存节点转移矩阵

    Parameters
    ----------
    pars_df : pd.DataFrame
        参数 DataFrame
    """
    print('\n' + '=' * 60)
    print('第二部分：计算节点转移矩阵')
    print('=' * 60)

    ensure_dir(OUT_PATH_TRANS)

    total = len(pars_df.index)
    success = 0

    for idx, run_id in enumerate(pars_df.index):
        print(f'\n处理 ID: {run_id} ({idx + 1}/{total})')
        try:
            Hs = read_edgelists_from_df(run_id, pars_df, IN_PATH)
            print(f'  读取了 {len(Hs)} 个时间步的超图数据')

            T = get_transition_matrix(Hs, max_k=MAX_K, normed=True)
            print(f'  计算出 {MAX_K}x{MAX_K} 的转移矩阵')

            out_fname = f'T_pars_id{run_id}.csv'
            transition_matrix_to_df(T).to_csv(
                OUT_PATH_TRANS + out_fname, header=True, index=False
            )
            print(f'  已保存到：{OUT_PATH_TRANS}{out_fname}')
            success += 1

        except FileNotFoundError as e:
            print(f'  错误：找不到文件 - {e}')
            print("  提示：请确保已经运行了 '1_Running_the_model.ipynb' 或 'run_model_batch.py'")
        except Exception as e:
            print(f'  错误：处理 ID {run_id} 时发生异常 - {e}')

    print(f'\n节点转移矩阵处理完成！')
    print(f'成功处理：{success}/{total}')
    print(f'结果保存在：{OUT_PATH_TRANS}')

    return success

def print_summary():
    """打印最终统计摘要"""
    print('\n' + '=' * 60)
    print('所有处理完成！')
    print('=' * 60)

    gsize_files = [f for f in os.listdir(OUT_PATH_GSIZE) if f.endswith('.csv')]
    gtrans_files = [f for f in os.listdir(OUT_PATH_TRANS) if f.endswith('.csv')]

    print(f'\n生成的文件统计:')
    print(f'  群组大小分布文件：{len(gsize_files)} 个')
    print(f'  节点转移矩阵文件：{len(gtrans_files)} 个')

    if gsize_files and gtrans_files:
        print('\n后续步骤:')
        print("  可以运行 '3_Fitting_model.ipynb' 或相应的.py 脚本进行模型拟合")
        print("  或者运行 '4_Processing_model_results_post_fit.ipynb' 处理拟合后的结果")
    else:
        print('\n警告：部分或全部文件未生成，请检查上方的错误信息')

def main():
    """主函数"""
    print('=' * 60)
    print('处理模型结果')
    print('=' * 60)

    # 读取参数
    pars_df = load_parameters(IN_PATH)
    print('\n读取的参数 DataFrame:')
    print(pars_df.head())
    print(f'\n总共有 {len(pars_df)} 个参数组合需要处理')

    # 计算群组大小分布
    compute_group_size_distributions(pars_df)

    # 计算节点转移矩阵
    compute_transition_matrices(pars_df)

    # 打印摘要
    print_summary()

if __name__ == '__main__':
    main()