"""
模型后拟合结果处理脚本

主要功能：
1️⃣ 自动识别最佳拟合参数
   - 读取 fit_model.py 生成的拟合结果
   - 自动找到联合 JSD 最小的最佳参数组合
   - 支持手动指定要分析的 ID

2️⃣ 群组持续时间分布 (Group Durations)
   - 计算每个群组大小的持续时间分布
   - 保存为 pickle 文件
   - 提供统计摘要（平均值、最大值、最小值）

3️⃣ 群组时间和分解/聚合矩阵
   - 群组时间: 记录每个群组的成员、创建时间和销毁时间
   - 分解矩阵 (D): 描述群组如何分裂成更小的群组
   - 聚合矩阵 (A): 描述小群组如何聚合成大群组
   - 分别计算：
     * 基于最大子群的矩阵
     * 基于所有子群的完整矩阵

4️⃣ 社会记忆 (Social Memory)
   - 测量节点在选择群组时的社会偏好
   - 比较节点选择的群组与随机群组的已知节点密度
   - 生成包含记忆分数的 DataFrame

5️⃣ 事件间隔时间 (Inter-event Times)
   - 计算连续群组事件之间的时间间隔
   - 用于分析时间动力学特征
   - 提供详细的统计信息

输出文件结构:
model/
├── results-gduration/
│   ├── gdurations_pars_id{ID}.csv      # 群组持续时间分布
│   └── interevent_times_id{ID}.p       # 事件间隔时间
├── results-gdisagg-mat/
│   ├── group_times_id{ID}.p            # 群组时间记录
│   ├── D_id{ID}.csv                    # 分解矩阵
│   ├── A_id{ID}.csv                    # 聚合矩阵
│   ├── Dfull_id{ID}.csv                # 完整分解矩阵
│   └── Afull_id{ID}.csv                # 完整聚合矩阵
└── results-social-memory/
    └── social_memory_id{ID}.csv.gz     # 社会记忆数据 (压缩)

前置要求
⚠️ 重要: 在运行此脚本之前，必须先运行:
fit_model.py - 生成拟合结果，确定最佳参数
确保 results/ 目录下有模型运行结果
"""

import os
import sys

import pandas as pd
import pickle

# 添加 code 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from code.model import read_edgelists_from_df  # noqa: E402
from code.model_analysis import (
    get_group_durations,
    get_group_times,
    get_dis_agg_matrices,
    get_full_dis_agg_matrices,
    dis_agg_matrix_to_df
)  # noqa: E402
from code.utils import get_cumulative_Gs_from_Hs, get_groups_dict_from_Hs  # noqa: E402
from code.data_analysis import measure_social_memory, get_interevent_times  # noqa: E402

# ----------------------------------------------------------------
# 常量配置
# ----------------------------------------------------------------
RESULTS_PATH = 'results/'
GSIZE_DURATION_PATH = 'results-gduration/'
GDISAGG_MAT_PATH = 'results-gdisagg-mat/'
SOCIAL_MEMORY_PATH = 'results-social-memory/'

MAX_K = 21

def load_parameters_and_fit_results():
    """加载参数和拟合结果"""
    # 读取参数 DataFrame
    pars_df = pd.read_csv(RESULTS_PATH + 'parameters.csv')
    pars_df.set_index('pars_id', inplace=True)
    print('\n读取的参数 DataFrame:')
    print(pars_df.head())
    print(f'总共有 {len(pars_df)} 个参数组合')

    # 读取拟合结果
    fit_df = pd.read_csv(RESULTS_PATH + 'parameters_fit_with_combined.csv')
    fit_df.set_index('pars_id', inplace=True)
    print('\n拟合结果 DataFrame:')
    print(fit_df[['alpha', 'n0', 'epsilon', 'JSD_gsize_log', 'JSD_T', 'JSD_combined']].head())

    # 找到最佳联合拟合的 ID
    best_fit_id = fit_df['JSD_combined'].idxmin()
    best_params = fit_df.loc[best_fit_id]

    print(f'\n--- 最佳拟合参数 ---')
    print(f'最佳 ID: {best_fit_id}')
    print(f'参数：alpha={best_params["alpha"]}, n0={best_params["n0"]}, epsilon={best_params["epsilon"]}')
    print(f'JSD_gsize_log={best_params["JSD_gsize_log"]:.4f}, JSD_T={best_params["JSD_T"]:.4f}')

    return pars_df, fit_df, best_fit_id

def ensure_dir(path):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'创建输出目录：{path}')

def compute_group_durations(pars_df, selected_ids):
    """计算并保存群组持续时间分布"""
    print('\n' + '=' * 60)
    print('第二部分：计算群组持续时间分布')
    print('=' * 60)

    ensure_dir(GSIZE_DURATION_PATH)

    for run_id in selected_ids:
        print(f'\n处理 ID: {run_id}')
        try:
            Hs = read_edgelists_from_df(run_id, pars_df, RESULTS_PATH)
            print(f'  读取了 {len(Hs)} 个时间步的超图数据')

            durations = get_group_durations(Hs)
            print(f'  计算出 {len(durations)} 个不同群组大小的持续时间分布')

            # 保存
            out_fname = f'gdurations_pars_id{run_id}.csv'
            pickle.dump(durations, open(GSIZE_DURATION_PATH + out_fname, 'wb'))
            print(f'  已保存到：{GSIZE_DURATION_PATH}{out_fname}')

            # 统计信息
            for k, dur_list in list(durations.items())[:5]:
                avg_dur = sum(dur_list) / len(dur_list) if dur_list else 0
                print(f'    群组大小 k={k}: 平均持续时间={avg_dur:.2f}, '
                      f'最大={max(dur_list) if dur_list else 0}, 最小={min(dur_list) if dur_list else 0}')
            if len(durations) > 5:
                print(f'    ... (还有 {len(durations) - 5} 个群组大小)')

        except FileNotFoundError as e:
            print(f'  错误：找不到文件 - {e}')
        except Exception as e:
            print(f'  错误：处理 ID {run_id} 时发生异常 - {e}')

    print('\n群组持续时间分布处理完成！')

def compute_group_times_and_matrices(pars_df, selected_ids):
    """计算并保存群组时间和分解/聚合矩阵"""
    print('\n' + '=' * 60)
    print('第三部分：计算群组时间和分解/聚合矩阵')
    print('=' * 60)

    ensure_dir(GDISAGG_MAT_PATH)

    for run_id in selected_ids:
        print(f'\n处理 ID: {run_id}')
        try:
            Hs = read_edgelists_from_df(run_id, pars_df, RESULTS_PATH)
            print(f'  读取了 {len(Hs)} 个时间步的超图数据')

            # 计算群组时间
            print('  计算群组时间...')
            groups_and_times = get_group_times(Hs)
            print(f'  识别出 {len(groups_and_times)} 个群组事件')

            # 保存群组时间
            out_fname = f'group_times_id{run_id}.p'
            pickle.dump(groups_and_times, open(GDISAGG_MAT_PATH + out_fname, 'wb'))
            print(f'  群组时间已保存到：{GDISAGG_MAT_PATH}{out_fname}')

            # 计算分解/聚合矩阵（最大子群）
            print('  计算分解/聚合矩阵（最大子群）...')
            D, A = get_dis_agg_matrices(Hs, groups_and_times, max_k=MAX_K, normed=True)

            df_D = dis_agg_matrix_to_df(D)
            df_A = dis_agg_matrix_to_df(A)

            df_D.to_csv(GDISAGG_MAT_PATH + f'D_id{run_id}.csv', header=True, index=False)
            df_A.to_csv(GDISAGG_MAT_PATH + f'A_id{run_id}.csv', header=True, index=False)
            print(f'  分解矩阵 (D) 和聚合矩阵 (A) 已保存')

            # 计算完整分解/聚合矩阵（所有子群）
            print('  计算完整分解/聚合矩阵（所有子群）...')
            D_full, A_full = get_full_dis_agg_matrices(Hs, groups_and_times, max_k=MAX_K, normed=True)

            df_D_full = dis_agg_matrix_to_df(D_full)
            df_A_full = dis_agg_matrix_to_df(A_full)

            df_D_full.to_csv(GDISAGG_MAT_PATH + f'Dfull_id{run_id}.csv', header=True, index=False)
            df_A_full.to_csv(GDISAGG_MAT_PATH + f'Afull_id{run_id}.csv', header=True, index=False)
            print(f'  完整分解矩阵 (Dfull) 和完整聚合矩阵 (Afull) 已保存')

            print(f'  ID {run_id} 的所有矩阵计算完成！')

        except FileNotFoundError as e:
            print(f'  错误：找不到文件 - {e}')
        except Exception as e:
            print(f'  错误：处理 ID {run_id} 时发生异常 - {e}')

    print('\n分解/聚合矩阵处理完成！')

def compute_social_memory(pars_df, selected_ids):
    """计算并保存社会记忆"""
    print('\n' + '=' * 60)
    print('第四部分：计算社会记忆')
    print('=' * 60)

    ensure_dir(SOCIAL_MEMORY_PATH)

    for run_id in selected_ids:
        print(f'\n处理 ID: {run_id}')
        try:
            # 读取超图数据
            Hs = read_edgelists_from_df(run_id, pars_df, RESULTS_PATH)
            print(f'  读取了 {len(Hs)} 个时间步的超图数据')

            # 加载群组时间记录
            group_times_file = f'group_times_id{run_id}.p'
            groups_and_times = pickle.load(open(GDISAGG_MAT_PATH + group_times_file, 'rb'))
            print(f'  加载了 {len(groups_and_times)} 个群组时间记录')

            # 计算累积网络
            Hs_dict = {k: v for k, v in enumerate(Hs)}
            Gs = get_cumulative_Gs_from_Hs(Hs_dict)
            print(f'  计算出 {len(Gs)} 个时间步的累积网络')

            # 构建群组时间字典
            groups_at_t_dict = get_groups_dict_from_Hs(Hs_dict)
            print('  构建群组时间字典完成')

            # 测量社会记忆
            print('  测量社会记忆...')
            memory_df = measure_social_memory(Hs_dict, groups_at_t_dict, Gs, groups_and_times)
            print(f'  社会记忆 DataFrame: {len(memory_df)} 行，{len(memory_df.columns)} 列')

            # 保存
            out_fname = f'social_memory_id{run_id}.csv.gz'
            memory_df.to_csv(SOCIAL_MEMORY_PATH + out_fname, header=True, index=False, compression='gzip')
            print(f'  社会记忆数据已保存到：{SOCIAL_MEMORY_PATH}{out_fname}')

            # 统计信息
            if 'memory_score' in memory_df.columns:
                stats = memory_df['memory_score'].describe()
                print(f'  社会记忆分数统计:')
                print(f'    平均值：{stats["mean"]:.4f}')
                print(f'    中位数：{stats["50%"]:.4f}')
                print(f'    标准差：{stats["std"]:.4f}')

        except FileNotFoundError as e:
            print(f'  错误：找不到文件 - {e}')
        except Exception as e:
            print(f'  错误：处理 ID {run_id} 时发生异常 - {e}')

    print('\n社会记忆计算完成！')

def compute_interevent_times(selected_ids):
    """计算并保存事件间隔时间"""
    print('\n' + '=' * 60)
    print('第五部分：计算事件间隔时间')
    print('=' * 60)

    ensure_dir(GSIZE_DURATION_PATH)

    for run_id in selected_ids:
        print(f'\n处理 ID: {run_id}')
        try:
            # 加载群组时间记录
            group_times_file = f'group_times_id{run_id}.p'
            groups_and_times = pickle.load(open(GDISAGG_MAT_PATH + group_times_file, 'rb'))
            print(f'  加载了 {len(groups_and_times)} 个群组时间记录')

            # 计算事件间隔时间
            interevent_times = get_interevent_times(groups_and_times)
            print(f'  计算出 {len(interevent_times)} 个事件间隔时间')

            # 保存
            out_fname = f'interevent_times_id{run_id}.p'
            pickle.dump(interevent_times, open(GSIZE_DURATION_PATH + out_fname, 'wb'))
            print(f'  事件间隔时间已保存到：{GSIZE_DURATION_PATH}{out_fname}')

            # 统计信息
            if interevent_times:
                sorted_times = sorted(interevent_times)
                median_idx = len(sorted_times) // 2
                print(f'  事件间隔时间统计:')
                print(f'    平均值：{sum(interevent_times) / len(interevent_times):.2f}')
                print(f'    中位数：{sorted_times[median_idx]}')
                print(f'    最大值：{max(interevent_times)}')
                print(f'    最小值：{min(interevent_times)}')

        except FileNotFoundError as e:
            print(f'  错误：找不到文件 - {e}')
        except Exception as e:
            print(f'  错误：处理 ID {run_id} 时发生异常 - {e}')

    print('\n事件间隔时间计算完成！')

def print_summary(selected_ids):
    """打印最终总结"""
    print('\n' + '=' * 60)
    print('所有后拟合处理完成！')
    print('=' * 60)

    run_id = selected_ids[0]
    print(f'\n生成的输出文件:')
    print(f'  群组持续时间：{GSIZE_DURATION_PATH}gdurations_pars_id{run_id}.csv')
    print(f'  群组时间：{GDISAGG_MAT_PATH}group_times_id{run_id}.p')
    print('  分解/聚合矩阵:')
    print(f'    - {GDISAGG_MAT_PATH}D_id{run_id}.csv')
    print(f'    - {GDISAGG_MAT_PATH}A_id{run_id}.csv')
    print(f'    - {GDISAGG_MAT_PATH}Dfull_id{run_id}.csv')
    print(f'    - {GDISAGG_MAT_PATH}Afull_id{run_id}.csv')
    print(f'  社会记忆：{SOCIAL_MEMORY_PATH}social_memory_id{run_id}.csv.gz')
    print(f'  事件间隔时间：{GSIZE_DURATION_PATH}interevent_times_id{run_id}.p')

    print('\n后续步骤:')
    print('  1. 分析群组持续时间分布的特征')
    print('  2. 研究分解/聚合矩阵的模式')
    print('  3. 探索社会记忆与经验数据的对比')
    print('  4. 分析事件间隔时间的分布特性')
    print('  5. 可以将这些结果与经验数据进行比较验证')

def main():
    """主函数"""
    print('=' * 60)
    print('模型后拟合结果处理')
    print('=' * 60)

    # 第一部分：读取参数和选择最佳拟合 ID
    pars_df, fit_df, best_fit_id = load_parameters_and_fit_results()

    # 选择要处理的 ID（使用最佳拟合或手动指定）
    selected_ids = [int(best_fit_id)]  # 使用最佳拟合
    # selected_ids = [169]  # 或者手动指定

    print(f'\n将处理以下 ID: {selected_ids}')

    # 第二部分：计算群组持续时间分布
    compute_group_durations(pars_df, selected_ids)

    # 第三部分：计算群组时间和分解/聚合矩阵
    compute_group_times_and_matrices(pars_df, selected_ids)

    # 第四部分：计算社会记忆
    compute_social_memory(pars_df, selected_ids)

    # 第五部分：计算事件间隔时间
    compute_interevent_times(selected_ids)

    # 总结
    print_summary(selected_ids)

if __name__ == '__main__':
    main()