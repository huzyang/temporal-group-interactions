#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNS 数据集的群体交互分析脚本

本脚本用于从 CNS 数据集中提取和分析群体交互模式，包括：
1. 提取不同情境下的群体交互数据
2. 计算群体规模分布
3. 计算节点转移矩阵
4. 计算群体持续时间分布
5. 计算群体解聚和聚合矩阵
6. 分析群体相似性、多重成员关系、社交记忆等

作者：Converted from Jupyter notebook
日期：2026-03-14
"""

import sys
import os
import pickle
from collections import Counter, OrderedDict
import numpy as np

# 添加代码目录到路径
sys.path.append("../code/")
from code.data_analysis import (
    groups_at_time_t,
    group_size_dist,
    get_transition_matrix,
    transition_matrix_to_df,
    get_group_durations,
    get_group_times,
    get_dis_agg_matrices,
    get_full_dis_agg_matrices,
    dis_agg_matrix_to_df,
    get_group_similarity,
    measure_social_memory,
    get_interevent_times,
    get_node_trajectory,
    get_probs_leaving_group,
)
from code.utils import get_Hs_from_groups_dict, get_cumulative_Gs_from_Hs, reduce_number_of_points

datasets = ["CNS"]
contexts = {
    "CNS": ['in-class', 'out-of-class', 'weekend'],
}

def create_output_directories():
    """创建输出目录"""
    print("=" * 60)
    print("步骤 0: 创建输出目录")
    print("=" * 60)

    directories = ["results/CNS"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  ✓ 创建目录：{directory}")
        else:
            print(f"  ✓ 目录已存在：{directory}")


def extract_cns_groups():
    """
    从 CNS 数据集中提取群体交互

    CNS 数据集包含蓝牙接触数据，按不同情境（课堂内、课堂外、周末）分组
    """
    print("\n" + "=" * 60)
    print("步骤 1.1: 提取 CNS 群体交互")
    print("=" * 60)

    dataset = "CNS"
    IN_PATH = f"../data-processed/{dataset}/"
    OUT_PATH = f"../data-analysis/results/{dataset}/"

    # 确保输出目录存在
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    # 读取处理后的数据
    FNAME = f"{dataset}_bluetooth_processed.csv.gz"
    print(f"读取数据：{IN_PATH + FNAME}")
    df = pd.read_csv(IN_PATH + FNAME)
    print(f"数据形状：{df.shape}")
    print(f"列名：{list(df.columns)}")

    # 定义三种情境
    contexts = ['in-class', 'out-of-class', 'weekend']

    # 根据不同情境筛选时间戳
    # 周末（周六和周日）
    weekends_timestamps = list(
        df[(df['DoW'] == 'Sunday') | (df['DoW'] == 'Saturday')]['# timestamp'].unique()
    )
    # 工作日课堂时间（8:00-17:00）
    workweek_class_timestamps = list(
        df[
            (df['DoW'] != 'Sunday') &
            (df['DoW'] != 'Saturday') &
            ((df['hour'] >= 8) & (df['hour'] <= 17))
            ]['# timestamp'].unique()
    )
    # 工作日非课堂时间
    workweek_noclass_timestamps = list(
        df[
            (df['DoW'] != 'Sunday') &
            (df['DoW'] != 'Saturday') &
            ((df['hour'] < 8) | (df['hour'] > 17))
            ]['# timestamp'].unique()
    )

    # 情境与时间戳的映射
    context_timestamps = {
        'in-class': workweek_class_timestamps,
        'out-of-class': workweek_noclass_timestamps,
        'weekend': weekends_timestamps
    }

    # 为每个情境提取群体
    for context in contexts:
        print(f"\n处理情境：{context} ({len(context_timestamps[context])} 个时间戳)")
        dfx = df[df['# timestamp'].isin(context_timestamps[context])]

        # 构建时间 - 群体字典
        groups_at_t_dict = {}
        timestamps = list(dfx['# timestamp'].unique())
        total = len(timestamps)

        for i, timestamp in enumerate(timestamps):
            groups_at_t_dict[timestamp] = groups_at_time_t(dfx, timestamp, dataset=dataset)

            # 显示进度
            if (i + 1) % 1000 == 0 or (i + 1) == total:
                print(f"  进度：{i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")

        # 保存结果
        FNAME = f"groups_at_t_{context}.p"
        with open(OUT_PATH + FNAME, "wb") as f:
            pickle.dump(groups_at_t_dict, f)
        print(f"  ✓ 保存到：{OUT_PATH + FNAME}")


def extract_dylnet_groups():
    """
    从 DyLNet 数据集中提取群体交互

    DyLNet 数据集包含学校儿童的互动数据，分为课堂内和课堂外两种情境
    """
    print("\n" + "=" * 60)
    print("步骤 1.2: 提取 DyLNet 群体交互")
    print("=" * 60)

    dataset = "DyLNet"
    IN_PATH = f"../data-processed/{dataset}/"
    OUT_PATH = f"../data-analysis/results/{dataset}/"

    # 确保输出目录存在
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    # 读取处理后的数据
    FNAME = f"{dataset}_processed.csv.gz"
    print(f"读取数据：{IN_PATH + FNAME}")
    df = pd.read_csv(IN_PATH + FNAME)
    print(f"数据形状：{df.shape}")
    print(f"列名：{list(df.columns)}")

    # 定义两种情境
    contexts = ['in-class', 'out-of-class']

    # 为每个情境提取群体
    for context in contexts:
        print(f"\n处理情境：{context}")
        dfx = df[df['context'] == context]

        # 构建时间 - 群体字典
        groups_at_t_dict = {}
        timestamps = list(dfx['timestamp'].unique())
        total = len(timestamps)

        for i, timestamp in enumerate(timestamps):
            groups_at_t_dict[timestamp] = groups_at_time_t(dfx, timestamp, dataset=dataset)

            # 显示进度
            if (i + 1) % 1000 == 0 or (i + 1) == total:
                print(f"  时间戳：{timestamp} ({i + 1}/{total})")

        # 保存结果
        FNAME = f"groups_at_t_{context}.p"
        with open(OUT_PATH + FNAME, "wb") as f:
            pickle.dump(groups_at_t_dict, f)
        print(f"  ✓ 保存到：{OUT_PATH + FNAME}")


def compute_group_size_distributions():
    """
    计算群体规模分布

    对于每个数据集和情境，计算不同规模 k 的群体出现概率 P(k)
    """
    print("\n" + "=" * 60)
    print("步骤 2.1: 计算群体规模分布")
    print("=" * 60)

    for dataset in datasets:
        print(f"\n数据集：{dataset}")
        IN_PATH = f"../data-analysis/results/{dataset}/"
        OUT_PATH = f"../data-analysis/results/{dataset}/"

        for context in contexts[dataset]:
            print(f"  情境：{context}")

            # 读取群体数据
            FNAME = f"groups_at_t_{context}.p"
            with open(IN_PATH + FNAME, "rb") as f:
                groups_at_t_dict = pickle.load(f)

            # 计算群体规模分布
            ks, Pks = group_size_dist(groups_at_t_dict)

            # 保存结果
            gsize_df = pd.DataFrame({'k': ks, 'Pk': Pks})
            FNAME = f"Pk_{context}.csv"
            gsize_df.to_csv(OUT_PATH + FNAME, header=True, index=False)
            print(f"    ✓ 保存到：{OUT_PATH + FNAME}")


def compute_transition_matrices():
    """
    计算节点转移矩阵

    转移矩阵描述了节点在不同规模的群体之间转移的概率
    """
    print("\n" + "=" * 60)
    print("步骤 2.2: 计算节点转移矩阵")
    print("=" * 60)

    for dataset in datasets:
        print(f"\n数据集：{dataset}")
        IN_PATH = f"../data-analysis/results/{dataset}/"
        OUT_PATH = f"../data-analysis/results/{dataset}/"

        for context in contexts[dataset]:
            print(f"  情境：{context}")

            # 读取群体数据
            FNAME = f"groups_at_t_{context}.p"
            with open(IN_PATH + FNAME, "rb") as f:
                groups_at_t_dict = pickle.load(f)

            # 转换为超图对象
            Hs = get_Hs_from_groups_dict(groups_at_t_dict)

            # 计算转移矩阵（最大群体规模为 20）
            T = get_transition_matrix(Hs, max_k=20, normed=True)

            # 转换为 DataFrame
            df_T = transition_matrix_to_df(T)

            # 保存结果
            OUT_FNAME = f"T_{context}.csv"
            df_T.to_csv(OUT_PATH + OUT_FNAME, header=True, index=False)
            print(f"    ✓ 保存到：{OUT_PATH + OUT_FNAME}")


def compute_group_durations():
    """
    计算群体持续时间分布

    记录每个群体从形成到解散的持续时间
    """
    print("\n" + "=" * 60)
    print("步骤 2.3: 计算群体持续时间分布")
    print("=" * 60)

    for dataset in datasets:
        print(f"\n数据集：{dataset}")
        IN_PATH = f"../data-analysis/results/{dataset}/"
        OUT_PATH = f"../data-analysis/results/{dataset}/"

        for context in contexts[dataset]:
            print(f"  情境：{context}")

            # 读取群体数据
            FNAME = f"groups_at_t_{context}.p"
            with open(IN_PATH + FNAME, "rb") as f:
                groups_at_t_dict = pickle.load(f)

            # 计算群体持续时间
            durations = get_group_durations(groups_at_t_dict)

            # 保存结果
            OUT_FNAME = f"gdurations_{context}.p"
            with open(OUT_PATH + OUT_FNAME, "wb") as f:
                pickle.dump(durations, f)
            print(f"    ✓ 保存到：{OUT_PATH + OUT_FNAME}")


def compute_group_times():
    """
    计算群体的时间信息

    对于每个群体，记录其成员以及形成和解散的时间
    """
    print("\n" + "=" * 60)
    print("步骤 2.4: 计算群体时间信息")
    print("=" * 60)

    for dataset in datasets:
        print(f"\n数据集：{dataset}")
        IN_PATH = f"../data-analysis/results/{dataset}/"
        OUT_PATH = f"../data-analysis/results/{dataset}/"

        for context in contexts[dataset]:
            print(f"  情境：{context}")

            # 读取群体数据
            FNAME = f"groups_at_t_{context}.p"
            with open(IN_PATH + FNAME, "rb") as f:
                groups_at_t_dict = pickle.load(f)

            print("    计算群体时间...")
            # 计算群体的开始和结束时间
            groups_and_times = get_group_times(groups_at_t_dict)

            # 保存结果
            OUT_FNAME = f"group_times_{context}.p"
            with open(OUT_PATH + OUT_FNAME, "wb") as f:
                pickle.dump(groups_and_times, f)
            print(f"    ✓ 保存到：{OUT_PATH + OUT_FNAME}")


def compute_disaggregation_aggregation_matrices():
    """
    计算群体解聚和聚合矩阵（方法 1：仅使用最大子群规模）

    解聚矩阵 D: 描述规模为 k 的群体解聚为较小群体的概率
    聚合矩阵 A: 描述群体聚合为较大群体的概率
    """
    print("\n" + "=" * 60)
    print("步骤 2.5.1: 计算解聚和聚合矩阵（最大子群）")
    print("=" * 60)

    for dataset in datasets:
        print(f"\n数据集：{dataset}")
        IN_PATH = f"../data-analysis/results/{dataset}/"
        OUT_PATH = f"../data-analysis/results/{dataset}/"

        for context in contexts[dataset]:
            print(f"  情境：{context}")

            # 读取群体时间信息
            FNAME = f"group_times_{context}.p"
            with open(IN_PATH + FNAME, "rb") as f:
                groups_and_times = pickle.load(f)

            # 读取群体数据
            FNAME = f"groups_at_t_{context}.p"
            with open(IN_PATH + FNAME, "rb") as f:
                groups_at_t_dict = pickle.load(f)

            # 计算解聚和聚合矩阵（最大群体规模为 21）
            D, A = get_dis_agg_matrices(
                groups_at_t_dict,
                groups_and_times,
                max_k=21,
                normed=True
            )

            # 转换为 DataFrame
            df_D = dis_agg_matrix_to_df(D)
            df_A = dis_agg_matrix_to_df(A)

            # 保存结果
            OUT_FNAME = f"D_{context}.csv"
            df_D.to_csv(OUT_PATH + OUT_FNAME, header=True, index=False)

            OUT_FNAME = f"A_{context}.csv"
            df_A.to_csv(OUT_PATH + OUT_FNAME, header=True, index=False)

            print(f"    ✓ 保存到：{OUT_PATH + OUT_FNAME}")


def compute_full_disaggregation_aggregation_matrices():
    """
    计算群体解聚和聚合矩阵（方法 2：使用所有子群规模）

    考虑所有子群的规模而不仅仅是最大的子群
    """
    print("\n" + "=" * 60)
    print("步骤 2.5.2: 计算解聚和聚合矩阵（所有子群）")
    print("=" * 60)

    for dataset in datasets:
        print(f"\n数据集：{dataset}")
        IN_PATH = f"../data-analysis/results/{dataset}/"
        OUT_PATH = f"../data-analysis/results/{dataset}/"

        for context in contexts[dataset]:
            print(f"  情境：{context}")

            # 读取群体时间信息
            FNAME = f"group_times_{context}.p"
            with open(IN_PATH + FNAME, "rb") as f:
                groups_and_times = pickle.load(f)

            # 读取群体数据
            FNAME = f"groups_at_t_{context}.p"
            with open(IN_PATH + FNAME, "rb") as f:
                groups_at_t_dict = pickle.load(f)

            # 计算完整的解聚和聚合矩阵
            D, A = get_full_dis_agg_matrices(
                groups_at_t_dict,
                groups_and_times,
                max_k=21,
                normed=True
            )

            # 转换为 DataFrame
            df_D = dis_agg_matrix_to_df(D)
            df_A = dis_agg_matrix_to_df(A)

            # 保存结果
            OUT_FNAME = f"Dfull_{context}.csv"
            df_D.to_csv(OUT_PATH + OUT_FNAME, header=True, index=False)

            OUT_FNAME = f"Afull_{context}.csv"
            df_A.to_csv(OUT_PATH + OUT_FNAME, header=True, index=False)

            print(f"    ✓ 保存到：{OUT_PATH + OUT_FNAME}")


def compute_group_similarity():
    """
    计算连续时刻的群体相似性（Jaccard 相似系数）

    用于衡量相邻时间点之间群体成员的重叠程度
    """
    print("\n" + "=" * 60)
    print("步骤 3.1: 计算群体相似性")
    print("=" * 60)

    for dataset in datasets:
        print(f"\n数据集：{dataset}")
        IN_PATH = f"../data-analysis/results/{dataset}/"
        OUT_PATH = f"../data-analysis/results/{dataset}/"

        for context in contexts[dataset]:
            print(f"  情境：{context}")

            # 读取群体数据
            FNAME = f"groups_at_t_{context}.p"
            with open(IN_PATH + FNAME, "rb") as f:
                groups_at_t_dict = pickle.load(f)

            # 转换为超图对象
            Hs = get_Hs_from_groups_dict(groups_at_t_dict)

            # 计算 Jaccard 相似性
            print("    计算 Jaccard 相似性...")
            J = get_group_similarity(Hs)

            # 保存完整结果
            OUT_FNAME = f"Jfull_{context}.p"
            with open(OUT_PATH + OUT_FNAME, "wb") as f:
                pickle.dump(J, f)
            print(f"    ✓ 保存到：{OUT_PATH + OUT_FNAME}")


def analyze_multiple_membership():
    """
    分析多重成员关系

    统计节点同时属于多个群体的情况
    """
    print("\n" + "=" * 60)
    print("步骤 3.2: 分析多重成员关系")
    print("=" * 60)

    deg_count_collection = {}

    for dataset in datasets:
        print(f"\n数据集：{dataset}")
        IN_PATH = f"../data-analysis/results/{dataset}/"

        deg_count_collection[dataset] = {}

        for context in contexts[dataset]:
            print(f"  情境：{context}")

            # 读取群体数据
            FNAME = f"groups_at_t_{context}.p"
            with open(IN_PATH + FNAME, "rb") as f:
                groups_at_t_dict = pickle.load(f)

            # 转换为超图对象
            Hs = get_Hs_from_groups_dict(groups_at_t_dict)

            # 收集所有节点在所有时刻的度
            flatten_degrees = []
            for t, H in Hs.items():
                for n, k in H.degree().items():
                    flatten_degrees.append(k)

            # 度的计数
            deg_count = Counter(flatten_degrees)
            deg_count_collection[dataset][context] = deg_count

            # 计算单一成员关系的比例（度为 1 的节点比例）
            single_member_ratio = deg_count[1] / sum(deg_count.values())
            print(f"    单一成员关系比例：{single_member_ratio:.4f}")

    return deg_count_collection


def measure_social_memory():
    """
    测量社交记忆

    分析节点在选择加入群体时，群体中已知节点的密度
    与随机选择群体的对比
    """
    print("\n" + "=" * 60)
    print("步骤 3.3: 测量社交记忆")
    print("=" * 60)

    for dataset in datasets:
        print(f"\n数据集：{dataset}")
        IN_PATH = f"../data-analysis/results/{dataset}/"
        OUT_PATH = f"../data-analysis/results/{dataset}/"

        for context in contexts[dataset]:
            print(f"  情境：{context}")

            # 读取群体数据
            FNAME = f"groups_at_t_{context}.p"
            with open(IN_PATH + FNAME, "rb") as f:
                groups_at_t_dict = pickle.load(f)

            # 转换为超图对象
            Hs = get_Hs_from_groups_dict(groups_at_t_dict)
            print("    超图对象已加载")

            # 读取群体时间信息
            FNAME = f"group_times_{context}.p"
            with open(IN_PATH + FNAME, "rb") as f:
                groups_and_times = pickle.load(f)
            print("    群体时间信息已加载")

            # 计算累积接触网络
            Gs = get_cumulative_Gs_from_Hs(Hs)
            print("    累积接触网络已计算")

            # 测量社交记忆
            memory_df = measure_social_memory(Hs, groups_at_t_dict, Gs, groups_and_times)
            print("    社交记忆数据框已计算")

            # 保存结果（使用 gzip 压缩）
            OUT_FNAME = f"social_memory_{context}.csv.gz"
            memory_df.to_csv(
                OUT_PATH + OUT_FNAME,
                header=True,
                index=False,
                compression="gzip"
            )
            print(f"    ✓ 保存到：{OUT_PATH + OUT_FNAME}")


def compute_interevent_times():
    """
    计算事件间隔时间分布

    分析群体形成事件之间的时间间隔
    """
    print("\n" + "=" * 60)
    print("步骤 3.4: 计算事件间隔时间")
    print("=" * 60)

    for dataset in datasets:
        print(f"\n数据集：{dataset}")
        IN_PATH = f"../data-analysis/results/{dataset}/"
        OUT_PATH = f"../data-analysis/results/{dataset}/"

        for context in contexts[dataset]:
            print(f"  情境：{context}")

            # 读取群体时间信息
            FNAME = f"group_times_{context}.p"
            with open(IN_PATH + FNAME, "rb") as f:
                groups_and_times = pickle.load(f)
            print("    群体时间信息已加载")

            # 计算事件间隔时间
            interevent_times = get_interevent_times(groups_and_times)
            print("    事件间隔时间已计算")

            # 保存结果
            OUT_FNAME = f"interevent_times_{context}.p"
            with open(OUT_PATH + OUT_FNAME, "wb") as f:
                pickle.dump(interevent_times, f)
            print(f"    ✓ 保存到：{OUT_PATH + OUT_FNAME}")


def compute_trajectories():
    """
    计算节点在群体规模间的轨迹

    追踪每个节点随时间在不同规模群体中的移动路径
    """
    print("\n" + "=" * 60)
    print("步骤 3.5: 计算节点轨迹")
    print("=" * 60)

    for dataset in datasets:
        print(f"\n数据集：{dataset}")
        IN_PATH = f"../data-analysis/results/{dataset}/"
        OUT_PATH = f"../data-analysis/results/{dataset}/"

        for context in contexts[dataset]:
            print(f"  情境：{context}")

            # 读取群体数据
            FNAME = f"groups_at_t_{context}.p"
            with open(IN_PATH + FNAME, "rb") as f:
                groups_at_t_dict = pickle.load(f)

            # 转换为超图对象
            Hs = get_Hs_from_groups_dict(groups_at_t_dict)
            print("    超图对象已加载")

            # 获取节点轨迹
            Traj, index_to_node = get_node_trajectory(Hs)
            print("    轨迹矩阵已计算")

            # 保存轨迹矩阵
            OUT_FNAME = f"trajectories_matrix_{context}.p"
            with open(OUT_PATH + OUT_FNAME, "wb") as f:
                pickle.dump(Traj, f)

            # 保存索引到节点的映射
            OUT_FNAME = f"trajectories_matrix_i2n_{context}.p"
            with open(OUT_PATH + OUT_FNAME, "wb") as f:
                pickle.dump(index_to_node, f)

            print(f"    ✓ 保存到：{OUT_PATH + OUT_FNAME}")


def compute_leaving_probabilities():
    """
    计算离开群体的概率

    计算节点在停留τ时间后离开规模为 k 的群体的概率
    这些概率将用于拟合模型中的 Logistic 函数
    """
    print("\n" + "=" * 60)
    print("步骤 3.6: 计算离开群体的概率")
    print("=" * 60)

    # 定义时间范围和群体规模范围
    taus = np.arange(1, 1000)
    gsizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for dataset in datasets:
        print(f"\n数据集：{dataset}")

        for context in contexts[dataset]:
            print(f"  情境：{context}")
            IN_PATH = f"../data-analysis/results/{dataset}/"
            OUT_PATH = f"../data-analysis/results/{dataset}/"

            # 读取群体持续时间
            IN_FNAME = f"gdurations_{context}.p"
            with open(IN_PATH + IN_FNAME, "rb") as f:
                durations = pickle.load(f)

            # 计算离开概率
            print("    计算概率...")
            prob_by_size = get_probs_leaving_group(durations, gsizes, taus)

            # 保存结果
            OUT_FNAME = f"Prob_leaving_group_sizek_after_tau_{context}.p"
            with open(OUT_PATH + OUT_FNAME, "wb") as f:
                pickle.dump(prob_by_size, f)
            print(f"    ✓ 保存到：{OUT_PATH + OUT_FNAME}")


def aggregate_leaving_probabilities():
    """
    聚合离开群体的概率（用于绘图）

    将不同群体规模的数据合并，并进行分箱处理以减少数据点数量
    结果将用于论文正文图 4 的绘制
    """
    print("\n" + "=" * 60)
    print("步骤 3.7: 聚合离开概率（用于绘图）")
    print("=" * 60)

    # datasets = ['CNS', 'DyLNet']
    context = 'out-of-class'

    for dataset in datasets:
        print(f"\n数据集：{dataset}")
        IN_PATH = f"results/{dataset}/"
        OUT_PATH = f"results/{dataset}/"

        # 读取概率数据
        IN_FNAME = f"Prob_leaving_group_sizek_after_tau_{context}.p"
        with open(IN_PATH + IN_FNAME, "rb") as f:
            prob = pickle.load(f)

        # 选择要聚合的群体规模
        ks = [1, 2, 3, 4]

        # 合并所有规模的数据
        x_data = list(np.arange(1, 1000)) * len(ks)
        y_data = []

        for k in ks:
            y_temp = prob[k]
            y_data = y_data + y_temp

        # 对数空间分箱
        xx_data, yy_data = reduce_number_of_points(
            x_data,
            y_data,
            bins=np.logspace(0, 3, 30)
        )

        # 将 0 值转换为 NaN 以避免垂直线
        yy_data[yy_data == 0] = np.nan

        # 移除 NaN 值以便拟合
        valid = ~(np.isnan(xx_data) | np.isnan(yy_data))

        # 保存结果
        OUT_FNAME = f"A_Binned_group_change_prob_{context}.p"
        with open(OUT_PATH + OUT_FNAME, "wb") as f:
            pickle.dump((xx_data[valid], yy_data[valid]), f)
        print(f"  ✓ 保存到：{OUT_PATH + OUT_FNAME}")


def main():
    """
    主函数：按顺序执行所有分析步骤
    """
    print("\n" + "#" * 60)
    print("# CNS 数据集分析流程")
    print("#" * 60)

    try:
        # 第 0 步：创建输出目录
        create_output_directories()

        # 第 1 步：提取群体交互
        extract_cns_groups()
        # extract_dylnet_groups()

        # 第 2 步：主要分析
        compute_group_size_distributions()
        compute_transition_matrices()
        compute_group_durations()
        compute_group_times()
        compute_disaggregation_aggregation_matrices()
        compute_full_disaggregation_aggregation_matrices()

        # 第 3 步：补充材料分析
        compute_group_similarity()
        analyze_multiple_membership()
        measure_social_memory()
        compute_interevent_times()
        compute_trajectories()
        compute_leaving_probabilities()
        aggregate_leaving_probabilities()

        print("\n" + "=" * 60)
        print("✓ 所有分析完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 错误：{e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # 需要 pandas，在这里导入以避免影响其他模块
    import pandas as pd

    main()

