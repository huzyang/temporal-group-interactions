#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
会议数据分析脚本 (Conferences Data Analysis)

功能说明:
    该脚本用于分析会议数据中的群体交互模式，包括：
    - 群体交互提取和统计
    - 群体规模分布分析
    - 转移矩阵计算
    - 群体持续时间分布
    - 群体聚合/解聚动力学
    - 社会记忆测量
    - 节点轨迹分析
    - 离开群体概率估计

输入:
    - ../data-processed/Confs/conf{16,17,18,19}_processed.csv.gz

输出:
    - results/Confs/目录下生成各种分析结果文件
"""

import sys
import os
import pickle
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd

# 添加代码路径
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
    get_probs_leaving_group
)
from code.utils import (
    get_Hs_from_groups_dict,
    get_cumulative_Gs_from_Hs,
    reduce_number_of_points
)


def setup_directories():
    """
    创建必要的输出目录

    Returns:
        str: 输出路径
    """
    directories = ["results/Confs"]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录：{directory}")

    return "../data-analysis/results/Confs/"


def extract_group_interactions(contexts, in_path, out_path):
    """
    从处理后的数据中提取每个时间步的群体结构

    参数:
        contexts (list): 会议上下文列表
        in_path (str): 输入路径
        out_path (str): 输出路径
    """
    print("\n=== 提取群体交互 ===")

    for context in contexts:
        print(f"\n处理 {context}...")

        # 读取数据
        in_fname = f"{context}_processed.csv.gz"
        df = pd.read_csv(in_path + in_fname)
        print(f"  读取数据：{len(df)} 条交互记录")

        # 提取每个时间步的群体
        groups_at_t_dict = {}
        timestamps = list(df['timestamp'].unique())
        print(f"  时间步数量：{len(timestamps)}")

        for timestamp in timestamps:
            groups_at_t_dict[timestamp] = groups_at_time_t(
                df, timestamp, dataset="Confs"
            )

        # 保存结果
        out_fname = f"groups_at_t_{context}.p"
        with open(out_path + out_fname, "wb") as f:
            pickle.dump(groups_at_t_dict, f)
        print(f"  保存群体数据到：{out_fname}")


def compute_group_size_distributions(contexts, in_path, out_path):
    """
    计算群体规模分布

    参数:
        contexts (list): 会议上下文列表
        in_path (str): 输入路径
        out_path (str): 输出路径
    """
    print("\n=== 计算群体规模分布 ===")

    for context in contexts:
        print(f"\n处理 {context}...")

        # 读取群体数据
        fname = f"groups_at_t_{context}.p"
        with open(in_path + fname, "rb") as f:
            groups_at_t_dict = pickle.load(f)

        # 计算规模分布
        ks, Pks = group_size_dist(groups_at_t_dict)
        print(f"  群体规模范围：{min(ks)}-{max(ks)}")

        # 保存为 DataFrame
        gsize_df = pd.DataFrame({'k': ks, 'Pk': Pks})
        fname = f"Pk_{context}.csv"
        gsize_df.to_csv(out_path + fname, header=True, index=False)
        print(f"  保存规模分布到：{fname}")


def compute_transition_matrices(contexts, in_path, out_path):
    """
    计算群体规模转移矩阵

    参数:
        contexts (list): 会议上下文列表
        in_path (str): 输入路径
        out_path (str): 输出路径
    """
    print("\n=== 计算转移矩阵 ===")

    for context in contexts:
        print(f"\n处理 {context}...")

        # 读取群体数据
        fname = f"groups_at_t_{context}.p"
        with open(in_path + fname, "rb") as f:
            groups_at_t_dict = pickle.load(f)

        # 转换为超图对象
        Hs = get_Hs_from_groups_dict(groups_at_t_dict)

        # 计算转移矩阵
        T = get_transition_matrix(Hs, max_k=20, normed=True)
        print(f"  转移矩阵维度：{T.shape}")

        # 转换为 DataFrame 并保存
        df_T = transition_matrix_to_df(T)
        out_fname = f"T_{context}.csv"
        df_T.to_csv(out_path + out_fname, header=True, index=False)
        print(f"  保存转移矩阵到：{out_fname}")


def compute_group_durations(contexts, in_path, out_path):
    """
    计算群体持续时间分布

    参数:
        contexts (list): 会议上下文列表
        in_path (str): 输入路径
        out_path (str): 输出路径
    """
    print("\n=== 计算群体持续时间 ===")

    for context in contexts:
        print(f"\n处理 {context}...")

        # 读取群体数据
        fname = f"groups_at_t_{context}.p"
        with open(in_path + fname, "rb") as f:
            groups_at_t_dict = pickle.load(f)

        # 计算持续时间
        durations = get_group_durations(groups_at_t_dict)
        print(f"  群体总数：{len(durations)}")

        # 保存结果
        out_fname = f"gdurations_{context}.p"
        with open(out_path + out_fname, "wb") as f:
            pickle.dump(durations, f)
        print(f"  保存持续时间到：{out_fname}")


def compute_group_times(contexts, in_path, out_path):
    """
    计算每个群体的创建和销毁时间

    参数:
        contexts (list): 会议上下文列表
        in_path (str): 输入路径
        out_path (str): 输出路径
    """
    print("\n=== 计算群体时间信息 ===")

    for context in contexts:
        print(f"\n处理 {context}...")

        # 读取群体数据
        fname = f"groups_at_t_{context}.p"
        with open(in_path + fname, "rb") as f:
            groups_at_t_dict = pickle.load(f)

        print("  计算群体时间...")
        # 计算群体的开始和结束时间
        groups_and_times = get_group_times(groups_at_t_dict)

        # 保存结果
        out_fname = f"group_times_{context}.p"
        with open(out_path + out_fname, "wb") as f:
            pickle.dump(groups_and_times, f)
        print(f"  保存群体时间到：{out_fname}")


def compute_disaggregation_matrices(contexts, in_path, out_path):
    """
    计算群体解聚和聚合矩阵

    参数:
        contexts (list): 会议上下文列表
        in_path (str): 输入路径
        out_path (str): 输出路径
    """
    print("\n=== 计算聚合/解聚矩阵 ===")

    for context in contexts:
        print(f"\n处理 {context}...")

        # 读取群体数据
        fname = f"groups_at_t_{context}.p"
        with open(in_path + fname, "rb") as f:
            groups_at_t_dict = pickle.load(f)

        # 读取群体时间
        fname = f"group_times_{context}.p"
        with open(in_path + fname, "rb") as f:
            groups_and_times = pickle.load(f)

        print("  计算矩阵...")
        # 计算解聚和聚合矩阵
        D, A = get_dis_agg_matrices(
            groups_at_t_dict,
            groups_and_times,
            max_k=15,
            normed=True
        )

        # 转换为 DataFrames
        df_D = dis_agg_matrix_to_df(D)
        df_A = dis_agg_matrix_to_df(A)

        # 保存结果
        out_fname = f"D_{context}.csv"
        df_D.to_csv(out_path + out_fname, header=True, index=False)
        print(f"  保存解聚矩阵到：{out_fname}")

        out_fname = f"A_{context}.csv"
        df_A.to_csv(out_path + out_fname, header=True, index=False)
        print(f"  保存聚合矩阵到：{out_fname}")


def analyze_multi_membership(contexts, in_path):
    """
    分析节点的多重成员关系

    参数:
        contexts (list): 会议上下文列表
        in_path (str): 输入路径

    Returns:
        dict: 每个上下文的度数计数
    """
    print("\n=== 分析多重成员关系 ===")

    deg_count_collection = {}

    for context in contexts:
        print(f"\n处理 {context}...")

        # 读取群体数据
        fname = f"groups_at_t_{context}.p"
        with open(in_path + fname, "rb") as f:
            groups_at_t_dict = pickle.load(f)

        # 转换为超图
        Hs = get_Hs_from_groups_dict(groups_at_t_dict)

        # 收集所有节点在所有时间步的度数
        flatten_degrees = []
        for t, H in Hs.items():
            for n, k in H.degree().items():
                flatten_degrees.append(k)

        # 度数计数
        deg_count = Counter(flatten_degrees)
        deg_count_collection[context] = deg_count

        # 打印单成员比例
        single_member_ratio = deg_count.get(1, 0) / sum(deg_count.values())
        print(f"  单成员比例：{single_member_ratio:.4f}")

    return deg_count_collection


def measure_social_memory(contexts, in_path, out_path):
    """
    测量社会记忆指标

    参数:
        contexts (list): 会议上下文列表
        in_path (str): 输入路径
        out_path (str): 输出路径
    """
    print("\n=== 测量社会记忆 ===")

    for context in contexts:
        print(f"\n处理 {context}...")

        # 读取群体数据
        fname = f"groups_at_t_{context}.p"
        with open(in_path + fname, "rb") as f:
            groups_at_t_dict = pickle.load(f)

        # 转换为超图
        Hs = get_Hs_from_groups_dict(groups_at_t_dict)
        print("  超图已读取")

        # 读取群体时间
        fname = f"group_times_{context}.p"
        with open(in_path + fname, "rb") as f:
            groups_and_times = pickle.load(f)
        print("  群体时间已读取")

        # 计算累积接触网络
        Gs = get_cumulative_Gs_from_Hs(Hs)
        print("  累积接触网络已计算")

        # 测量社会记忆
        memory_df = measure_social_memory(
            Hs,
            groups_at_t_dict,
            Gs,
            groups_and_times
        )

        # 保存结果
        out_fname = f"social_memory_{context}.csv.gz"
        memory_df.to_csv(
            out_path + out_fname,
            header=True,
            index=False,
            compression="gzip"
        )
        print(f"  保存社会记忆数据到：{out_fname}")


def compute_interevent_times(contexts, in_path, out_path):
    """
    计算事件间隔时间

    参数:
        contexts (list): 会议上下文列表
        in_path (str): 输入路径
        out_path (str): 输出路径
    """
    print("\n=== 计算事件间隔时间 ===")

    for context in contexts:
        print(f"\n处理 {context}...")

        # 读取群体时间
        fname = f"group_times_{context}.p"
        with open(in_path + fname, "rb") as f:
            groups_and_times = pickle.load(f)
        print("  群体时间已读取")

        # 计算事件间隔时间
        interevent_times = get_interevent_times(groups_and_times)

        # 保存结果
        out_fname = f"interevent_times_{context}.p"
        with open(out_path + out_fname, "wb") as f:
            pickle.dump(interevent_times, f)
        print(f"  保存事件间隔时间到：{out_fname}")


def compute_node_trajectories(contexts, in_path, out_path):
    """
    计算节点在不同群体规模间的轨迹

    参数:
        contexts (list): 会议上下文列表
        in_path (str): 输入路径
        out_path (str): 输出路径
    """
    print("\n=== 计算节点轨迹 ===")

    for context in contexts:
        print(f"\n处理 {context}...")

        # 读取群体数据
        fname = f"groups_at_t_{context}.p"
        with open(in_path + fname, "rb") as f:
            groups_at_t_dict = pickle.load(f)

        # 转换为超图
        Hs = get_Hs_from_groups_dict(groups_at_t_dict)
        print("  超图已读取")

        # 计算轨迹矩阵
        Traj, index_to_node = get_node_trajectory(Hs)
        print(f"  轨迹矩阵维度：{Traj.shape}")

        # 保存结果
        out_fname = f"trajectories_matrix_{context}.p"
        with open(out_path + out_fname, "wb") as f:
            pickle.dump(Traj, f)
        print(f"  保存轨迹矩阵到：{out_fname}")

        out_fname = f"trajectories_matrix_i2n{context}.p"
        with open(out_path + out_fname, "wb") as f:
            pickle.dump(index_to_node, f)
        print(f"  保存索引到节点映射到：{out_fname}")


def compute_leaving_probabilities(contexts, in_path, out_path):
    """
    计算离开群体的概率（用于 Logistic 函数拟合）

    参数:
        contexts (list): 会议上下文列表
        in_path (str): 输入路径
        out_path (str): 输出路径
    """
    print("\n=== 计算离开群体概率 ===")

    # 定义参数范围
    taus = np.arange(1, 1000)
    gsizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for context in contexts:
        print(f"\n处理 {context}...")

        # 读取持续时间数据
        in_fname = f"gdurations_{context}.p"
        with open(in_path + in_fname, "rb") as f:
            durations = pickle.load(f)

        print("  计算概率...")
        # 计算不同群体规模下的离开概率
        prob_by_size = get_probs_leaving_group(durations, gsizes, taus)

        # 保存结果
        out_fname = f"Prob_leaving_group_sizek_after_tau_{context}.p"
        with open(out_path + out_fname, "wb") as f:
            pickle.dump(prob_by_size, f)
        print(f"  保存概率数据到：{out_fname}")


def main():
    """
    主函数：执行完整的会议数据分析流程
    """
    print("=" * 60)
    print("会议数据分析脚本")
    print("=" * 60)

    # 设置数据集和上下文
    dataset = "Confs"
    contexts = ["conf16", "conf17", "conf18", "conf19"]

    # 设置路径
    in_path = f"../data-processed/{dataset}/"
    out_path = f"../data-analysis/results/{dataset}/"

    # 创建输出目录
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # 执行分析流程
    try:
        # 1. 提取群体交互
        extract_group_interactions(contexts, in_path, out_path)

        # 2. 计算群体规模分布
        compute_group_size_distributions(contexts, in_path, out_path)

        # 3. 计算转移矩阵
        compute_transition_matrices(contexts, in_path, out_path)

        # 4. 计算群体持续时间
        compute_group_durations(contexts, in_path, out_path)

        # 5. 计算群体时间信息
        compute_group_times(contexts, in_path, out_path)

        # 6. 计算聚合/解聚矩阵
        compute_disaggregation_matrices(contexts, in_path, out_path)

        # 7. 分析多重成员关系
        deg_counts = analyze_multi_membership(contexts, in_path)

        # 8. 测量社会记忆
        measure_social_memory(contexts, in_path, out_path)

        # 9. 计算事件间隔时间
        compute_interevent_times(contexts, in_path, out_path)

        # 10. 计算节点轨迹
        compute_node_trajectories(contexts, in_path, out_path)

        # 11. 计算离开群体概率
        compute_leaving_probabilities(contexts, in_path, out_path)

        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误：{e}")
        raise


if __name__ == "__main__":
    main()
