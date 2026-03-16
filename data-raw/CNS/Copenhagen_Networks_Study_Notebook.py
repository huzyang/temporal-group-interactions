#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
哥本哈根网络研究数据处理脚本
Copenhagen Networks Study Data Processing

该脚本用于加载和探索哥本哈根网络研究的数据集，
包括蓝牙接触、通话、短信、Facebook 好友和性别信息。

数据来源：Copenhagen Networks Study
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import itertools


def setup_directories():
    """
    设置数据目录

    Returns:
        str: 数据目录路径
    """
    data_dir = "./"
    print(f"数据目录：{data_dir}")
    return data_dir


def define_data_files():
    """
    定义数据文件名

    Returns:
        dict: 包含各种数据类型文件名的字典
    """
    data_filenames = {
        "bluetooth": "bt_symmetric.csv",
        "calls": "calls.csv",
        "sms": "sms.csv",
        "facebook_friends": "fb_friends.csv",
        "genders": "genders.csv"
    }
    return data_filenames


def load_data(data_filename, data_dir):
    """
    从指定目录加载数据文件

    Args:
        data_filename (str): 数据文件名
        data_dir (str): 数据目录路径

    Returns:
        pd.DataFrame: 加载的数据 DataFrame
    """
    try:
        file_path = os.path.join(data_dir, data_filename)
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到")
        raise
    except Exception as e:
        print(f"加载文件时出错：{e}")
        raise


def display_data_preview(df, dataset_name):
    """
    显示数据集的前几行预览

    Args:
        df (pd.DataFrame): 要显示的数据框
        dataset_name (str): 数据集名称
    """
    print(f"\n{dataset_name}")
    print(df.head())


def visualize_network_graph(df_edges, source_col, target_col, title="Network Graph"):
    """
    可视化网络图

    Args:
        df_edges (pd.DataFrame): 包含边信息的 DataFrame
        source_col (str): 源节点列名
        target_col (str): 目标节点列名
        title (str): 图表标题
    """
    try:
        G = nx.from_pandas_edgelist(df_edges, source=source_col, target=target_col)

        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        nx.draw_networkx_nodes(G, pos, node_size=50, alpha=0.6, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color='gray', width=0.5)

        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}.pdf", format='pdf', bbox_inches='tight')
        plt.show()

        print(f"网络图已保存为 {title.replace(' ', '_')}.pdf")
    except Exception as e:
        print(f"可视化网络图时出错：{e}")


def calculate_network_statistics(df_edges, source_col, target_col):
    """
    计算网络统计指标

    Args:
        df_edges (pd.DataFrame): 包含边信息的 DataFrame
        source_col (str): 源节点列名
        target_col (str): 目标节点列名

    Returns:
        dict: 包含各种网络统计指标的字典
    """
    G = nx.from_pandas_edgelist(df_edges, source=source_col, target=target_col)

    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'avg_degree': 2 * G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    }

    try:
        stats['avg_clustering'] = nx.average_clustering(G)
    except:
        stats['avg_clustering'] = None

    try:
        connected_components = list(nx.connected_components(G))
        stats['num_connected_components'] = len(connected_components)
        stats['largest_component_size'] = max(len(cc) for cc in connected_components) if connected_components else 0
    except:
        stats['num_connected_components'] = None
        stats['largest_component_size'] = None

    return stats


def analyze_gender_distribution(df_genders):
    """
    分析性别分布

    Args:
        df_genders (pd.DataFrame): 包含性别信息的 DataFrame

    Returns:
        dict: 包含性别分布统计的字典
    """
    total = len(df_genders)
    females = df_genders['female'].sum()
    males = total - females

    gender_stats = {
        'total': total,
        'females': int(females),
        'males': int(males),
        'female_percentage': (females / total * 100) if total > 0 else 0,
        'male_percentage': (males / total * 100) if total > 0 else 0
    }

    return gender_stats


def plot_gender_distribution(gender_stats):
    """
    绘制性别分布饼图

    Args:
        gender_stats (dict): 性别分布统计字典
    """
    labels = ['Male', 'Female']
    sizes = [gender_stats['males'], gender_stats['females']]

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, colors=['lightskyblue', 'lightcoral'],
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title('Gender Distribution', fontsize=16)
    plt.axis('equal')
    plt.savefig('Gender_Distribution.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    print("性别分布图已保存为 Gender_Distribution.pdf")


def main():
    """
    主函数：执行完整的数据加载和分析流程
    """
    print("=" * 80)
    print("哥本哈根网络研究 - 数据处理与分析")
    print("=" * 80)

    try:
        # 设置目录和文件名
        print("\n[步骤 1/7] 设置数据目录...")
        data_dir = setup_directories()

        print("\n[步骤 2/7] 定义数据文件...")
        data_filenames = define_data_files()

        # 加载所有数据集
        print("\n[步骤 3/7] 加载数据集...")
        print("-" * 80)

        print("\n加载蓝牙接触数据...")
        df_bt = load_data(data_filenames["bluetooth"], data_dir)
        display_data_preview(df_bt, "蓝牙接触数据")

        print("\n加载通话记录数据...")
        df_calls = load_data(data_filenames["calls"], data_dir)
        display_data_preview(df_calls, "通话记录数据")

        print("\n加载短信数据...")
        df_sms = load_data(data_filenames["sms"], data_dir)
        display_data_preview(df_sms, "短信数据")

        print("\n加载 Facebook 好友数据...")
        df_facebook_friends = load_data(data_filenames["facebook_friends"], data_dir)
        display_data_preview(df_facebook_friends, "Facebook 好友数据")

        print("\n加载性别信息数据...")
        df_genders = load_data(data_filenames["genders"], data_dir)
        display_data_preview(df_genders, "性别信息数据")

        # 网络统计分析
        print("\n[步骤 4/7] 计算网络统计指标...")
        print("-" * 80)

        # 蓝牙网络
        print("\n蓝牙接触网络统计:")
        bt_stats = calculate_network_statistics(df_bt, 'user_a', 'user_b')
        for key, value in bt_stats.items():
            print(f"  {key}: {value}")

        # 通话网络
        print("\n通话网络统计:")
        calls_stats = calculate_network_statistics(df_calls, 'caller', 'callee')
        for key, value in calls_stats.items():
            print(f"  {key}: {value}")

        # 短信网络
        print("\n短信网络统计:")
        sms_stats = calculate_network_statistics(df_sms, 'sender', 'recipient')
        for key, value in sms_stats.items():
            print(f"  {key}: {value}")

        # Facebook 好友网络
        print("\nFacebook 好友网络统计:")
        fb_stats = calculate_network_statistics(df_facebook_friends, 'user_a', 'user_b')
        for key, value in fb_stats.items():
            print(f"  {key}: {value}")

        # 性别分析
        print("\n[步骤 5/7] 分析性别分布...")
        print("-" * 80)
        gender_stats = analyze_gender_distribution(df_genders)
        print(f"\n总人数：{gender_stats['total']}")
        print(f"男性：{gender_stats['males']} ({gender_stats['male_percentage']:.1f}%)")
        print(f"女性：{gender_stats['females']} ({gender_stats['female_percentage']:.1f}%)")

        # 可视化
        print("\n[步骤 6/7] 生成可视化图表...")
        print("-" * 80)

        print("\n生成蓝牙网络图...")
        visualize_network_graph(df_bt.head(1000), 'user_a', 'user_b', "Bluetooth_Network")

        print("\n生成通话网络图...")
        visualize_network_graph(df_calls.head(1000), 'caller', 'callee', "Calls_Network")

        print("\n生成短信网络图...")
        visualize_network_graph(df_sms.head(1000), 'sender', 'recipient', "SMS_Network")

        print("\n生成 Facebook 好友网络图...")
        visualize_network_graph(df_facebook_friends.head(1000), 'user_a', 'user_b', "Facebook_Network")

        print("\n生成性别分布图...")
        plot_gender_distribution(gender_stats)

        # 完成
        print("\n[步骤 7/7] 完成!")
        print("=" * 80)
        print("数据处理和分析已完成。")
        print("所有图表已保存为 PDF 格式。")
        print("=" * 80)

    except Exception as e:
        print(f"\n程序执行过程中出错：{e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
