#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CNS 数据集预处理脚本

本脚本用于对 CNS（Complex Network Systems）蓝牙接触数据进行完整的预处理，包括：
1. 数据加载和初步清洗
2. RSSI 信号强度过滤
3. 时间戳重缩放
4. 用户 ID 映射
5. 填充单个时间间隙
6. 移除散点交互
7. 三角闭包处理
8. 添加孤立节点
9. 最终数据整合
"""

import pandas as pd
import pickle
import numpy as np
from itertools import combinations
import networkx as nx
import sys
import os
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# import palettable as pltt
import matplotlib as mplt

# 设置中文字体
mplt.rcParams['font.sans-serif'] = 'Avenir'

# 设置图表样式
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['xtick.minor.width'] = 1.2
plt.rcParams['ytick.minor.size'] = 2
plt.rcParams['ytick.minor.width'] = 1.2
plt.rcParams['axes.linewidth'] = 1.2

# 添加代码目录到路径
sys.path.append("../code/")
from code.data_processing import LoadData, get_triadic_closure_links_to_add_at_time_t


def create_directories():
    """创建必要的输出目录"""
    print("=" * 60)
    print("步骤 0: 创建输出目录")
    print("=" * 60)

    directories = ["figures", "results", "temp", "../data-processed/CNS/"]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  ✓ 创建目录：{directory}")
        else:
            print(f"  ✓ 目录已存在：{directory}")


def load_and_preprocess_data():
    """
    加载并初步预处理蓝牙数据

    返回：
        df: 预处理后的 DataFrame
    """
    print("\n" + "=" * 60)
    print("步骤 1: 加载和初步预处理数据")
    print("=" * 60)

    # 设置数据目录和文件名
    DATA_DIR = "../data-raw/CNS/"
    DATA_FILENAMES = {
        "bluetooth": "bt_symmetric.csv",
        "calls": "calls.csv",
        "sms": "sms.csv",
        "facebook_friends": "fb_friends.csv",
        "genders": "genders.csv"
    }

    # 加载蓝牙数据
    print("加载蓝牙数据...")
    df = LoadData(DATA_FILENAMES["bluetooth"], DATA_DIR)
    print(f"  原始记录数：{len(df)}")
    print(f"  列名：{list(df.columns)}")

    # 清理错误的 RSSI 值（RSSI=20 不合理）
    print("清理错误的 RSSI 值...")
    df = df[df['rssi'] <= 0]
    print(f"  清理后记录数：{len(df)}")

    # 时间重缩放：将 300 秒（5 分钟）的时间间隔转换为 1
    print("时间重缩放（5 分钟→1 个单位）...")
    df['# timestamp'] = (df['# timestamp'] / 300).astype(int)

    return df


def map_user_ids(df):
    """
    映射用户 ID 为连续的整数

    参数：
        df: 输入 DataFrame

    返回：
        df: ID 映射后的 DataFrame
        old_id_to_new: 旧 ID 到新 ID 的映射字典
        new_id_to_old: 新 ID 到旧 ID 的映射字典
    """
    print("\n" + "=" * 60)
    print("步骤 2: 映射用户 ID")
    print("=" * 60)

    # 提取所有用户 ID
    users = set(df['user_a'].unique()).union(set(df['user_b'].unique()))

    # 移除占位符 -1（空扫描）和 -2（非研究设备）
    users.remove(-1)
    users.remove(-2)
    users = list(users)

    print(f"  真实用户数：{len(users)}")
    print(f"  最大原始 ID: {max(users)}")

    # 创建 ID 映射字典
    old_id_to_new = {}
    new_id_to_old = {}

    for new_id, old_id in enumerate(users):
        old_id_to_new[old_id] = new_id
        new_id_to_old[new_id] = old_id

    # 映射用户列表
    users = [old_id_to_new[ID] for ID in users]
    print(f"  映射后最大 ID: {max(users)}")

    # 保存映射字典
    output_path = "../data-processed/CNS/"

    with open(output_path + "old_id_to_new_dict.p", "wb") as f:
        pickle.dump(old_id_to_new, f)

    with open(output_path + "new_id_to_old_dict.p", "wb") as f:
        pickle.dump(new_id_to_old, f)

    print(f"  ✓ 保存 ID 映射字典到：{output_path}")

    # 替换 DataFrame 中的 ID
    df.replace({'user_a': old_id_to_new}, inplace=True)
    df.replace({'user_b': old_id_to_new}, inplace=True)

    return df, old_id_to_new, new_id_to_old


def analyze_user_interactions(df, N=706):
    """
    分析不同类型用户的比例随时间变化

    参数：
        df: 输入 DataFrame
        N: 真实用户总数
    """
    print("\n" + "=" * 60)
    print("步骤 3: 分析用户交互类型")
    print("=" * 60)

    users_int_data = []
    timestamps = df['# timestamp'].unique()
    total = len(timestamps)

    for i, timestamp in enumerate(timestamps):
        dft = df[df['# timestamp'] == timestamp]

        # 统计有真实交互的用户
        users_int = set(dft['user_a'].unique()).union(set(dft['user_b'].unique()))
        users_int.discard(-1)
        users_int.discard(-2)
        N_users_int = len(users_int)

        # 统计只与外部用户交互的用户
        users_ext = set(dft[dft['user_b'] == -2]['user_a'].unique())
        users_ext = users_ext.difference(users_int)
        N_users_ext = len(users_ext)

        # 统计声明为孤立的用户
        users_iso = set(dft[dft['user_b'] == -1]['user_a'].unique())
        users_iso = users_iso.difference(users_int)
        N_users_iso = len(users_iso)

        # 计算缺失数据的用户
        N_users_missing = N - (N_users_int + N_users_ext + N_users_iso)

        row = [timestamp, N_users_int, N_users_ext, N_users_iso, N_users_missing]
        users_int_data.append(row)

        # 显示进度
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"  进度：{i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")

    users_int_df = pd.DataFrame(
        users_int_data,
        columns=['# timestamp', 'N_users_int', 'N_users_ext', 'N_users_iso', 'N_users_missing']
    )

    # 可视化
    print("生成用户类型分布图...")
    plt.figure(figsize=(6.5, 4))
    ax = plt.subplot(111)

    users_int_df.plot.area(
        x='# timestamp', ax=ax,
        color={
            'N_users_missing': '#b2df8a',
            'N_users_int': '#a6cee3',
            'N_users_iso': '#1f78b4',
            'N_users_ext': 'lightgray'
        }
    )

    ax.set_xlabel('Time', size=18)
    ax.set_ylabel('Number of users', size=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xlim(0, users_int_df['# timestamp'].max())
    ax.set_ylim(0, N)

    custom_lines = [
        Line2D([0], [0], color='#a6cee3', lw=8),
        Line2D([0], [0], color='lightgray', lw=8),
        Line2D([0], [0], color='#1f78b4', lw=8),
        Line2D([0], [0], color='#b2df8a', lw=8)
    ]

    ax.legend(
        custom_lines,
        ["Interacting internally", "Interacting (only) externally", "Isolated", "No data"],
        fontsize=14, handlelength=1, borderpad=0.8
    )

    plt.tight_layout()
    plt.savefig("figures/CNS_users_class_in_time.pdf", bbox_inches='tight', dpi=150)
    print(f"  ✓ 保存到：figures/CNS_users_class_in_time.pdf")

    plt.close()


def filter_rssi(df, threshold=-90):
    """
    根据 RSSI 阈值过滤数据

    参数：
        df: 输入 DataFrame
        threshold: RSSI 阈值（默认 -90dBm）

    返回：
        df: 过滤后的 DataFrame
    """
    print("\n" + "=" * 60)
    print(f"步骤 4: 过滤 RSSI (阈值：{threshold}dBm)")
    print("=" * 60)

    # 移除外部分用户（user_b = -2）
    print("移除外部用户交互...")
    df = df[df['user_b'] != -2]
    print(f"  记录数：{len(df)}")

    # 获取所有非零 RSSI 值
    values = df[df['rssi'] != 0]['rssi'].values
    filtered_values = df[(df['rssi'] != 0) & (df['rssi'] >= threshold)]['rssi'].values

    # 可视化 RSSI 分布
    print("生成 RSSI 过滤分布图...")
    plt.figure(figsize=(12, 3.5))

    # 左图：线性坐标
    ax = plt.subplot(121)
    ax.hist(values, histtype='stepfilled', color='lightgray',
            bins=np.linspace(-100, -20, 33), label='Unfiltered', clip_on=False)
    ax.hist(filtered_values, histtype='stepfilled', color='C0', ec="k",
            bins=np.linspace(-100, -20, 33), label='Filtered', clip_on=False)

    ax.set_xlabel('Received Signal Strength Indication (RSSI)', size=18)
    ax.set_ylabel('Num. records', size=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim(-105, -20)
    ax.set_ylim(0, 3e5)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(frameon=False, fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.annotate('A', xy=(-0.1, -0.29), xycoords="axes fraction", fontsize=18, fontweight="bold")

    # 右图：对数坐标
    ax = plt.subplot(122)
    ax.hist(values, histtype='stepfilled', color='lightgray',
            bins=np.linspace(-100, -20, 33), label='Unfiltered')
    ax.hist(filtered_values, histtype='stepfilled', color='C0', ec="k",
            bins=np.linspace(-100, -20, 33), label='Filtered')

    ax.set_xlabel('Received Signal Strength Indication (RSSI)', size=18)
    ax.set_ylabel('Num. records', size=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.set_xlim(-105, -20)
    ax.set_yscale('log')
    ax.set_ylim(1e1, 1e6)
    ax.set_yticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    ax.legend(frameon=False, fontsize=14)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.annotate('B', xy=(-0.1, -0.29), xycoords="axes fraction", fontsize=18, fontweight="bold")

    plt.tight_layout()
    plt.savefig("figures/CNS_RSSI_filtering.pdf", bbox_inches='tight', dpi=150)
    print(f"  ✓ 保存到：figures/CNS_RSSI_filtering.pdf")

    plt.close()

    # 应用 RSSI 过滤
    df = df[df['rssi'] >= threshold]

    # 保存中间结果
    output_path = "temp/"
    df.to_csv(output_path + "CNS_bluetooth_partially_processed.csv", sep=',', header=True, index=False)

    # 打印统计信息
    print("\n数据统计:")
    print(f"  所有记录（rssi≠0）: {len(df[df['rssi'] != 0])}")
    print(f"  成对交互总数：{len(df)}")
    print(f"  空扫描（rssi=0）: {len(df[df['rssi'] == 0])}")

    return df


def fill_single_gaps(df):
    """
    填充单个时间间隙
    （在 t 和 t+2 时刻存在但在 t+1 时刻缺失的交互）

    参数：
        df: 输入 DataFrame

    返回：
        df: 填充后的 DataFrame
    """
    print("\n" + "=" * 60)
    print("步骤 5: 填充单个时间间隙")
    print("=" * 60)

    # 获取所有唯一的用户对（排除 user_b=-1 的记录）
    unique_pairs = [
        (user_a, user_b)
        for (user_a, user_b), _ in df.groupby(by=['user_a', 'user_b'])
        if user_b != -1
    ]

    print(f"  唯一用户对数量：{len(unique_pairs)}")

    # 存储需要添加的条目
    entries_to_add = []

    for user_a, user_b in unique_pairs:
        df_ab = df[(df['user_a'] == user_a) & (df['user_b'] == user_b)]

        for t in df_ab['# timestamp']:
            # 检查 t+2 时刻存在但 t+1 时刻不存在
            if ((df_ab['# timestamp'] == t + 2).any() and
                    (not (df_ab['# timestamp'] == t + 1).any())):
                # 计算平均 RSSI
                rssi_t = int(df_ab[df_ab['# timestamp'] == t]['rssi'])
                rssi_t2 = int(df_ab[df_ab['# timestamp'] == t + 2]['rssi'])
                avg_rssi = (rssi_t + rssi_t2) / 2
                entries_to_add.append([t + 1, user_a, user_b, avg_rssi])

    print(f"  需要添加的条目数：{len(entries_to_add)}")

    # 保存条目
    with open("temp/entries_to_add.p", "wb") as f:
        pickle.dump(entries_to_add, f)

    # 将新条目添加到 DataFrame
    filled_df = df.copy()

    entries_to_add_dict = [
        {'# timestamp': entry[0], 'user_a': entry[1],
         'user_b': entry[2], 'rssi': entry[3]}
        for entry in entries_to_add
    ]

    entries_to_add_df = pd.DataFrame.from_records(entries_to_add_dict)
    filled_df = pd.concat([df, entries_to_add_df], ignore_index=True)

    print(f"  原始记录：{len(df)}")
    print(f"  填充后记录：{len(filled_df)}")
    print(f"  新增记录：{len(filled_df) - len(df)}")

    # 检查并移除重复项（某个节点在 t 时刻已被标记为孤立的情况）
    duplicates = filled_df[filled_df.duplicated(subset=['# timestamp', 'user_a'], keep=False)]
    to_remove = duplicates[duplicates['rssi'] == 0]

    print(f"  发现冲突的孤立记录：{len(to_remove)}")

    filled_df_corrected = filled_df.drop(to_remove.index)
    print(f"  修正后记录：{len(filled_df_corrected)}")

    # 保存结果
    output_path = "temp/"
    filled_df_corrected.to_csv(
        output_path + "CNS_bluetooth_thresholded_and_filled.csv.gz",
        sep=',', header=True, index=False, compression='gzip'
    )
    print(f"  ✓ 保存到：{output_path}CNS_bluetooth_thresholded_and_filled.csv.gz")

    return filled_df_corrected


def remove_scattered_interactions(df):
    """
    移除散点交互（仅持续一个时间步的交互）

    参数：
        df: 输入 DataFrame

    返回：
        df: 清理后的 DataFrame
        indices_to_remove: 被移除的索引列表
    """
    print("\n" + "=" * 60)
    print("步骤 6: 移除散点交互")
    print("=" * 60)

    # 获取所有唯一的用户对
    unique_pairs = [
        (user_a, user_b)
        for (user_a, user_b), _ in df.groupby(by=['user_a', 'user_b'])
        if user_b != -1
    ]

    # 存储需要移除的索引
    indices_to_remove = []

    total_pairs = len(unique_pairs)

    for i, (user_a, user_b) in enumerate(unique_pairs):
        df_ab = df[(df['user_a'] == user_a) & (df['user_b'] == user_b)]

        for t in df_ab['# timestamp']:
            # 检查是否在 t-1 或 t+1 时刻存在交互
            if (df_ab['# timestamp'] == t - 1).any() or (df_ab['# timestamp'] == t + 1).any():
                continue
            else:
                # 这是孤立的时间点，需要移除
                index_to_drop = df_ab[df_ab['# timestamp'] == t].index
                indices_to_remove.append(index_to_drop)

        # 显示进度
        if (i + 1) % 1000 == 0 or (i + 1) == total_pairs:
            print(f"  进度：{i + 1}/{total_pairs} ({(i + 1) / total_pairs * 100:.1f}%)")

    # 保存索引
    with open("temp/indices_to_remove.p", "wb") as f:
        pickle.dump(indices_to_remove, f)

    # 移除这些条目
    clean_indices_to_remove = [ii[0] for ii in indices_to_remove]
    df_final = df.drop(clean_indices_to_remove)

    print(f"  移除的散点交互数：{len(df) - len(df_final)}")
    print(f"  剩余记录数：{len(df_final)}")

    # 保存结果
    output_path = "temp/"
    df_final.to_csv(
        output_path + "CNS_bluetooth_thresholded_filled_and_nosingletons.csv.gz",
        sep=',', header=True, index=False, compression='gzip'
    )
    print(f"  ✓ 保存到：{output_path}CNS_bluetooth_thresholded_filled_and_nosingletons.csv.gz")

    return df_final, indices_to_remove


def apply_triadic_closure(df):
    """
    应用三角闭包算法

    参数：
        df: 输入 DataFrame

    返回：
        df_triclo: 应用三角闭包后的 DataFrame
    """
    print("\n" + "=" * 60)
    print("步骤 7: 应用三角闭包")
    print("=" * 60)

    triclo_entries_to_add = []
    timestamps = df['# timestamp'].unique()
    total = len(timestamps)

    for i, timestamp in enumerate(timestamps):
        # 获取当前时间步需要添加的三角闭包链接
        new_links = get_triadic_closure_links_to_add_at_time_t(df, timestamp)
        triclo_entries_to_add.extend(new_links)

        # 显示进度
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"  时间步：{timestamp} ({i + 1}/{total}), 已收集 {len(triclo_entries_to_add)} 条链接")

    # 创建 DataFrame 并合并
    triclo_entries_to_add_df = pd.DataFrame.from_records(triclo_entries_to_add)
    df_triclo = pd.concat([df, triclo_entries_to_add_df], ignore_index=True)

    print(f"\n  原始记录：{len(df)}")
    print(f"  三角闭包后：{len(df_triclo)}")
    print(f"  新增三角形链接：{len(df_triclo) - len(df)}")

    # 保存结果
    output_path = "temp/"
    df_triclo.to_csv(
        output_path + "CNS_bluetooth_thresholded_filled_nosingletons_and_triclo.csv.gz",
        sep=',', header=True, index=False, compression='gzip'
    )
    print(f"  ✓ 保存到：{output_path}CNS_bluetooth_thresholded_filled_nosingletons_and_triclo.csv.gz")

    return df_triclo


def analyze_triclo_thresholds(df, df_triclo):
    """
    分析不同 RSSI 阈值下的三角闭包效果

    参数：
        df: 原始 DataFrame（三角闭包前）
        df_triclo: 三角闭包后的 DataFrame
    """
    print("\n" + "=" * 60)
    print("步骤 8: 分析三角闭包阈值")
    print("=" * 60)

    # 提取三角闭包添加的链接
    df_tri = df_triclo.drop(df.index, axis=0, inplace=False)

    # 不同的 RSSI 阈值
    triclo_thresholds = np.arange(-90, -30, 5)
    num_added_links = [len(df_tri[df_tri['rssi'] >= th]) for th in triclo_thresholds]

    # 可视化
    plt.figure(figsize=(6, 4.5))
    ax = plt.subplot(111)

    ax.plot(triclo_thresholds, num_added_links, 'o-', mec='black', ms=8, lw=4, clip_on=False)

    ax.set_xlabel('RSSI threshold for triadic closure, $\\phi$', size=18)
    ax.set_ylabel('Number of closed triangles', size=18)
    ax.tick_params(axis='both', which='major', labelsize=18)

    ax.set_yscale('log')
    ax.set_ylim(1e0, 1e7)
    ax.set_yticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ax.set_xticks([-90, -80, -70, -60, -50, -40, -30])

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.tight_layout()
    plt.savefig("figures/CNS_scaling_triclo_RSSI.pdf", bbox_inches='tight', dpi=150,
                transparent=False, facecolor='white')
    print(f"  ✓ 保存到：figures/CNS_scaling_triclo_RSSI.pdf")

    plt.close()


def create_datetime_converter(df):
    """
    创建时间戳到 datetime 的转换器

    参数：
        df: 包含时间戳的 DataFrame

    返回：
        dt_converter: 时间戳到 datetime 的字典
    """
    print("\n" + "=" * 60)
    print("步骤 9: 创建日期时间转换器")
    print("=" * 60)

    # 已知数据从 2013 年 3 月 3 日周日午夜开始
    starting_index_t = 0
    ending_index_t = df['# timestamp'].astype(int).max()
    starting_dt = dt.datetime(year=2013, month=3, day=3, hour=0, minute=0)
    Delta_t = dt.timedelta(minutes=5)
    current_dt = starting_dt

    dt_converter = {}

    for index_t in range(starting_index_t, ending_index_t + 1):
        dt_converter[index_t] = current_dt
        current_dt += Delta_t

    print(f"  时间范围：{starting_dt} 到 {dt_converter[ending_index_t]}")
    print(f"  总时间步数：{len(dt_converter)}")

    return dt_converter


def process_with_threshold(df, df_triclo, dt_converter, threshold=-75):
    """
    使用选定的阈值处理数据，并添加 datetime 信息

    参数：
        df: 原始 DataFrame
        df_triclo: 三角闭包后的 DataFrame
        dt_converter: 时间戳到 datetime 的转换器
        threshold: RSSI 阈值（默认 -75）

    返回：
        df_processed: 处理后的 DataFrame
    """
    print("\n" + "=" * 60)
    print(f"步骤 10: 使用阈值 {threshold}dBm 处理数据")
    print("=" * 60)

    # 提取三角闭包添加的链接
    df_tri = df_triclo.drop(df.index, axis=0, inplace=False)

    # 应用阈值
    thresholds = [-90, -85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35]

    for th in thresholds:
        print(f"  处理阈值：{th}dBm")

        # 应用阈值的 DataFrame
        df_triclo_th = pd.concat([df, df_tri[df_tri['rssi'] >= th]])

        # 转换时间戳为整数
        df_triclo_th['# timestamp'] = df_triclo_th['# timestamp'].astype(int)

        # 转换为 datetime
        df_triclo_th['datetime'] = df_triclo_th['# timestamp'].apply(lambda x: dt_converter[x])
        df_triclo_th['DoW'] = df_triclo_th['datetime'].apply(lambda x: x.strftime('%A'))
        df_triclo_th['hour'] = df_triclo_th['datetime'].apply(lambda x: x.hour)

        # 保存
        output_path = "temp/"
        fname = f"CNS_bluetooth_thresholded_filled_nosingletons_and_triclo_thr{abs(th)}_datetime.csv.gz"
        df_triclo_th.to_csv(
            output_path + fname,
            sep=',', header=True, index=False, compression='gzip'
        )

    # 返回选定阈值的结果
    df_selected = pd.concat([df, df_tri[df_tri['rssi'] >= threshold]])
    df_selected['# timestamp'] = df_selected['# timestamp'].astype(int)
    df_selected['datetime'] = df_selected['# timestamp'].apply(lambda x: dt_converter[x])
    df_selected['DoW'] = df_selected['datetime'].apply(lambda x: x.strftime('%A'))
    df_selected['hour'] = df_selected['datetime'].apply(lambda x: x.hour)

    return df_selected


def add_back_isolated_nodes(df_original, df_processed, removed_indices, dt_converter, threshold=-75):
    """
    重新添加孤立节点（在移除散点交互时被错误删除的节点）

    参数：
        df_original: 原始 DataFrame（移除散点前）
        df_processed: 处理后的 DataFrame（包含三角闭包）
        removed_indices: 之前移除的索引列表
        dt_converter: 时间戳到 datetime 的转换器
        threshold: RSSI 阈值

    返回：
        final_df: 最终的 DataFrame
    """
    print("\n" + "=" * 60)
    print(f"步骤 11: 重新添加孤立节点（阈值：{threshold}dBm）")
    print("=" * 60)

    # 获取被移除的条目详情
    removed_entries = []

    for ix_removed in removed_indices:
        df_ix = df_original.iloc[ix_removed]
        t = df_ix['# timestamp'].values[0]
        a = df_ix['user_a'].values[0]
        b = df_ix['user_b'].values[0]
        removed_entries.append([t, a, b])

    print(f"  被移除的链接数：{len(removed_entries)}")

    # 找出需要重新添加的孤立节点
    isolated_nodes_to_add_back = []
    total = len(removed_entries)

    for i, entry in enumerate(removed_entries):
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"  进度：{i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")

        t, a, b = entry

        # 检查节点 a 在 t 时刻是否还有其他交互
        df_t_a = df_processed[
            (df_processed['# timestamp'] == t) &
            ((df_processed['user_a'] == a) | (df_processed['user_b'] == a))
            ]

        if len(df_t_a) == 0:
            # 该链接是 a 在 t 时刻的唯一交互，需要添加为孤立节点
            isolated_nodes_to_add_back.append([t, a, -1, 0])

        # 检查节点 b 在 t 时刻是否还有其他交互
        df_t_b = df_processed[
            (df_processed['# timestamp'] == t) &
            ((df_processed['user_a'] == b) | (df_processed['user_b'] == b))
            ]

        if len(df_t_b) == 0:
            # 该链接是 b 在 t 时刻的唯一交互，需要添加为孤立节点
            isolated_nodes_to_add_back.append([t, b, -1, 0])

    df_isolated = pd.DataFrame(
        isolated_nodes_to_add_back,
        columns=['# timestamp', 'user_a', 'user_b', 'rssi']
    )

    # 添加 datetime 信息
    df_isolated['# timestamp'] = df_isolated['# timestamp'].astype(int)
    df_isolated['datetime'] = df_isolated['# timestamp'].apply(lambda x: dt_converter[x])
    df_isolated['DoW'] = df_isolated['datetime'].apply(lambda x: x.strftime('%A'))
    df_isolated['hour'] = df_isolated['datetime'].apply(lambda x: x.hour)

    print(f"  需要重新添加的孤立节点记录：{len(df_isolated)}")

    # 保存孤立节点
    output_path = "temp/"
    fname = f"CNS_isolated_nodes_to_add_back_after_triclo_thr{abs(threshold)}_datetime.csv.gz"
    df_isolated.to_csv(
        output_path + fname,
        sep=',', header=True, index=False, compression='gzip'
    )
    print(f"  ✓ 保存到：{output_path}{fname}")

    return df_isolated


def merge_final_data(df_processed, df_isolated):
    """
    合并所有数据生成最终结果

    参数：
        df_processed: 处理后的 DataFrame
        df_isolated: 孤立节点 DataFrame

    返回：
        final_df: 最终的 DataFrame
    """
    print("\n" + "=" * 60)
    print("步骤 12: 合并最终数据")
    print("=" * 60)

    # 合并数据
    final_df = pd.concat([df_processed, df_isolated])
    print(f"  合并后记录数：{len(final_df)}")

    # 移除重复项（安全性检查）
    final_df = final_df.drop_duplicates(
        subset=['# timestamp', 'user_a', 'user_b', 'rssi']
    )
    print(f"  去重后记录数：{len(final_df)}")

    # 修复数据类型
    final_df['user_a'] = final_df['user_a'].astype(int)
    final_df['user_b'] = final_df['user_b'].astype(int)
    final_df['rssi'] = final_df['rssi'].astype(int)

    print("\n最终数据统计:")
    print(f"  总记录数：{len(final_df)}")
    print(f"  列名：{list(final_df.columns)}")
    print(f"  时间范围：{final_df['datetime'].min()} 到 {final_df['datetime'].max()}")

    # 保存最终结果
    output_path = "../data-processed/CNS/"
    fname = "CNS_bluetooth_processed.csv.gz"
    final_df.to_csv(
        output_path + fname,
        sep=',', header=True, index=False, compression='gzip'
    )
    print(f"\n  ✓ 最终结果保存到：{output_path}{fname}")

    return final_df


def main():
    """主函数：执行完整的 CNS 数据预处理流程"""
    print("\n" + "#" * 60)
    print("# CNS 数据集完整预处理流程")
    print("#" * 60)

    try:
        # 步骤 0: 创建目录
        create_directories()

        # 步骤 1: 加载和初步预处理
        df = load_and_preprocess_data()

        # 步骤 2: 映射用户 ID
        df, old_id_to_new, new_id_to_old = map_user_ids(df)

        # 步骤 3: 分析用户交互类型
        analyze_user_interactions(df)

        # 步骤 4: RSSI 过滤
        df = filter_rssi(df, threshold=-90)

        # 步骤 5: 填充单个时间间隙
        df = fill_single_gaps(df)

        # 步骤 6: 移除散点交互
        df_no_singletons, removed_indices = remove_scattered_interactions(df)

        # 步骤 7: 应用三角闭包
        df_triclo = apply_triadic_closure(df_no_singletons)

        # 步骤 8: 分析三角闭包阈值
        analyze_triclo_thresholds(df_no_singletons, df_triclo)

        # 步骤 9: 创建日期时间转换器
        dt_converter = create_datetime_converter(df_triclo)

        # 步骤 10: 使用选定阈值处理（这里使用 -75dBm）
        threshold = -75
        df_processed = process_with_threshold(
            df_no_singletons, df_triclo, dt_converter, threshold
        )

        # 步骤 11: 重新添加孤立节点
        df_isolated = add_back_isolated_nodes(
            df, df_processed, removed_indices, dt_converter, threshold
        )

        # 步骤 12: 合并最终数据
        final_df = merge_final_data(df_processed, df_isolated)

        print("\n" + "=" * 60)
        print("✓ CNS 数据预处理完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 错误：{e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
