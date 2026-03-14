# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DyLNet 数据预处理脚本
=====================
功能：对 DyLNet 数据集进行预处理，包括数据合并、清洗、时间聚合和孤立节点添加
输出文件
temp/DyLNet_processed_step*.csv.gz - 中间结果
../data-processed/DyLNet/DyLNet_processed.csv.gz - 最终结果
figures/DyLNet_duration_distribution.pdf - 可视化图表
results/DyLNet_preprocessing_summary.txt - 摘要报告
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns


def setup_directories():
    """
    创建必要的输出目录
    """
    print("=" * 60)
    print("步骤 1: 创建输出目录")
    print("=" * 60)

    dirs_to_create = ["figures", "results", "temp", "../data-processed/DyLNet/"]

    for directory in dirs_to_create:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ 创建目录：{directory}")
        else:
            print(f"✓ 目录已存在：{directory}")

    print()


def load_and_merge_data():
    """
    加载并合并所有 DyLNet 数据文件

    Returns:
        DataFrame: 合并后的数据框
    """
    print("=" * 60)
    print("步骤 2: 加载并合并数据文件")
    print("=" * 60)

    root_dir = "../data-raw/DyLNet/"

    weeks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    week_filenames = [
        "1_WEEK39/", "1_WEEK41/", "1_WEEK46/", "1_WEEK50/",
        "2_WEEK03/", "2_WEEK06/", "2_WEEK11/", "2_WEEK14/",
        "2_WEEK20/", "2_WEEK25/"
    ]
    week_dir = {w: fn for w, fn in zip(weeks, week_filenames)}

    dir_tnets = "HD_tnet_reconstructed/"
    days = [1, 2, 3, 4, 5]
    days_ma = ["M", "A"]
    contexts = ["in-class", "out-of-class"]
    context_labels = {
        "in-class": ["CCCC1"],
        "out-of-class": ["FFFF1", "FFFF0"]
    }

    dfs = []
    total_files = len(weeks) * len(days) * len(days_ma) * len(contexts)
    processed_count = 0

    print(f"开始处理 {total_files} 个文件...")

    for week in weeks:
        for day in days:
            for ma in days_ma:
                for context in contexts:
                    fname = (root_dir + week_dir[week] + dir_tnets +
                             "%i-%s-blstm_RSSI.csv" % (day, ma))
                    cnames = ['timestamp', 'user_a', 'user_b', 'duration',
                              'interaction_type']

                    try:
                        df = pd.read_csv(fname, names=cnames)

                        df = df[df['interaction_type'].isin(context_labels[context])]
                        df.drop('interaction_type', axis=1, inplace=True)

                        df['week'] = week
                        df['day'] = day
                        df['morning-afternoon'] = {'M': 'morning', 'A': 'afternoon'}[ma]
                        df['context'] = context

                        dfs.append(df)
                        processed_count += 1

                        if processed_count % 20 == 0:
                            print(f"  已处理 {processed_count}/{total_files} 个文件")

                    except FileNotFoundError:
                        continue

    df = pd.concat(dfs, axis=0, ignore_index=True)

    print(f"\n✓ 成功合并 {len(df):,} 条记录")
    print(f"  列：{list(df.columns)}")
    print(f"  时间范围：{df['timestamp'].min()} - {df['timestamp'].max()}")
    print()

    return df


def visualize_duration_distribution(df):
    """
    可视化持续时间分布

    Args:
        df: 输入数据框
    """
    print("可视化持续时间分布...")

    plt.figure(figsize=(10, 6))
    df.hist('duration')
    plt.yscale('log')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency (log scale)')
    plt.title('Distribution of Interaction Durations')
    plt.savefig('figures/DyLNet_duration_distribution.pdf', bbox_inches='tight')
    print("✓ 图表已保存至：figures/DyLNet_duration_distribution.pdf")
    plt.close()
    print()


def remove_adults(df):
    """
    移除成人（教师）数据

    Args:
        df: 输入数据框

    Returns:
        DataFrame: 过滤后的数据框
    """
    print("=" * 60)
    print("步骤 3: 数据清洗 - 移除成人数据")
    print("=" * 60)

    initial_count = len(df)
    df = df[(df['user_a'] < 1000) & (df['user_b'] < 1000)]
    final_count = len(df)

    print(f"移除前记录数：{initial_count:,}")
    print(f"移除后记录数：{final_count:,}")
    print(f"移除了 {initial_count - final_count:,} 条涉及成人的记录")
    print()

    return df


def add_same_class_info(df, root_dir="../data-raw/DyLNet/"):
    """
    添加同班级信息列

    Args:
        df: 输入数据框
        root_dir: 数据根目录

    Returns:
        DataFrame: 添加了 same_class 列的数据框
    """
    print("添加同班级信息...")

    info = pd.read_csv(root_dir + 'SOCIODEMOLING/SocioDemoLing_Data.csv', sep=';')
    info.set_index('ID', inplace=True)

    def check_same_class(i, j):
        try:
            return info.loc[i]['class'] == info.loc[j]['class']
        except KeyError:
            return False

    df['same_class'] = df.apply(
        lambda row: check_same_class(row.user_a, row.user_b),
        axis=1
    )

    print(f"✓ 已添加 same_class 列")
    print(f"  同班级互动：{(df['same_class'] == True).sum():,} 条")
    print(f"  跨班级互动：{(df['same_class'] == False).sum():,} 条")
    print()

    return df


def remove_cross_class_interactions(df):
    """
    移除跨班级的课内互动

    Args:
        df: 输入数据框

    Returns:
        DataFrame: 过滤后的数据框
    """
    print("=" * 60)
    print("步骤 4: 移除跨班级的课内互动")
    print("=" * 60)

    initial_count = len(df)
    df = df[~((df['context'] == 'in-class') & (df['same_class'] == False))]
    df.drop('same_class', axis=1, inplace=True)
    final_count = len(df)

    print(f"移除前记录数：{initial_count:,}")
    print(f"移除后记录数：{final_count:,}")
    print(f"移除了 {initial_count - final_count:,} 条跨班级课内互动记录")
    print()

    return df


def filter_by_duration(df, min_duration=10):
    """
    按持续时间过滤数据

    Args:
        df: 输入数据框
        min_duration: 最小持续时间（秒）

    Returns:
        DataFrame: 过滤后的数据框
    """
    print("=" * 60)
    print(f"步骤 5: 过滤短时互动 (< {min_duration}秒)")
    print("=" * 60)

    initial_count = len(df)
    df = df[df['duration'] >= min_duration]
    final_count = len(df)

    print(f"移除前记录数：{initial_count:,}")
    print(f"移除后记录数：{final_count:,}")
    print(f"移除了 {initial_count - final_count:,} 条短时互动记录")
    print()

    return df


def expand_duration_to_timesteps(df):
    """
    将持续时间展开为时间步长（创建非平衡面板）

    Args:
        df: 输入数据框

    Returns:
        DataFrame: 展开后的数据框
    """
    print("=" * 60)
    print("步骤 6: 将持续时间展开为时间步长")
    print("=" * 60)

    initial_count = len(df)

    ext_df = df.loc[df.index.repeat(df.duration)]
    ext_df.drop('duration', axis=1, inplace=True)

    ext_df['increment'] = ext_df.groupby(ext_df.index).timestamp.cumcount()
    ext_df['timestamp'] = ext_df['timestamp'] + ext_df['increment']
    ext_df.drop('increment', axis=1, inplace=True)
    ext_df = ext_df.reset_index(drop=True)

    final_count = len(ext_df)

    print(f"展开前记录数：{initial_count:,}")
    print(f"展开后记录数：{final_count:,}")
    print(f"平均每条记录展开为 {final_count / initial_count:.1f} 个时间步")
    print()

    return ext_df


def temporal_aggregation(df, window_sampling=10):
    """
    时间聚合 - 按采样窗口聚合时间戳

    Args:
        df: 输入数据框
        window_sampling: 采样窗口大小（秒）

    Returns:
        DataFrame: 聚合后的数据框
    """
    print("=" * 60)
    print(f"步骤 7: 时间聚合 (窗口={window_sampling}秒)")
    print("=" * 60)

    initial_count = len(df)

    df['agg_timestamp'] = df.apply(
        lambda row: round(row.timestamp / window_sampling),
        axis=1
    )

    df.drop('timestamp', axis=1, inplace=True)
    df.rename(columns={"agg_timestamp": "timestamp"}, inplace=True)

    df.drop_duplicates(inplace=True)
    df = df.reset_index(drop=True)

    final_count = len(df)

    print(f"聚合前记录数：{initial_count:,}")
    print(f"聚合后记录数：{final_count:,}")
    print(f"去重了 {initial_count - final_count:,} 条重复记录")
    print()

    return df


def add_isolated_nodes(df, window_sampling=10):
    """
    添加孤立节点（用 user_b=-1 表示）

    Args:
        df: 输入数据框
        window_sampling: 采样窗口大小（秒）

    Returns:
        DataFrame: 添加了孤立节点的数据框
    """
    print("=" * 60)
    print("步骤 8: 添加孤立节点")
    print("=" * 60)

    entries_to_add = []

    weeks = sorted(df['week'].unique())
    days = sorted(df['day'].unique())

    total_periods = len(weeks) * len(days) * 2 * 2
    processed_periods = 0

    print(f"开始处理 {total_periods} 个时间段...")

    for week in weeks:
        print(f"\n  处理周 {week}:")
        for day in days:
            for ma in df['morning-afternoon'].unique():
                for context in df['context'].unique():
                    dfx = df[
                        (df['week'] == week) &
                        (df['day'] == day) &
                        (df['morning-afternoon'] == ma) &
                        (df['context'] == context)
                        ]

                    if len(dfx) == 0:
                        continue

                    active_nodes = set(dfx['user_a'].unique()).union(
                        set(dfx['user_b'].unique())
                    )

                    for t in dfx['timestamp'].unique():
                        dfx_t = dfx[dfx['timestamp'] == t]
                        active_nodes_t = set(dfx_t['user_a'].unique()).union(
                            set(dfx_t['user_b'].unique())
                        )
                        isolated_nodes_t = active_nodes.difference(active_nodes_t)

                        for iso_t in isolated_nodes_t:
                            entries_to_add.append([
                                iso_t, -1, week, day, ma, context, t
                            ])

                    processed_periods += 1
                    if processed_periods % 10 == 0:
                        print(f"    已处理 {processed_periods}/{total_periods} 个时间段", end='\r')

    print(f"\n  ✓ 完成所有时间段处理")

    df_isolates = pd.DataFrame(
        columns=['user_a', 'user_b', 'week', 'day', 'morning-afternoon',
                 'context', 'timestamp'],
        data=entries_to_add
    )

    initial_count = len(df)
    isolates_count = len(df_isolates)

    df_with_isolates = pd.concat([df, df_isolates], ignore_index=True)
    df_with_isolates = df_with_isolates.reset_index(drop=True)

    final_count = len(df_with_isolates)

    print(f"\n原始记录数：{initial_count:,}")
    print(f"孤立节点记录数：{isolates_count:,}")
    print(f"合并后总记录数：{final_count:,}")
    print(f"添加了 {isolates_count:,} 条孤立节点记录 ({isolates_count / final_count * 100:.1f}%)")
    print()

    return df_with_isolates


def save_intermediate_results(df, step_name, compression="gzip"):
    """
    保存中间结果

    Args:
        df: 要保存的数据框
        step_name: 步骤名称
        compression: 压缩方式
    """
    filename = f'temp/DyLNet_processed_{step_name}.csv.gz'
    df.to_csv(filename, index=False, header=True, compression=compression)
    print(f"✓ 中间结果已保存：{filename} ({len(df):,} 条记录)")


def save_final_results(df, filename='../data-processed/DyLNet/DyLNet_processed.csv.gz'):
    """
    保存最终结果

    Args:
        df: 要保存的数据框
        filename: 输出文件名
    """
    df.to_csv(filename, index=False, header=True, compression="gzip")
    print(f"✓ 最终结果已保存：{filename}")
    print(f"  记录数：{len(df):,}")
    print(f"  列：{list(df.columns)}")
    print()


def generate_summary_report(df):
    """
    生成摘要报告

    Args:
        df: 最终处理后的数据框
    """
    print("=" * 60)
    print("生成摘要报告")
    print("=" * 60)

    report = []
    report.append("DyLNet 数据预处理摘要报告")
    report.append("=" * 60)
    report.append("")
    report.append(f"总记录数：{len(df):,}")
    report.append(f"列数：{len(df.columns)}")
    report.append(f"列名：{', '.join(df.columns)}")
    report.append("")

    report.append("数据统计:")
    report.append(f"  - 唯一用户数 (user_a): {df['user_a'].nunique():,}")
    report.append(f"  - 唯一用户数 (user_b): {df['user_b'].nunique():,}")
    report.append(f"  - 唯一时间戳数：{df['timestamp'].nunique():,}")
    report.append(f"  - 周数：{df['week'].nunique()}")
    report.append(f"  - 天数：{df['day'].nunique()}")
    report.append("")

    report.append("上下文分布:")
    context_counts = df['context'].value_counts()
    for context, count in context_counts.items():
        report.append(f"  - {context}: {count:,} ({count / len(df) * 100:.1f}%)")
    report.append("")

    report.append("时间段分布:")
    ma_counts = df['morning-afternoon'].value_counts()
    for ma, count in ma_counts.items():
        report.append(f"  - {ma}: {count:,} ({count / len(df) * 100:.1f}%)")
    report.append("")

    isolated_nodes = (df['user_b'] == -1).sum()
    report.append(f"孤立节点记录数：{isolated_nodes:,} ({isolated_nodes / len(df) * 100:.1f}%)")
    report.append("")
    report.append("=" * 60)

    report_text = "\n".join(report)
    print(report_text)

    with open('results/DyLNet_preprocessing_summary.txt', 'w') as f:
        f.write(report_text)

    print("\n✓ 摘要报告已保存：results/DyLNet_preprocessing_summary.txt")
    print()


def main():
    """
    主函数：执行完整的 DyLNet 数据预处理流程
    """
    print("\n" + "=" * 60)
    print("DyLNet 数据预处理流程启动")
    print("=" * 60 + "\n")

    try:
        setup_directories()

        df = load_and_merge_data()

        visualize_duration_distribution(df)

        save_intermediate_results(df, "step1_raw_merged")

        df = remove_adults(df)
        save_intermediate_results(df, "step2_no_adults")

        df = add_same_class_info(df)

        df = remove_cross_class_interactions(df)
        save_intermediate_results(df, "step3_no_cross_class")

        df = filter_by_duration(df, min_duration=10)
        save_intermediate_results(df, "step4_filtered_duration")

        df = expand_duration_to_timesteps(df)
        save_intermediate_results(df, "step5_expanded")

        df = temporal_aggregation(df, window_sampling=10)
        save_intermediate_results(df, "step6_aggregated_10sec")

        df = add_isolated_nodes(df, window_sampling=10)

        save_final_results(
            df,
            filename='../data-processed/DyLNet/DyLNet_processed.csv.gz'
        )

        generate_summary_report(df)

        print("=" * 60)
        print("✓ DyLNet 数据预处理流程完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
