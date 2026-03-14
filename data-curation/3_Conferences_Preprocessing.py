# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
会议数据预处理脚本

输出文件
../data-processed/Confs/conf*_processed.csv.gz - 4 个会议的处理后数据
figures/Confs_raw_data_distribution.pdf - 原始数据时间分布图
results/Confs_preprocessing_summary.txt - 摘要报告
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def setup_directories():
    """
    创建必要的输出目录
    """
    print("=" * 60)
    print("步骤 1: 创建输出目录")
    print("=" * 60)

    dirs_to_create = ["figures", "results", "temp", "../data-processed/Confs/"]

    for directory in dirs_to_create:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ 创建目录：{directory}")
        else:
            print(f"✓ 目录已存在：{directory}")

    print()


def load_and_visualize_data(root_dir="../data-raw/Confs/"):
    """
    加载会议数据并可视化时间分布

    Args:
        root_dir: 数据根目录

    Returns:
        dict: 包含原始数据的字典
    """
    print("=" * 60)
    print("步骤 2: 加载并可视化原始数据")
    print("=" * 60)

    contexts = ["conf16", "conf17", "conf18", "conf19"]
    conferences = {
        "conf16": "WS16",
        "conf17": "ICCSS17",
        "conf18": "ECSS18",
        "conf19": "ECIR19"
    }
    colors = {
        "conf16": "#a6cee3",
        "conf17": "#1f78b4",
        "conf18": "#b2df8a",
        "conf19": "#33a02c"
    }

    dfs = {}

    print("加载原始数据...")
    for context in contexts:
        df = pd.read_csv(
            root_dir + "%s.txt" % context,
            names=["timestamp", "user_a", "user_b"],
            sep=" "
        )
        dfs[context] = df
        print(f"  {context} ({conferences[context]}): {len(df):,} 条记录")

    print("\n可视化时间分布...")
    plt.figure(figsize=(12, 12))

    for i, context in enumerate(contexts):
        df = dfs[context]
        ax = plt.subplot(4, 2, i + 1)
        df.groupby("timestamp").size().plot(ax=ax, color=colors[context])
        ax.set_ylabel("Number of pairwise links")
        ax.set_title(conferences[context])

    plt.tight_layout()
    plt.savefig("figures/Confs_raw_data_distribution.pdf", bbox_inches='tight')
    print("✓ 图表已保存至：figures/Confs_raw_data_distribution.pdf")
    plt.close()
    print()

    return dfs


def temporal_aggregation(dfs, window_sampling=25):
    """
    时间聚合 - 按采样窗口聚合时间戳

    说明：数据已被重新缩放，1 个时间步增量对应 20 秒
    聚合到 5 分钟窗口 = 25 × 20 秒

    Args:
        dfs: 原始数据框字典
        window_sampling: 采样窗口大小（默认 25，即 5 分钟）

    Returns:
        dict: 聚合后的数据框字典
    """
    print("=" * 60)
    print(f"步骤 3: 时间聚合 (窗口={window_sampling}×20 秒={window_sampling * 20}秒)")
    print("=" * 60)

    aggregated_dfs = {}

    for context, df in dfs.items():
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

        aggregated_dfs[context] = df

        removed = initial_count - final_count
        print(f"{context}:")
        print(f"  聚合前：{initial_count:,} 条记录")
        print(f"  聚合后：{final_count:,} 条记录")
        print(f"  移除重复：{removed:,} 条记录 ({removed / initial_count * 100:.1f}%)")

    print()
    return aggregated_dfs


def add_isolated_nodes(dfs):
    """
    添加孤立节点（用 user_b=-1 表示）

    重要说明：节点在不同时间加入数据集，因此只在节点首次出现后
    才开始添加为孤立节点

    Args:
        dfs: 聚合后的数据框字典

    Returns:
        dict: 添加了孤立节点的数据框字典
    """
    print("=" * 60)
    print("步骤 4: 添加孤立节点")
    print("=" * 60)

    processed_dfs = {}

    for context, df in dfs.items():
        print(f"\n处理 {context}...")

        entries_to_add = []

        active_nodes = set(df['user_a'].unique()).union(set(df['user_b'].unique()))
        active_nodes_up_to_t = set()

        timestamps = sorted(df['timestamp'].unique())
        total_timestamps = len(timestamps)

        print(f"  总节点数：{len(active_nodes)}")
        print(f"  总时间步数：{total_timestamps}")

        for idx, t in enumerate(timestamps):
            df_t = df[df['timestamp'] == t]
            active_nodes_t = set(df_t['user_a'].unique()).union(set(df_t['user_b'].unique()))

            active_nodes_up_to_t = active_nodes_up_to_t.union(active_nodes_t)

            missing_nodes_t = active_nodes.difference(active_nodes_t)
            isolated_nodes_t = missing_nodes_t.intersection(active_nodes_up_to_t)

            for iso_t in isolated_nodes_t:
                entries_to_add.append([t, iso_t, -1])

            if (idx + 1) % 1000 == 0 or (idx + 1) == total_timestamps:
                print(f"  已处理 {idx + 1}/{total_timestamps} 个时间步 "
                      f"(当前活跃节点：{len(active_nodes_up_to_t)}/{len(active_nodes)})",
                      end='\r')

        print()

        df_isolates = pd.DataFrame(
            columns=["timestamp", "user_a", "user_b"],
            data=entries_to_add
        )

        initial_count = len(df)
        isolates_count = len(df_isolates)

        df_with_isolates = pd.concat([df, df_isolates], ignore_index=True)
        df_with_isolates = df_with_isolates.reset_index(drop=True)

        final_count = len(df_with_isolates)

        processed_dfs[context] = df_with_isolates

        print(f"  原始记录数：{initial_count:,}")
        print(f"  孤立节点记录数：{isolates_count:,}")
        print(f"  合并后总记录数：{final_count:,}")
        print(f"  孤立节点占比：{isolates_count / final_count * 100:.1f}%")

    print()
    return processed_dfs


def save_results(dfs, output_dir='../data-processed/Confs/'):
    """
    保存处理后的结果

    Args:
        dfs: 处理后的数据框字典
        output_dir: 输出目录
    """
    print("=" * 60)
    print("步骤 5: 保存结果")
    print("=" * 60)

    for context, df in dfs.items():
        filename = output_dir + '%s_processed.csv.gz' % context
        df.to_csv(filename, index=False, header=True, compression="gzip")
        print(f"✓ {context}: 已保存至 {filename} ({len(df):,} 条记录)")

    print()


def verify_results(contexts, output_dir='../data-processed/Confs/'):
    """
    验证保存的结果

    Args:
        contexts: 上下文列表
        output_dir: 输出目录
    """
    print("=" * 60)
    print("步骤 6: 验证结果")
    print("=" * 60)

    for context in contexts:
        df = pd.read_csv(output_dir + '%s_processed.csv.gz' % context)
        print(f"{context}: {len(df):,} 条记录")

    print()


def generate_summary_report(dfs):
    """
    生成摘要报告

    Args:
        dfs: 最终处理后的数据框字典
    """
    print("=" * 60)
    print("生成摘要报告")
    print("=" * 60)

    report = []
    report.append("Conferences 数据预处理摘要报告")
    report.append("=" * 60)
    report.append("")

    total_records = 0
    total_isolated = 0

    for context, df in dfs.items():
        records = len(df)
        total_records += records

        isolated = (df['user_b'] == -1).sum()
        total_isolated += isolated

        unique_users = set(df['user_a'].unique()).union(set(df['user_b'].unique()))
        unique_timestamps = df['timestamp'].nunique()

        report.append(f"{context}:")
        report.append(f"  总记录数：{records:,}")
        report.append(f"  唯一用户数：{len(unique_users)}")
        report.append(f"  唯一时间步数：{unique_timestamps:,}")
        report.append(f"  孤立节点记录数：{isolated:,} ({isolated / records * 100:.1f}%)")
        report.append("")

    report.append("总计:")
    report.append(f"  总记录数：{total_records:,}")
    report.append(f"  总孤立节点记录数：{total_isolated:,} ({total_isolated / total_records * 100:.1f}%)")
    report.append("")
    report.append("=" * 60)

    report_text = "\n".join(report)
    print(report_text)

    with open('results/Confs_preprocessing_summary.txt', 'w') as f:
        f.write(report_text)

    print("\n✓ 摘要报告已保存：results/Confs_preprocessing_summary.txt")
    print()


def main():
    """
    主函数：执行完整的 Conferences 数据预处理流程
    """
    print("\n" + "=" * 60)
    print("Conferences 数据预处理流程启动")
    print("=" * 60 + "\n")

    try:
        setup_directories()

        contexts = ["conf16", "conf17", "conf18", "conf19"]

        dfs = load_and_visualize_data(root_dir="../data-raw/Confs/")

        dfs = temporal_aggregation(dfs, window_sampling=25)

        dfs = add_isolated_nodes(dfs)

        save_results(dfs, output_dir='../data-processed/Confs/')

        verify_results(contexts, output_dir='../data-processed/Confs/')

        generate_summary_report(dfs)

        print("=" * 60)
        print("✓ Conferences 数据预处理流程完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
