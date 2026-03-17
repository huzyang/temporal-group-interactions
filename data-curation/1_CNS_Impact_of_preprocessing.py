"""
主要分析了 CNS（Complex Network Systems）数据预处理的影响，包括：
导入依赖库和设置 matplotlib 参数
创建必要的目录
计算三角闭包前后的群组
比较不同时间步的交互数量
可视化预处理的影响
输出文件：
results/CNS_groups_at_t_dict_after_triclo.p: 三角闭包后的群组数据
results/CNS_groups_at_t_dict_before_triclo.p: 三角闭包前的群组数据
figures/CNS_triclo_impact_comparison.pdf: 链接数量对比图
figures/CNS_group_size_distribution.pdf: 群组大小分布图
results/CNS_preprocessing_impact_summary.txt: 分析摘要报告
"""
import sys
import pandas as pd
import numpy as np
from collections import Counter
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# import palettable as pltt

# 导入自定义模块
sys.path.append("../code/")
from code.data_processing import groups_at_time_t

# 设置 matplotlib 参数
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.minor.size'] = 2
plt.rcParams['xtick.minor.width'] = 1.2
plt.rcParams['ytick.minor.size'] = 2
plt.rcParams['ytick.minor.width'] = 1.2
plt.rcParams['axes.linewidth'] = 1.2

import matplotlib as mplt

mplt.rcParams['figure.dpi'] = 300
mplt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mplt.rcParams['axes.unicode_minus'] = False


def create_directories():
    """创建必要的输出目录"""
    dirs_to_create = ["figures", "results", "temp"]
    for directory in dirs_to_create:
        if not os.path.exists(directory):
            os.makedirs(directory)
    print("已创建必要的输出目录")


def compute_groups_after_triclo():
    """计算三角闭包后的群组"""
    print("=" * 60)
    print("步骤 1: 计算三角闭包后的群组")
    print("=" * 60)

    input_path = "../data-processed/CNS/"
    fname = "CNS_bluetooth_processed.csv.gz"

    # 加载处理后的数据
    df = pd.read_csv(input_path + fname, sep=',')
    print(f"加载数据：{len(df)} 条交互记录")
    print(df.head())

    # 计算每个时间步的群组
    groups_at_t_dict = {}
    for timestamp in df['# timestamp'].unique():
        groups_at_t_dict[timestamp] = groups_at_time_t(df, timestamp)

    print(f"计算了 {len(groups_at_t_dict)} 个时间步的群组")

    # 保存结果
    output_path = "results/"
    fname_output = "CNS_groups_at_t_dict_after_triclo.p"
    pickle.dump(groups_at_t_dict, open(output_path + fname_output, "wb"))
    print(f"群组数据已保存到 {output_path + fname_output}")

    return groups_at_t_dict


def compute_groups_before_triclo():
    """计算三角闭包前的群组"""
    print("\n" + "=" * 60)
    print("步骤 2: 计算三角闭包前的群组")
    print("=" * 60)

    input_path = "temp/"
    fname = "CNS_bluetooth_thresholded_filled_and_nosingletons.csv.gz"

    # 加载三角闭包前的数据
    df = pd.read_csv(input_path + fname, sep=',')
    print(f"加载数据：{len(df)} 条交互记录")
    print(df.head())

    # 计算每个时间步的群组
    groups_at_t_dict = {}
    for timestamp in df['# timestamp'].unique():
        groups_at_t_dict[timestamp] = groups_at_time_t(df, timestamp)

    print(f"计算了 {len(groups_at_t_dict)} 个时间步的群组")

    # 保存结果
    output_path = "results/"
    fname_output = "CNS_groups_at_t_dict_before_triclo.p"
    pickle.dump(groups_at_t_dict, open(output_path + fname_output, "wb"))
    print(f"群组数据已保存到 {output_path + fname_output}")

    return groups_at_t_dict


def analyze_link_counts():
    """分析三角闭包前后添加的链接数量"""
    print("\n" + "=" * 60)
    print("步骤 3: 分析三角闭包前后添加的链接数量")
    print("=" * 60)

    # 三角闭包前
    output_path = "temp/"
    fname_before = "CNS_bluetooth_thresholded_filled_and_nosingletons.csv.gz"
    df_before = pd.read_csv(output_path + fname_before)

    # 移除孤立节点
    df_before = df_before[df_before['user_b'] != -1]

    # 统计每个时间步的交互数量
    df_count_before = df_before.groupby('# timestamp').size().reset_index(name='counts')
    print("三角闭包前（前 5 个时间步）:")
    print(df_count_before.head())

    # 三角闭包后（使用选定的阈值 φ=-75）
    th = -75
    fname_after = f"CNS_bluetooth_thresholded_filled_nosingletons_and_triclo_thr{abs(th)}_datetime.csv.gz"
    df_after = pd.read_csv(output_path + fname_after)

    # 移除孤立节点
    df_after = df_after[df_after['user_b'] != -1]

    # 统计每个时间步的交互数量
    df_count_after = df_after.groupby('# timestamp').size().reset_index(name='counts')
    print(f"\n三角闭包后（阈值={th}，前 5 个时间步）:")
    print(df_count_after.head())

    # 计算差异
    merged_counts = pd.merge(
        df_count_before,
        df_count_after,
        on='# timestamp',
        suffixes=('_before', '_after')
    )
    merged_counts['difference'] = merged_counts['counts_after'] - merged_counts['counts_before']

    print(f"\n平均每个时间步添加的链接数：{merged_counts['difference'].mean():.2f}")
    print(f"最大添加链接数：{merged_counts['difference'].max()}")
    print(f"最小添加链接数：{merged_counts['difference'].min()}")

    return df_count_before, df_count_after, merged_counts


def plot_link_comparison(df_count_before, df_count_after, merged_counts):
    """绘制三角闭包前后链接数量对比图"""
    print("\n" + "=" * 60)
    print("步骤 4: 绘制链接数量对比图")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制三角闭包前后的链接数量
    ax.plot(df_count_before['# timestamp'], df_count_before['counts'],
            label='Before Triadic Closure', alpha=0.7, linewidth=2)
    ax.plot(df_count_after['# timestamp'], df_count_after['counts'],
            label=f'After Triadic Closure (φ=-75)', alpha=0.7, linewidth=2)

    ax.set_xlabel('Timestamp', fontsize=12)
    ax.set_ylabel('Number of Links', fontsize=12)
    ax.set_title('Impact of Triadic Closure on Network Links', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 保存图表
    output_path = "figures/"
    fname_output = "CNS_triclo_impact_comparison.pdf"
    plt.tight_layout()
    plt.savefig(output_path + fname_output, dpi=300, bbox_inches='tight')
    print(f"对比图已保存到 {output_path + fname_output}")
    plt.show()


def visualize_group_statistics(groups_before, groups_after):
    """可视化群组统计信息"""
    print("\n" + "=" * 60)
    print("步骤 5: 可视化群组统计信息")
    print("=" * 60)

    # 计算群组大小统计
    group_sizes_before = []
    for timestamp, groups in groups_before.items():
        for group in groups:
            if len(group) > 1:  # 只考虑非单元素群组
                group_sizes_before.append(len(group))

    group_sizes_after = []
    for timestamp, groups in groups_after.items():
        for group in groups:
            if len(group) > 1:  # 只考虑非单元素群组
                group_sizes_after.append(len(group))

    # 绘制群组大小分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(group_sizes_before, bins=range(1, max(group_sizes_before) + 2),
                 alpha=0.7, label='Before Triadic Closure', edgecolor='black')
    axes[0].set_xlabel('Group Size', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Group Size Distribution (Before)', fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(group_sizes_after, bins=range(1, max(group_sizes_after) + 2),
                 alpha=0.7, label='After Triadic Closure', color='orange', edgecolor='black')
    axes[1].set_xlabel('Group Size', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Group Size Distribution (After)', fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 保存图表
    output_path = "figures/"
    fname_output = "CNS_group_size_distribution.pdf"
    plt.tight_layout()
    plt.savefig(output_path + fname_output, dpi=300, bbox_inches='tight')
    print(f"群组大小分布图已保存到 {output_path + fname_output}")
    plt.show()

    # 打印统计信息
    print(f"\n三角闭包前:")
    print(f"  平均群组大小：{np.mean(group_sizes_before):.2f}")
    print(f"  最大群组大小：{max(group_sizes_before)}")
    print(f"  总群组数：{len(group_sizes_before)}")

    print(f"\n三角闭包后:")
    print(f"  平均群组大小：{np.mean(group_sizes_after):.2f}")
    print(f"  最大群组大小：{max(group_sizes_after)}")
    print(f"  总群组数：{len(group_sizes_after)}")


def generate_summary_report(df_count_before, df_count_after, groups_before, groups_after):
    """生成分析摘要报告"""
    print("\n" + "=" * 60)
    print("摘要报告")
    print("=" * 60)

    report = []
    report.append("CNS 数据预处理影响分析")
    report.append("=" * 60)

    # 链接统计
    total_links_before = df_count_before['counts'].sum()
    total_links_after = df_count_after['counts'].sum()
    link_increase = total_links_after - total_links_before
    percentage_increase = (link_increase / total_links_before) * 100 if total_links_before > 0 else 0

    report.append(f"\n1. 链接数量统计:")
    report.append(f"   - 三角闭包前总链接数：{total_links_before}")
    report.append(f"   - 三角闭包后总链接数：{total_links_after}")
    report.append(f"   - 增加的链接数：{link_increase} ({percentage_increase:.2f}%)")

    # 群组统计
    num_groups_before = sum(len(groups) for groups in groups_before.values())
    num_groups_after = sum(len(groups) for groups in groups_after.values())

    report.append(f"\n2. 群组数量统计:")
    report.append(f"   - 三角闭包前总群组数：{num_groups_before}")
    report.append(f"   - 三角闭包后总群组数：{num_groups_after}")

    # 结论
    report.append(f"\n3. 结论:")
    report.append(f"   - 三角闭包显著增加了网络中的链接数量")
    report.append(f"   - 这可能导致更大的群组和更密集的网络结构")
    report.append(f"   - 对于动态群体交互分析，三角闭包是一个重要的预处理步骤")

    # 打印报告
    for line in report:
        print(line)

    # 保存报告
    output_path = "results/"
    fname_output = "CNS_preprocessing_impact_summary.txt"
    with open(output_path + fname_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"\n摘要报告已保存到 {output_path + fname_output}")


def main():
    """主函数：执行完整的 CNS 预处理影响分析流程"""
    print("开始 CNS 数据预处理影响分析...")
    print("=" * 60)

    try:
        # 创建目录
        create_directories()

        # 计算三角闭包前后的群组
        groups_after = compute_groups_after_triclo()
        groups_before = compute_groups_before_triclo()

        # 分析链接数量
        df_count_before, df_count_after, merged_counts = analyze_link_counts()

        # 绘制对比图
        plot_link_comparison(df_count_before, df_count_after, merged_counts)

        # 可视化群组统计
        visualize_group_statistics(groups_before, groups_after)

        # 生成摘要报告
        generate_summary_report(df_count_before, df_count_after, groups_before, groups_after)

        print("\n" + "=" * 60)
        print("CNS 数据预处理影响分析完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n发生错误：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
