"""
时序超图模型运行示例

演示如何：
- 初始化和运行时序超图模型
- 计算和可视化群组大小分布
- 从 DataFrame 保存和加载模型结果
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('../code/')

from code.model import TemporalHypergraphModel  # noqa: E402
from code.model import run_from_df_and_save_edgelists, read_edgelists_from_df  # noqa: E402
from code.model_analysis import group_size_dist, get_transition_matrix  # noqa: E402

def setup_matplotlib_fonts():
    """设置 Matplotlib 中文字体支持"""
    try:
        plt.rcParams['font.sans-serif'] = [
            'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans'
        ]
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f'警告：字体设置出现问题：{e}')
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

def plot_group_size_distribution(ks, Ps, filename, title='Group Size Distribution'):
    """绘制群组大小分布图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(ks, Ps)
    ax.set_yscale('log')
    ax.set_xlabel('Group Size (k)')
    ax.set_ylabel('Probability P(k)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f'图表已保存为：{filename}')
    plt.show()

def print_statistics(Hs, pars_dict, ks):
    """打印模型统计信息"""
    print('\n' + '=' * 50)
    print('统计信息:')
    print('=' * 50)
    print(f"总时间步数：{len(Hs)}")
    print(f"节点数量：{pars_dict['N']}")
    print(f"最大模拟时间：{pars_dict['t_max']}")
    print(f'不同的群组大小数量：{len(ks)}')
    print(f'最大群组大小：{max(ks) if ks else 0}')
    print(f'最小群组大小：{min(ks) if ks else 0}')

def run_model_example():
    """运行模型示例"""
    # 模型参数
    pars_dict = {
        'N': 300,
        't_max': 1000,
        'beta': 0.8,
        'epsilon': 10,
        'alpha': 0.25,
        'n0': 10,
        'L': 1,
        'verbose': False,
        'verbose_light': False
    }

    print('=' * 50)
    print('初始化模型...')
    print('=' * 50)

    # 创建并运行模型
    model = TemporalHypergraphModel()
    model.set_parameters(pars_dict)
    model.reset()

    print('开始运行模型...')
    Hs = model.run()
    print('模型运行完成!')

    # 计算并绘制群组大小分布
    print('\n计算群组大小分布...')
    ks, Ps = group_size_dist(Hs)
    plot_group_size_distribution(ks, Ps, 'group_size_distribution.png')

    # 打印统计信息
    print_statistics(Hs, pars_dict, ks)

    return pars_dict, Hs

def demonstrate_dataframe_workflow(pars_dict):
    """演示从 DataFrame 运行和保存模型"""
    print('\n' + '=' * 50)
    print('演示如何从 DataFrame 运行模型...')
    print('=' * 50)

    pars_df = pd.DataFrame.from_dict({k: [v] for k, v in pars_dict.items()})
    print('\n参数 DataFrame:')
    print(pars_df)

    # 创建结果目录
    output_dir = 'results/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'\n创建结果目录：{output_dir}')

    # 运行并保存边列表
    run_id = 0
    print(f'\n运行 ID {run_id} - 保存边列表...')
    run_from_df_and_save_edgelists(run_id, pars_df, output_dir)
    print('边列表保存完成!')

    # 读取保存的结果
    print('\n读取保存的边列表...')
    Hs_loaded = read_edgelists_from_df(run_id, pars_df, output_dir)
    print(f'成功读取 {len(Hs_loaded)} 个时间步的超图数据')

    # 验证读取的数据
    print('\n验证读取的数据...')
    ks_loaded, Ps_loaded = group_size_dist(Hs_loaded)
    plot_group_size_distribution(
        ks_loaded,
        Ps_loaded,
        'group_size_distribution_loaded.png',
        'Group Size Distribution (Loaded Data)'
    )

def main():
    """主函数"""
    # 设置字体
    setup_matplotlib_fonts()

    # 运行模型示例
    pars_dict, Hs = run_model_example()

    # 演示 DataFrame 工作流
    demonstrate_dataframe_workflow(pars_dict)

    print('\n' + '=' * 50)
    print('所有任务完成!')
    print('=' * 50)

if __name__ == '__main__':
    main()