# coding: utf-8
"""
Temporal Group Interactions Model Code Package

这个包包含了时间群组交互模型的所有核心代码模块。
"""

# 导入主要模块
from .model import (
    TemporalHypergraphModel,
    run_from_df_and_save_edgelists,
    read_edgelists_from_df
)

from .model_analysis import (
    group_size_dist,
    get_transition_matrix,
    transition_matrix_to_df,
    get_group_durations,
    get_group_times,
    get_dis_agg_matrices,
    get_full_dis_agg_matrices,
    dis_agg_matrix_to_df
)

from .fitting import (
    compute_JSD,
    fit_gsize_dist,
    compute_JSD_trans_mat,
    fit_gtrans_mat,
    extract_matrix_vector,
    get_distance_from_vec
)

from .data_analysis import (
    measure_social_memory,
    get_interevent_times
)

from .data_processing import (
    LoadData,
    get_triadic_closure_links_to_add_at_time_t
)

from .utils import (
    logistic,
    normalize_dict,
    normalize_by_row,
    get_jaccard,
    change_width,
    centered_np_hist,
    get_cumulative_Gs_from_Hs,
    get_groups_dict_from_Hs
)

# 包版本信息
__version__ = '1.0.0'
__all__ = [
    # 模型核心类
    'TemporalHypergraphModel',

    # 模型运行和读取函数
    'run_from_df_and_save_edgelists',
    'read_edgelists_from_df',

    # 模型分析函数
    'group_size_dist',
    'get_transition_matrix',
    'transition_matrix_to_df',
    'get_group_durations',
    'get_group_times',
    'get_dis_agg_matrices',
    'get_full_dis_agg_matrices',
    'dis_agg_matrix_to_df',

    # 拟合函数
    'compute_JSD',
    'fit_gsize_dist',
    'compute_JSD_trans_mat',
    'fit_gtrans_mat',
    'extract_matrix_vector',
    'get_distance_from_vec',

    # 数据分析函数
    'measure_social_memory',
    'get_interevent_times',

    # 工具函数
    'logistic',
    'normalize_dict',
    'normalize_by_row',
    'get_jaccard',
    'change_width',
    'centered_np_hist',
    'get_cumulative_Gs_from_Hs',
    'get_groups_dict_from_Hs',
]
