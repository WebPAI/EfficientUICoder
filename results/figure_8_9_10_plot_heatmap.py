import matplotlib
# matplotlib.use('TKAgg')
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import seaborn as sns
import pandas as pd

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap_step_lamda(dataset_name, metric_name):
    # 读取JSON文件
    with open(f'data/{dataset_name}_para.json', 'r') as f:
        data = json.load(f)

    # 提取step和lambda值
    steps = sorted([int(step) for step in data.keys()])
    lambdas = sorted([float(lam) for lam in data[list(data.keys())[0]].keys()], reverse=True)

    # 创建clip_score矩阵
    clip_scores = []
    for step in steps:
        row = []
        for lam in lambdas:
            # clip_score = data[str(step)][str(lam)]['clip_score']
            clip_score = data[str(step)][str(lam)][metric_name]
            # clip_score = round(data[str(step)][str(lam)][metric_name])
            row.append(clip_score)
        clip_scores.append(row)

    # 转换为numpy数组
    clip_scores_array = np.array(clip_scores)

    # 创建DataFrame用于更好的标签显示
    # df = pd.DataFrame(clip_scores_array,
    #                   index=[f'Step {step}' for step in steps],
    #                   columns=[f'λ={lam}' for lam in lambdas])

    # lambdas = ["$\\frac{7}{8}$", "3/4","1/2", "1/3"]

    lambdas = ["$\\frac{7}{8}$", "$\\frac{3}{4}$", "$\\frac{1}{2}$", "$\\frac{1}{3}$"]
    df = pd.DataFrame(clip_scores_array,
                      index=[f'{step}' for step in steps],
                      columns=[f'{lam}' for lam in lambdas])

    plt.figure(figsize=(10, 8))
    colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', '#1E88E5', '#1976D2', '#1565C0',
              '#0D47A1']
    custom_cmap = plt.cm.Blues


    if metric_name == "avg_time":
        heatmap = sns.heatmap(df,
                              annot=True,  # 显示数值
                              # fmt='.4f',  # 数值格式
                              fmt='.2f',  # 数值格式
                              cmap='Greens',  # 颜色映射
                              # cmap='YlGnBu',  # 颜色映射
                              # cbar_kws={'label': 'CLIP Score'},
                              linewidths=0.5,
                              annot_kws={'fontsize': 30}
                              )
    else:
        heatmap = sns.heatmap(df,
                              annot=True,  # 显示数值
                              fmt='.4f',  # 数值格式
                              # fmt='.2f',  # 数值格式
                              # cmap='YlOrRd',  # 颜色映射
                              cmap='YlGnBu',  # 颜色映射
                              # cbar_kws={'label': 'CLIP Score'},
                              linewidths=0.5,
                              annot_kws={'fontsize': 30}
                              )



    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([])  # 移除所有刻度
    cbar.set_label('')  # 移除标签
    # cbar.ax.tick_params(labelsize=30)
    # plt.title('CLIP Score Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Decay Factor (λ)', fontsize=30)
    plt.ylabel('Suppression Steps (s)', fontsize=30)

    # 旋转x轴标签以避免重叠
    # plt.xticks(fontsize=20, rotation=45)
    plt.xticks(fontsize=30, rotation=0)
    plt.yticks(fontsize=30, rotation=0)

    # 调整布局
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_{metric_name}_para.pdf")
    # 显示图形
    plt.show()

    # 可选：保存图形
    # plt.savefig('clip_score_heatmap.png', dpi=300, bbox_inches='tight')

    # 打印一些统计信息
    print(f"{metric_name} 统计信息:")
    print(f"最小值: {clip_scores_array.min():.4f}")
    print(f"最大值: {clip_scores_array.max():.4f}")
    print(f"平均值: {clip_scores_array.mean():.4f}")
    print(f"标准差: {clip_scores_array.std():.4f}")

    # 找出最佳参数组合
    max_idx = np.unravel_index(clip_scores_array.argmax(), clip_scores_array.shape)
    best_step = steps[max_idx[0]]
    best_lambda = lambdas[max_idx[1]]
    best_score = clip_scores_array[max_idx]

    print(f"\n最佳参数组合:")
    print(f"Step: {best_step}, Lambda: {best_lambda}, {metric_name}: {best_score:.4f}")





if __name__ == "__main__":

    plot_heatmap_step_lamda(dataset_name="webcode2m", metric_name="block_match")
    plot_heatmap_step_lamda(dataset_name="webcode2m", metric_name="text_match")
    plot_heatmap_step_lamda(dataset_name="webcode2m", metric_name="clip_score")
    plot_heatmap_step_lamda(dataset_name="webcode2m", metric_name="bleu")

    # plot_heatmap_step_lamda(dataset_name="design2code", metric_name="block_match")
    # plot_heatmap_step_lamda(dataset_name="design2code", metric_name="text_match")
    # plot_heatmap_step_lamda(dataset_name="design2code", metric_name="clip_score")
    # plot_heatmap_step_lamda(dataset_name="design2code", metric_name="bleu")

    plot_heatmap_step_lamda(dataset_name="webcode2m", metric_name="avg_time")
    # plot_heatmap_step_lamda(dataset_name="design2code", metric_name="avg_time")

    pass


