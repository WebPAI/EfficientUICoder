
import matplotlib
# matplotlib.use('TKAgg')
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import json
import matplotlib.pyplot as plt




def plot_line_ratio_add(dataset_name):
    # 读取JSON文件
    with open(f'data/{dataset_name}_ratio_add.json', 'r') as f:
        data = json.load(f)

    # 选择的ratio值
    # ratios = [0.025, 0.05, 0.1, 0.2]
    # ratios = [0, 0.05, 0.1, 0.2, 0.3]
    ratios = ["add0", "add1", "add2", "add3", "add4", "add5"]
    # ratio_labels = ["0", "10", "20", "30", "40", "50"]
    ratio_labels = ["0", "20", "40", "60", "80", "100"]
    # ratios = [0, 0.05, 0.1, 0.2]
    # 初始化指标数据
    metrics = {
        'block_match': [],
        'text_match': [],
        'position_match': [],
        'text_color_match': [],
        'clip_score': [],
        'bleu':[]
    }

    # 提取数据
    for ratio in ratios:
        ratio_str = str(ratio)
        for metric in metrics.keys():
            metrics[metric].append(data[ratio_str][metric])

    # 创建折线图
    plt.figure(figsize=(10, 8))

    # 定义颜色和标记样式
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', "black"]
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', "#8c564b"]
    markers = ['o', 's', '^', 'D', 'v', 'p']
    line_styles = ['-', '--', '-.', ':', '-', '--']

    labels = ["Block", "Text", "Position", "Color", "CLIP", "Bleu"]

    # ratio_ticks = [0, 10, 20, 30, 40, 50]
    ratio_ticks = [0, 20, 40, 60, 80, 100]
    # 绘制每个指标的折线
    for i, (metric, values) in enumerate(metrics.items()):
        plt.plot(ratio_ticks, values,
                 color=colors[i],
                 marker=markers[i],
                 linestyle=line_styles[i],
                 linewidth=4,
                 markersize=15,
                 # label=metric,
                 label=labels[i],
                 markerfacecolor='white',
                 markeredgecolor=colors[i],
                 markeredgewidth=2)

    # 设置图形属性
    plt.xlabel('Ratio (%)', fontsize=40)
    # plt.ylabel('Metric Value', fontsize=14, fontweight='bold')
    # plt.title('Metrics vs Ratio', fontsize=16, fontweight='bold', pad=20)

    # 设置x轴刻度
    plt.xticks(ratio_ticks, ratio_labels, fontsize=35)
    plt.yticks(fontsize=40)

    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')

    # 添加图例
    # plt.legend(fontsize=20, loc='best', frameon=False, fancybox=True, shadow=True, ncol=3)

    plt.legend(fontsize=30, loc='upper center', bbox_to_anchor=(0.5, 1.3),
               ncol=3, frameon=False, fancybox=True, shadow=True)

    # 设置y轴范围以更好地显示数据
    # plt.ylim(0.0, 1.0)
    # plt.ylim(0.1, 0.85)
    if dataset_name == "webcode2m":
        plt.ylim(0.1, 0.82)
    else:
        plt.ylim(0.2, 0.85)

    # 调整布局
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_ratio_add.pdf")

    # 显示图形
    plt.show()

    # 可选：保存图形
    # plt.savefig('metrics_line_plot.png', dpi=300, bbox_inches='tight')

    # 打印数据摘要
    print("Data Summary:")
    for metric, values in metrics.items():
        print(f"{metric}: {values}")
        print(f"  - Max: {max(values):.4f} at ratio {ratios[values.index(max(values))]}")
        print(f"  - Min: {min(values):.4f} at ratio {ratios[values.index(min(values))]}")
        print()



if __name__ == "__main__":

    # plot_line_ratio()
    plot_line_ratio_add(dataset_name="webcode2m")
    plot_line_ratio_add(dataset_name="design2code")