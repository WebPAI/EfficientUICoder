import json
import matplotlib
# matplotlib.use('TKAgg')
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np


# 读取JSON文件
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


# 绘制双Y轴折线图
def plot_dual_axis_metrics(dataset_name="webcode2m"):
    # 读取数据
    json_file_path = f'data/{dataset_name}_ratio_add.json'
    data = load_json_data(json_file_path)

    # 提取数据
    versions = []
    flops_data = []
    prefill_time_data = []

    for key in sorted(data.keys()):  # 确保按add0, add1, add2...的顺序
        versions.append(key)
        flops_data.append(data[key]['avg_all_FLOPs'])
        prefill_time_data.append(data[key]['avg_prefill_time'] * 1000)  # 转换为ms
    # versions = [0, 10, 20, 30, 40, 50]
    versions = [0, 20, 40, 60, 80, 100]
    # 创建图形和第一个y轴
    fig, ax1 = plt.subplots(figsize=(11, 8))

    # 绘制FLOPS数据（左y轴）
    color1 = 'tab:blue'

    # ax1.set_xlabel('The ratio added from unselected tokens')
    # ax1.set_xlabel('The ratio added from unselected tokens (%)', fontsize=30)
    ax1.set_xlabel('Ratio (%)', fontsize=30)

    ax1.set_ylabel('FLOPs (T)', color=color1, fontsize=30)
    line1 = ax1.plot(versions, flops_data, 'o-', color=color1, linewidth=2, markersize=6, label='FLOPs')
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=30)
    ax1.tick_params(axis='x', labelsize=30)
    # ax1.set_xticks([0, 10, 20, 30, 40, 50])
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.grid(True, alpha=0.3)
    if dataset_name == "design2code":
        ax1.set_ylim([15, 30])
    else:
        ax1.set_ylim([18, 30])
    # 创建第二个y轴
    ax2 = ax1.twinx()

    # 绘制prefill time数据（右y轴）
    color2 = 'tab:red'
    ax2.set_ylabel('Prefill Time (ms)', color=color2, fontsize=30)
    line2 = ax2.plot(versions, prefill_time_data, 's-', color=color2, linewidth=2, markersize=6,
                     label='Prefill Time')
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=30)
    # ax2.tick_params(axis='x', values=[0, 10, 20, 30, 40, 50], labelsize=30)

    # 添加标题
    # plt.title('Model Performance Metrics: FLOPs vs Prefill Time', fontsize=14, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=30, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2)


    plt.tight_layout()
    plt.savefig(f"{dataset_name}_ratio_add_efficiency.pdf")
    plt.show()


if __name__ == "__main__":

    plot_dual_axis_metrics(dataset_name="design2code")
    plot_dual_axis_metrics(dataset_name="webcode2m")


