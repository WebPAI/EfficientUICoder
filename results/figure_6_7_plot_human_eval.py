import json
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np


def load_data(filename='data/human_eval.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def prepare_plot_data(data, model_size, dataset_name):
    methods = []
    win_percentages = []
    tie_percentages = []
    lose_percentages = []

    for model_name, model_data in data.items():
        if model_size not in model_name:
            continue
        for task_name, task_data in model_data.items():
            if dataset_name not in task_name:
                continue
            for method_name, results in task_data.items():
                # 计算百分比
                total = results['total']
                win_pct = (results['win'] / total) * 100
                tie_pct = (results['tie'] / total) * 100
                lose_pct = (results['lose'] / total) * 100

                # 创建标签（模型-方法）
                # label = f"{model_name.replace('llava-v1.6-', 'LLaVA-')} {method_name}"
                label = method_name

                methods.append(label)
                win_percentages.append(win_pct)
                tie_percentages.append(tie_pct)
                lose_percentages.append(lose_pct)

    return methods, win_percentages, tie_percentages, lose_percentages




def plot_performance_chart_styled(model_size, dataset_name):
    data = load_data()
    methods, win_pct, tie_pct, lose_pct = prepare_plot_data(data, model_size=model_size, dataset_name=dataset_name)

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {
        'Win': '#c5e0b4',  # 蓝色
        'Tie': '#87cefa',  # 浅蓝绿色
        'Lose': '#f08080'  # 红色
    }

    y_pos = np.arange(len(methods))

    bars1 = ax.barh(y_pos, win_pct, color=colors['Win'], label='Win', height=0.6)
    bars2 = ax.barh(y_pos, tie_pct, left=win_pct, color=colors['Tie'], label='Tie', height=0.6)
    bars3 = ax.barh(y_pos, lose_pct, left=np.array(win_pct) + np.array(tie_pct),
                    color=colors['Lose'], label='Lose', height=0.6)

    for i, (w, t, l) in enumerate(zip(win_pct, tie_pct, lose_pct)):
        if w > 3:
            ax.text(w / 2, i, f'{w:.0f}%', ha='center', va='center',
                    color='black', fontweight='bold', fontsize=28)
        if t > 3:
            ax.text(w + t / 2, i, f'{t:.0f}%', ha='center', va='center',
                    color='black', fontweight='bold', fontsize=28)
        if l > 3:
            ax.text(w + t + l / 2, i, f'{l:.0f}%', ha='center', va='center',
                    color='black', fontweight='bold', fontsize=28)

    ax.set_yticks(y_pos)

    # ax.set_yticklabels(methods, fontsize=30)
    ax.set_yticklabels(["EUC", "Visionzip", "Pdrop", "FastV", "Random"], fontsize=30)
    ax.tick_params(axis='x', labelsize=40)
    ax.tick_params(axis='y', labelsize=40)
    ax.set_xlabel('Percentage (%)', fontsize=40)
    ax.set_xlim(0, 100)

    # 添加图例（在右上角）
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), frameon=False, fancybox=True, shadow=True, ncol=3,
              fontsize=35)

    # 移除上边框和右边框
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # 设置背景和网格
    # ax.set_facecolor('white')
    ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f"llava{model_size}_{dataset_name}_humaneval.pdf")
    plt.show()


if __name__ == "__main__":
    plot_performance_chart_styled(model_size="7b", dataset_name="d2c")
    plot_performance_chart_styled(model_size="34b", dataset_name="d2c")
    plot_performance_chart_styled(model_size="7b", dataset_name="webcode2m")
    plot_performance_chart_styled(model_size="34b", dataset_name="webcode2m")
