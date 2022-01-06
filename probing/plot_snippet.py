import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from pandas.core import base 
import seaborn as sns 

def add_lines(ax, baseline_df, comp_df, x_name, y_name):
    for baseline_idx, baseline_row in baseline_df.iterrows():
        comp_row = comp_df.iloc[baseline_idx]
        bx, by = baseline_row[x_name], baseline_row[y_name]
        cx, cy = comp_row[x_name], comp_row[y_name]
        assert baseline_row['object'] == comp_row['object']
        try:
            ax.plot([bx,cx], [by, cy], color='grey', linewidth=0.5)
        except ValueError:
            pass


def linked_point_plot(ax, df, key = "model", left_name="bert-base", right_name="bert-large", title=""):
    baseline_idxs = df[key] == left_name
    comp_idxs = df[key] == right_name
    df = df[baseline_idxs | comp_idxs]
    df['x_value'] = [0 if row[1][key] == left_name else 1 for row in df.iterrows()]

    sns.scatterplot(data = df, x="x_value", y = "spearman", hue="object", ax = ax, legend = False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([left_name, right_name], rotation=15)

    # add lines 
    baseline_df = df[baseline_idxs].reset_index()
    comp_df = df[comp_idxs].reset_index()
    print(comp_df.head())
    add_lines(ax, baseline_df, comp_df, "x_value", "spearman")

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def draw_linked_plot():
    # 4 plots across horizontally, one per (color, shape, material, co-occurrence)
    fig, axs = plt.subplots(1,4, figsize=(10,4))
    df = pd.read_csv("probing/pt_corrs_sm.csv", index_col=False)
    key, left, right = ["model", "bert-base", "oscar-base"]
    #key, left, right = ["has_prompt_tuning", False, True]
    for idx, rel in enumerate(['color', 'shape', 'material', 'cooccur']):
        # assumed CSV format: 
        #   - column names: model, relation, object, has_prompt_tuning, spearman
        #   - example column: "bert-base", "color", "apple", "True", 0.36
        rel_df = df.loc[df['relation'] == rel]
        if key == 'model':
            rel_df = rel_df.loc[rel_df['has_prompt_tuning'] == True]
        if key == 'has_prompt_tuning':
            rel_df = rel_df.loc[rel_df['model'] == "oscar-base"]

        # Example: left-most plot is comparing before and after prompt tuning for color 
        linked_point_plot(axs[idx], rel_df, key=key, left_name=left, right_name=right, title=rel)

    axs[0].set_ylabel("Spearman correlation")
    fig.tight_layout()
    plt.savefig(f'probing/plots/linked_plot_{key}.pdf')



# reference: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def plot_heatmap(**kwargs):
    models = ['BERT', 'Oscar', 'Distilled', 'RoBERTa', 'Vokenization']  #  'ALBERT',
    data = ['CoDa', 'VG', 'Wiki']
    df = pd.read_csv('heatmap_data.csv', sep=',', index_col=0)
    df_values = df.values*100
    sns.set(font_scale = 1)
    ax = sns.heatmap(df_values, annot=True, fmt = '.1f', xticklabels=data, yticklabels=models)
    ax.collections[0].colorbar.set_label("Spearman correlation")

    plt.tight_layout()
    plt.savefig('probing/plots/heatmap.pdf')


if __name__ == '__main__':
    draw_linked_plot()
    #plot_heatmap(cmap="YlGn")
