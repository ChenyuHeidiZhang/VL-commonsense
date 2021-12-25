import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# reference: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def plot_heatmap(**kwargs):
    models = ['BERT', 'Oscar', 'Distilled', 'RoBERTa', 'ALBERT', 'Vokenization']
    data = ['CoDa', 'VG', 'Wiki']
    df = pd.read_csv('heatmap_data.csv', sep=',', index_col=0)
    df_values = df.values*100
    sns.set(font_scale = 1)
    ax = sns.heatmap(df_values, annot=True, fmt = '.1f', xticklabels=data, yticklabels=models)
    ax.collections[0].colorbar.set_label("Spearman correlation")

    plt.tight_layout()
    plt.savefig('probing/plots/heatmap.pdf')


if __name__ == '__main__':
    plot_heatmap(cmap="YlGn")
