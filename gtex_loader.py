import pandas as pd
import numpy as np
import umap
import pyarrow.parquet as pq
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyensembl import EnsemblRelease
import plotly.graph_objects as go

ensemble_data = EnsemblRelease(96)
GTEX_EXPRESSIONS_PATH = './data/v8_expressions.parquet'
GTEX_SAMPLES_PATH = './data/v8_samples.parquet'
TRAIN_SIZE = 4500
TEST_SIZE = 1100


# load gene expression data
def get_expressions(path=GTEX_EXPRESSIONS_PATH):
    if path.endswith(".parquet"):
        genes_to_choose = pd.read_csv('data/aging_significant_genes.csv')['ids'].values
        return pq.read_table(path).to_pandas().set_index("Name")[genes_to_choose]
    else:
        genes_to_choose = pd.read_csv('data/aging_significant_genes.csv')['ids'].values
        separator = "," if path.endswith(".csv") else "\t"
        return pd.read_csv(path, sep=separator).set_index("Name")[genes_to_choose]


# load additional metadata of the dataset
def get_samples(path=GTEX_SAMPLES_PATH):
    samples = pd.read_parquet(path, engine='pyarrow')
    samples["Death"].fillna(-1.0, inplace=True)
    samples = samples.set_index("Name")
    samples["Sex"].replace([1, 2], ['male', 'female'], inplace=True)
    samples["Death"].replace([-1, 0, 1, 2, 3, 4],
                                  ['alive/NA', 'ventilator case', '<10 min.', '<1 hr', '1-24 hr.', '>1 day'],
                                  inplace=True)
    return samples


# load whole dataset
def get_gtex_dataset(problem='classification'):
    samples = get_samples()
    expressions = get_expressions()

    first_100_genes = list(expressions.columns)[:1000]
    expressions = expressions[first_100_genes]

    #expressions = get_expressions()[get_genes_of_interest()]
    data = samples.join(expressions, on="Name", how="inner")
    #data = data[(data['Avg_age'] == 64.5)]
    #print(data['Death'].unique())
    #data = data[(data['Death'] != '>1 day') & (data['Death'] != 'alive/NA')]
    #data = data[(data['Age'] == '20-29') | (data['Age'] == '60-69')]

    if problem == 'classification':
        Y = data['Age'].values
    else:
        Y = data['Avg_age'].values

    # removing labels

    columns_to_drop = ["Tissue", "Sex", "Age", "Death", "Subtissue", "Avg_age"]
    valid_columns = data.columns.drop(columns_to_drop)

    # normalize expression data for nn
    steps = [('standardization', StandardScaler()), ('normalization', MinMaxScaler())]
    pre_processing_pipeline = Pipeline(steps)
    transformed_data = pre_processing_pipeline.fit_transform(data[valid_columns])

    # save data to dataframe
    scaled_df = pd.DataFrame(transformed_data, columns=valid_columns)

    X = scaled_df.values

    '''pd.concat([
        scaled_df, # scaled expressions
        pd.get_dummies(data['Tissue'].values, prefix='tissue'), # one-hot-encoded tissues
        pd.get_dummies(data['Sex'].values, prefix='sex'), # one-hot-encoded gender
        pd.get_dummies(data['Death'].values, prefix='death')], # one-hot-encoded death type
        axis=1).values'''

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y)

    syn_x = pd.read_csv('data/expressions_synthetic_2000.csv')
    syn_y = pd.read_csv('data/samples_synthetic_2000.csv')

    # plot_dataset_in_3d_space(scaled_df.values, Y)
    find_max_latent_space_size(syn_x.values, 10)
    find_max_latent_space_size(syn_x.values, 50)
    find_max_latent_space_size(syn_x.values, 100)
    find_max_latent_space_size(syn_x.values, 150)
    find_max_latent_space_size(syn_x.values, 200)

    tsne_with_plotly(scaled_df.values, Y, classes=True)
    #tsne_with_plotly(syn_x.values, list(syn_y.values), classes=True)

    gene_names = [ensemble_data.gene_name_of_gene_id(c) for c in list(scaled_df.columns)]

    return (X_train, Y_train), (X_test, Y_test), scaled_df.values, gene_names, Y


# limit expressions to only 50 genes
def get_genes_of_interest():
    with open('./data/selected_genes.txt') as f:
        content = [x.strip() for x in f.readlines()]
    return content


def find_max_latent_space_size(X, components):
    pca = PCA(n_components=components)
    pca.fit(X)
    print('With', components, 'explained variance is', np.sum(pca.explained_variance_ratio_))


def tsne_with_plotly(X, Y, classes=False):
    pca = PCA(n_components=3)
    transformed_x = pca.fit_transform(X)

    X_3d = transformed_x

    #tsne_model = TSNE(n_components=3, init='pca')
    #X_3d = tsne_model.fit_transform(X)
    #np.save('models/tsne_sampled_space.npy', X_3d)

    #X_3d = np.load('models/tsne_full_space.npy')

    if classes:
        colors_dict = {
            "['20-29']": 'blue',
            "['30-39']": 'orange',
            "['40-49']": 'red',
            "['50-59']": 'purple',
            "['60-69']": 'yellow',
            "['70-79']": 'green',
            "20-29": 'blue',
            "30-39": 'orange',
            "40-49": 'red',
            "50-59": 'purple',
            "60-69": 'yellow',
            "70-79": 'green'
        }
    else:
        colors_dict = {
            '24.5': 'blue',
            '34.5': 'orange',
            '44.5': 'red',
            '54.5': 'purple',
            '64.5': 'yellow',
            '74.5': 'green'
        }

    class_colors = list(map(lambda y: colors_dict [str(y)], Y))

    x_vals = list(np.array(X_3d[:, 0:1]).flatten())
    y_vals = list(np.array(X_3d[:, 1:2]).flatten())
    z_vals = list(np.array(X_3d[:, 2:3]).flatten())

    print(x_vals[0:5])
    print(y_vals[0:5])
    print(z_vals[0:5])

    x, y, z = x_vals, y_vals, z_vals

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=class_colors  # set color to an array/list of desired values
        )
    )])

    # tight layout
    fig.show()