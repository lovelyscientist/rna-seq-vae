import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyensembl import EnsemblRelease

ensemble_data = EnsemblRelease(96)
GTEX_EXPRESSIONS_PATH = './data/v8_expressions.parquet'
GTEX_SAMPLES_PATH = './data/v8_samples.parquet'
TRAIN_SIZE = 4500
TEST_SIZE = 1100

# load gene expression data
def get_expressions(path=GTEX_EXPRESSIONS_PATH):
    if path.endswith(".parquet"):
        return pq.read_table(path).to_pandas().set_index("Name")
    else:
        separator = "," if path.endswith(".csv") else "\t"
        return pd.read_csv(path, sep=separator).set_index("Name")

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
def get_gtex_dataset(problem='regression'):
    samples = get_samples()
    expressions = get_expressions()

    first_100_genes = list(expressions.columns)[:1000]
    expressions = expressions[first_100_genes]

    #expressions = get_expressions()[get_genes_of_interest()]
    data = samples.join(expressions, on="Name", how="inner")
    #data = data[(data['Avg_age'] == 64.5)]

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


    '''X = pd.concat([
        scaled_df, # scaled expressions
        pd.get_dummies(data['Tissue'].values, prefix='tissue'), # one-hot-encoded tissues
        pd.get_dummies(data['Sex'].values, prefix='sex'), # one-hot-encoded gender
        pd.get_dummies(data['Death'].values, prefix='death'), # one-hot-encoded death type
        pd.DataFrame(data=MinMaxScaler().fit_transform(Y.reshape(-1, 1)), columns=['Age'])],
        #pd.get_dummies(Y, prefix='age')],  # one-hot-encoded death type
        #pd.DataFrame(data=Y, columns=['Age'])], # age
    axis=1).values'''


    X_train, X_test, Y_train, Y_test = train_test_split(scaled_df.values, Y, test_size = 0.2, stratify=Y)

    plot_dataset_in_3d_space(scaled_df.values, Y)

    gene_names = [ensemble_data.gene_name_of_gene_id(c) for c in list(scaled_df.columns)]

    return (X_train[:TRAIN_SIZE], Y_train[:TRAIN_SIZE]), (X_test[:TEST_SIZE], Y_test[:TEST_SIZE]), scaled_df.values, gene_names, Y

# limit expressions to only 50 genes
def get_genes_of_interest():
    with open('./data/selected_genes.txt') as f:
        content = [x.strip() for x in f.readlines()]
    return content

def plot_dataset_in_3d_space(X, Y):
    tsne_model = TSNE(n_components=3)
    X_3d = tsne_model.fit_transform(X, Y)
    np.save('models/tsne_full_space.npy', X_3d)
    #print(pca_model.explained_variance_ratio_)


    #X_3d = np.load('models/tsne_full_space.npy')

    colors_dict = {
        '24.5': 'blue',
        '34.5': 'orange',
        '44.5': 'red',
        '54.5': 'purple',
        '64.5': 'yellow',
        '74.5': 'green'
    }

    class_colors = list(map(lambda y: colors_dict[str(y)], Y))

    fig = plt.figure()

    ax = Axes3D(fig)

    x_vals = X_3d[:, 0:1]
    y_vals = X_3d[:, 1:2]
    z_vals = X_3d[:, 2:3]

    ax.scatter(x_vals, y_vals, z_vals, c=class_colors, alpha=0.4)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()
