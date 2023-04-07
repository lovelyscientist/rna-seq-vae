import pandas as pd
import numpy as np
import umap
import pyarrow.parquet as pq
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from pyensembl import EnsemblRelease
import plotly.graph_objects as go
from plotly import graph_objs

ensemble_data = EnsemblRelease(96)
GTEX_EXPRESSIONS_PATH = './data/v8_expressions.parquet'
GTEX_SAMPLES_PATH = './data/v8_samples.parquet'
TRAIN_SIZE = 4500
TEST_SIZE = 1100


# load gene expression data
def get_expressions(path=GTEX_EXPRESSIONS_PATH):
    if path.endswith(".parquet"):
        # genes_to_choose = pd.read_csv('data/aging_significant_genes.csv')['ids'].values
        return pq.read_table(path).to_pandas().set_index("Name")
    else:
        # genes_to_choose = pd.read_csv('data/aging_significant_genes.csv')['ids'].values
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
def get_gtex_dataset(label='tissue', problem='classification'):
    samples = get_samples()
    expressions = get_expressions()

    first_1000_genes = list(expressions.columns)[:1000]
    expressions = expressions[first_1000_genes]
    data = samples.join(expressions, on="Name", how="inner")

    if label == 'age':
        if problem == 'classification':
            Y = data['Age'].values
        else:
            Y = data['Avg_age'].values
    else:
        Y = data["Tissue"].values

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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
    gene_names = [ensemble_data.gene_name_of_gene_id(c) for c in list(scaled_df.columns)]

    return {'train_set': (X_train, Y_train),
            'test_set': (X_test, Y_test),
            'X_df': scaled_df.values,
            'Y': Y,
            'gene_names': gene_names}


def get_3d_embeddings(method='umap', dataset='real', label='tissue', file_pattern=None, save=False):
    if dataset == 'real':
        data = get_gtex_dataset(problem='classification', label=label)
        x_values = data['X_df']
        y_values = data['Y']
    else:
        x_values = pd.read_csv('data/{}_expressions.csv'.format(file_pattern)).values
        y_values = pd.read_csv('data/{}_labels.csv'.format(file_pattern))['label'].values

    if method == 'umap':
        embedding = umap.UMAP(n_components=3).fit_transform(x_values)

    if method == 'tsne':
        embedding = TSNE(n_components=3, init='pca').fit_transform(x_values)

    if method == 'pca':
        embedding = PCA(n_components=3).fit_transform(x_values)

    if save:
        np.save('data/{0}_{1}.npy'.format(method, dataset), embedding)

    title = '3D embedding of {0} GTEx gene expressions by {1} coloured by {2}'.format(dataset, method, label)
    return embedding, y_values, title


# limit expressions to only 50 genes
def get_genes_of_interest():
    with open('./data/selected_genes.txt') as f:
        content = [x.strip() for x in f.readlines()]
    return content


def find_max_latent_space_size(X, components):
    pca = PCA(n_components=components)
    pca.fit(X)
    print('With', components, 'explained variance is', np.sum(pca.explained_variance_ratio_))


def plot_3d_embeddings(x_values, labels, title):
    uniq_y = list(set(labels))
    label_name = 'tissue' if 'tissue' in title else 'age'
    colors_dict = {}
    for y in uniq_y:
        colors_dict[y] = get_random_plotly_color()

    colors = list(map(lambda y: colors_dict[y], labels))
    x_vals = list(np.array(x_values[:, 0:1]).flatten())
    y_vals = list(np.array(x_values[:, 1:2]).flatten())
    z_vals = list(np.array(x_values[:, 2:3]).flatten())

    df = pd.DataFrame(labels, columns=[label_name])
    df['x'] = x_vals
    df['y'] = y_vals
    df['z'] = z_vals
    df['color'] = colors

    fig = go.Figure(data=[go.Scatter3d(
        x=df[df[label_name] == label]['x'].values,
        y=df[df[label_name] == label]['y'].values,
        z=df[df[label_name] == label]['z'].values,
        name=label,
        mode='markers',
        marker=dict(
            size=5,
            color=colors_dict[label]
        )
    ) for label in uniq_y], layout=go.Layout(
        title=title,
        width=1000,
        showlegend=True,
        scene=graph_objs.Scene(
            xaxis=graph_objs.layout.scene.XAxis(title='x axis title'),
            yaxis=graph_objs.layout.scene.YAxis(title='y axis title'),
            zaxis=graph_objs.layout.scene.ZAxis(title='z axis title')
        )))

    fig.show()


def get_random_plotly_color():
    colors = '''
            aliceblue, antiquewhite, aqua, aquamarine, azure,
            beige, bisque, black, blanchedalmond, blue,
            blueviolet, brown, burlywood, cadetblue,
            chartreuse, chocolate, coral, cornflowerblue,
            cornsilk, crimson, cyan, darkblue, darkcyan,
            darkgoldenrod, darkgray, darkgrey, darkgreen,
            darkkhaki, darkmagenta, darkolivegreen, darkorange,
            darkorchid, darkred, darksalmon, darkseagreen,
            darkslateblue, darkslategray, darkslategrey,
            darkturquoise, darkviolet, deeppink, deepskyblue,
            dimgray, dimgrey, dodgerblue, firebrick,
            floralwhite, forestgreen, fuchsia, gainsboro,
            ghostwhite, gold, goldenrod, gray, grey, green,
            greenyellow, honeydew, hotpink, indianred, indigo,
            ivory, khaki, lavender, lavenderblush, lawngreen,
            lemonchiffon, lightblue, lightcoral, lightcyan,
            lightgoldenrodyellow, lightgray, lightgrey,
            lightgreen, lightpink, lightsalmon, lightseagreen,
            lightskyblue, lightslategray, lightslategrey,
            lightsteelblue, lightyellow, lime, limegreen,
            linen, magenta, maroon, mediumaquamarine,
            mediumblue, mediumorchid, mediumpurple,
            mediumseagreen, mediumslateblue, mediumspringgreen,
            mediumturquoise, mediumvioletred, midnightblue,
            mintcream, mistyrose, moccasin, navajowhite, navy,
            oldlace, olive, olivedrab, orange, orangered,
            orchid, palegoldenrod, palegreen, paleturquoise,
            palevioletred, papayawhip, peachpuff, peru, pink,
            plum, powderblue, purple, red, rosybrown,
            royalblue, saddlebrown, salmon, sandybrown,
            seagreen, seashell, sienna, silver, skyblue,
            slateblue, slategray, slategrey, snow, springgreen,
            steelblue, tan, teal, thistle, tomato, turquoise,
            violet, wheat, white, whitesmoke, yellow,
            yellowgreen
            '''
    colors_list = colors.split(',')
    colors_list = [c.replace('\n', '').replace(' ', '') for c in colors_list]
    return colors_list[np.random.choice(len(colors_list))]


#x, y, t = get_3d_embeddings(method='pca', dataset='real', label='tissue', file_pattern='trial_1_embedding')
#plot_3d_embeddings(x, y, t)
