import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

GTEX_EXPRESSIONS_PATH = './data/v8_expressions.parquet'
GTEX_SAMPLES_PATH = './data/v8_samples.parquet'

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
    expressions = get_expressions()[get_genes_of_interest()]
    data = samples.join(expressions, on="Name", how="inner")

    if problem == 'classification':
        Y = data['Age'].values
    else:
        Y = data['Avg_age'].values

    columns_to_drop = ["Tissue", "Sex", "Age", "Death", "Subtissue", "Avg_age"]
    valid_columns = data.columns.drop(columns_to_drop)
    scaled_df = pd.DataFrame(MinMaxScaler().fit_transform(data[valid_columns]), columns=valid_columns)

    X = pd.concat([
        scaled_df, # scaled expressions
        pd.get_dummies(data['Tissue'].values, prefix='tissue'), # one-hot-encoded tissues
        pd.get_dummies(data['Sex'].values, prefix='sex'), # one-hot-encoded gender
        pd.get_dummies(data['Death'].values, prefix='death'), # one-hot-encoded death type
        pd.DataFrame(data=MinMaxScaler().fit_transform(Y.reshape(-1, 1)), columns=['Age'])],
        #pd.get_dummies(Y, prefix='age')],  # one-hot-encoded death type
        #pd.DataFrame(data=Y, columns=['Age'])], # age
    axis=1).values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify=Y)

    print(X_train[0])

    return (X_train, Y_train), (X_test, Y_test)

# limit expressions to only 50 genes
def get_genes_of_interest():
    with open('./data/selected_genes.txt') as f:
        content = [x.strip() for x in f.readlines()]
    return content