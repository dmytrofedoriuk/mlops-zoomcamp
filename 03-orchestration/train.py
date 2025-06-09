import prefect
import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
import mlflow.models

from prefect import flow, task, get_run_logger
from mlflow.models.signature import infer_signature
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from pathlib import Path


@task(name="Print Tool Info")
def print_tool_info():
    """Prints the tool information used in this project."""
    logger = get_run_logger()
    logger.info(f'Q1: Tools used in this project: Prefect')
    logger.info(f'Q2: Prefect version: {prefect.__version__}')


@task(name="Read Data")
def read_data(url):
    """Reads parquet data from a given URL and processes it."""
    logger = get_run_logger()
    df = pd.read_parquet(url, engine="pyarrow")
    logger.info(f'Q3: {df.shape[0]} rows loaded...')

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    logger.info(f'Q4: {df.shape[0]} rows after processing...')
    return df


@task(name="Train Model")
def train_model(df):
    logger = get_run_logger()
    with mlflow.start_run():
        df['PULocationID'] = df['PULocationID'].astype('str')
        df['DOLocationID'] = df['DOLocationID'].astype('str')
        df_to_ohe = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')
        
        v = DictVectorizer()
        X = v.fit_transform(df_to_ohe)
        y = df['duration']

        reg = LinearRegression().fit(X, y)
        logger.info(f'Q5: Intercept of the model: {reg.intercept_}')

        input_example = v.transform([{'PULocationID': '1', 'DOLocationID': '2'}])
        signature = infer_signature(input_example, reg.predict(input_example))
        
        mlflow.sklearn.log_model(
            reg,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        return reg, v


@task(name="Get Model Size")
def get_latest_model_size():
    logger = get_run_logger()
    mlruns_dir = Path("mlruns/0")
    run_dirs = [d for d in mlruns_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    latest_run = max(run_dirs, key=lambda x: x.stat().st_ctime)
    mlmodel_path = latest_run / "artifacts/model/MLmodel"
    
    if mlmodel_path.exists():
        with open(mlmodel_path) as f:
            for line in f:
                if "model_size_bytes" in line:
                    logger.info(f'Q6: The model size is {int(line.split(":")[1].strip())} bytes')


@flow(name="Train Pipeline")
def main(url: str):
    print_tool_info()
    df = read_data(url)
    model, vectorizer = train_model(df)
    get_latest_model_size()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read parquet data from a URL")
    parser.add_argument('--url', type=str, required=True, help='URL to the parquet file')
    args = parser.parse_args()

    main(args.url)
