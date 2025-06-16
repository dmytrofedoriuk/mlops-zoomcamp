import click
import pickle
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from sklearn import pipeline


def read_data(filename: str, categorical: list):
    """Read and preprocess the input data"""
    print(f"Reading data from {filename}")
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    print(f"Processed {len(df)} rows")
    return df


def load_model():
    """Load the pretrained model"""
    print("Loading model...")
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def make_pipeline(dv, lr):
    """Create a scikit-learn pipeline"""
    return pipeline.Pipeline([
        ('dv', dv),
        ('model', lr)
    ])


def prepare_features(df, categorical, year, month):
    """Prepare features for prediction"""
    print(f"Preparing features for {len(df)} rows")
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    dicts = df[categorical].to_dict(orient='records')
    return dicts


def score_model(pipe, dicts):
    """Make predictions using the pipeline"""
    print("Scoring model...")
    X_val = pipe.named_steps['dv'].transform(dicts)
    y_pred = pipe.named_steps['model'].predict(X_val)
    print(f"Generated predictions for {len(y_pred)} rows")
    return y_pred


def save_results(df_result, account_name: str, container_name: str, blob_name: str):
    """Save results to Azure Blob Storage using DefaultAzureCredential"""
    print(f"Saving results to Azure Blob Storage: {blob_name}")
    account_url = f"https://{account_name}.blob.core.windows.net"
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url, credential=credential)
    container_client = blob_service_client.get_container_client(container_name)
    
    local_path = "predictions.parquet"
    df_result.to_parquet(local_path, engine='pyarrow', compression=None, index=False)
    
    with open(local_path, "rb") as data:
        container_client.upload_blob(name=blob_name, data=data, overwrite=True)
    print("Results saved successfully")


def main(year: int, month: int, account_name: str, container_name: str):
    """Main prediction pipeline"""
    categorical = ['PULocationID', 'DOLocationID']

    print('Reading data...')
    df = read_data(
        f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet',
        categorical
    )

    print('Loading model...')
    dv, model = load_model()

    print('Making predictions...')
    pipe = make_pipeline(dv, model)
    dicts = prepare_features(df, categorical, year, month)
    y_pred = score_model(pipe, dicts)

    print(f'Mean predicted duration: {y_pred.mean():.2f}')

    print('Saving results...')
    df['predicted_duration'] = y_pred
    df_result = df[['ride_id', 'predicted_duration']]
    
    blob_name = f'predictions_{year:04d}_{month:02d}.parquet'
    save_results(df_result, account_name, container_name, blob_name)


@click.command()
@click.option('--year', type=int, required=True, help='Year to process')
@click.option('--month', type=int, required=True, help='Month to process')
@click.option(
    '--account-name',
    required=True,
    help='Azure Storage account name',
    envvar='AZURE_STORAGE_ACCOUNT'
)
@click.option(
    '--container-name',
    required=True,
    help='Azure Storage container name',
    default='taxi-predictions'
)
def run_pipeline(year: int, month: int, account_name: str, container_name: str):
    """Run the prediction pipeline"""
    main(year, month, account_name, container_name)


if __name__ == '__main__':
    run_pipeline()