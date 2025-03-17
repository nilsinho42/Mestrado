from google.cloud import aiplatform
from google.cloud import storage
from pathlib import Path
import yaml
import os

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def upload_model_to_gcs(model_path, bucket_name):
    """Upload model to Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"models/{Path(model_path).name}")
    blob.upload_from_filename(model_path)
    return f"gs://{bucket_name}/{blob.name}"

def deploy_to_gcp(model_path, config):
    """Deploy model to Google Cloud AI Platform."""
    try:
        # Set GCP project ID
        project_id = os.environ.get('GCP_PROJECT_ID')
        region = config['cloud']['gcp']['region']
        
        if not project_id:
            raise ValueError("GCP_PROJECT_ID environment variable not set")
        
        # Initialize AI Platform client
        aiplatform.init(project=project_id, location=region)
        
        # Upload model to GCS
        gcs_model_path = upload_model_to_gcs(
            model_path,
            config['cloud']['gcp']['bucket']
        )
        
        # Create model resource
        model = aiplatform.Model.upload(
            display_name="object-detection-model",
            artifact_uri=gcs_model_path,
            serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/pytorch-gpu.1-13",
            serving_container_predict_route="/predict",
            serving_container_health_route="/health"
        )
        
        # Deploy model
        endpoint = model.deploy(
            machine_type=config['cloud']['gcp']['machine_type'],
            accelerator_type=config['cloud']['gcp']['accelerator'],
            min_replica_count=1,
            max_replica_count=1,
            sync=True
        )
        
        print(f"Model deployed successfully to endpoint: {endpoint.resource_name}")
        return endpoint.resource_name
        
    except Exception as e:
        print(f"Error deploying to Google Cloud AI Platform: {str(e)}")
        raise

if __name__ == "__main__":
    config = load_config()
    model_path = "models/yolo_final.pt"
    deploy_to_gcp(model_path, config) 