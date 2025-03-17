import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from pathlib import Path
import yaml
import os

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def deploy_to_sagemaker(model_path, config):
    """Deploy model to AWS SageMaker."""
    try:
        # Initialize SageMaker session
        sagemaker_session = sagemaker.Session()
        aws_role = os.environ.get('AWS_ROLE_ARN')
        
        if not aws_role:
            raise ValueError("AWS_ROLE_ARN environment variable not set")
        
        # Upload model to S3
        model_data = sagemaker_session.upload_data(
            model_path,
            bucket=config['cloud']['aws']['bucket'],
            key_prefix='models'
        )
        
        # Create PyTorch model for deployment
        model = PyTorchModel(
            model_data=model_data,
            role=aws_role,
            framework_version='2.0.0',
            py_version='py39',
            entry_point='inference.py',
            source_dir=str(Path(__file__).parent / 'scripts'),
            name='object-detection-model'
        )
        
        # Deploy model to endpoint
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=config['cloud']['aws']['instance_type'],
            endpoint_name='object-detection-endpoint'
        )
        
        print(f"Model deployed successfully to endpoint: {predictor.endpoint_name}")
        return predictor.endpoint_name
        
    except Exception as e:
        print(f"Error deploying to SageMaker: {str(e)}")
        raise

if __name__ == "__main__":
    config = load_config()
    model_path = "models/yolo_final.pt"
    deploy_to_sagemaker(model_path, config) 