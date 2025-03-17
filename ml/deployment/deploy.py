import argparse
from pathlib import Path
import yaml

from aws_deploy import deploy_to_sagemaker
from azure_deploy import deploy_to_azure
from gcp_deploy import deploy_to_gcp

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def deploy_model(model_path, platform, config):
    """
    Deploy model to specified cloud platform.
    
    Args:
        model_path: Path to the model file
        platform: Cloud platform to deploy to ('aws', 'azure', or 'gcp')
        config: Configuration dictionary
    """
    if platform == 'aws':
        return deploy_to_sagemaker(model_path, config)
    elif platform == 'azure':
        return deploy_to_azure(model_path, config)
    elif platform == 'gcp':
        return deploy_to_gcp(model_path, config)
    else:
        raise ValueError(f"Unsupported platform: {platform}")

def main():
    parser = argparse.ArgumentParser(description='Deploy object detection model to cloud platforms')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the model file')
    parser.add_argument('--platform', type=str, choices=['aws', 'azure', 'gcp'],
                      required=True, help='Cloud platform to deploy to')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Deploy model
    try:
        endpoint = deploy_model(args.model, args.platform, config)
        print(f"\nDeployment successful!")
        print(f"Endpoint: {endpoint}")
    except Exception as e:
        print(f"\nDeployment failed: {str(e)}")

if __name__ == "__main__":
    main() 