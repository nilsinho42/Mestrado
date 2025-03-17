from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.identity import DefaultAzureCredential
from pathlib import Path
import yaml
import os

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def deploy_to_azure(model_path, config):
    """Deploy model to Azure ML."""
    try:
        # Initialize Azure ML client
        credential = DefaultAzureCredential()
        subscription_id = os.environ.get('AZURE_SUBSCRIPTION_ID')
        resource_group = os.environ.get('AZURE_RESOURCE_GROUP')
        workspace_name = os.environ.get('AZURE_WORKSPACE_NAME')
        
        if not all([subscription_id, resource_group, workspace_name]):
            raise ValueError("Azure environment variables not set")
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        # Register model
        model = ml_client.models.create_or_update(
            Model(
                path=model_path,
                name="object-detection-model",
                description="YOLOv8 model for object detection"
            )
        )
        
        # Create online endpoint
        endpoint = ManagedOnlineEndpoint(
            name="object-detection-endpoint",
            description="Endpoint for object detection",
            auth_mode="key"
        )
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        # Create deployment
        deployment = ManagedOnlineDeployment(
            name="object-detection-deployment",
            endpoint_name=endpoint.name,
            model=model.id,
            instance_type=config['cloud']['azure']['vm_size'],
            instance_count=1,
            environment_variables={
                "AZUREML_ENTRY_SCRIPT": "inference.py"
            }
        )
        
        ml_client.online_deployments.begin_create_or_update(deployment).result()
        
        # Set deployment as default
        ml_client.online_endpoints.update_traffic(
            endpoint.name,
            {deployment.name: 100}
        )
        
        print(f"Model deployed successfully to endpoint: {endpoint.name}")
        return endpoint.name
        
    except Exception as e:
        print(f"Error deploying to Azure ML: {str(e)}")
        raise

if __name__ == "__main__":
    config = load_config()
    model_path = "models/yolo_final.pt"
    deploy_to_azure(model_path, config) 