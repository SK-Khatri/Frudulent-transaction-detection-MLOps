"""
Azure ML Environment Setup
Creates and manages the compute and environment for the ML pipeline.
"""

import logging
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    AmlCompute,
    Environment,
    BuildContext,
)
from azure.identity import DefaultAzureCredential

from azure_ml.config import AzureMLConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def get_ml_client(config: AzureMLConfig) -> MLClient:
    """Create an authenticated Azure ML client."""
    config.validate()
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=config.subscription_id,
        resource_group_name=config.resource_group,
        workspace_name=config.workspace_name,
    )
    logger.info(
        "Connected to Azure ML workspace: %s/%s/%s",
        config.subscription_id,
        config.resource_group,
        config.workspace_name,
    )
    return ml_client


def create_compute(ml_client: MLClient, config: AzureMLConfig) -> str:
    """Create or retrieve the compute cluster."""
    try:
        compute = ml_client.compute.get(config.compute_name)
        logger.info("Found existing compute cluster: %s", config.compute_name)
    except Exception:
        logger.info("Creating compute cluster: %s", config.compute_name)
        compute = AmlCompute(
            name=config.compute_name,
            type="amlcompute",
            size=config.compute_vm_size,
            min_instances=config.compute_min_nodes,
            max_instances=config.compute_max_nodes,
            idle_time_before_scale_down=300,
            tier="Dedicated",
        )
        ml_client.compute.begin_create_or_update(compute).result()
        logger.info("Compute cluster created: %s", config.compute_name)

    return config.compute_name


def create_environment(ml_client: MLClient, config: AzureMLConfig) -> Environment:
    """Create or retrieve the training environment."""
    env_name = config.environment_name

    try:
        env = ml_client.environments.get(env_name, label="latest")
        logger.info("Found existing environment: %s", env_name)
        return env
    except Exception:
        pass

    logger.info("Creating environment: %s", env_name)
    env = Environment(
        name=env_name,
        description="Fraud detection training environment with XGBoost, scikit-learn, and MLflow",
        conda_file={
            "name": "fraud-detection-env",
            "channels": ["defaults", "conda-forge"],
            "dependencies": [
                "python=3.10",
                "pip",
                {
                    "pip": [
                        "xgboost==2.0.3",
                        "scikit-learn==1.4.0",
                        "imbalanced-learn==0.12.0",
                        "pandas==2.2.0",
                        "numpy==1.26.3",
                        "mlflow==2.10.0",
                        "azureml-mlflow==1.55.0",
                        "kaggle==1.6.3",
                        "azure-ai-ml==1.13.0",
                        "azure-identity==1.15.0",
                    ]
                },
            ],
        },
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    )
    env = ml_client.environments.create_or_update(env)
    logger.info("Environment created: %s (version %s)", env.name, env.version)
    return env


def setup_infrastructure(config: AzureMLConfig = None) -> dict:
    """Set up all required Azure ML infrastructure."""
    config = config or AzureMLConfig()
    ml_client = get_ml_client(config)
    compute_name = create_compute(ml_client, config)
    environment = create_environment(ml_client, config)

    return {
        "ml_client": ml_client,
        "compute_name": compute_name,
        "environment": environment,
        "config": config,
    }


if __name__ == "__main__":
    result = setup_infrastructure()
    print(f"Compute: {result['compute_name']}")
    print(f"Environment: {result['environment'].name}")
