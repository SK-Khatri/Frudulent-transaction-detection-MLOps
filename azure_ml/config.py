"""
Azure ML Configuration
Centralizes workspace and compute configuration for the ML pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AzureMLConfig:
    """Configuration for Azure ML resources."""

    # Workspace
    subscription_id: str = field(
        default_factory=lambda: os.environ.get("AZURE_SUBSCRIPTION_ID", "")
    )
    resource_group: str = field(
        default_factory=lambda: os.environ.get("AZURE_RESOURCE_GROUP", "fraud-detection-rg")
    )
    workspace_name: str = field(
        default_factory=lambda: os.environ.get("AZURE_ML_WORKSPACE", "fraud-detection-ws")
    )

    # Compute
    compute_name: str = "fraud-compute-cluster"
    compute_vm_size: str = "Standard_DS3_v2"
    compute_min_nodes: int = 0
    compute_max_nodes: int = 4

    # Environment
    environment_name: str = "fraud-detection-env"
    environment_version: str = "1.0"

    # Model
    model_name: str = "fraud-detection-model"
    model_description: str = "XGBoost credit card fraud detection model"

    # Experiment
    experiment_name: str = "fraud-detection-experiment"

    # Pipeline
    pipeline_name: str = "fraud-detection-pipeline"

    # Data
    dataset_name: str = "creditcard-fraud-data"

    # AKS Deployment
    aks_cluster_name: str = field(
        default_factory=lambda: os.environ.get("AKS_CLUSTER_NAME", "fraud-aks-cluster")
    )
    aks_service_name: str = "fraud-detection-service"

    # Kaggle
    kaggle_username: str = field(
        default_factory=lambda: os.environ.get("KAGGLE_USERNAME", "")
    )
    kaggle_key: str = field(
        default_factory=lambda: os.environ.get("KAGGLE_KEY", "")
    )

    def validate(self) -> None:
        """Validate that all required configuration values are set."""
        required = {
            "subscription_id": self.subscription_id,
            "resource_group": self.resource_group,
            "workspace_name": self.workspace_name,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(
                f"Missing required Azure ML configuration: {', '.join(missing)}. "
                "Set the corresponding environment variables."
            )
