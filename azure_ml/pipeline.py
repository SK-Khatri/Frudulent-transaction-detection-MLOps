"""
Azure ML Pipeline Definition
Defines the end-to-end ML pipeline: ingestion → preprocessing → training → evaluation → registration.
Orchestrates all steps using Azure ML SDK v2.
"""

import os
import sys
import logging
from pathlib import Path

from azure.ai.ml import MLClient, Input, Output, command, dsl
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import DefaultAzureCredential

from azure_ml.config import AzureMLConfig
from azure_ml.environment import get_ml_client, create_compute, create_environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_data_ingestion_step(environment_name: str, compute_name: str):
    """Step 1: Download dataset from Kaggle."""
    return command(
        name="data_ingestion",
        display_name="Data Ingestion (Kaggle Download)",
        description="Download Credit Card Fraud Detection dataset from Kaggle API",
        command=(
            "export KAGGLE_USERNAME=${{inputs.kaggle_username}} && "
            "export KAGGLE_KEY=${{inputs.kaggle_key}} && "
            "python data_ingestion_step.py "
            "--output-dir ${{outputs.raw_data}}"
        ),
        environment=environment_name,
        compute=compute_name,
        code=str(PROJECT_ROOT / "azure_ml" / "steps"),
        inputs={
            "kaggle_username": Input(type="string"),
            "kaggle_key": Input(type="string"),
        },
        outputs={
            "raw_data": Output(type="uri_folder", mode=InputOutputModes.RW_MOUNT),
        },
    )


def create_preprocessing_step(environment_name: str, compute_name: str):
    """Step 2: Feature engineering, scaling, and splitting."""
    return command(
        name="preprocessing",
        display_name="Data Preprocessing",
        description="Feature engineering, scaling, SMOTE balancing, and train/test split",
        command=(
            "python preprocessing_step.py "
            "--input-dir ${{inputs.raw_data}} "
            "--output-dir ${{outputs.processed_data}} "
            "--artifacts-dir ${{outputs.artifacts}}"
        ),
        environment=environment_name,
        compute=compute_name,
        code=str(PROJECT_ROOT / "azure_ml" / "steps"),
        inputs={
            "raw_data": Input(type="uri_folder"),
        },
        outputs={
            "processed_data": Output(type="uri_folder", mode=InputOutputModes.RW_MOUNT),
            "artifacts": Output(type="uri_folder", mode=InputOutputModes.RW_MOUNT),
        },
    )


def create_training_step(environment_name: str, compute_name: str):
    """Step 3: Train the XGBoost model."""
    return command(
        name="training",
        display_name="Model Training",
        description="Train XGBoost classifier with hyperparameter configuration",
        command=(
            "python training_step.py "
            "--data-dir ${{inputs.processed_data}} "
            "--artifacts-dir ${{inputs.artifacts}} "
            "--model-dir ${{outputs.model}}"
        ),
        environment=environment_name,
        compute=compute_name,
        code=str(PROJECT_ROOT / "azure_ml" / "steps"),
        inputs={
            "processed_data": Input(type="uri_folder"),
            "artifacts": Input(type="uri_folder"),
        },
        outputs={
            "model": Output(type="uri_folder", mode=InputOutputModes.RW_MOUNT),
        },
    )


def create_evaluation_step(environment_name: str, compute_name: str):
    """Step 4: Evaluate the trained model."""
    return command(
        name="evaluation",
        display_name="Model Evaluation",
        description="Evaluate model performance and run quality gates",
        command=(
            "python evaluation_step.py "
            "--data-dir ${{inputs.processed_data}} "
            "--model-dir ${{inputs.model}} "
            "--report-dir ${{outputs.report}}"
        ),
        environment=environment_name,
        compute=compute_name,
        code=str(PROJECT_ROOT / "azure_ml" / "steps"),
        inputs={
            "processed_data": Input(type="uri_folder"),
            "model": Input(type="uri_folder"),
        },
        outputs={
            "report": Output(type="uri_folder", mode=InputOutputModes.RW_MOUNT),
        },
    )


@dsl.pipeline(
    name="fraud-detection-pipeline",
    description="End-to-end fraud detection ML pipeline: ingest → preprocess → train → evaluate",
    default_compute="fraud-compute-cluster",
)
def fraud_detection_pipeline(
    kaggle_username: str,
    kaggle_key: str,
):
    """Define the complete ML pipeline using DSL."""
    env_name = "fraud-env:latest"
    compute = "fraud-compute-cluster"
    
    # Step 1: Data Ingestion
    ingest = create_data_ingestion_step(env_name, compute)(
        kaggle_username=kaggle_username,
        kaggle_key=kaggle_key,
    )

    # Step 2: Preprocessing
    preprocess = create_preprocessing_step(env_name, compute)(
        raw_data=ingest.outputs.raw_data,
    )

    # Step 3: Training
    train = create_training_step(env_name, compute)(
        processed_data=preprocess.outputs.processed_data,
        artifacts=preprocess.outputs.artifacts,
    )

    # Step 4: Evaluation
    evaluate = create_evaluation_step(env_name, compute)(
        processed_data=preprocess.outputs.processed_data,
        model=train.outputs.model,
    )

    return {
        "model": train.outputs.model,
        "report": evaluate.outputs.report,
    }


def register_model(
    ml_client: MLClient,
    config: AzureMLConfig,
    model_path: str,
    metrics: dict,
) -> Model:
    """Register the trained model in Azure ML registry."""
    model = Model(
        path=model_path,
        name=config.model_name,
        description=config.model_description,
        type=AssetTypes.CUSTOM_MODEL,
        properties={
            "roc_auc": str(metrics.get("roc_auc", "")),
            "f1_score": str(metrics.get("f1_score", "")),
            "precision": str(metrics.get("precision", "")),
            "recall": str(metrics.get("recall", "")),
        },
        tags={
            "framework": "xgboost",
            "task": "fraud-detection",
            "dataset": "creditcard-fraud",
        },
    )
    registered = ml_client.models.create_or_update(model)
    logger.info("Model registered: %s (version %s)", registered.name, registered.version)
    return registered


def submit_pipeline(config: AzureMLConfig = None) -> str:
    """Submit the pipeline for execution."""
    config = AzureMLConfig()
    config.validate()

    ml_client = get_ml_client(config)

    # Ensure infrastructure exists
    create_compute(ml_client, config)
    create_environment(ml_client, config)

    # Build pipeline
    pipeline_job = fraud_detection_pipeline(
        kaggle_username=config.kaggle_username,
        kaggle_key=config.kaggle_key,
    )

    pipeline_job.settings.default_compute = config.compute_name
    pipeline_job.experiment_name = config.experiment_name

    # Submit
    submitted_job = ml_client.jobs.create_or_update(pipeline_job)
    logger.info("Pipeline submitted: %s", submitted_job.name)
    logger.info("Studio URL: %s", submitted_job.studio_url)

    return submitted_job.name


if __name__ == "__main__":
    job_name = submit_pipeline()
    print(f"Pipeline job submitted: {job_name}")
