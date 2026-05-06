from setuptools import setup, find_packages

setup(
    name="fraud-detection-mlops",
    version="1.0.0",
    description="Credit Card Fraud Detection with MLOps on Azure ML Pipelines",
    author="MLOps Team",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
    install_requires=[
        "xgboost>=2.0.0",
        "scikit-learn>=1.4.0",
        "imbalanced-learn>=0.12.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "pydantic>=2.6.0",
        "prometheus-client>=0.20.0",
        "mlflow>=2.10.0",
        "kaggle>=1.6.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "azure": [
            "azure-ai-ml>=1.13.0",
            "azure-identity>=1.15.0",
        ],
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.23.0",
            "httpx>=0.26.0",
            "flake8>=7.0.0",
            "black>=24.1.0",
            "mypy>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fraud-train=src.train:run_training",
            "fraud-serve=api.app:main",
        ],
    },
)
