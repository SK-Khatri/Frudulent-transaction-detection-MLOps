#!/bin/bash
# ============================================================================
# Infrastructure Setup Script
# Creates Azure resources for the MLOps pipeline
# ============================================================================

set -euo pipefail

# Configuration (override via environment variables)
RG="${AZURE_RESOURCE_GROUP:-fraud-detection-rg}"
LOCATION="${AZURE_LOCATION:-eastus}"
WS="${AZURE_ML_WORKSPACE:-fraud-detection-ws}"
ACR="${ACR_NAME:-frauddetectionacr}"
AKS="${AKS_CLUSTER_NAME:-fraud-aks-cluster}"

echo "================================================"
echo "  Fraud Detection MLOps - Infrastructure Setup"
echo "================================================"
echo "  Resource Group: $RG"
echo "  Location:       $LOCATION"
echo "  ML Workspace:   $WS"
echo "  ACR:            $ACR"
echo "  AKS Cluster:    $AKS"
echo "================================================"

# Login check
echo "[1/7] Checking Azure login..."
az account show > /dev/null 2>&1 || { echo "Please run 'az login' first"; exit 1; }
SUBSCRIPTION=$(az account show --query id -o tsv)
echo "  Subscription: $SUBSCRIPTION"

# Resource Group
echo "[2/7] Creating resource group..."
az group create --name "$RG" --location "$LOCATION" --output none
echo "  Resource group '$RG' ready"

# Azure Container Registry
echo "[3/7] Creating Azure Container Registry..."
az acr create \
    --resource-group "$RG" \
    --name "$ACR" \
    --sku Standard \
    --admin-enabled true \
    --output none
echo "  ACR '$ACR' ready"

# Azure ML Workspace
echo "[4/7] Creating Azure ML Workspace..."
az ml workspace create \
    --resource-group "$RG" \
    --name "$WS" \
    --location "$LOCATION" \
    --output none
echo "  ML Workspace '$WS' ready"

# AKS Cluster
echo "[5/7] Creating AKS cluster..."
az aks create \
    --resource-group "$RG" \
    --name "$AKS" \
    --node-count 3 \
    --node-vm-size Standard_DS2_v2 \
    --enable-managed-identity \
    --attach-acr "$ACR" \
    --generate-ssh-keys \
    --output none
echo "  AKS cluster '$AKS' ready"

# Get AKS credentials
echo "[6/7] Configuring kubectl..."
az aks get-credentials --resource-group "$RG" --name "$AKS" --overwrite-existing
kubectl get nodes

# Create Kubernetes secrets
echo "[7/7] Creating Kubernetes secrets..."
if [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
    kubectl create secret generic kaggle-credentials \
        --from-literal=username="$KAGGLE_USERNAME" \
        --from-literal=key="$KAGGLE_KEY" \
        --dry-run=client -o yaml | kubectl apply -f -
    echo "  Kaggle credentials secret created"
fi

# ACR credentials for K8s
ACR_PASSWORD=$(az acr credential show --name "$ACR" --query "passwords[0].value" -o tsv)
kubectl create secret docker-registry acr-secret \
    --docker-server="${ACR}.azurecr.io" \
    --docker-username="$ACR" \
    --docker-password="$ACR_PASSWORD" \
    --dry-run=client -o yaml | kubectl apply -f -
echo "  ACR pull secret created"

echo ""
echo "================================================"
echo "  Infrastructure setup complete!"
echo "================================================"
echo ""
echo "  Next steps:"
echo "  1. Configure GitHub Secrets:"
echo "     - AZURE_CREDENTIALS (service principal JSON)"
echo "     - AZURE_SUBSCRIPTION_ID ($SUBSCRIPTION)"
echo "     - KAGGLE_USERNAME"
echo "     - KAGGLE_KEY"
echo "  2. Push code to GitHub"
echo "  3. CI/CD pipeline will trigger automatically"
