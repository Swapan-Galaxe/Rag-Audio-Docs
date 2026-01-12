# CI/CD Setup Guide

## Overview
Automatic deployment to Azure Container Apps on every push to `main` branch.

## Setup Steps

### 1. Create Azure Service Principal

```bash
az ad sp create-for-rbac \
  --name "rag-chatbot-github" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/rag-chatbot-rg \
  --sdk-auth
```

Copy the JSON output (you'll need it for GitHub secrets).

### 2. Configure GitHub Secrets

Go to: **GitHub Repo → Settings → Secrets and variables → Actions**

Add these secrets:

**AZURE_CREDENTIALS**
```json
{
  "clientId": "xxx",
  "clientSecret": "xxx",
  "subscriptionId": "xxx",
  "tenantId": "xxx"
}
```
(Paste the JSON from step 1)

**OPENAI_API_KEY**
```
your-openai-api-key
```

### 3. Initial Azure Setup (One-time)

Run this once to create Azure resources:

```bash
export OPENAI_API_KEY="your-key"
bash deploy_azure.sh
```

### 4. Push Code to GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

The CI/CD pipeline will automatically:
1. Build Docker image
2. Push to Azure Container Registry
3. Deploy to Azure Container Apps
4. Display the app URL

## Workflow Triggers

- **Automatic**: Push to `main` branch
- **Manual**: Go to Actions tab → Run workflow

## Monitor Deployment

1. Go to GitHub repo → **Actions** tab
2. Click on latest workflow run
3. View logs and deployment status

## Update Deployment

Just push code:
```bash
git add .
git commit -m "Update chatbot"
git push origin main
```

Auto-deploys in ~3-5 minutes.

## Rollback

```bash
# List previous images
az acr repository show-tags --name ragchatbotacr --repository rag-chatbot

# Deploy specific version
az containerapp update \
  --name rag-chatbot \
  --resource-group rag-chatbot-rg \
  --image ragchatbotacr.azurecr.io/rag-chatbot:{commit-sha}
```

## Cost
- GitHub Actions: Free (2,000 minutes/month)
- Azure: Same as before (~$35-55/month)
