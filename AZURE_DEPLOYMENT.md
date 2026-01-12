# Deploy to Azure Container Apps

## Prerequisites
- Azure CLI installed
- Azure subscription
- OpenAI API key

## Quick Deploy

### 1. Set Environment Variable
```bash
export OPENAI_API_KEY="your-openai-key"
```

### 2. Run Deployment Script
```bash
bash deploy_azure.sh
```

## Manual Deployment

### 1. Login to Azure
```bash
az login
```

### 2. Create Resource Group
```bash
az group create --name rag-chatbot-rg --location eastus
```

### 3. Create Container Registry
```bash
az acr create --resource-group rag-chatbot-rg --name ragchatbotacr --sku Basic --admin-enabled true
```

### 4. Build and Push Image
```bash
az acr build --registry ragchatbotacr --image rag-chatbot:latest .
```

### 5. Create Container Apps Environment
```bash
az containerapp env create \
  --name rag-chatbot-env \
  --resource-group rag-chatbot-rg \
  --location eastus
```

### 6. Deploy Container App
```bash
az containerapp create \
  --name rag-chatbot \
  --resource-group rag-chatbot-rg \
  --environment rag-chatbot-env \
  --image ragchatbotacr.azurecr.io/rag-chatbot:latest \
  --registry-server ragchatbotacr.azurecr.io \
  --target-port 7860 \
  --ingress external \
  --cpu 1.0 \
  --memory 2.0Gi \
  --min-replicas 1 \
  --max-replicas 3 \
  --secrets openai-key=$OPENAI_API_KEY \
  --env-vars OPENAI_API_KEY=secretref:openai-key
```

### 7. Get App URL
```bash
az containerapp show --name rag-chatbot --resource-group rag-chatbot-rg --query properties.configuration.ingress.fqdn -o tsv
```

## Update Deployment

```bash
# Rebuild image
az acr build --registry ragchatbotacr --image rag-chatbot:latest .

# Update container app
az containerapp update \
  --name rag-chatbot \
  --resource-group rag-chatbot-rg \
  --image ragchatbotacr.azurecr.io/rag-chatbot:latest
```

## Monitor

```bash
# View logs
az containerapp logs show --name rag-chatbot --resource-group rag-chatbot-rg --follow

# View metrics
az monitor metrics list --resource /subscriptions/{subscription-id}/resourceGroups/rag-chatbot-rg/providers/Microsoft.App/containerApps/rag-chatbot
```

## Cleanup

```bash
az group delete --name rag-chatbot-rg --yes --no-wait
```

## Cost Estimate
- Container Apps: ~$30-50/month (1 vCPU, 2GB RAM)
- Container Registry: ~$5/month (Basic tier)
- Total: ~$35-55/month
