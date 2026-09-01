@description('Nombre base usado para generar los nombres de los recursos. La longitud maxima esta establecida porque de este valor se deriva el nombre de la cuenta de almacenamiento (nombreBase + "st", guiones removidos), y las cuentas de almacenamiento de Azure exigen nombres de 3 a 24 caracteres alfanumericos en minusculas.')
@minLength(1)
@maxLength(20)
param nombreBase string = 'phishing-tesis'

@description('Region de despliegue')
param location string = resourceGroup().location

@description('Object ID del usuario/administrador al que se le da acceso de escritura de secretos en el Key Vault')
param adminObjectId string

@description('Tenant ID de Entra ID donde esta registrada la App (el mismo tenant del buzon monitoreado)')
param graphTenantId string

@description('Client (application) ID de la App Registration usada para autenticarse contra Microsoft Graph')
param graphClientId string

@description('Direccion del buzon que se va a monitorear (el mismo que se restringio con la Application Access Policy)')
param correoMonitoreado string

@description('Direccion de correo que recibe las alertas de phishing detectado')
param correoDestinoNotificacion string

@description('Id de la suscripcion de Microsoft Graph ya creada (README-azure.md paso 8). Vacio en el primer despliegue -- todavia no existe la suscripcion porque requiere que la Function ya este publicada. En redespliegues posteriores, pasar el valor real para no pisarlo con uno vacio.')
param graphSubscriptionId string = ''

var nombreStorage = toLower(replace(replace(replace(replace('${nombreBase}st', '-', ''), '_', ''), '.', ''), ' ', ''))
var nombreFunctionApp = '${nombreBase}-func'
var nombrePlan = '${nombreBase}-plan'
var nombreAppInsights = '${nombreBase}-ai'
var nombreLogAnalytics = '${nombreBase}-law'
var nombreKeyVault = '${nombreBase}-kv'
// Se calcula a partir del nombre de la Function App (no de
// functionApp.properties.defaultHostName) para poder usarla dentro de los
// appSettings de la propia Function App sin crear una referencia
// circular sobre el recurso que la contiene. Asume nube publica de Azure
// y sin dominio personalizado, que es el caso de este despliegue.
var urlNotificacion = 'https://${nombreFunctionApp}.azurewebsites.net/api/recibir_notificacion'

resource storage 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: nombreStorage
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
}

// Log Analytics workspace: requerido para crear Application Insights en su
// forma moderna "workspace-based". Microsoft dejo de soportar la creacion
// de componentes classic (sin WorkspaceResourceId) en febrero de 2024;
// muchas suscripciones ya lo bloquean por politica. SKU/retencion por
// defecto son suficientes para un despliegue de bajo volumen como este.
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: nombreLogAnalytics
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: nombreAppInsights
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

resource plan 'Microsoft.Web/serverfarms@2023-01-01' = {
  name: nombrePlan
  location: location
  sku: { name: 'Y1', tier: 'Dynamic' }
  properties: { reserved: true }
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: nombreKeyVault
  location: location
  properties: {
    sku: { family: 'A', name: 'standard' }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
  }
}

resource functionApp 'Microsoft.Web/sites@2023-01-01' = {
  name: nombreFunctionApp
  location: location
  kind: 'functionapp,linux'
  identity: { type: 'SystemAssigned' }
  properties: {
    serverFarmId: plan.id
    httpsOnly: true
    siteConfig: {
      linuxFxVersion: 'Python|3.12'
      appSettings: [
        { name: 'AzureWebJobsStorage', value: 'DefaultEndpointsProtocol=https;AccountName=${storage.name};AccountKey=${storage.listKeys().keys[0].value};EndpointSuffix=core.windows.net' }
        { name: 'FUNCTIONS_EXTENSION_VERSION', value: '~4' }
        { name: 'FUNCTIONS_WORKER_RUNTIME', value: 'python' }
        { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
        { name: 'RUTA_MODELO', value: 'modelo/svm_phishing.pkl' }
        { name: 'RUTA_VECTORIZADOR', value: 'modelo/vectorizador_tfidf.pkl' }
        { name: 'GRAPH_CLIENT_SECRET', value: '@Microsoft.KeyVault(SecretUri=${keyVault.properties.vaultUri}secrets/graph-client-secret/)' }
        { name: 'WEBHOOK_CLIENT_STATE', value: '@Microsoft.KeyVault(SecretUri=${keyVault.properties.vaultUri}secrets/webhook-client-state/)' }
        // Usada por el Timer trigger de renovacion como fallback: si el
        // PATCH de renovar_suscripcion falla (suscripcion ya expirada o
        // eliminada), se recrea desde cero apuntando a esta misma URL.
        { name: 'GRAPH_NOTIFICATION_URL', value: urlNotificacion }
        { name: 'GRAPH_TENANT_ID', value: graphTenantId }
        { name: 'GRAPH_CLIENT_ID', value: graphClientId }
        { name: 'CORREO_MONITOREADO', value: correoMonitoreado }
        { name: 'CORREO_DESTINO_NOTIFICACION', value: correoDestinoNotificacion }
        // Vacio en el primer despliegue (ver descripcion del parametro);
        // el Timer trigger de renovacion recien lo necesita despues del
        // paso 8 de README-azure.md, cuando ya se creo la suscripcion.
        { name: 'GRAPH_SUBSCRIPTION_ID', value: graphSubscriptionId }
      ]
    }
  }
}

resource rolKeyVaultSecretsUser 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  scope: subscription()
  name: '4633458b-17de-408a-b874-0445c86b69e6'
}

resource rolKeyVaultSecretsOfficer 'Microsoft.Authorization/roleDefinitions@2022-04-01' existing = {
  scope: subscription()
  name: 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7'
}

resource asignacionRolFunction 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, functionApp.id, 'kv-secrets-user')
  scope: keyVault
  properties: {
    roleDefinitionId: rolKeyVaultSecretsUser.id
    principalId: functionApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

resource asignacionRolAdmin 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, adminObjectId, 'kv-secrets-officer')
  scope: keyVault
  properties: {
    roleDefinitionId: rolKeyVaultSecretsOfficer.id
    principalId: adminObjectId
    principalType: 'User'
  }
}

output functionAppName string = functionApp.name
output keyVaultName string = keyVault.name
output notificationUrl string = urlNotificacion
