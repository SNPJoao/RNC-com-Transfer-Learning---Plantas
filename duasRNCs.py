import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# ==========================================
# PARTE 1: PIPELINE DE DADOS (TRANSFORMS E DATALOADERS)
# ==========================================
DIRETORIO_DADOS = "PlantVillage-Dataset-master/data_distribution_for_SVM"

BATCH_SIZE = 32

# O PyTorch exige que nós definamos as transformações matemáticas das imagens
transformacoes = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # Transforma a imagem em matriz matemática (Tensor)
    # A normalização abaixo é o padrão exigido pelas redes treinadas na base ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

print("Carregando datasets...")
dados_treino = datasets.ImageFolder(os.path.join(DIRETORIO_DADOS, 'train'), transform=transformacoes)
dados_validacao = datasets.ImageFolder(os.path.join(DIRETORIO_DADOS, 'test'), transform=transformacoes)

NUM_CLASSES = len(dados_treino.classes) # Deve resultar em 38 para o dataset de plantas

# Os DataLoaders são os responsáveis por entregar os "lotes" (batches) de imagens para a rede
loader_treino = torch.utils.data.DataLoader(dados_treino, batch_size=BATCH_SIZE, shuffle=True)
loader_validacao = torch.utils.data.DataLoader(dados_validacao, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# PARTE 2: MONTANDO O MODELO (TRANSFER LEARNING)
# ==========================================
# Configurando o hardware (Usa GPU se disponível, senão usa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Treinando no dispositivo: {device}")

# 2.1 Importando a ResNet50 já treinada
modelo = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 2.2 Congelando a base (Evita destruir o conhecimento prévio)
for param in modelo.parameters():
    param.requires_grad = False

# 2.3 Trocando a "cabeça" da rede
# A última camada da ResNet50 original no PyTorch chama-se "fc" (Fully Connected).
# Pegamos o número de neurônios que entram nela e criamos uma nova saída com 38 neurônios.
num_features = modelo.fc.in_features
modelo.fc = nn.Linear(num_features, NUM_CLASSES)

# Envia o modelo para o processador ou placa de vídeo
modelo = modelo.to(device)

# ==========================================
# PARTE 3: COMPILADOR E LOOP DE TREINAMENTO
# ==========================================
criterio_erro = nn.CrossEntropyLoss() # A função matemática de perda
otimizador = optim.Adam(modelo.fc.parameters(), lr=0.001) # Otimiza APENAS a nova camada final

EPOCAS = 10

print("Iniciando o Treinamento...")

for epoca in range(EPOCAS):
    modelo.train() # Avisa ao PyTorch que estamos no modo de treino
    erro_treino_acumulado = 0.0
    acertos_treino = 0
    total_treino = 0
    
    # --- LOOP DE TREINO ---
    for imagens, rotulos in loader_treino:
        imagens, rotulos = imagens.to(device), rotulos.to(device)
        
        otimizador.zero_grad()      # 1. Zera a memória de erros do passo anterior
        saidas = modelo(imagens)    # 2. A rede tenta adivinhar a doença (Forward Pass)
        erro = criterio_erro(saidas, rotulos) # 3. Calcula o quão errada ela foi
        erro.backward()             # 4. Calcula a correção necessária (Backpropagation)
        otimizador.step()           # 5. Aplica a correção nos pesos da última camada
        
        # Guardando métricas
        erro_treino_acumulado += erro.item()
        _, previsoes = torch.max(saidas, 1)
        acertos_treino += (previsoes == rotulos).sum().item()
        total_treino += rotulos.size(0)
        
    acuracia_treino = acertos_treino / total_treino
    
    # --- LOOP DE VALIDAÇÃO ---
    modelo.eval() # Avisa ao PyTorch que estamos apenas testando (desliga Dropouts, etc)
    acertos_val = 0
    total_val = 0
    
    with torch.no_grad(): # Desliga o cálculo de gradientes para economizar memória e ficar rápido
        for imagens, rotulos in loader_validacao:
            imagens, rotulos = imagens.to(device), rotulos.to(device)
            saidas = modelo(imagens)
            
            _, previsoes = torch.max(saidas, 1)
            acertos_val += (previsoes == rotulos).sum().item()
            total_val += rotulos.size(0)
            
    acuracia_val = acertos_val / total_val
    
    print(f"Época [{epoca+1}/{EPOCAS}] | Acurácia Treino: {acuracia_treino:.4f} | Acurácia Validação: {acuracia_val:.4f}")

print("Treinamento finalizado!")
