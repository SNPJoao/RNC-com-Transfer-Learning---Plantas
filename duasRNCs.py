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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Treinando no dispositivo: {device}")

# ==========================================
# PARTE 2: FUNÇÃO PARA TREINAR OS MODELOS 
# ==========================================

def treinar_modelo(modelo, nome_modelo, learning_rate, epocas):
    print(f"\n---Treinando modelo {nome_modelo} com learning rate {learning_rate}")
    modelo = modelo.to(device)
    criterio_erro = nn.CrossEntropyLoss() # A função matemática de perda
    parametros_treinaveis = filter(lambda p: p.requires_grad, modelo.parameters())
    otimizador = optim.Adam(parametros_treinaveis, lr=learning_rate) # Otimiza APENAS a nova camada final

    melhor_acuracia = 0.0

    for epoca in range(epocas):
        modelo.train() # Avisa ao PyTorch que estamos no modo de treino
    
        # --- LOOP DE TREINO ---
        for imagens, rotulos in loader_treino:
            imagens, rotulos = imagens.to(device), rotulos.to(device)
        
            otimizador.zero_grad()      # 1. Zera a memória de erros do passo anterior
            saidas = modelo(imagens)    # 2. A rede tenta adivinhar a doença (Forward Pass)
            erro = criterio_erro(saidas, rotulos) # 3. Calcula o quão errada ela foi
            erro.backward()             # 4. Calcula a correção necessária (Backpropagation)
            otimizador.step()           # 5. Aplica a correção nos pesos da última camada
        
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
        melhor_acuracia = max(melhor_acuracia, acuracia_val) 
        print(f"Época [{epoca+1}/{epocas}] | Acurácia Validação: {acuracia_val:.4f}")

    print("Treinamento finalizado!")
    return melhor_acuracia

# ===================================================
# PARTE 3: MODELOS PARA TREINAR E COMPARAR RESULTADOS 
# ===================================================

resultados = {}

densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
for params in densenet.parameters():
    params.requires_grad = False
num_features = densenet.classifier.in_features
densenet.classifier = nn.Linear(num_features, NUM_CLASSES)

mobilenetv2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
for param in mobilenetv2.parameters():
    param.requires_grad = False
num_features = mobilenetv2.classifier[1].in_features
mobilenetv2.classifier[1] = nn.Linear(num_features, NUM_CLASSES)

EPOCAS = 5

resultados["DesneNet121 (Lr = 0.001)"] = treinar_modelo(densenet, "DenseNet121", 0.001, EPOCAS)
resultados["MobileNetV2 (Lr = 0.0005)"] = treinar_modelo(mobilenetv2, "MobileNetV2", 0.0005, EPOCAS)

for nome, acuracia in resultados:
    print(f"Modelo {nome} | Melhor acurácia = {acuracia:.4f}")
