import random
import numpy as np
import tensorflow as tf

from tensorflow import keras

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CARROCERIAS = ["hatch", "sedan", "suv"]
COMBUSTIVEIS = ["flex", "diesel", "eletrico"]
SEGMENTOS = ["entrada", "intermediario", "premium"]

# Hiperparâmetros possíveis
NEURONIOS = [8, 16, 32, 64]
TAXAS = [0.001, 0.005, 0.01]

POPULACAO = 6
GERACOES = 3
EPOCAS = 12

# ==========================================
# 2. BASE DE VEÍCULOS FICTÍCIOS
# ==========================================

def gerar_base():
    dados = []
    rotulos = []

    # 200 exemplos por segmento
    for classe in range(3):
        for _ in range(200):

            if classe == 0:
                preco = random.randint(35000, 65000)
            elif classe == 1:
                preco = random.randint(75000, 115000)
            else:
                preco = random.randint(125000, 190000)

            ano = random.randint(2015, 2026)
            km = random.randint(0, 150000)
            carroceria = random.randrange(3)
            combustivel = random.randrange(3)

            # Normalização de dados numéricos
            entrada = [
                (ano - 2015) / 11,
                preco / 200000,
                km / 150000,
            ]

            # One-hot encoding
            entrada += [
                int(carroceria == i)
                for i in range(3)
            ]

            entrada += [
                int(combustivel == i)
                for i in range(3)
            ]

            dados.append(entrada)
            rotulos.append(classe)

    X = np.array(dados, dtype="float32")
    y = np.array(rotulos, dtype="int32")

    indices = np.random.permutation(len(X))

    return X[indices], y[indices]


X, y = gerar_base()

# Treino: 70%
# Validação: 15%
# Teste: 15%
X_treino, y_treino = X[:420], y[:420]
X_val, y_val = X[420:510], y[420:510]
X_teste, y_teste = X[510:], y[510:]

# ==========================================
# 3. CRIAÇÃO DA REDE NEURAL
# ==========================================

def criar_rede(neuronios, taxa):

    modelo = keras.Sequential([
        keras.Input(shape=(9,)),

        keras.layers.Dense(
            neuronios,
            activation="relu"
        ),

        keras.layers.Dense(
            3,
            activation="softmax"
        )
    ])

    modelo.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=taxa
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modelo


# ==========================================
# 4. FUNÇÃO FITNESS
# ==========================================

def avaliar(individuo):

    neuronios, taxa = individuo

    # Cada indivíduo cria sua própria rede
    tf.keras.backend.clear_session()

    modelo = criar_rede(neuronios, taxa)

    modelo.fit(
        X_treino,
        y_treino,
        epochs=EPOCAS,
        batch_size=32,
        verbose=0
    )

    # Fitness = acurácia de validação
    _, acuracia = modelo.evaluate(
        X_val,
        y_val,
        verbose=0
    )

    return float(acuracia)


# ==========================================
# 5. ALGORITMO GENÉTICO
# ==========================================

def criar_individuo():
    return (
        random.choice(NEURONIOS),
        random.choice(TAXAS)
    )


def cruzar(pai, mae):
    # Herda os neurônios do pai
    # e a taxa de aprendizado da mãe
    return (pai[0], mae[1])


def mutar(individuo):
    neuronios, taxa = individuo

    if random.random() < 0.3:
        neuronios = random.choice(NEURONIOS)

    if random.random() < 0.3:
        taxa = random.choice(TAXAS)

    return (neuronios, taxa)


populacao = [
    criar_individuo()
    for _ in range(POPULACAO)
]

melhor = None
melhor_fitness = -1.0

for geracao in range(GERACOES):

    print(f"\nGERAÇÃO {geracao + 1}")

    resultados = []

    for individuo in populacao:

        fitness = avaliar(individuo)

        resultados.append(
            (fitness, individuo)
        )

        print(
            f"Neurônios={individuo[0]} "
            f"Taxa={individuo[1]} "
            f"Fitness={fitness:.2%}"
        )

    resultados.sort(
        key=lambda resultado: resultado[0],
        reverse=True
    )

    if resultados[0][0] > melhor_fitness:
        melhor_fitness, melhor = resultados[0]

    # Seleciona os 2 melhores
    pais = [
        resultado[1]
        for resultado in resultados[:2]
    ]

    # Elitismo: preserva os melhores
    nova_populacao = pais.copy()

    while len(nova_populacao) < POPULACAO:

        filho = cruzar(pais[0], pais[1])
        filho = mutar(filho)

        nova_populacao.append(filho)

    populacao = nova_populacao


# ==========================================
# 6. MELHOR CONFIGURAÇÃO
# ==========================================

print("\nMELHOR CONFIGURAÇÃO")
print("Neurônios:", melhor[0])
print("Learning rate:", melhor[1])
print("Fitness:", f"{melhor_fitness:.2%}")

# Treina novamente utilizando treino + validação
X_final = np.concatenate([X_treino, X_val])
y_final = np.concatenate([y_treino, y_val])

modelo_final = criar_rede(
    melhor[0],
    melhor[1]
)

modelo_final.fit(
    X_final,
    y_final,
    epochs=EPOCAS,
    batch_size=32,
    verbose=0
)

# Teste reservado, não usado na seleção
_, acuracia_teste = modelo_final.evaluate(
    X_teste,
    y_teste,
    verbose=0
)

print("Acurácia teste:", f"{acuracia_teste:.2%}")


# ==========================================
# 7. CLASSIFICAR UM NOVO VEÍCULO
# ==========================================

def classificar_veiculo(
    ano,
    preco,
    km,
    carroceria,
    combustivel
):
    entrada = [
        (ano - 2015) / 11,
        preco / 200000,
        km / 150000,
    ]

    entrada += [
        int(carroceria == c)
        for c in CARROCERIAS
    ]

    entrada += [
        int(combustivel == c)
        for c in COMBUSTIVEIS
    ]

    entrada = np.array(
        [entrada],
        dtype="float32"
    )

    probabilidades = modelo_final.predict(
        entrada,
        verbose=0
    )[0]

    for segmento, probabilidade in zip(
        SEGMENTOS,
        probabilidades
    ):
        print(
            f"{segmento}: {probabilidade:.2%}"
        )

    escolhido = int(np.argmax(probabilidades))

    print(
        "Classificação:",
        SEGMENTOS[escolhido]
    )


# ==========================================
# 8. CENÁRIO DE TESTE
# ==========================================

print("\nTESTE: VEÍCULO")

classificar_veiculo(
    ano=2024,
    preco=145000,
    km=25000,
    carroceria="suv",
    combustivel="flex"
)
