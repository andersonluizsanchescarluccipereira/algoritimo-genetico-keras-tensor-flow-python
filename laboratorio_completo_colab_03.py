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


# =================================================================
# TESTES EXTRAS — 20 CENÁRIOS
# =================================================================

# ===========================================================
# LABORATÓRIO EXTRA — 20 CENÁRIOS DE TESTE
# Cole esta célula DEPOIS de executar seu código original.
# ===========================================================

cenarios = [
    # 6 cenários na faixa de ENTRADA
    {"id": 1,  "nome": "Econômico e rodado",                     "ano": 2017, "preco": 35000,  "km": 150000, "carroceria": "hatch", "combustivel": "flex",     "esperado": "entrada"},
    {"id": 2,  "nome": "Barato e recente",                      "ano": 2024, "preco": 42000,  "km": 20000,  "carroceria": "sedan", "combustivel": "eletrico", "esperado": "entrada"},
    {"id": 3,  "nome": "SUV barato",                            "ano": 2022, "preco": 50000,  "km": 60000,  "carroceria": "suv",   "combustivel": "diesel",   "esperado": "entrada"},
    {"id": 4,  "nome": "Barato, zero km",                       "ano": 2026, "preco": 60000,  "km": 0,      "carroceria": "hatch", "combustivel": "flex",     "esperado": "entrada"},
    {"id": 5,  "nome": "Limite superior de entrada",             "ano": 2015, "preco": 65000,  "km": 145000, "carroceria": "suv",   "combustivel": "eletrico", "esperado": "entrada"},
    {"id": 6,  "nome": "Quase no limite de entrada",            "ano": 2023, "preco": 64000,  "km": 8000,   "carroceria": "sedan", "combustivel": "diesel",   "esperado": "entrada"},

    # 6 cenários na faixa INTERMEDIÁRIA
    {"id": 7,  "nome": "Limite inferior de intermediário",      "ano": 2017, "preco": 75000,  "km": 130000, "carroceria": "sedan", "combustivel": "flex",     "esperado": "intermediario"},
    {"id": 8,  "nome": "SUV intermediário",                     "ano": 2020, "preco": 85000,  "km": 60000,  "carroceria": "suv",   "combustivel": "eletrico", "esperado": "intermediario"},
    {"id": 9,  "nome": "Hatch recente intermediário",           "ano": 2025, "preco": 95000,  "km": 15000,  "carroceria": "hatch", "combustivel": "diesel",   "esperado": "intermediario"},
    {"id": 10, "nome": "Intermediário mais caro",              "ano": 2018, "preco": 105000, "km": 95000,  "carroceria": "sedan", "combustivel": "flex",     "esperado": "intermediario"},
    {"id": 11, "nome": "Limite superior de intermediário",      "ano": 2026, "preco": 115000, "km": 0,      "carroceria": "suv",   "combustivel": "eletrico", "esperado": "intermediario"},
    {"id": 12, "nome": "Intermediário bem rodado",             "ano": 2016, "preco": 76000,  "km": 150000, "carroceria": "hatch", "combustivel": "flex",     "esperado": "intermediario"},

    # 6 cenários na faixa PREMIUM
    {"id": 13, "nome": "Limite inferior de premium",            "ano": 2022, "preco": 125000, "km": 65000,  "carroceria": "suv",   "combustivel": "flex",     "esperado": "premium"},
    {"id": 14, "nome": "Mesmo carro do teste original",        "ano": 2024, "preco": 145000, "km": 25000,  "carroceria": "suv",   "combustivel": "flex",     "esperado": "premium"},
    {"id": 15, "nome": "Premium antigo e rodado",              "ano": 2015, "preco": 165000, "km": 145000, "carroceria": "hatch", "combustivel": "diesel",   "esperado": "premium"},
    {"id": 16, "nome": "Premium novo e zero km",              "ano": 2026, "preco": 190000, "km": 0,      "carroceria": "sedan", "combustivel": "eletrico", "esperado": "premium"},
    {"id": 17, "nome": "Premium bastante rodado",              "ano": 2019, "preco": 180000, "km": 110000, "carroceria": "suv",   "combustivel": "diesel",   "esperado": "premium"},
    {"id": 18, "nome": "Premium próximo do limite inferior",   "ano": 2025, "preco": 130000, "km": 5000,   "carroceria": "hatch", "combustivel": "flex",     "esperado": "premium"},

    # 2 cenários de EXPLORAÇÃO: preços nunca vistos na base
    {"id": 19, "nome": "Lacuna: entrada / intermediário",       "ano": 2024, "preco": 70000,  "km": 30000,  "carroceria": "sedan", "combustivel": "flex",     "esperado": None},
    {"id": 20, "nome": "Lacuna: intermediário / premium",       "ano": 2023, "preco": 120000, "km": 30000,  "carroceria": "suv",   "combustivel": "flex",     "esperado": None},
]

# O código original precisa ter criado esta variável:
assert "modelo_final" in globals(), (
    "Execute primeiro a célula principal: a rede ainda não foi treinada."
)

acertos = 0
avaliados = 0

for c in cenarios:
    print("\n" + "=" * 65)
    print(f"CENÁRIO {c['id']:02d} — {c['nome']}")
    print("=" * 65)
    print(f"Ano: {c['ano']} | Preço: R$ {c['preco']:,.0f} | Km: {c['km']:,}")
    print(f"Carroceria: {c['carroceria']} | Combustível: {c['combustivel']}")
    print(f"Expectativa didática: {c['esperado'] or 'INDEFINIDA (lacuna)'}")
    print("--- Saída da sua função original ---")

    # Reutilizamos sua função, SEM alterar o laboratório principal.
    classificar_veiculo(
        ano=c["ano"],
        preco=c["preco"],
        km=c["km"],
        carroceria=c["carroceria"],
        combustivel=c["combustivel"],
    )

    # Como a função original só IMPRIME e não retorna o resultado,
    # repetimos a preparação da entrada para obter o rótulo de forma
    # automática e comparar com a expectativa didática.
    vetor = [
        (c["ano"] - 2015) / 11,
        c["preco"] / 200000,
        c["km"] / 150000,
    ]
    vetor += [int(c["carroceria"] == opcao) for opcao in CARROCERIAS]
    vetor += [int(c["combustivel"] == opcao) for opcao in COMBUSTIVEIS]

    previsao = modelo_final.predict(
        np.array([vetor], dtype="float32"), verbose=0
    )[0]
    classe_prevista = SEGMENTOS[int(np.argmax(previsao))]

    if c["esperado"] is None:
        print("RESULTADO DO EXPERIMENTO: sem gabarito; observe a resposta.")
    else:
        avaliados += 1
        acertou = classe_prevista == c["esperado"]
        acertos += int(acertou)
        print("COMPARAÇÃO DIDÁTICA:", "✅ COINCIDIU" if acertou else "❌ DIFERIU")

print("\n" + "=" * 65)
print("RESUMO DOS 20 CENÁRIOS")
print(f"Cenários executados: {len(cenarios)}")
print(f"Com expectativa definida: {avaliados}")
print(f"Coincidiram com a expectativa: {acertos}/{avaliados}")
print("Cenários exploratórios sem gabarito: 2")
print("=" * 65)
