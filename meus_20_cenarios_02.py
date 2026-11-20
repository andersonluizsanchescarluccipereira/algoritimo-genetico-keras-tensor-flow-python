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
