# 🧬🚗 Algoritmo Genético + Rede Neural: classificando veículos

> **Laboratório didático em Python, TensorFlow/Keras e Google Colab**  
> **Explicação nível: “nunca vi isso antes e quero entender cada pedaço”** 😄

Neste laboratório, você vai usar **duas técnicas trabalhando juntas**:

- **Rede neural**: aprende, a partir de exemplos, a classificar veículos em `entrada`, `intermediario` ou `premium`.
- **Algoritmo genético**: procura uma **configuração melhor para essa rede**, escolhendo quantos neurônios ela terá e qual será sua taxa de aprendizado.

**Objetivo:** entender a lógica do código que já funcionou no Google Colab, conseguir alterar um veículo de teste e executar **20 cenários prontos**.

---

## 📚 Índice

1. [Resumo em 30 segundos](#1-resumo-em-30-segundos)
2. [A história da escola de carros](#2-a-história-da-escola-de-carros)
3. [O que instalar no Colab](#3-o-que-instalar-no-colab)
4. [Explicação das 8 partes do código](#4-explicação-das-8-partes-do-código)
5. [Como a informação entra e sai da rede](#5-como-a-informação-entra-e-sai-da-rede)
6. [O que significam os resultados](#6-o-que-significam-os-resultados)
7. [Como testar um carro sozinho](#7-como-testar-um-carro-sozinho)
8. [20 cenários de teste](#8-20-cenários-de-teste)
9. [Código pronto: executar os 20 cenários no Colab](#9-código-pronto-executar-os-20-cenários-no-colab)
10. [Dicionário de termos difíceis](#10-dicionário-de-termos-difíceis)
11. [Pegadinhas e limitações importantes](#11-pegadinhas-e-limitações-importantes)
12. [Exercícios para aprender de verdade](#12-exercícios-para-aprender-de-verdade)

---

## 1. Resumo em 30 segundos

Imagine que você tem **600 fichas de veículos fictícios** e quer ensinar um computador a separar os carros em três caixas:

| Caixa | Nome no código | Preços usados para criar os exemplos |
|---|---|---|
| 🟢 0 | `entrada` | R$ 35.000 até R$ 65.000 |
| 🟡 1 | `intermediario` | R$ 75.000 até R$ 115.000 |
| 🔵 2 | `premium` | R$ 125.000 até R$ 190.000 |

**Detalhe importantíssimo:** essas faixas foram **inventadas para o exercício**. Não são a classificação real do mercado automotivo.

O programa faz isto:

```text
600 veículos fictícios
          |
          v
   Divide os dados
          |
          +--> 420: ensinar a rede (TREINO)
          +-->  90: medir durante a busca (VALIDAÇÃO)
          +-->  90: prova final (TESTE)
          |
          v
Algoritmo genético cria 6 configurações
          |
          v
Treina uma rede para cada configuração
          |
          v
Mede quem acertou mais na validação
          |
          v
Escolhe 2 configurações como "pais"
          |
          v
Cria filhos por cruzamento + mutação
          |
          v
Repete até completar 3 gerações
          |
          v
Escolhe a melhor configuração encontrada
          |
          v
Cria e treina uma nova rede com 510 exemplos
          |
          v
Faz uma prova final com os 90 de teste
          |
          v
Recebe um carro novo e mostra 3 probabilidades
```

**Guarde a frase:** o **algoritmo genético escolhe a receita**; a **rede neural aprende com os exemplos**.

---

## 2. A história da escola de carros

Pense na seguinte situação:

- Você é dono de uma **escola**.
- A **rede neural** é um **aluno**.
- Os **600 veículos** são as fichas de exercícios.
- Os **neurônios** são uma parte da estrutura do aluno para aprender padrões (não são neurônios biológicos).
- A **taxa de aprendizado** controla o tamanho dos ajustes que ele faz nos pesos durante o treinamento.
- O **algoritmo genético** é um **coordenador** que testa diferentes configurações de alunos e escolhe as melhores.

O coordenador pergunta:

> “Será que uma rede com 8 neurônios e taxa 0,001 aprende melhor ou pior que outra com 64 neurônios e taxa 0,01?”

Ele testa, compara, combina configurações e tenta melhorar.

**Cuidado com uma confusão comum:** o algoritmo genético **não faz a classificação diretamente**. Quem olha as características e produz as probabilidades é a **rede neural**.

---

## 3. O que instalar no Colab

1. Abra [Google Colab](https://colab.research.google.com/).
2. Crie um notebook Python.
3. Cole **todo o seu código original** em uma célula e execute pelo botão **▶**.
4. Depois que o treinamento terminar, crie **uma segunda célula** para os testes do [item 9](#9-código-pronto-executar-os-20-cenários-no-colab).

O Colab costuma incluir `numpy` e `tensorflow`, então normalmente não precisa instalar nada.

Se quiser conferir as versões, execute em uma célula:

```python
import sys
import numpy as np
import tensorflow as tf

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("TensorFlow:", tf.__version__)
print("GPUs disponíveis:", tf.config.list_physical_devices("GPU"))
```

**Se aparecer `GPUs disponíveis: []`, tudo bem.** Esse laboratório pequeno pode rodar na CPU. Não exige GPU.

> **Regra do Colab:** primeiro execute a célula do programa original; **só depois** execute a célula com os 20 testes. Se reiniciar a sessão, será preciso treinar novamente, pois `modelo_final` deixa de existir na memória.

---

## 4. Explicação das 8 partes do código

### 4.1. Configurações: as regras da brincadeira

```python
SEED = 42
NEURONIOS = [8, 16, 32, 64]
TAXAS = [0.001, 0.005, 0.01]

POPULACAO = 6
GERACOES = 3
EPOCAS = 12
```

**Tradução bem simples:**

| Variável | Explicação para quem está começando |
|---|---|
| `SEED = 42` | Controla a aleatoriedade para tentar reproduzir resultados. Não é garantia absoluta de resultados idênticos em todo ambiente. |
| `NEURONIOS` | Quantidades de neurônios que o algoritmo genético pode testar na camada escondida. |
| `TAXAS` | Taxas de aprendizado que ele pode experimentar. |
| `POPULACAO = 6` | Cada geração terá 6 **configurações candidatas**. |
| `GERACOES = 3` | O processo de seleção e reprodução acontece 3 vezes. |
| `EPOCAS = 12` | Cada rede é treinada passando 12 vezes pelos seus dados de treino. |

**O que é uma geração?** Uma rodada de competição entre 6 configurações.

**O que é uma época?** Uma passagem pelos dados durante o aprendizado. Não confunda **gerações do genético** com **épocas da rede**.

Uma configuração (também chamada de **indivíduo**) pode ser:

```python
(32, 0.005)
```

Isso significa:

- **32 neurônios** na camada escondida;
- **taxa de aprendizado 0,005**.

Há `4 × 3 = 12` combinações possíveis de configurações na lista, embora o programa possa testar combinações repetidas.

### 4.2. A base fictícia: fabricando 600 carros

```python
for classe in range(3):
    for _ in range(200):
        ...
```

Esse trecho produz:

- `classe = 0` → 200 exemplos de **entrada**;
- `classe = 1` → 200 exemplos de **intermediário**;
- `classe = 2` → 200 exemplos de **premium**.

**200 + 200 + 200 = 600 carros.**

No seu programa, a classe é escolhida **antes** de gerar o preço, e o preço muda conforme a classe:

```python
if classe == 0:
    preco = random.randint(35000, 65000)
elif classe == 1:
    preco = random.randint(75000, 115000)
else:
    preco = random.randint(125000, 190000)
```

**O grande segredo deste laboratório:** o **preço define o rótulo** dos dados fictícios. Ano, quilometragem, carroceria e combustível são gerados aleatoriamente, independentemente do segmento. Assim, a rede tem motivo para aprender principalmente o **preço**, e não deve ser interpretada como uma avaliadora real de veículos.

Exemplos de fichas fictícias (apenas ilustrações):

| Ano | Preço | Km | Carroceria | Combustível | Classe criada |
|---|---:|---:|---|---|---|
| 2020 | R$ 45.000 | 90.000 | SUV | Elétrico | Entrada |
| 2023 | R$ 95.000 | 20.000 | Hatch | Flex | Intermediário |
| 2016 | R$ 170.000 | 140.000 | Sedan | Diesel | Premium |

**Não estranhe um SUV elétrico de R$ 45 mil:** a base é artificial. No seu código, essa combinação é possível.

### 4.3. Normalização: colocar números grandes em uma escala menor

Veja a transformação:

```python
entrada = [
    (ano - 2015) / 11,
    preco / 200000,
    km / 150000,
]
```

Por que isso existe? Porque **2024**, **145000** e **25000** são números de tamanhos muito diferentes. A normalização ajuda a rede a trabalhar com escalas comparáveis.

Exemplo, para um veículo de 2024, R$ 145.000, 25.000 km:

| Campo | Conta | Valor aproximado |
|---|---|---:|
| Ano | `(2024 - 2015) / 11` | `0.8182` |
| Preço | `145000 / 200000` | `0.7250` |
| Km | `25000 / 150000` | `0.1667` |

O computador aprende a partir dos **números transformados**, não das strings `"R$ 145.000"` ou `"SUV"`.

A normalização usada é simples e depende das faixas escolhidas. Ela **não impede** que o usuário informe valores fora do intervalo.

### 4.4. One-hot encoding: transformar palavras em 0 e 1

A rede não recebe diretamente `"suv"` ou `"flex"`. Ela recebe sinais numéricos.

Para a carroceria:

| Carroceria | hatch | sedan | suv |
|---|---:|---:|---:|
| `hatch` | 1 | 0 | 0 |
| `sedan` | 0 | 1 | 0 |
| `suv` | 0 | 0 | 1 |

Para o combustível:

| Combustível | flex | diesel | eletrico |
|---|---:|---:|---:|
| `flex` | 1 | 0 | 0 |
| `diesel` | 0 | 1 | 0 |
| `eletrico` | 0 | 0 | 1 |

Portanto, **cada veículo tem 9 valores de entrada**:

```text
3 números:   ano, preço, quilometragem
3 posições: hatch, sedan, suv
3 posições: flex, diesel, eletrico
---------------------------------
TOTAL:       9 posições
```

Um SUV flex de 2024, R$ 145.000 e 25.000 km vira aproximadamente:

```python
[
    0.8182, 0.7250, 0.1667,  # ano, preço, km
    0, 0, 1,                 # SUV
    1, 0, 0                  # flex
]
```

**Se você escrever `"SUV"` com letras maiúsculas, seu código atual não reconhecerá o valor!** Ele exige `"suv"` exatamente. Isso será explicado nas limitações.

### 4.5. Separação treino / validação / teste: estudar, simular, fazer prova

```python
X_treino, y_treino = X[:420], y[:420]
X_val, y_val = X[420:510], y[420:510]
X_teste, y_teste = X[510:], y[510:]
```

Pense numa prova de escola:

| Conjunto | Quantidade | Função |
|---|---:|---|
| **Treino** | 420 (70%) | Exercícios que a rede usa para **ajustar seus pesos**. |
| **Validação** | 90 (15%) | Simulado para **comparar configurações** durante o genético. |
| **Teste** | 90 (15%) | Prova final, **não usada na seleção**. |

`X` são as características do veículo; `y` é a resposta correta da ficha (`0`, `1` ou `2`).

A linha `np.random.permutation(...)` embaralha as fichas **antes** da divisão. Assim, os conjuntos não ficam simplesmente separados por classe.

Mais tarde, a configuração escolhida é treinada com **treino + validação** (`420 + 90 = 510`), e a avaliação final continua nos **90 exemplos reservados**.

### 4.6. Rede neural: o aluno que aprende

```python
modelo = keras.Sequential([
    keras.Input(shape=(9,)),
    keras.layers.Dense(neuronios, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])
```

Sua rede é pequena:

```text
ENTRADA                 CAMADA ESCONDIDA             SAÍDA
9 características ----> 8, 16, 32 ou 64 neurônios --> 3 segmentos
                         (ReLU)                     (Softmax)

                                                       entrada
                                                       intermediario
                                                       premium
```

- **`Input(shape=(9,))`**: um carro chega como **9 números**.
- **`Dense(neuronios)`**: camada de neurônios, quantidade escolhida pelo algoritmo genético.
- **`relu`**: função que zera valores negativos e mantém os positivos, introduzindo não linearidade.
- **`Dense(3)`**: produz **3 saídas**, uma para cada segmento.
- **`softmax`**: transforma as 3 saídas em números não negativos que **somam 100%**.

Exemplo **inventado apenas para ilustrar**:

```text
entrada:        2%
intermediario:  8%
premium:       90%
```

O programa vai escolher `premium` por ser a maior saída. **Isso não significa 90% de certeza estatística garantida**: são pontuações do modelo convertidas pela Softmax; podem estar mal calibradas.

### 4.7. Como a rede aprende: optimizer, loss e accuracy

```python
modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=taxa),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

| Nome assustador | O que significa no dia a dia |
|---|---|
| `Adam` | Método que ajusta os **pesos internos** da rede para aprender. |
| `learning_rate` | Controla quanto esses pesos podem mudar em cada atualização. |
| `loss` | Mede o **erro da previsão**, penalizando previsões ruins. |
| `sparse_categorical_crossentropy` | Função de erro adequada quando há **uma classe correta**, marcada como número `0`, `1` ou `2`. |
| `accuracy` | Fração de exemplos em que a classe escolhida coincide com o rótulo correto. |

E o treinamento acontece aqui:

```python
modelo.fit(
    X_treino,
    y_treino,
    epochs=EPOCAS,
    batch_size=32,
    verbose=0
)
```

Em português:

> “Aluno, estude as 420 fichas por 12 épocas, em pequenos lotes de 32 exemplos; não imprima a evolução de cada época.”

**O que `verbose=0` faz?** Só esconde o progresso na tela. O treinamento acontece normalmente.

### 4.8. Fitness: a nota do aluno

```python
_, acuracia = modelo.evaluate(X_val, y_val, verbose=0)
return float(acuracia)
```

**Fitness = acurácia na validação.**

Se a rede acertar 81 das 90 fichas de validação, seu fitness será:

```text
81 / 90 = 0,90 = 90%
```

A palavra **fitness** não significa que um carro é bom: é a **nota da configuração de rede** naquela avaliação.

### 4.9. O algoritmo genético: competição de receitas

Agora sim, o outro protagonista do laboratório:

#### (A) Criar indivíduos

```python
def criar_individuo():
    return (
        random.choice(NEURONIOS),
        random.choice(TAXAS)
    )
```

O código sorteia uma configuração, por exemplo `(16, 0.01)`.

#### (B) Avaliar a população

```text
Geração 1 — exemplos ILUSTRATIVOS (não são resultados reais do seu Colab)

( 8, 0.001)  -> fitness 61%
(16, 0.005)  -> fitness 80%
(32, 0.010)  -> fitness 89%
(64, 0.001)  -> fitness 68%
( 8, 0.010)  -> fitness 82%
(32, 0.005)  -> fitness 86%
```

O programa ordena do maior fitness para o menor e escolhe **as 2 melhores configurações como pais**.

#### (C) Cruzamento: misturar partes de pai e mãe

```python
def cruzar(pai, mae):
    return (pai[0], mae[1])
```

Imagine:

```text
Pai:   (32 neurônios, taxa 0.010)
Mãe:   (16 neurônios, taxa 0.005)
Filho: (32 neurônios, taxa 0.005)
```

**Nesse código específico**, o filho pega sempre os **neurônios do primeiro pai** e a **taxa da segunda mãe**. Não é uma mistura aleatória de todos os genes.

#### (D) Mutação: dar uma chance a algo diferente

```python
if random.random() < 0.3:
    neuronios = random.choice(NEURONIOS)

if random.random() < 0.3:
    taxa = random.choice(TAXAS)
```

Cada gene tem **30% de chance de ser sorteado novamente**.

Exemplo:

```text
Antes:  (32, 0.005)
Depois: (64, 0.005)  <- mudou a quantidade de neurônios
```

**Atenção:** mutação não significa obrigatoriamente mudar o valor: o sorteio pode escolher o mesmo número.

#### (E) Elitismo: manter os dois melhores

```python
nova_populacao = pais.copy()
```

A nova geração começa com as **2 configurações vencedoras**. Depois, o código preenche o restante das 6 posições com filhos.

Isso ajuda a evitar perder as melhores **configurações**. Mas o código **não preserva os pesos já treinados** daquelas redes: cada avaliação treina uma rede nova, então uma mesma configuração pode receber fitness diferente em outra rodada.

#### (F) Quantos treinamentos acontecem?

```text
6 indivíduos × 3 gerações = 18 avaliações de redes
+ 1 novo treinamento da rede final
= 19 treinamentos de redes no total
```

Esse é o total previsto na execução normal **desse código**, mesmo quando o mesmo par de hiperparâmetros reaparece.

### 4.10. Escolhendo a melhor receita e fazendo a prova final

```python
X_final = np.concatenate([X_treino, X_val])
y_final = np.concatenate([y_treino, y_val])

modelo_final = criar_rede(melhor[0], melhor[1])
modelo_final.fit(X_final, y_final, epochs=EPOCAS, batch_size=32, verbose=0)
```

O algoritmo seleciona a **melhor combinação observada** de neurônios e taxa. Depois cria uma **nova rede**, do zero, com essa receita, e a ensina usando 510 fichas.

A prova final:

```python
_, acuracia_teste = modelo_final.evaluate(X_teste, y_teste, verbose=0)
```

Essa nota é uma estimativa mais honesta de como a rede funciona com fichas **não usadas para escolher a configuração**. Ainda é uma prova numa base **fictícia**, não no mercado real.

---

## 5. Como a informação entra e sai da rede

A função final é:

```python
classificar_veiculo(
    ano=2024,
    preco=145000,
    km=25000,
    carroceria="suv",
    combustivel="flex"
)
```

Passo a passo:

1. Recebe 5 informações de um veículo.
2. Converte ano, preço e km em números normalizados.
3. Transforma `"suv"` e `"flex"` em posições com zeros e uns.
4. Monta um vetor com **9 números**.
5. Envia o vetor à rede `modelo_final`.
6. Obtém as 3 saídas com `modelo_final.predict(...)`.
7. Imprime o percentual atribuído a cada classe.
8. Usa `np.argmax(...)` para localizar a **maior saída** e imprimir o segmento escolhido.

**O modelo não pesquisa preços na internet, não consulta FIPE, não reconhece marcas e não considera segurança ou acabamento do carro.** Ele só usa o padrão da base fictícia.

---

## 6. O que significam os resultados

Imagine que o terminal apresente algo como:

```text
MELHOR CONFIGURAÇÃO
Neurônios: 32
Learning rate: 0.01
Fitness: 95.56%
Acurácia teste: 96.67%

TESTE: VEÍCULO
entrada: 1.00%
intermediario: 6.00%
premium: 93.00%
Classificação: premium
```

> **Esses números são fictícios e apenas ilustrativos.** A execução do seu Colab pode retornar valores diferentes.

- **Neurônios = 32:** a melhor configuração observada usou 32 neurônios ocultos.
- **Learning rate = 0,01:** taxa escolhida para treinar a rede.
- **Fitness = 95,56%:** acurácia que a melhor configuração obteve em **uma avaliação de validação**.
- **Acurácia teste = 96,67%:** acurácia da nova rede nos **90 veículos reservados**.
- **`premium: 93%`:** saída Softmax para esse veículo específico, **não é a acurácia geral**.

**Três coisas diferentes:**

```text
FITNESS          -> nota de validação da configuração
ACURÁCIA TESTE   -> nota da rede final em carros reservados
PROBABILIDADES  -> saídas para um único carro
```

A acurácia de teste pode ser **maior ou menor** que o fitness: são avaliações diferentes e o modelo final é treinado novamente.

---

## 7. Como testar um carro sozinho

Depois de executar **o código original**, abra uma **nova célula** no Colab e digite:

```python
classificar_veiculo(
    ano=2020,
    preco=50000,
    km=80000,
    carroceria="hatch",
    combustivel="flex"
)
```

Isso imprime as três saídas e a classificação. Você pode alterar os cinco argumentos.

**Valores válidos esperados pelo laboratório:**

| Argumento | Como preencher |
|---|---|
| `ano` | Número inteiro; o treino usa 2015 a 2026. |
| `preco` | Número em reais **sem `R$` e sem ponto de milhar**. Ex.: `145000`. |
| `km` | Número de quilômetros; o treino usa 0 a 150000. |
| `carroceria` | Exatamente `"hatch"`, `"sedan"` ou `"suv"`. |
| `combustivel` | Exatamente `"flex"`, `"diesel"` ou `"eletrico"` (sem acento). |

**Não é preciso reexecutar as 3 gerações cada vez que você quiser testar um novo veículo.** Basta chamar `classificar_veiculo(...)` novamente enquanto `modelo_final` estiver na memória do Colab.

---

## 8. 20 cenários de teste

Os primeiros **18 testes** têm preços dentro das faixas usadas para gerar a base. A **expectativa didática** é que a rede aprenda as classes correspondentes; **isso não é uma garantia de acerto**.

Os testes **19 e 20** caem em intervalos de preço que **não existem na base de treinamento**. Neles, não há um gabarito definido pelo código: queremos observar como a rede se comporta **entre as faixas**.

| Nº | Cenário | Ano | Preço (R$) | Km | Carroceria | Combustível | Expectativa didática |
|---:|---|---:|---:|---:|---|---|---|
| 01 | Econômico e rodado | 2017 | 35.000 | 150.000 | hatch | flex | Entrada |
| 02 | Barato e recente | 2024 | 42.000 | 20.000 | sedan | eletrico | Entrada |
| 03 | SUV barato | 2022 | 50.000 | 60.000 | suv | diesel | Entrada |
| 04 | Barato, zero km | 2026 | 60.000 | 0 | hatch | flex | Entrada |
| 05 | Limite superior de entrada | 2015 | 65.000 | 145.000 | suv | eletrico | Entrada |
| 06 | Quase no limite de entrada | 2023 | 64.000 | 8.000 | sedan | diesel | Entrada |
| 07 | Limite inferior de intermediário | 2017 | 75.000 | 130.000 | sedan | flex | Intermediário |
| 08 | SUV intermediário | 2020 | 85.000 | 60.000 | suv | eletrico | Intermediário |
| 09 | Hatch recente intermediário | 2025 | 95.000 | 15.000 | hatch | diesel | Intermediário |
| 10 | Intermediário mais caro | 2018 | 105.000 | 95.000 | sedan | flex | Intermediário |
| 11 | Limite superior de intermediário | 2026 | 115.000 | 0 | suv | eletrico | Intermediário |
| 12 | Intermediário bem rodado | 2016 | 76.000 | 150.000 | hatch | flex | Intermediário |
| 13 | Limite inferior de premium | 2022 | 125.000 | 65.000 | suv | flex | Premium |
| 14 | Mesmo carro do teste original | 2024 | 145.000 | 25.000 | suv | flex | Premium |
| 15 | Premium antigo e rodado | 2015 | 165.000 | 145.000 | hatch | diesel | Premium |
| 16 | Premium novo e zero km | 2026 | 190.000 | 0 | sedan | eletrico | Premium |
| 17 | Premium bastante rodado | 2019 | 180.000 | 110.000 | suv | diesel | Premium |
| 18 | Premium próximo do limite inferior | 2025 | 130.000 | 5.000 | hatch | flex | Premium |
| 19 | **Lacuna entre entrada e intermediário** | 2024 | 70.000 | 30.000 | sedan | flex | **Indefinido: observar** |
| 20 | **Lacuna entre intermediário e premium** | 2023 | 120.000 | 30.000 | suv | flex | **Indefinido: observar** |

### Por que os testes 19 e 20 são especiais?

Na base original, não existe:

- Nenhum exemplo entre **R$ 65.001 e R$ 74.999**.
- Nenhum exemplo entre **R$ 115.001 e R$ 124.999**.

A rede **ainda vai dar uma resposta** para R$ 70.000 ou R$ 120.000, pois a Softmax sempre apresenta suas três saídas. Só que **não existe uma regra de classificação ensinada diretamente nessas lacunas**. É uma boa demonstração de que **uma resposta confiante não prova que ela foi bem fundamentada**.

---

## 9. Código pronto: executar os 20 cenários no Colab

### ✅ Onde colar?

1. **Primeira célula:** seu código original completo, incluindo o treinamento e a função `classificar_veiculo`.
2. **Segunda célula:** cole o bloco abaixo e execute **▶**.
3. Veja as **3 saídas e a classificação para cada cenário**.
4. Para repetir os 20 testes, execute **somente a segunda célula**. Não precisa treinar de novo.

```python
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
```

**Por que prever duas vezes por cenário?** A sua função `classificar_veiculo(...)` **só imprime**, não devolve a classe prevista. Para exibir a comparação automática, a célula reconstrói a entrada e chama `predict` outra vez. Isso aumenta um pouco o processamento, mas permite **testar o código original sem modificá-lo**.

**Esse resumo não substitui a `acuracia_teste`:** você escolheu os 20 cenários manualmente, e apenas 18 têm expectativa de classe. É um **check didático**, não uma nova medição estatística imparcial.

### 🔎 Exemplo do que observar

```text
CENÁRIO 14 — Mesmo carro do teste original
Expectativa didática: premium
--- Saída da sua função original ---
entrada: ...%
intermediario: ...%
premium: ...%
Classificação: ...
COMPARAÇÃO DIDÁTICA: ✅ COINCIDIU  # somente se a rede prever premium
```

Os `...%` são propositalmente espaços ilustrativos: o seu Colab vai calcular os valores reais.

---

## 10. Dicionário de termos difíceis

| Termo | Tradução para a vida real |
|---|---|
| **Machine Learning** | Fazer o computador aprender padrões a partir de exemplos. |
| **Rede neural** | Modelo com camadas e pesos ajustáveis que aprende relações entre entradas e saídas. |
| **TensorFlow** | Biblioteca para construir e treinar modelos de aprendizado de máquina. |
| **Keras** | Interface de alto nível usada para montar e treinar redes neurais. |
| **Dataset** | Conjunto de fichas/dados para aprender e testar. |
| **Feature** | Característica de entrada: ano, preço, km etc. |
| **Label / rótulo** | Resposta correta da ficha: entrada, intermediário ou premium. |
| **Normalização** | Ajustar a escala dos números, como dividir preço por 200.000. |
| **One-hot** | Representar uma categoria com várias posições de 0 e uma de 1. |
| **Neurônio** | Unidade matemática que combina valores e aplica uma função. |
| **Peso** | Número interno que a rede ajusta ao aprender. |
| **ReLU** | Função que mantém valores positivos e zera negativos. |
| **Softmax** | Converte saídas em valores que somam 1 (100%). |
| **Época** | Uma passagem de treino pelo conjunto de exemplos. |
| **Batch** | Pequeno lote de exemplos processado de cada vez. |
| **Learning rate** | Intensidade dos ajustes no treinamento. |
| **Loss** | Medida de quanto a previsão errou. |
| **Accuracy / acurácia** | Proporção de respostas classificadas corretamente. |
| **Hiperparâmetro** | Escolha feita antes do treino: neurônios e taxa são exemplos. |
| **Algoritmo genético** | Busca que imita seleção, cruzamento e mutação de configurações. |
| **Indivíduo** | Uma configuração candidata: `(neurônios, taxa)`. |
| **População** | Grupo de indivíduos candidatos da geração. |
| **Geração** | Uma rodada de avaliação e seleção. |
| **Fitness** | Nota usada para comparar os indivíduos. |
| **Cruzamento** | Combinar informações de dois indivíduos para gerar um filho. |
| **Mutação** | Sortear uma modificação em partes do indivíduo. |
| **Elitismo** | Preservar os melhores indivíduos para a geração seguinte. |
| **Inferência / predict** | Usar uma rede já treinada para classificar um novo exemplo. |
| **`argmax`** | Encontrar a posição da maior saída da rede. |

---

## 11. Pegadinhas e limitações importantes

### 11.1. Alta acurácia não quer dizer inteligência sobre o mercado

A classe foi determinada pela **faixa de preço inventada**. Se a rede conseguir 99% nessa base, ela mostrou que aprendeu a tarefa simulada; **não** que entende veículos reais ou qualidade de um carro.

### 11.2. Os percentuais da Softmax não são uma garantia

`premium: 95%` não significa que o veículo está “95% comprovadamente” na categoria premium. Para interpretar esses valores como probabilidades confiáveis, seria necessário verificar a **calibração** do modelo com dados adequados.

### 11.3. Ano, km, carroceria e combustível têm pouca informação real nessa base

Eles foram sorteados independentemente da classe. Pode parecer que a rede aprendeu a “preferir SUV”, mas isso pode ser **ruído** de uma amostra pequena. Alterar apenas o tipo de carroceria não deveria ser interpretado como avaliação real do veículo.

### 11.4. Lacunas: R$ 70 mil e R$ 120 mil

Não há fichas de treino nessas regiões de preço. Mesmo assim, a rede será obrigada a escolher um dos três segmentos. **Por isso os cenários 19 e 20 não têm gabarito definido**.

### 11.5. O código não valida valores absurdos

Se você tentar:

```python
classificar_veiculo(ano=2035, preco=-5000, km=-10,
                    carroceria="SUV", combustivel="gasolina")
```

A rede ainda pode devolver percentuais — **mas a entrada não representa um exemplo válido do laboratório**. Pior: `"SUV"` e `"gasolina"` não correspondem às listas, então seus blocos one-hot ficam todos zerados. **Isso não é um teste válido de precisão.**

Em uma aplicação real, adicione validação de tipos, limites e valores permitidos antes do `predict`.

### 11.6. O algoritmo genético testa configurações, não evolui pesos

Ele só evolui os **dois hiperparâmetros**: `neuronios` e `taxa`. Os pesos da rede são aprendidos pelo otimizador **Adam** durante cada treinamento. O genético não cria novos pesos misturando redes.

### 11.7. Fitness e rede final não são a mesma rede

O melhor fitness pertence a **uma avaliação** com treino de 420 exemplos. A rede final é criada e treinada **novamente**, agora usando 510 exemplos. Portanto, a acurácia de teste pode diferir da de validação.

### 11.8. Mesmo indivíduo, novo resultado

O mesmo par `(32, 0.005)` pode ser avaliado em duas gerações. O código cria e treina uma nova rede a cada vez. Por causa de inicialização aleatória, embaralhamento e outros detalhes, as notas podem mudar. A configuração é preservada pelo elitismo, **não os pesos treinados**.

### 11.9. Há poucas configurações possíveis

Com 4 opções de neurônios e 3 taxas, existem **somente 12 combinações**. Para esse laboratório pequeno, testar todas as 12 diretamente seria uma alternativa simples ao algoritmo genético. O genético aqui é útil sobretudo para **aprender o conceito**; ele ganha mais sentido quando a busca fica maior e mais cara.

### 11.10. Não há garantia de encontrar a melhor configuração absoluta

Com 6 indivíduos, 3 gerações e mutação aleatória, o algoritmo genético encontra a **melhor configuração observada durante a busca**, não necessariamente a melhor de todas as possíveis, e não necessariamente a melhor para qualquer base futura.

---

## 12. Exercícios para aprender de verdade

Faça na ordem, sem pressa:

1. **Preço manda?** Pegue o cenário 14 e mantenha o preço em `145000`, mas troque `suv` por `hatch`. Compare as saídas. Repita trocando `flex` por `diesel`. O que mudou?
2. **Teste a fronteira.** Compare os cenários 05 (`65000`), 19 (`70000`) e 07 (`75000`). Veja como as saídas variam entre classes.
3. **Mais treinamento.** Troque `EPOCAS = 12` por `EPOCAS = 30` e execute **tudo novamente**. A acurácia necessariamente melhora? Não: compare os resultados.
4. **Mude a população.** Troque `POPULACAO = 6` por `POPULACAO = 10`. Quantas avaliações ocorrerão em 3 gerações? `10 × 3 = 30`, além do treino final.
5. **Remova pistas inúteis (desafio).** Crie uma versão da base que use **apenas preço** como entrada. Ajuste `Input(shape=(1,))` e o preparo dos dados e da previsão. Compare com o modelo de 9 características. Isso testa a hipótese de que **preço contém a regra de classificação**.

**Resultado esperado do aprendizado:** você deve saber explicar a diferença entre **dado de treino**, **rede neural**, **fitness**, **algoritmo genético**, **hiperparâmetro**, **teste final** e **predição de um carro novo**.

---

## 🧠 Cola final de uma linha

> **O algoritmo genético testa receitas de redes; o TensorFlow treina cada rede; a validação decide quais receitas parecem melhores; o teste final avalia a rede escolhida; e a função `classificar_veiculo` apresenta o resultado para um carro novo.**

**Pronto: agora é só executar os 20 testes do item 9 no Colab e observar as diferenças.** 🚗
