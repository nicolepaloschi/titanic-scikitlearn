import streamlit as st
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
plt.style.use('fivethirtyeight') # estilo FiveThirtyEight para gráficos mais limpos e profissionais
warnings.filterwarnings('ignore') # não exibe os warnings

# Title
st.title("Previsão de Sobreviventes do Titanic com Machine Learning")

# Markdown
st.markdown("""
### Introdução
RMS Titanic foi o navio de passageiros de luxo britânico que afundou entre 14 e 15 de abril de 1912 durante sua
viagem inaugural de Southampton na Inglaterra até Nova York, matando cerca de 1500 passageiros e tripulantes. Uma
das tragédias mais famosas da história moderna, inspirou inúmeras histórias, filmes e musicais e, até hoje, é objeto
de muitas bolsas de estudos e especulações científicas. [Britannica, 2024] (https://www.britannica.com/topic/Titanic)

### Objetivo
O objetivo desta atividade é desenvolver um modelo de machine learning capaz de prever a sobrevivência dos passageiros
do Titanic com base em um conjunto de dados com atributos demográficos (idade, sexo) e classe socioeconômica (bilhete, classe).
Analisando essas informações, busca-se identificar padrões que podem influenciar as taxas de sobrevivência e, subsequentemente, 
utilizar esses dados para fazer previsões.

### Materiais e Métodos
Utilizando o algoritmo de classificação Random Forest, busca-se construir um modelo preditivo que permite estimar a probabilidade de 
sobrevivência de cada indivíduo a bordo do Titanic.

O conjunto de dados, bem como a proposta da análise, são disponibilizados pela [Kaggle](https://www.kaggle.com/competitions/titanic).

Para a análise foram utilizadas as bibliotecas open source Python:
- **scikit-learn:** uma biblioteca para análises preditivas baseada em _NumPy_, _SciPy_ e _matplotlib_ (https://scikit-learn.org/)
- **Streamlit:** uma biblioteca que facilita a criação e compartilhamento de aplicativos web interativos (https://streamlit.io/)
""")

train_file = st.file_uploader("Selecione o arquivo de treinamento", type=["csv", "xlsx"])

if train_file is not None:
    train = pd.read_csv(train_file)
    st.write(train)
    st.write(f"Número de linhas e colunas no conjunto de treinamento: {train.shape}")

test_file = st.file_uploader("Selecione o arquivo de teste", type=["csv", "xlsx"])

if test_file is not None:
    test = pd.read_csv(test_file)
    st.write(test)
    st.write(f"Número de linhas e colunas no conjunto de teste: {test.shape}")

if test_file is not None and test_file is not None:
    # Definir a paleta de cores
    musk_green = "#1C8A63"
    crimson = "#DC143C"

    # Título e descrição
    st.markdown("<h2>Visualização dos Dados</h2>", unsafe_allow_html=True)
    st.markdown("<h3>Tendências de sobrevivência e demografia dos passageiros</h3>", unsafe_allow_html=True)

    # Criar o gráfico
    f, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Gráfico de pizza para a sobrevivência
    train['Survived'].value_counts().plot.pie(
        explode=[0, 0.1],   # separa as fatias em 0.1
        autopct='%1.1f%%',  # formata as fatias em %
        ax=ax[0],           # primeiro subplot
        shadow=False,
        colors=[musk_green, crimson]
    )

    ax[0].set_ylabel('')  # remove o rótulo do eixo y

    # countplot da sobrevivência
    sns.countplot(x='Survived', data=train, ax=ax[1], color=musk_green)

    # anotações no gráfico de barras
    for p in ax[1].patches:
        ax[1].annotate(
            f'{int(p.get_height())}',  # formato como número inteiro
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', va='center',
            fontsize=12, color='black',
            xytext=(0, 8), textcoords='offset points'
        )

    ax[1].set_ylabel('Quantidade')  # rótulo do eixo y
    ax[1].set_xlabel('')  # remove o rótulo do eixo x

    f.suptitle('Mortos (0) e Sobreviventes (1)', fontsize=16)

    # Exibir o gráfico no Streamlit
    st.pyplot(f)

    st.markdown("<h3>Análise do impacto do sexo na taxa de sobrevivência</h3>", unsafe_allow_html=True)

    # Criação da figura e dos subplots
    f, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Agrupando por sexo e calculando a média de sobreviventes
    survival_by_sex = train[['Sex', 'Survived']].groupby(['Sex']).mean()

    # Alterando as colunas para 'feminino' e 'masculino'
    survival_by_sex = survival_by_sex.rename(index={'female': 'feminino', 'male': 'masculino'})

    # Criando o gráfico de barras
    survival_by_sex.plot.bar(ax=ax[0])

    # Definindo as cores para cada barra manualmente
    bars = ax[0].patches  # Obtém as barras geradas pelo gráfico
    bars[0].set_facecolor(crimson)  # Primeira barra: carmesim
    bars[1].set_facecolor(musk_green)  # Segunda barra: verde

    # Ocultando a legenda
    ax[0].legend().set_visible(False)

    # Adicionando o título
    ax[0].set_title('Média de Sobreviventes por Sexo', fontsize=14)

    # Rotacionando os rótulos para se alinharem ao eixo x
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0)
    ax[0].set_xlabel('')

    # Criando o gráfico de barras (countplot)
    sns.countplot(x='Sex', hue='Survived', data=train, ax=ax[1], palette={0: crimson, 1: musk_green})

    # Ajuste do título e rótulos
    ax[1].set_ylabel('')
    ax[1].set_xlabel('')

    # Redefinindo os rótulos dos grupos de barras
    ax[1].set_xticklabels(['masculino', 'feminino'])

    # Ajuste do título e rótulos da legenda
    plt.legend(title='', labels=["Mortos (0)", "Sobreviventes(1)"], fontsize=11, title_fontsize=11)

    ax[1].set_title('Quantidade de Mortos e Sobreviventes por Sexo', fontsize=14)

    # Exibir os gráficos no Streamlit
    st.pyplot(f)

    # Title and description for the section
    st.markdown("<h2>Otimizando os dados para o treinamento do modelo</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p>Nesta seção os dados são refinados ao remover atributos irrelevantes e transformar os dados categóricos em numéricos.
    As etapas executadas serão:</p>
    <ul>
        <li>Remoção dos atributos redundantes: remover colunas 'Cabin' e 'Ticket', que oferecem valor preditivo limitado</li>
        <li>Transformação dos dados: converter dados categóricos em numéricos.</li>
    </ul>
                
    <p>Como visto anteriormente, existem lacunas em 'Embarked'. Vamos substituir os valores NULL com 'S', uma vez que a quantidade de embarques com 'S' é maior que as demais categorias.</p>
    """, unsafe_allow_html=True)

    # Step 1: remove as colunas 'Cabin' e 'Ticket'
    train = train.drop(columns=['Cabin', 'Ticket'], axis=1)
    test = test.drop(columns=['Cabin', 'Ticket'], axis=1)
    
    # Step 2: converter valores NULL em 'Embarked' para 'S'
    train = train.fillna({'Embarked': 'S'})

    st.write(train)
    
    st.markdown("""
    <p>Em seguida, vamos classificar as idades (que também possuem lacunas) em grupos. Combinaremos as faixas etárias das pessoas e as categorizaremos nos mesmos grupos. Fazer isso resulta em menos categorias e uma previsão melhor, considerando que o conjunto de dados será categórico.</p>
    """, unsafe_allow_html=True)

    # Step 3: distribuição das idades em categorias lógicas
    train["Age"] = train["Age"].fillna(-0.5)
    test["Age"] = test["Age"].fillna(-0.5)
    bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
    labels = ['Unknown', 'Baby', 'Child', 'Teenager',
            'Student', 'Young Adult', 'Adult', 'Senior']
    train['AgeGroup'] = pd.cut(train["Age"], bins, labels=labels)
    test['AgeGroup'] = pd.cut(test["Age"], bins, labels=labels)

    st.markdown("""
    <p>Também vamos categorizar a coluna 'title' dos conjuntos train e test em um mesmo número de classes. Em seguida vamos designar valores numéricos ao título, para facilitar o treinamento do modelo.</p>
    """, unsafe_allow_html=True)

    # cria um conjunto de dados combinando train e test
    combine = [train, test]

    # extrai o título para cada nome em train e test
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\\.', expand=False)

    pd.crosstab(train['Title'], train['Sex'])

    # substitui os títulos com nomes comuns
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
                                                    'Don', 'Dr', 'Major',
                                                    'Rev', 'Jonkheer', 'Dona'],
                                                    'Rare')

        dataset['Title'] = dataset['Title'].replace(
            ['Countess', 'Lady', 'Sir'], 'Royal')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

    # mapeia cada grupo de títulos em um valor numérico
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3,
                    "Master": 4, "Royal": 5, "Rare": 6}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    mr_age = train[train["Title"] == 1]["AgeGroup"].mode()    # Jovem adulto
    miss_age = train[train["Title"] == 2]["AgeGroup"].mode()  # Estudante
    mrs_age = train[train["Title"] == 3]["AgeGroup"].mode()   # Adulto
    master_age = train[train["Title"] == 4]["AgeGroup"].mode()  # Bebê
    royal_age = train[train["Title"] == 5]["AgeGroup"].mode()  # Adulto
    rare_age = train[train["Title"] == 6]["AgeGroup"].mode()  # Adulto

    age_title_mapping = {1: "Young Adult", 2: "Student",
                        3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

    for x in range(len(train["AgeGroup"])):
        if train["AgeGroup"][x] == "Unknown":
            train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]

    for x in range(len(test["AgeGroup"])):
        if test["AgeGroup"][x] == "Unknown":
            test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]

    st.markdown("""
    <p>Uma vez que temos a idade mapeada em categorias, não precisamos mais do atributo idade e, portanto, podemos excluí-los. A coluna 'Name' também será excluída, pois sua informação relevante (título) já foi extraída.</p>
    """, unsafe_allow_html=True)

    # mapeia cada valor em Age em um valor numérico
    age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3,
                'Student': 4, 'Young Adult': 5, 'Adult': 6, 
                'Senior': 7}
    train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
    test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

    train.head()

    # excluindo as coluna 'Age' e 'Name'
    train = train.drop(['Age', 'Name'], axis=1)
    test = test.drop(['Age', 'Name'], axis=1)

    st.markdown("""
    <p>'Sex' e 'Embarked' também devem ser categorizados numericamente. E os valores de bilhre (Fare Value) são preenchidos no conjunto teste com base no valor médio da classe (P-class) a que pertencem.</p>
    """, unsafe_allow_html=True)

    # categorizando 'Sex' e 'Embarked'
    sex_mapping = {"male": 0, "female": 1}
    train['Sex'] = train['Sex'].map(sex_mapping)
    test['Sex'] = test['Sex'].map(sex_mapping)

    embarked_mapping = {"S": 1, "C": 2, "Q": 3}
    train['Embarked'] = train['Embarked'].map(embarked_mapping)
    test['Embarked'] = test['Embarked'].map(embarked_mapping)

    for x in range(len(test["Fare"])):
        if pd.isnull(test["Fare"][x]):
            pclass = test["Pclass"][x]  # Pclass = 3
            test["Fare"][x] = round(
                train[train["Pclass"] == pclass]["Fare"].mean(), 4)

    # mapeia os valores de bilhete em grupos numéricos
    train['FareBand'] = pd.qcut(train['Fare'], 4, 
                                labels=[1, 2, 3, 4])
    test['FareBand'] = pd.qcut(test['Fare'], 4, 
                            labels=[1, 2, 3, 4])

    # exclui a coluna 'Fare', uma vez que não é mais necessária
    train = train.drop(['Fare'], axis=1)
    test = test.drop(['Fare'], axis=1)

    st.write(train)

    # Title and description for the section
    st.markdown("<h2>Treinamento do modelo</h2>", unsafe_allow_html=True)
    st.markdown("<h3>Construindo o modelo preditivo</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p>Nesta etapa, vamos empregar o Random Forest como algoritmo para treinar o modelo para previsão de sobrevivência.
    As principais etapas incluem:</p>
    <ul>
        <li>Divisão dos dados: dividir os dados em dois subconjuntos (80% treinamento e 20% teste) utilizando o train_test_split() da biblioteca scikit-klearn</li>
        <li>Seleção do modelo: aplicação do algoritmo Random Forest, conhecido pela sua robustez e capacidade de lidar com dados diversos</li>
        <li>Avaliação de performance: avaliar a precisão do modelo treinado nos dados de teste para assegurar que ele generaliza adequadamente</li>
    </ul>
    """, unsafe_allow_html=True)

    # remoção das colunas 'Survived' e 'PassengerId' do conjunto teste
    predictors = train.drop(['Survived', 'PassengerId'], axis=1)
    target = train["Survived"]
    x_train, x_val, y_train, y_val = train_test_split(
    predictors, target, test_size=0.2, random_state=0)
   
   # Adiciona um título e explicação sobre o algoritmo Random Forest
    st.markdown("<h3>Algoritmo Random Forest</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p>O algoritmo **Random Forest** é um método de aprendizado supervisionado baseado em árvores de decisão. Ele cria uma floresta de árvores de decisão, onde cada árvore é construída com um subconjunto aleatório dos dados de treinamento. A previsão final é obtida por meio de um processo de votação (para classificação) ou média (para regressão) das previsões feitas por todas as árvores.</p>
    """, unsafe_allow_html=True)
    
    image_path = "C:\\Users\\Nicole\\Downloads\\random-forest.png"
    st.image(image_path, caption="(Geeks for Geeks - Random Forest Algorithm in Machine Learning)", use_column_width=True)

    st.markdown("""
    <p>Entre as vantagens do **Random Forest** estão:
    - **Robustez**: Ele é resistente ao overfitting (sobreajuste) e pode lidar com grandes volumes de dados e dados com características complexas.
    - **Versatilidade**: Pode ser utilizado tanto para tarefas de classificação quanto de regressão.
    - **Importância das variáveis**: O algoritmo é capaz de fornecer informações sobre a importância de cada variável no modelo, o que pode ser útil para análise de dados.

    Em geral, o Random Forest é um dos algoritmos mais poderosos e amplamente utilizados em projetos de machine learning, especialmente para dados complexos e de alta dimensionalidade.</p>
    """, unsafe_allow_html=True)

    code = """
        randomforest = RandomForestClassifier()

        # ajusta os dados de treinamento com o seu output
        randomforest.fit(x_train, y_train)
        y_pred = randomforest.predict(x_val)

        # encontra o score de acurácia do modelo
        acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2) 
    """

    randomforest = RandomForestClassifier()

    # ajusta os dados de treinamento com o seu output
    randomforest.fit(x_train, y_train)
    y_pred = randomforest.predict(x_val)

    # encontra o score de acurácia do modelo
    acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

    # Exibe o código formatado no Streamlit
    st.code(code, language='python')

    # Exibe o resultado no Streamlit
    st.write(f"Acurácia do modelo Random Forest: {acc_randomforest}%")

    # Adiciona o título e a explicação sobre a etapa de previsão
    st.markdown("<h2>Previsão: Gerando Previsões de Sobrevivência nos Dados Teste</h2>", unsafe_allow_html=True)

    st.markdown("""
    Nesta etapa final, usamos o modelo Random Forest treinado para fazer previsões no dataset de teste. Os passos-chave são:

    - **Rodar previsões**: Inserir o conjunto de dados de teste no modelo treinado para prever os resultados de sobrevivência.
    - **Preparar resultados**: Armazenar o 'PassengerId' do conjunto de teste e a previsão de sobrevivência.
    - **Salvar o output**: Exportar os dados de previsão em um arquivo .csv com duas colunas:
    - **PassengerId**: ID de cada passageiro do conjunto de teste.
    - **Survival**: Status de sobrevivência previsto (0 = não sobreviveu, 1 = sobreviveu).
    """)

    # Exibe o código formatado no Streamlit
    code = """
    # Obter os IDs dos passageiros do conjunto de dados de teste
    ids = test['PassengerId']

    # Fazer as previsões de sobrevivência
    predictions = randomforest.predict(test.drop('PassengerId', axis=1))

    # Criar um DataFrame com os resultados
    output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})

    # Salvar os resultados em um arquivo CSV
    output.to_csv('resultfile.csv', index=False)
    """

    # Exibe o código formatado
    st.code(code, language='python')

    ids = test['PassengerId']
    predictions = randomforest.predict(test.drop('PassengerId', axis=1))
    output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
    output.to_csv('resultfile.csv', index=False)

    st.write(output)

    # Create a countplot to visualize the distribution of survival predictions
    fig, ax = plt.subplots(figsize=(8, 6))

    # Using seaborn to create a countplot for the 'Survived' column
    sns.countplot(x='Survived', data=output, ax=ax)

    # Display percentages on top of the bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}',  # Display the count of passengers
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 8), textcoords='offset points')

    # Add title and labels
    ax.set_title('Distribuição das Previsões de Sobrevivência', fontsize=16)
    ax.set_xlabel('Sobreviveu?', fontsize=12)
    ax.set_ylabel('Quantidade de Passageiros', fontsize=12)

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Obter o PassengerId e fazer previsões no conjunto de dados de teste
    ids = test['PassengerId']
    predictions = randomforest.predict(test.drop('PassengerId', axis=1))

    # Adicionar uma coluna de "Survived" no dataframe de teste
    test['Survived'] = predictions

    # Exibir o título
    st.markdown("<h2>Previsão de Sobrevivência do Passageiro</h2>", unsafe_allow_html=True)

    # Criar um seletor para escolher o PassengerId
    passenger_id = st.selectbox(
        "Selecione o ID do passageiro:",
        test['PassengerId'].tolist()  # Passa a lista de PassengerId do conjunto de dados de teste
    )

    # Encontrar o passageiro selecionado no dataframe
    passenger = test[test['PassengerId'] == passenger_id].iloc[0]

    # Exibir o status de sobrevivência
    survival_status = "sobreviveu" if passenger['Survived'] == 1 else "não sobreviveu"
    st.write(f"Passageiro ID: {passenger['PassengerId']} - Status de sobrevivência: {survival_status}")

    st.markdown("<h2>Conclusão</h2>", unsafe_allow_html=True)

    st.markdown("""
    Neste projeto, um classificador Random Forest foi construído para prever as chances de sobrevivência dos passageiros do Titanic. 
    Por meio do pré-processamento de dados, engenharia de atributos, imputação e treinamento de modelo, foi possível criar um modelo 
    robusto com **84.36%** de acurácia no conjunto de treinamento.
    """)