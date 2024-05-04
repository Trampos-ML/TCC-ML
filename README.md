# TCC-ML

Trabalho Final de Conclusão de Curso de Ciência de Dados e Machine Learning.

Recomendador de Filmes baseado em Conteúdo.
Link do Repo do projeto
### [Repo do Projeto](https://github.com/Trampos-ML/TCC-ML)


## Alunos
### [Arthur Sanchez Fortuna](https://github.com/Trampos-ML/TCC-ML/tree/pamonha)

### [Arthur Carvalho de Mario](https://github.com/Trampos-ML/TCC-ML/tree/girafa/recomendation)

### [Luciano Duarte Gonzalez](https://github.com/Trampos-ML/TCC-ML/tree/bebo)

## Importar e Iniciar
```py
import re
import nltk
import time
import psutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
%matplotlib inline

from flask import Flask, request
from nltk.corpus import stopwords
import plotly.graph_objects as go

from nltk.tokenize import word_tokenize
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Ignorar mensagens de aviso durante a execução do código
warnings.filterwarnings("ignore")

# Caminho para o arquivo CSV
caminho_arquivo = 'Data/netflix_titles.csv'

# Carregar o arquivo CSV em um dataframe
df = pd.read_csv('Data/netflix_titles.csv')

# Exibir o dataframe
df
```
## Pré-Processamento
```python
# Dropando colunas que não serão utilizadas
df = df.drop(columns=['show_id','type','cast','date_added','rating', 'director'])

# Dropando valores nulos
df = df.dropna()

# Função para limpar o texto
def limpar_texto(texto):
    # Converter o texto para minúsculo
    texto = texto.lower()
    # Remover caracteres especiais
    texto = re.sub(r'[^a-z0-9]', ' ', texto)
    return texto

# Aplicar a função de limpeza ao dataframe
df['description'] = df['description'].apply(limpar_texto)
df['description']

# Criar uma máscara booleana para identificar linhas que contêm 'Season' ou 'Seasons' na coluna 'duration'
mask = df['duration'].str.contains(r'Season|Seasons', case=False)

# Dropar as linhas que correspondem à máscara
df = df[~mask]

# Remover a palavra 'min' dos valores na coluna 'duration'
df['duration'] = df['duration'].str.replace(r'\bmin\b', '', regex=True)

# Mudar para tipo INT
df['duration'] = df['duration'].apply(int)

# Renomear todas as colunas de uma vez
df = df.rename(columns={
    'title': 'Titulo',
    'release_year': 'Ano',
    'duration': 'Minutagem',
    'listed_in': 'Genero',
    'description': 'Sinopse',
    'country': "Pais"
})
```

## Analise Estatisctica e Graficos
```py
#1
# Converter a coluna 'Ano' para numérica
df['Ano'] = pd.to_numeric(df['Ano'], errors='coerce')

# Remover os valores nulos na coluna 'Ano'
df = df.dropna(subset=['Ano'])

# Criar o gráfico de dispersão com Plotly
fig = px.scatter(df, x='Ano', y='Minutagem', title='Correlação entre Ano de Lançamento e Duração',
                 labels={'Ano': 'Ano de Lançamento', 'Minutagem': 'Duração (minutos)'},
                 opacity=0.7, trendline="ols", trendline_color_override="red")

# Personalizar o layout do gráfico
fig.update_traces(marker=dict(size=5))  # Ajusta o tamanho dos pontos
fig.update_layout(title_font_size=20, title_font_family="Arial", title_font_color="navy", title_x=0.5, title_y=0.95,
                  xaxis_title_font_size=16, yaxis_title_font_size=16, font_family="Arial", font_color="black",
                  legend_title_font_size=14, legend_font_size=12, legend_font_family="Arial")

# Exibir o gráfico
fig.show()

#2
# Filtrar os filmes gravados em um único país
filmes_unico_pais = df[df['Pais'].str.count(',') == 0]

# Contagem de filmes por país
contagem_por_pais = filmes_unico_pais['Pais'].value_counts()

# Selecionar os top 10 países
top_10_paises = contagem_por_pais.head(10)

# Criar o gráfico de barras
fig = px.bar(top_10_paises, 
             x=top_10_paises.values, 
             y=top_10_paises.index, 
             orientation='h',  # Orientação horizontal
             labels={'x':'Número de Filmes', 'y':'País'},
             title='Top 10 Países com Maior Número de Filmes (Gravados em 1 Único País)',
             color=top_10_paises.values,  # Colorir as barras de acordo com o número de filmes
             color_continuous_scale='rainbow')  # Paleta de cores aleatórias
fig.update_yaxes(categoryorder='total ascending')  # Ordenar as barras de acordo com o total
fig.show()

#3
# Calculando a quantidade de filmes por ano
filmes_por_ano = df['Ano'].value_counts().sort_index()

# Definindo cores RGB para as barras
cores_rgb = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 
             'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 
             'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

# Criando o gráfico de barras
fig = go.Figure()

# Adicionando as barras ao gráfico com cores diferentes para cada barra
for i, (ano, freq) in enumerate(zip(filmes_por_ano.index, filmes_por_ano.values)):
    fig.add_trace(go.Bar(x=[ano], y=[freq], marker_color=cores_rgb[i % len(cores_rgb)]))

# Atualizando o layout do gráfico
fig.update_layout(title='Distribuição de Filmes por Ano',
                  xaxis_title='Ano',
                  yaxis_title='Frequência',
                  xaxis=dict(tickangle=-45, tickmode='linear', tick0=1940, dtick=5),  # Definindo intervalo de 5 anos
                  height=600,  # Altura do gráfico
                  width=1130,  # Largura do gráfico
                  showlegend=False,
                  margin=dict(l=100, r=100, t=50, b=50))  # Definindo as margens

# Mostrando o gráfico
fig.show()

#4
# Ordenar a contagem de gêneros do menor para o maior
genero_contagem = df['Genero'].str.split(', ', expand=True).stack().value_counts()
genero_contagem = genero_contagem.sort_values()

# Definindo uma paleta de cores
cores_rgb = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 
             'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 
             'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

# Criando o gráfico de barras
fig = go.Figure()

# Adicionando as barras ao gráfico com cores diferentes para cada barra
for i, (genero, contagem) in enumerate(zip(genero_contagem.index, genero_contagem.values)):
    fig.add_trace(go.Bar(x=[genero], y=[contagem], marker_color=cores_rgb[i % len(cores_rgb)]))

# Atualizando o layout do gráfico
fig.update_layout(title='Contagem de Títulos por Gênero (1942-2021)',
                  xaxis_title='Gênero',
                  yaxis_title='Contagem de Títulos',
                  xaxis=dict(tickangle=-45, tickmode='array', tickvals=genero_contagem.index, ticktext=genero_contagem.index),  # Definindo texto do eixo x
                  height=600,  # Altura do gráfico
                  width=1130,  # Largura do gráfico
                  showlegend=False,
                  margin=dict(l=100, r=100, t=50, b=50))  # Definindo as margens

# Mostrando o gráfico
fig.show()

#5
# Ordenar a contagem de gêneros do menor para o maior
genero_contagem = df['Genero'].str.split(', ', expand=True).stack().value_counts()
genero_contagem = genero_contagem.sort_values()

# Criando o gráfico de pizza
fig = go.Figure()

# Adicionando as fatias de pizza ao gráfico
fig.add_trace(go.Pie(labels=genero_contagem.index, values=genero_contagem.values))

# Atualizando o layout do gráfico
fig.update_layout(title='Contagem de Títulos por Gênero (1942-2021)',
                  height=600,  # Altura do gráfico
                  width=1100)  # Largura do gráfico

# Mostrando o gráfico
fig.show()

#6
# Agrupamento por ano e contagem de gêneros
generos_por_ano = df.groupby('Ano')['Genero'].apply(lambda x: ','.join(x)).reset_index()

# Criando um dataframe com dummies para os gêneros
generos_dummies = generos_por_ano['Genero'].str.get_dummies(sep=',')

# Lista de anos
anos = generos_por_ano['Ano']

# Criando lista de figuras para os subplots
fig = go.Figure()

# Adicionando um trace para cada gênero
for genero in generos_dummies.columns:
    fig.add_trace(go.Bar(x=anos, y=generos_dummies[genero], name=genero, showlegend=False))

# Atualizando o layout do gráfico
fig.update_layout(
    title='Distribuição de Gêneros por Ano',
    xaxis_title='Ano',
    yaxis_title='Contagem de Títulos',
    xaxis=dict(tickangle=-45, dtick=5),  # Definindo o intervalo de tick como 1 para mostrar todos os anos
    barmode='stack',
    height=600,  # Altura do gráfico
    width=1100,  # Largura do gráfico
    margin=dict(l=10, r=10, t=50, b=50)  # Margens (esquerda, direita, topo, baixo)
)

# Mostrando o gráfico
fig.show()

#7
# Contar quantos gêneros únicos existem no DataFrame
generos_unicos = df['Genero'].str.split(', ').explode().unique()
numero_de_generos = len(generos_unicos)

# Selecionar apenas os dados que possuem um único gênero na coluna "Genero"
dados_um_genero = df[df['Genero'].str.split(', ').apply(len) == 1]

print("Número de gêneros únicos:", numero_de_generos)
print("Dados com apenas um gênero na coluna 'Genero':")
dados_um_genero

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Agrupar os dados por Ano e Genero e contar o número de filmes
heatmap_data = dados_um_genero.groupby(['Ano', 'Genero']).size().unstack(fill_value=0)

# Plotar o mapa de calor
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Número de Filmes com Apenas um Gênero por Ano e Gênero')
plt.xlabel('Gênero')
plt.ylabel('Ano')
plt.show()

#8
# Função para calcular a década correspondente a um determinado ano
def calcular_decada(ano):
    return str(ano // 10 * 10) + "s"

# Criar uma nova coluna 'Decada' com base no ano
dados_um_genero['Decada'] = dados_um_genero['Ano'].apply(calcular_decada)

# Agrupar os dados por Decada e Genero e contar o número de filmes
heatmap_data = dados_um_genero.groupby(['Decada', 'Genero']).size().unstack(fill_value=0)

# Plotar o mapa de calor
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Número de Filmes com Apenas um Gênero por Década e Gênero')
plt.xlabel('Gênero')
plt.ylabel('Década')
plt.show()

#9
# Dividir os gêneros em uma lista e expandir o DataFrame
generos_expandidos = df['Genero'].str.split(', ', expand=True)

# Concatenar os gêneros expandidos com o DataFrame original
df_expandido = pd.concat([df, generos_expandidos], axis=1)

# Mapear as colunas geradas para um único nome de gênero
df_expandido = df_expandido.melt(id_vars=['Titulo', 'Ano', 'Minutagem', 'Sinopse'], value_vars=[0, 1, 2], value_name='Genero')

# Remover linhas com valores nulos na coluna 'Genero'
df_expandido.dropna(subset=['Genero'], inplace=True)

# Exibir as primeiras linhas do DataFrame expandido
df_expandido

# Lista vazia para armazenar os trace do boxplot
data = []

# Iterar sobre os gêneros únicos
for genero in df_expandido['Genero'].unique():
    # Selecionar os dados de duração para o gênero atual
    dados_genero = df_expandido[df_expandido['Genero'] == genero]['Minutagem']
    # Adicionar um trace de boxplot para o gênero atual
    data.append(go.Box(y=dados_genero, name=genero))

# Layout do gráfico
layout = go.Layout(title='Duração dos Filmes por Gênero',
                   yaxis=dict(title='Duração (minutos)'),
                   boxmode='group')  # Para agrupar os boxplots lado a lado

# Criar a figura e mostrar o gráfico
fig = go.Figure(data=data, layout=layout)
fig.show()

#10
# Calcular a média da duração dos filmes por década e gênero
media_por_decada_genero = df_expandido.groupby([df_expandido['Ano'] // 10 * 10, 'Genero'])['Minutagem'].mean().reset_index()

# Lista vazia para armazenar os trace do boxplot
data = []

# Iterar sobre os gêneros únicos
for genero in media_por_decada_genero['Genero'].unique():
    # Selecionar os dados de média de duração para o gênero atual
    dados_genero = media_por_decada_genero[media_por_decada_genero['Genero'] == genero]
    # Adicionar um trace de boxplot para a média de duração do gênero atual
    data.append(go.Box(y=dados_genero['Minutagem'], name=genero, line=dict(width=8)))  # Ajuste da espessura da linha

# Layout do gráfico
layout = go.Layout(title='Média da Duração dos Filmes por Gênero e Década',
                   yaxis=dict(title='Duração Média (minutos)'),
                   boxmode='group')  # Para agrupar os boxplots lado a lado

# Criar a figura e mostrar o gráfico
fig = go.Figure(data=data, layout=layout)
fig.show()
```
### Modelo
```py
# Início da contagem de tempo
start_time = time.time()

# Função para medir o uso de memória
def memory_usage():
    # Obtém o uso de memória atual em MB
    mem_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
    return mem_usage_mb

# Imprime o uso de memória inicial
print("Uso de memória inicial:", memory_usage(), "MB")

# 3. Combinar título, sinopse e gênero em uma única coluna
df['Features'] = df['Titulo'] + ' ' + df['Sinopse'] + ' ' + df['Genero'] + ' ' + df['Pais'] + ' ' + df['Ano'].astype(str) + ' ' + df['Minutagem'].astype(str)

# Imprime o uso de memória após carregar o DataFrame
print("Uso de memória após carregar DataFrame:", memory_usage(), "MB")

# Criar um vetorizador TF-IDF para converter o texto em recursos numéricos
vectorizer = TfidfVectorizer(stop_words='english')  
X = vectorizer.fit_transform(df['Features'])

# Imprime o uso de memória após criar o vetorizador TF-IDF
print("Uso de memória após criar o vetorizador TF-IDF:", memory_usage(), "MB")

# Criar e treinar o modelo KNN
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(X)

# Imprime o uso de memória após criar e treinar o modelo KNN
print("Uso de memória após criar e treinar o modelo KNN:", memory_usage(), "MB")

# Função para recomendar filmes com base na entrada do usuário
def recommend_movies(user_input):
    # Vetorizar a entrada do usuário
    entrada_usuario_vector = vectorizer.transform([user_input])

    # Encontrar os filmes mais próximos da entrada do usuário
    distances, indices = knn.kneighbors(entrada_usuario_vector)
    
    # Criar um DataFrame com os filmes recomendados
    recommended_movies_df = df.iloc[indices[0]][['Titulo', 'Sinopse', 'Genero']] # SÃO AS COLUNAS QUE INTERESSAM + PRO USUARIO

    return recommended_movies_df

# Entrada do usuário
user_input = "Naruto"

# Recomendar filmes com base na entrada do usuário
recommended_movies = recommend_movies(user_input)

# Exibir os títulos e sinopses dos filmes recomendados
print(recommended_movies)

# Fim da contagem de tempo
end_time = time.time()

# Tempo total de execução
execution_time = end_time - start_time
print("Tempo de execução:", execution_time, "segundos")

# Uso de memória final
print("Uso de memória final:", memory_usage(), "MB")
```

### Matriz e Acuracia
```py
# Filtrar filmes com gênero único
df = df[df['Genero'].apply(lambda x: len(x.split(',')) == 1)]

# Dividir os dados em conjunto de treinamento e teste
X = df.drop('Genero', axis=1)
y = df['Genero']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vetorizar a sinopse usando TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train['Sinopse'])
X_test = vectorizer.transform(X_test['Sinopse'])

# Treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Avaliar o modelo
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculando manualmente os valores de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos
tp = conf_matrix[1, 1]
fp = conf_matrix[0, 1]
tn = conf_matrix[0, 0]
fn = conf_matrix[1, 0]

# Imprimir os valores de verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos
print("Verdadeiros Positivos:", tp)
print("Falsos Positivos:", fp)
print("Verdadeiros Negativos:", tn)
print("Falsos Negativos:", fn)

# Calcular métricas de precisão, recall e F1-score
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Imprimir métricas
print("Precisão:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Acurácia:", accuracy)


# Plotar a matriz de confusão usando Plotly
fig = go.Figure(data=go.Heatmap(z=conf_matrix,
                                 x=knn.classes_,
                                 y=knn.classes_,
                                 colorscale='Blues',
                                 colorbar=dict(title='Contagem'),
                                 hoverongaps=False))
fig.update_layout(title='Matriz de Confusão',
                  xaxis=dict(title='Gênero Previsto', automargin=True),
                  yaxis=dict(title='Gênero Verdadeiro', automargin=True),
                  width=800, height=600,
                  font=dict(size=12),
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  hoverlabel=dict(bgcolor="white", font_size=12, font_family="Rockwell"))
for i in range(len(conf_matrix)):
    for j in range(len(conf_matrix[0])):
        if conf_matrix[i][j] != 0:
            fig.add_annotation(x=knn.classes_[j], y=knn.classes_[i], text=str(conf_matrix[i][j]),
                               showarrow=False, font=dict(color='black', size=12))
fig.show()
```
### API
```py
# Inicializar o aplicativo Flask
app = Flask(__name__)

# Carregar o DataFrame com os dados dos filmes
df = pd.read_csv('Models/dados.csv')

# Combinar título, sinopse e gênero em uma única coluna
df['Features'] = df['Titulo'] + ' ' + df['Sinopse'] + ' ' + df['Genero'] + ' ' + df['Pais'] + ' ' + df['Ano'].astype(
    str) + ' ' + df['Minutagem'].astype(str)

# Criar um vetorizador TF-IDF para converter o texto em recursos numéricos
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Features'])

# Criar e treinar o modelo KNN
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(X)


# Função para recomendar filmes com base na entrada do usuário
def recommend_movies(user_input):
    # Vetorizar a entrada do usuário
    entrada_usuario_vector = vectorizer.transform([user_input])

    # Encontrar os filmes mais próximos da entrada do usuário
    distances, indices = knn.kneighbors(entrada_usuario_vector)

    # Criar um DataFrame com os filmes recomendados
    recommended_movies_df = df.iloc[indices[0]][['Titulo', 'Sinopse', 'Genero']]

    return recommended_movies_df


@app.route('/recommend', methods=['GET'])
def get_recommendations():
    # Obter a entrada do usuário da query string
    user_input = request.args.get('input')

    # Verificar se a entrada do usuário está presente
    if not user_input:
        return 'No input provided', 400

    # Obter recomendações com base na entrada do usuário
    recommended_movies = recommend_movies(user_input)

    # Criar uma tabela HTML com os filmes recomendados
    html_content = """
    <html>
    <head>
        <title>Filmes Recomendados</title>
        <style>
            table {
                font-family: Arial, sans-serif;
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid #dddddd;
                text-align: left;
                padding: 8px;
            }
            th {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        <h1>Filmes Recomendados</h1>
        <table>
            <tr>
                <th>Título</th>
                <th>Sinopse</th>
                <th>Gênero</th>
            </tr>
    """

    # Iterar sobre os filmes recomendados e adicioná-los à tabela HTML
    for index, movie in recommended_movies.iterrows():
        html_content += f"<tr>"
        html_content += f"<td>{movie['Titulo']}</td>"
        html_content += f"<td>{movie['Sinopse']}</td>"
        html_content += f"<td>{movie['Genero']}</td>"
        html_content += f"</tr>"

    html_content += """
        </table>
    </body>
    </html>
    """

    # Retornar a tabela HTML como resposta
    return html_content


# Executar o aplicativo Flask
if __name__ == '__main__':
    host = 'localhost'
    port = 8080
    print(f"Aplicação rodando em http://{host}:{port}/")
    app.run(debug=True, use_reloader=True, host=host, port=port)
```
