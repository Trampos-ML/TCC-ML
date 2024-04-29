# TCC-ML

Trabalho Final de Conclusão de Curso de Ciência de Dados e Machine Learning.

Recomendador de Filmes baseado em Conteúdo.

## Alunos
### [Arthur Sanchez Fortuna](https://github.com/Trampos-ML/TCC-ML/tree/pamonha)

### Arthur Carvalho de Mario

### [Luciano Duarte Gonzalez](https://github.com/Trampos-ML/TCC-ML/tree/bebo)

## Importar e Iniciar
```py
import re
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

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
