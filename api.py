import pandas as pd
from flask import Flask, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

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
