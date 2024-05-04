# TCC-ML

Trabalho Final de Conclusão de Curso de Ciência de Dados e Machine Learning.

Recomendador de Filmes baseado em Conteúdo.

## Links Uteis

- https://semantix.ai/tf-idf-entenda-o-que-e-frequency-inverse-document-frequency/#:~:text=O%20TF%2DIDF%20%C3%A9%20um,import%C3%A2ncia%20ao%20longo%20do%20corpus.

- 
## Como rodar o projeto
```bash
# Clone este repositório
$ git clone

# Acesse a pasta do projeto no terminal/cmd
$ cd TCC-ML

# Inicie a aplicação
$ python api.py
```
Há duas formas de acessar os resultados da recomendação de filmes.

O primeiro é acessando o endereço http://localhost:8080/recommend?input=Romantic no navegador.
O segundo é utilizando o comando curl no terminal assim como o exemplo abaixo:
``` bash
 curl -X GET "http://localhost:8080/recommend?input=Romantic"
```
A inserção de um gênero de filme é obrigatória para a recomendação de filmes. Ela deverá ser feita no campo "input" da requisição.
