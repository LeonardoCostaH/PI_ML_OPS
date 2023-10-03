import pandas as pd
from fastapi import FastAPI
from typing import Optional
import uvicorn
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

ruta_parquet_games = "datasets/data_steam_games.parquet.gz"
ruta_parquet_reviews = "datasets/data_reviews.parquet.gz"
ruta_parquet_items = "datasets/data_items.parquet.gz"

df_games = pd.read_parquet(ruta_parquet_games)
df_reviews = pd.read_parquet(ruta_parquet_reviews)
df_items = pd.read_parquet(ruta_parquet_items)



@app.get("/")
async def index():
    return {"message": "Hola"}


@app.get("/PlayTimeGenre/{genero}")
def PlayTimeGenre(genero:str):
    # Filtrar df_games para obtener solo las filas que contienen el género especificado
    filtered_games = df_games[df_games['genres'].str.contains(genero.capitalize(), case=False, na=False)]

    # Combinar los DataFrames filtrados en uno solo usando "item_id" como clave
    combined_df = pd.merge(df_items, filtered_games, on="item_id", how="inner")

    # Agrupar por año y encontrar el año con más horas jugadas
    result_df = combined_df.groupby('year')['playtime_forever'].sum().reset_index()
    max_year = result_df.loc[result_df['playtime_forever'].idxmax()]

    return max_year.to_dict()

@app.get("/UserForGenre/{genero}")
def UserForGenre(genero:str):
    # Filtrar df_games para obtener solo las filas que contienen el género especificado
    filtered_games = df_games[df_games['genres'].str.contains(genero, case=False, na=False)]

    # Combinar los DataFrames filtrados en uno solo usando "item_id" como clave
    combined_df = pd.merge(df_items, filtered_games, on="item_id", how="inner")

    # Agrupar por usuario y año, sumar las horas jugadas y encontrar el usuario con más horas jugadas
    result_df = combined_df.groupby(['user_id', 'year'])['playtime_forever'].sum().reset_index()
    max_user = result_df.loc[result_df['playtime_forever'].idxmax()]

    # Convertir las horas jugadas de minutos a horas
    result_df['playtime_forever'] = round(result_df['playtime_forever'] / 60, 0)

    # Crear una lista de acumulación de horas jugadas por año
    accumulation = result_df.groupby('year')['playtime_forever'].sum().reset_index()
    accumulation = accumulation.rename(columns={'year': 'Año', 'playtime_forever': 'Horas'})
    accumulation_list = accumulation.to_dict(orient='records')

    return {"Usuario con más horas jugadas para el género " + genero: max_user['user_id'], "Horas jugadas": accumulation_list} 

@app.get("/UsersRecommend/{year}")
def UsersRecommend(df_reviews, df_items, year:int):
    # Filtrar las revisiones para el año dado y que son recomendadas (recommend = True)
    filtered_reviews = df_reviews[(df_reviews['posted'] == year) & (df_reviews['recommend'] == True)]

    # Filtrar las revisiones con comentarios positivos o neutrales (sentiment_analysis = 1 o 2)
    positive_reviews = filtered_reviews[filtered_reviews['sentiment_analysis'].isin([1, 2])]

    # Realizar la fusión de los DataFrames para obtener el nombre del juego
    merged_reviews = pd.merge(positive_reviews, df_items, on="item_id", how="left")

    # Contar el número de revisiones positivas para cada juego
    positive_reviews_count = merged_reviews['item_name'].value_counts().reset_index()
    positive_reviews_count.columns = ['item_name', 'count']

    # Obtener los top 3 juegos más recomendados
    top_3_most_recommended = positive_reviews_count.nlargest(3, 'count')

    # Crear una lista con el resultado en el formato deseado
    result = [{"Puesto " + str(i + 1): game} for i, game in enumerate(top_3_most_recommended['item_name'])]

    return result

@app.get("/UsersNotRecommend/{year}")
def UsersNotRecommend(df_reviews, df_items, year:int):
    # Filtrar las revisiones para el año dado y que no son recomendadas (recommend = False)
    filtered_reviews = df_reviews[(df_reviews['posted'] == year) & (df_reviews['recommend'] == False)]

    # Filtrar las revisiones con comentarios negativos (sentiment_analysis = 0)
    negative_reviews = filtered_reviews[filtered_reviews['sentiment_analysis'] == 0]

    # Realizar la fusión de los DataFrames para obtener el nombre del juego
    merged_reviews = pd.merge(negative_reviews, df_items, on="item_id", how="left")

    # Contar el número de revisiones negativas para cada juego
    negative_reviews_count = merged_reviews['item_name'].value_counts().reset_index()
    negative_reviews_count.columns = ['item_name', 'count']

    # Obtener los top 3 juegos menos recomendados
    top_3_least_recommended = negative_reviews_count.nlargest(3, 'count')

    # Crear una lista con el resultado en el formato deseado
    result = [{"Puesto " + str(i + 1): game} for i, game in enumerate(top_3_least_recommended['item_name'])]

    return result

@app.get("/sentiment_analysis/{year}")
def sentiment_analysis(df_reviews, year:int):
    # Filtrar las revisiones para el año dado
    filtered_reviews = df_reviews[df_reviews['posted'] == year]

    # Contar la cantidad de registros de reseñas para cada categoría de análisis de sentimiento
    sentiment_counts = filtered_reviews['sentiment_analysis'].value_counts().to_dict()

    # Crear un diccionario con los resultados
    result = {
        "Negative": sentiment_counts.get(0, 0),  # Contar las reseñas negativas (sentiment_analysis = 0)
        "Neutral": sentiment_counts.get(1, 0),   # Contar las reseñas neutrales (sentiment_analysis = 1)
        "Positive": sentiment_counts.get(2, 0)   # Contar las reseñas positivas (sentiment_analysis = 2)
    }
    return result


# Crear una matriz de usuario-ítem
user_item_matrix = df_reviews.pivot(index='user_id', columns='item_id', values='recommend').fillna(0)

# Calcular la similitud coseno entre usuarios
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

@app.get("/recomendacion_usuario/{user_id}")
def recomendacion_usuario(user_id: int):
    # Obtener las calificaciones del usuario dado
    user_ratings = user_item_matrix.loc[user_id]

    # Calcular la similitud entre el usuario dado y todos los demás usuarios
    user_similarities = user_similarity_df[user_id]

    # Calcular la puntuación ponderada de juegos recomendados
    recommended_scores = user_item_matrix.mul(user_similarities, axis=0).sum()

    # Filtrar juegos que el usuario ya ha calificado
    recommended_scores = recommended_scores[user_ratings == 0]

    # Ordenar juegos por puntuación
    recommended_games = recommended_scores.sort_values(ascending=False).index.tolist()[:5]

    return {"juegos_recomendados": recommended_games}


# Crear una matriz de características de género por juego
game_genre_matrix = df_games[['item_id', 'genres']].set_index('item_id')

# Aplicar la codificación one-hot para los géneros
game_genre_matrix = game_genre_matrix['genres'].str.get_dummies(sep=', ')

# Calcular la similitud del coseno entre juegos basada en género
item_item_similarity = cosine_similarity(game_genre_matrix, game_genre_matrix)
item_item_similarity_df = pd.DataFrame(item_item_similarity, index=game_genre_matrix.index, columns=game_genre_matrix.index)

@app.get("/recomendacion_juego/{game_id}")
def recomendacion_juego(game_id: int):
    # Obtener la fila de similitudes para el juego dado
    game_similarity_row = item_item_similarity_df.loc[game_id]

    # Ordenar juegos por similitud
    recommended_games = game_similarity_row.sort_values(ascending=False).index.tolist()[1:6]

    return {"juegos_recomendados": recommended_games}
