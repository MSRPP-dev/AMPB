# %%
'''
------------------------------------------------
Análisis de modelos predictivos en bolsa: NVIDIA
MSRPP
Copyright (C) 2024-2025 MegaStorm Systems

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

------------------------------------------------------------------------------------------------
SA/GetData_Reddit v1.3
  - Permite leer datos de Reddit dando un subforo y unas palabras clave.
  - Lo almacena en la base de datos MongoDB listo para ser analizado.
  - Exportado a GetDataRD_SA.py
------------------------------------------------------------------------------------------------'''

# %%
# Importar librerías
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import praw 
import time
from pymongo import MongoClient
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
from ampblib import AMPBConfig, getMongoCollection, closeMongoClient
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

# %%
# 1. CONEXIÓN A REDDIT
REDDIT_USER_AGENT = "AMPB-SentimentAnalysis/1.0 (by /u/Upstairs_Repair9008)"
reddit_client = None
try:
    reddit_client = praw.Reddit(
        client_id=AMPBConfig.REDDIT_CLIENT_ID,
        client_secret=AMPBConfig.REDDIT_CLIENT_SECRET,
        username=AMPBConfig.REDDIT_USERNAME,
        password=AMPBConfig.REDDIT_PASSWORD,
        user_agent=REDDIT_USER_AGENT,
    )
    print(f"Autenticado en Reddit como: {reddit_client.user.me()}")
except Exception as e:
    print(f"Error conectando a Reddit: {e}. No se obtendrán datos de Reddit.")

# %%
# 2. FUNCIÓN DE BUSQUEDA EN REDDIT
def searchRedditSentimentNews(
    praw_client,
    subreddit_name,
    keywords,
    limit=1000  
):
   
    # Obtener el subreddit
    subreddit = praw_client.subreddit(subreddit_name)
   
    created_utc = subreddit.created_utc
    created_date = datetime.utcfromtimestamp(created_utc).strftime("%Y-%m-%d")  
    print(f"{AMPBConfig.COLOR_VALUE}-------------------------------------------------------------------{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_MSG}Buscando posts en Reddit{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_VALUE} Últimos {limit} posts en Subreddit 'r/{subreddit_name}' creado el {created_date}'{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_VALUE} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_VALUE}-------------------------------------------------------------------{AMPBConfig.COLOR_RESET}")

    # Inicializar lista para almacenar resultados
    results = []
   
    # Contadores
    processed = 0
    new_posts = 0
    duplicated_posts = 0
    
    # Obtener colección de MongoDB "market_sentiment"
    collection = getMongoCollection("market_sentiment")
   
    try:
        # Obtener los posts más recientes
        for post in subreddit.new(limit=limit):
            processed += 1
           
            # Mostrar progreso cada 100 posts
            if processed % 100 == 0:
                print(f"  Procesados {processed} posts hasta ahora...")
           
            # Convertir el título y texto a minúsculas para búsqueda case-insensitive
            title_lower = post.title.lower()
            selftext_lower = post.selftext.lower() if hasattr(post, 'selftext') else ""
           
            # Verificar si alguno de los keywords está en el título o texto
            matching_keywords = []
            for keyword in keywords:
                if keyword.lower() in title_lower or keyword.lower() in selftext_lower:
                    matching_keywords.append(keyword)
           
            # Si encontramos algún keyword, guardar el post
            if matching_keywords:
                # Convertir timestamp a datetime
                post_date = datetime.fromtimestamp(post.created_utc)
               
                # Extraer datos del post
                post_data = {
                    'fecha': post_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'tipo': 'reddit',
                    'subtipo': subreddit_name,
                    'titulo': post.title,
                    'url': f"https://www.reddit.com{post.permalink}",
                    'texto': post.selftext[:1000] if hasattr(post, 'selftext') else '',
                    'keywords': ', '.join(matching_keywords),
                    'sentiment_algorithm': 'none', # Con 'none' indica que no ha sido procesado aun
                    'value': 0  
                }
                
                # Verificar si el post ya existe en la base de datos
                existing_post = collection.find_one({'url': post_data['url']})
                
                if existing_post:
                    duplicated_posts += 1
                else:
                    # Guardar en MongoDB
                    collection.insert_one(post_data)
                    new_posts += 1
                    
                    # Añadir a resultados para el DataFrame
                    results.append(post_data)                
   
    except Exception as e:
        print(f"Error al recuperar posts: {str(e)}")
   
    # Convertir resultados a DataFrame
    df = pd.DataFrame(results) if results else pd.DataFrame(columns=['fecha', 'tipo', 'subtipo', 'titulo', 'url', 'texto', 'keywords', 'value'])
   
    # Mostrar estadísticas
    if len(df) > 0:
        oldest_date = df['fecha'].min()
        print(f" Resultados:")
        print(f" - Posts totales procesados: {processed}")
        print(f" - Posts con keywords encontrados: {len(df) + duplicated_posts}")
        print(f" - Posts nuevos guardados en MongoDB: {new_posts} (fecha del más antiguo: {oldest_date})")
        print(f" - Posts duplicados (ya existentes en MongoDB): {duplicated_posts}")
    else:
        print(f" No se encontraron posts con las keywords especificadas o todos son duplicados.")
        print(f" - Posts totales procesados: {processed}")
        print(f" - Posts con keywords encontrados: {len(df) + duplicated_posts}")
        print(f" - Posts duplicados (ya existentes en MongoDB): {duplicated_posts}")
    
    print()
    # Cerramos conexion a MongoDB y devolvemos el dataframe
    closeMongoClient()    
    return df

# %%
# 3. INGESTIÓN DE DATOS 
keywords = ["NVIDIA", "INTEL", "AMD", "tariff", "Rate", "BREAKING", "inflation", "interest", "market", "economy", "crash", "breaking", "Earnings"] 
limit = 1000

# Buscamos en el primer foro
subreddit_to_scan = "stocks"
results_df = searchRedditSentimentNews(reddit_client, subreddit_to_scan, keywords, limit)

# Buscamos en el segundo foro
subreddit_to_scan = "wallstreetbets"
results_df = searchRedditSentimentNews(reddit_client, subreddit_to_scan, keywords, limit)


