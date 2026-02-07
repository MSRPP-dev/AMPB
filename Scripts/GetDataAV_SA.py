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
SA/GetData_AlphaVantage v1.3
  - Permite extraer noticias de Alpha Vantage por palabra clave y fecha.
  - Lo almacena en la base de datos MongoDB con el indice de sentimiento de Alpha Vantage.
  - Exportado a GetDataAV_SA.py
------------------------------------------------------------------------------------------------'''

# %%
# Importar librerías
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from pymongo import MongoClient
from pymongo import DESCENDING
import random
import requests
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
from ampblib import AMPBConfig, getMongoCollection, closeMongoClient
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

# %%
# 1. FUNCIÓN DE BUSQUEDA EN ALPHA VANTAGE
def searchAlphaVantageSentimentNews(ticker, api_key, time_from, time_to):
    print(f"{AMPBConfig.COLOR_VALUE}-------------------------------------------------------------------{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_MSG}Buscando noticias de Alpha Vantage para el ticker '{ticker}'{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_VALUE} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_VALUE}-------------------------------------------------------------------{AMPBConfig.COLOR_RESET}")
    
    # Inicializar variables
    results = []
    processed = 0
    relevant_news = 0
    non_relevant_news = 0
    saved_to_mongodb = 0
    duplicated_news = 0
    
    # Obtener colección de MongoDB "market_sentiment"
    collection = getMongoCollection("market_sentiment")
    
    # Convertir time_from y time_to de AV a MongoDB inline
    dt_from = datetime.strptime(time_from, "%Y%m%dT%H%M")
    mongo_from = dt_from.strftime("%Y-%m-%d %H:%M:%S")
    dt_to   = datetime.strptime(time_to,   "%Y%m%dT%H%M")
    mongo_to   = dt_to.strftime("%Y-%m-%d %H:%M:%S")
    
    # Busca la noticia más reciente en el rango 
    last_cursor = (
            collection
            .find({
                'tipo':    'alpha_vantage',
                'keywords': ticker,
                'fecha':   {'$gte': mongo_from, '$lte': mongo_to}
            })
            .sort('fecha', DESCENDING)
            .limit(1)
        )
    last = list(last_cursor)
    if last:
        # Ajustar time_from a partir de la última fecha MongoDB
        last_fecha = last[0]['fecha']  # 'YYYY-MM-DD HH:MM:SS'
        dt_last = datetime.strptime(last_fecha, "%Y-%m-%d %H:%M:%S")
        time_from = dt_last.strftime("%Y%m%dT%H%M")  # de vuelta a AV
        print(f"  Ajustando start_date a: {dt_last} (Alpha Vantage {time_from})")
    else:
        print(f"  No hay noticias previas entre {mongo_from} y {mongo_to} (Alpha Vantage {time_from} a {time_to})")
    
    try:
        # Construir la URL de Alpha Vantage
        base_url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ticker,
            'apikey': api_key,
            'limit': 1000,  # Alpha Vantage limita a 1000 resultados máximo
            'time_from': time_from,
            'time_to': time_to,
            'sort': 'EARLIEST'  # Ordenar desde más antiguo a más reciente
        }
        # Breve pausa para no sobrecargar
        time.sleep(random.uniform(0.1, 0.3))
     
        # Realizar la petición        
        response = requests.get(base_url, params=params)
        
        # Verificar si la respuesta es exitosa
        if response.status_code != 200:
            print(f"  Error en la petición: {response.status_code} - {response.text}")
            return pd.DataFrame()
            
        # Parsear el JSON de respuesta
        data = response.json()
        
        # Verificar si hay un mensaje de error de la API
        if 'Error Message' in data:
            print(f"  Error de Alpha Vantage: {data['Error Message']}")
            return pd.DataFrame()
            
        # Verificar si hay datos de noticias
        if 'feed' not in data or not data['feed']:
            print("  No se encontraron noticias en la respuesta.")
            return pd.DataFrame()
            
        # Mostrar información sobre las noticias encontradas
        feed = data['feed']
        print(f"  Se encontraron {len(feed)} noticias en total.")
        
        last_processed_date = None
        
        # Procesar cada noticia
        for item in feed:
            processed += 1
            
            try:
                # Extraer información básica
                title = item.get('title', '')
                news_url = item.get('url', '')
                summary = item.get('summary', '')
                #print(item) Para debug
                # Mostrar los primeros 50 caracteres del título
                title_preview = title[:50] + "..." if len(title) > 50 else title
                print(f"\nNoticia #{processed}: '{title_preview}'")
                          
                # Extraer y convertir fecha a datetime
                published_date = datetime.now()  # Valor predeterminado
                if 'time_published' in item:
                    try:
                        # El formato típico es YYYYMMDDTHHMM
                        date_str = item['time_published']
                        published_date = datetime.strptime(date_str, '%Y%m%dT%H%M%S')
                    except (ValueError, TypeError):
                        print(f"  Error al parsear fecha {item.get('time_published')}, usando fecha actual")
                
                # Convertir a string formateado
                news_date_formatted = published_date.strftime('%Y-%m-%d %H:%M:%S')
                last_processed_date = news_date_formatted
                print(f"  Fecha: {news_date_formatted}")
                
                # Buscar el sentimiento específico para nuestro ticker
                ticker_sentiments = item.get('ticker_sentiment', [])
                ticker_data = None
                
                for sentiment in ticker_sentiments:
                    if sentiment.get('ticker') == ticker:
                        ticker_data = sentiment
                        break
                
                # Verificar si el ticker es relevante (relevant_score >= 0.7)
                is_relevant = False
                sentiment_score = 0
                relevant_score = 0
                
                if ticker_data:
                    relevant_score = float(ticker_data.get('relevance_score', 0))
                    sentiment_score = float(ticker_data.get('ticker_sentiment_score', 0))
                              
                    is_relevant = relevant_score >= 0.7
                    
                    print(f"  Relevancia para {ticker}: {relevant_score:.2f} - Sentimiento: {sentiment_score:.2f}")
                else:
                    print(f"  No se encontró información de sentimiento para {ticker}")
                
                # Si es relevante, guardar la información
                if is_relevant:
                    relevant_news += 1
                                       
                    # Preparar datos de la noticia
                    news_data = {
                        'fecha': news_date_formatted,
                        'tipo': 'alpha_vantage',
                        'subtipo': 'market news and sentiment',
                        'titulo': title,
                        'url': news_url,
                        'texto': summary[:1000],  # Limitar a 1000 caracteres aunque ira mucho menos
                        'keywords': ticker,
                        'alpha_vantage_value': sentiment_score,
                        'sentiment_algorithm': 'none', # Con 'none' indica que no ha sido procesado aun
                        'value': 0
                    }
                    
                    # Verificar si la noticia ya existe en la base de datos  
                    existing_news =  collection.find_one({'url': news_data['url']})
                    
                    if existing_news:
                        print(f"\033[38;2;184;134;11m  Noticia duplicada (ya existe en MongoDB)\033[0m")
                        duplicated_news += 1
                    else:
                        # Guardar en MongoDB
                        collection.insert_one(news_data)
                        saved_to_mongodb += 1
                        
                        # Añadir a resultados para el DataFrame
                        results.append(news_data)
                        print(f"\033[92m  Noticia relevante guardada\033[0m")
                else:
                    non_relevant_news += 1
                    print(f"\033[91m  Noticia no relevante para {ticker} (score < 0.7)\033[0m")              
            
            except Exception as e:
                print(f"  Error al procesar noticia #{processed}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Error al recuperar noticias de Alpha Vantage: {str(e)}")
        if last_processed_date:
            print(f"La ejecución se detuvo en la fecha: {last_processed_date}")
    
    # Convertir resultados a DataFrame
    columns = ['fecha', 'tipo', 'subtipo', 'titulo', 'url', 'fuente', 'texto', 
               'keywords', 'sentiment_algorithm', 'value']
    
    df = pd.DataFrame(results) if results else pd.DataFrame(columns=columns)
    
            # Mostrar estadísticas
    if len(df) > 0:
        oldest_date = df['fecha'].min()
        newest_date = df['fecha'].max()
        print(f"\nResultados:")
        print(f" - Noticias totales procesadas: {processed}")
        print(f" - Noticias relevantes para {ticker} (relevance_score >= 0.7): {relevant_news}")
        print(f" - Noticias no relevantes (relevance_score < 0.7): {non_relevant_news}")
        print(f" - Noticias guardadas en MongoDB: {saved_to_mongodb}")
        print(f" - Noticias duplicadas (ya existían en MongoDB): {duplicated_news}")
        print(f" - Fecha de la noticia más antigua: {oldest_date}")
        print(f" - Fecha de la noticia más reciente: {newest_date}")
        print(f" - Rango de tiempo solicitado: {time_from if time_from else 'No especificado'} a {time_to if time_to else 'No especificado'}")
        print(f" - Rango de tiempo procesado: {mongo_from} a {mongo_to}")
    else:
        print(f"\nNo se encontraron nuevas noticias relevantes para {ticker}.")
        print(f" - Noticias totales procesadas: {processed}")
        print(f" - Noticias no relevantes: {non_relevant_news}")
        print(f" - Noticias duplicadas verificadas: {duplicated_news}")
    
    print()
    # Cerramos conexión a MongoDB y devolvemos el dataframe
    closeMongoClient()
    return df

# %%
# 2. INGESTIÓN DE DATOS 

# Opcional: especificar fechas
start_date = "20220405T0130"
end_date = "20300615T0130"

# Coleccion de tickers
tickers = ["NVDA","INTC", "AMD"]
    
for ticker in tickers:
    result = searchAlphaVantageSentimentNews(
        ticker=ticker,
        api_key=AMPBConfig.AV_API_KEY,
        time_from=start_date,
        time_to=end_date
    )


