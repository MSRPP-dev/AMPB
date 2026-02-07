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
SA/GetData_YahooFinance v1.3
  - Permite leer datos de Yahoo Finance para un ticker.
  - Lo almacena en la base de datos MongoDB listo para ser analizado.
  - Problema: solo obtiene las 10 ultimas (restriccion de datos libres).
  - Exportado a GetDataYF_SA.py
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
import yfinance as yf
import warnings
import random
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
from ampblib import AMPBConfig, getMongoCollection, closeMongoClient
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

# %%
# 2. FUNCIÓN DE BUSQUEDA EN YAHOO FINANCE
def searchYahooFinance(ticker):

    print(f"{AMPBConfig.COLOR_VALUE}-------------------------------------------------------------------{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_MSG}Buscando noticias de Yahoo Finance para el ticker '{ticker}'{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_VALUE} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_VALUE}-------------------------------------------------------------------{AMPBConfig.COLOR_RESET}")

    # Inicializar variables
    results = []
    processed = 0
    new_news = 0
    duplicated_news = 0
    day_news_count = {}  # Contador de noticias por día
    
    # Obtener colección de MongoDB "market_sentiment"
    collection = getMongoCollection("market_sentiment")
    
    try:
        # Crear un objeto Ticker de yfinance
        ticker_obj = yf.Ticker(ticker)
        
        # Obtener las noticias
        print(f"Obteniendo noticias para {ticker}...")
        news = ticker_obj.news
        
        print(f"Se encontraron {len(news)} noticias en total.")
        
        # Procesar cada noticia
        for item in news:
            processed += 1
            
            try:
                # Extraer información de la noticia, teniendo en cuenta la estructura anidada
                if 'content' in item:
                    content = item['content']
                else:
                    content = item  # Por si la estructura cambia en el futuro
                
                title = content.get('title', '')
                
                # Para la URL, podría estar en diferentes lugares según la estructura
                news_url = ''
                if 'url' in content:
                    news_url = content.get('url', '')
                elif 'link' in content:
                    news_url = content.get('link', '')
                else:
                    # En algunos casos, podría ser necesario construir la URL
                    news_id = content.get('id', '')
                    if news_id:
                        news_url = f"https://finance.yahoo.com/news/{news_id}"
                
                # Mostrar los primeros 50 caracteres del título
                title_preview = title[:50] + "..." if len(title) > 50 else title
                print(f"  Noticia #{processed}: '{title_preview}'")
                
                # Extraer y convertir fecha ISO a datetime
                timestamp = None
                if 'pubDate' in content:
                    pub_date_str = content.get('pubDate')
                    if pub_date_str:
                        try:
                            # Convertir fecha ISO a objeto datetime
                            news_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                        except (ValueError, TypeError):
                            # Si falla, usar la fecha actual
                            news_date = datetime.now()
                            print(f"  Error al convertir fecha '{pub_date_str}' para la noticia #{processed}, usando fecha actual")
                else:
                    # Si no hay pubDate, usar la fecha actual
                    news_date = datetime.now()
                    print(f"  No se encontró fecha para la noticia #{processed}, usando fecha actual")
                
                # Convertir a string formateado
                news_date_formatted = news_date.strftime('%Y-%m-%d %H:%M:%S')
                news_date_day = news_date.strftime('%Y-%m-%d')  # Solo la fecha para contar por día
                
                    
                # Obtener resumen de la noticia
                summary = content.get('summary', '')
                if not summary:
                    # Intentar con description si no hay summary
                    summary = content.get('description', '')
                
                # Preparar datos de la noticia
                news_data = {
                    'fecha': news_date_formatted,
                    'tipo': 'yahoo_finance',
                    'subtipo': ticker,
                    'titulo': title,
                    'url': news_url,
                    'texto': summary[:1000],  # Limitar a 1000 caracteres
                    'sentiment_algorithm': 'none',  # Indica que no ha sido procesado aún
                    'value': 0
                }
                
                # Verificar si la noticia ya existe en la base de datos (comentado)
                existing_news = collection.find_one({'url': news_data['url']})
                
                if existing_news:
                    duplicated_news += 1
                    print(f"  Noticia duplicada (ya existe en MongoDB)")
                else:
                    # Guardar en MongoDB (comentado)
                    collection.insert_one(news_data)
                    new_news += 1
                    
                    # Añadir a resultados para el DataFrame
                    results.append(news_data)
                    
                    # Breve pausa para no sobrecargar
                    time.sleep(random.uniform(0.1, 0.3))
            
            except Exception as e:
                print(f"  Error al procesar noticia #{processed}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Error al recuperar noticias: {str(e)}")
    
    # Convertir resultados a DataFrame
    df = pd.DataFrame(results) if results else pd.DataFrame(columns=['fecha', 'tipo', 'subtipo', 'titulo', 'url', 'texto', 'sentiment_algorithm', 'value'])
    
    # Mostrar estadísticas
    if len(df) > 0:
        oldest_date = df['fecha'].min()
        print(f"\nResultados:")
        print(f" - Noticias totales procesadas: {processed}")
        print(f" - Noticias que cumplen criterios: {len(df) + duplicated_news}")
        print(f" - Noticias nuevas guardadas en MongoDB: {new_news} (fecha de la más antigua: {oldest_date})")
        print(f" - Noticias duplicadas (ya existentes en MongoDB): {duplicated_news}")
    else:
        print(f"\nNo se encontraron noticias que cumplan los criterios o todas son duplicadas.")
        print(f" - Noticias totales procesadas: {processed}")
        print(f" - Noticias duplicadas (ya existentes en MongoDB): {duplicated_news}")
    
    print("")
    # Cerramos conexión a MongoDB y devolvemos el dataframe (comentado)
    closeMongoClient()
    return df

# %%
# 2. INGESTIÓN DE DATOS 

# Coleccion de tickers
tickers = ["NVDA","INTC", "AMD"]
    
for ticker in tickers:
    result = searchYahooFinance(ticker=ticker)


