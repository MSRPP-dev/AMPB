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
GetData de Yahoo Finance v1.2
  - Ingestador automatico de datos en MongoDB desde Yahoo Finance.
  - Exportado a GetDataYF.py
------------------------------------------------------------------------------------------------'''

# %%
# Importar librerías
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import pymongo
from pymongo import MongoClient
import os
from ampblib import AMPBConfig, getMongoCollection, closeMongoClient, calculateTechnicalIndicators

# %%
# 1. El ingestor principal de Yahoo Finance a MongoDB
def setupCollectionIndexes(collection_name):
    collection = getMongoCollection(collection_name)
    try:
        collection.create_index([("Name", 1), ("Date", -1)])
        collection.create_index([("Name", 1), ("Date", 1)], unique=True)
        return True
    except Exception as e:
        print(f" Error creando índices en {collection_name}: {e}")
        return False


def getLastDateInCollection(collection, ticker_name):
    try:
        last_doc = collection.find(
            {"Name": ticker_name},
            {"Date": 1, "_id": 0}
        ).sort("Date", -1).limit(1)
        
        last_doc_list = list(last_doc)
        if last_doc_list:
            return last_doc_list[0]["Date"]
        return None
    except Exception as e:
        print(f" Error obteniendo última fecha: {e}")
        return None

def getExistingDatesInRange(collection, ticker_name, start_date, end_date):
    try:
        existing_docs = list(collection.find(
            {"Name": ticker_name, "Date": {"$gte": start_date, "$lte": end_date}},
            {"Date": 1, "_id": 0}
        ))
        existing_dates = set([doc["Date"] for doc in existing_docs])
        return existing_dates
    except Exception as e:
        print(f" Error consultando fechas existentes: {e}")
        return set()
    

def getDataYF(ticker_name, start_date, collection_name, field_mapping, calculate_indicators=True):
    current_date = date.today().strftime('%Y-%m-%d')
    
    print(f"{AMPBConfig.COLOR_VALUE}-------------------------------------------------------------------{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_MSG}Ingestando datos de Yahoo Finance para el ticker '{ticker_name}' en '{collection_name}'{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_VALUE} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_VALUE}-------------------------------------------------------------------{AMPBConfig.COLOR_RESET}")
    
    try:
        # Conectar a MongoDB y configurar índices
        collection = getMongoCollection(collection_name)
        setupCollectionIndexes(collection_name)
        
        # Encontrar la última fecha en la base de datos
        last_date_in_db = getLastDateInCollection(collection, ticker_name)
        
        if last_date_in_db:
            # Si existe data, descargar solo desde el día siguiente a la última fecha
            download_start_date = (datetime.strptime(last_date_in_db, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Última fecha en BD: {last_date_in_db}")
            print(f"Descargando solo desde: {download_start_date}")
        else:
            # Si no existe data, usar start_date original
            download_start_date = start_date
            print(f"No hay datos previos, descargando desde: {download_start_date}")
        
        # Solo descargar si hay fechas nuevas que obtener
        if download_start_date <= current_date:
            print("Descargando datos de Yahoo Finance...")
            temp_data = yf.download(
                ticker_name, 
                start=download_start_date, 
                end=current_date, 
                progress=False,
                auto_adjust=True  # Silenciar el warning
            )
        else:
            print("No hay fechas nuevas para descargar")
            temp_data = pd.DataFrame()
        
        if temp_data.empty:
            print(f" No se encontraron datos nuevos para {ticker_name}")
            return {
                "ticker": ticker_name,
                "collection": collection_name,
                "status": "no_new_data",
                "message": "No hay datos nuevos para descargar",
                "records_updated": 0,
                "records_inserted": 0,
                "records_skipped": 0
            }
        
        # Convertir MultiIndex a columnas planas si es necesario
        if isinstance(temp_data.columns, pd.MultiIndex):
            temp_data.columns = [col[0] for col in temp_data.columns]
        
        # Calcular indicadores técnicos si es necesario
        if calculate_indicators:
            temp_data = calculateTechnicalIndicators(temp_data)
        
        # MEJORA: Obtener fechas de trading y filtrar solo las >= download_start_date
        trading_dates_all = set(temp_data.index.strftime('%Y-%m-%d').tolist())
        trading_dates = set()
        
        # Filtrar solo fechas que son realmente nuevas
        for date_str in trading_dates_all:
            if date_str >= download_start_date:
                trading_dates.add(date_str)
        
        today = date.today().strftime('%Y-%m-%d')
        update_today = today in trading_dates
        
        print(f"Fechas descargadas de Yahoo: {len(trading_dates_all)}")
        print(f"Fechas filtradas (>= {download_start_date}): {len(trading_dates)}")
        
        # MEJORA: Consultar fechas existentes usando función especializada
        if trading_dates:
            range_start = min(trading_dates)
            range_end = max(trading_dates)
            existing_dates = getExistingDatesInRange(collection, ticker_name, range_start, range_end)
        else:
            existing_dates = set()
        
        missing_dates = trading_dates - existing_dates
        
        if update_today:
            missing_dates.discard(today)  # Procesaremos por separado
        
        print(f"Fechas nuevas disponibles: {len(trading_dates)}")
        print(f"Fechas faltantes para insertar: {len(missing_dates)}")
        print(f"¿Actualizar datos de hoy ({today})?: {'Sí' if update_today else 'No'}")
        
        # Contadores
        records_inserted = 0
        records_updated = 0
        records_skipped = len(existing_dates) - (1 if update_today else 0)
        
        # Procesar fechas faltantes (insertar nuevos registros)
        if missing_dates:
            print(f"Procesando {len(missing_dates)} fechas faltantes...")
            
            missing_dates_list = sorted(list(missing_dates))
            filtered_data = temp_data[temp_data.index.strftime('%Y-%m-%d').isin(missing_dates_list)].copy()
            
            # Preparar documentos para inserción
            documents_to_insert = []
            for date_idx, row in filtered_data.iterrows():
                doc = {"Date": date_idx.strftime('%Y-%m-%d'), "Name": ticker_name}
                
                # Aplicar mapeo de campos
                for mongo_field, source_field in field_mapping.items():
                    if source_field in row.index:
                        value = row[source_field]
                        if not pd.isna(value):
                            if mongo_field == "Volume":
                                doc[mongo_field] = int(value)
                            elif mongo_field == "Trend":
                                doc[mongo_field] = int(value)
                            else:
                                doc[mongo_field] = float(value)
                        else:
                            doc[mongo_field] = None
                
                documents_to_insert.append(doc)
            
            # MEJORA: Insertar en lotes con manejo robusto de duplicados
            if documents_to_insert:
                try:
                    collection.insert_many(documents_to_insert, ordered=False)
                    records_inserted = len(documents_to_insert)
                    print(f"Insertados {records_inserted} nuevos registros")
                except Exception as e:
                    # Manejar errores de duplicados individualmente
                    if "duplicate key error" in str(e).lower():
                        print(" Detectados duplicados, insertando uno por uno...")
                        successful_inserts = 0
                        duplicate_count = 0
                        
                        for doc in documents_to_insert:
                            try:
                                collection.insert_one(doc)
                                successful_inserts += 1
                            except Exception as single_error:
                                if "duplicate key error" in str(single_error).lower():
                                    duplicate_count += 1
                                    print(f" Duplicado omitido: {doc['Date']}")
                                else:
                                    print(f" Error insertando {doc['Date']}: {single_error}")
                        
                        records_inserted = successful_inserts
                        records_skipped += duplicate_count
                        print(f"Insertados {successful_inserts} registros, {duplicate_count} duplicados omitidos")
                    else:
                        raise e
        
        # Procesar datos del día actual (actualizar si existe)
        if update_today and today in trading_dates:
            print(f"Actualizando datos del día actual ({today})...")
            
            today_data = temp_data[temp_data.index.strftime('%Y-%m-%d') == today]
            
            if not today_data.empty:
                row = today_data.iloc[0]
                update_doc = {}
                
                # Aplicar mapeo de campos
                for mongo_field, source_field in field_mapping.items():
                    if source_field in row.index:
                        value = row[source_field]
                        if not pd.isna(value):
                            if mongo_field == "Volume":
                                update_doc[mongo_field] = int(value)
                            elif mongo_field == "Trend":
                                update_doc[mongo_field] = int(value)
                            else:
                                update_doc[mongo_field] = float(value)
                        else:
                            update_doc[mongo_field] = None
                
                # Actualizar o insertar
                result = collection.update_one(
                    {"Name": ticker_name, "Date": today},
                    {"$set": update_doc},
                    upsert=True
                )
                
                if result.matched_count > 0:
                    records_updated = 1
                    print(f"Actualizado registro del día actual ({today})")
                else:
                    records_inserted += 1
                    print(f"Insertado nuevo registro del día actual ({today})")
        
        # Resumen final
        total_processed = records_inserted + records_updated
        print(f"{AMPBConfig.COLOR_VALUE}Resumen de operación para '{ticker_name}' en '{collection_name}'{AMPBConfig.COLOR_RESET}")
        print(f"{AMPBConfig.COLOR_VALUE}Registros insertados: {records_inserted}{AMPBConfig.COLOR_RESET}")
        print(f"{AMPBConfig.COLOR_VALUE}Registros actualizados: {records_updated}{AMPBConfig.COLOR_RESET}")
        print(f"{AMPBConfig.COLOR_VALUE}Registros omitidos (ya existían): {records_skipped}{AMPBConfig.COLOR_RESET}")
        print(f"{AMPBConfig.COLOR_VALUE}Total procesados: {total_processed}{AMPBConfig.COLOR_RESET}\n")
        
        return {
            "ticker": ticker_name,
            "collection": collection_name,
            "status": "success",
            "start_date": start_date,
            "download_start_date": download_start_date,
            "end_date": current_date,
            "records_inserted": records_inserted,
            "records_updated": records_updated,
            "records_skipped": records_skipped,
            "total_processed": total_processed
        }
        
    except Exception as e:
        print(f"{AMPBConfig.COLOR_INFO}Error en ingestData para '{ticker_name}' en '{collection_name}': {str(e)}{AMPBConfig.COLOR_RESET}\n")
        return {
            "ticker": ticker_name,
            "collection": collection_name,
            "status": "error",
            "message": str(e),
            "records_updated": 0,
            "records_inserted": 0,
            "records_skipped": 0
        }

# %%
# 2. INGESTIÓN DE DATOS 
start_date = "2015-01-05"

# 2.1 Coleccion "tickers"
field_mapping_tickers = {
        "Close": "Close",
        "High": "High",
        "Low": "Low",
        "Open": "Open",
        "Volume": "Volume",
        "SMA_20": "SMA_20",
        "SMA_50": "SMA_50",
        "SMA_200": "SMA_200",
        "EMA_20": "EMA_20",
        "RSI_14": "RSI_14",
        "MACD": "MACD",
        "MACD_signal": "MACD_signal",
        "ATR_14": "ATR_14",
        "BB_upper": "BB_upper",
        "BB_lower": "BB_lower",
        "Range": "Range",
        "OC_Change": "OC_Change",
        "Chaikin_Osc": "Chaikin_Osc",
        "Trend": "Trend"
    }
tickers_tickers = ["GOOGL", "AMZN", "AAPL", "META", "MSFT", "NVDA", "TSLA", "AMD", "INTC"]
    
for ticker in tickers_tickers:
    result = getDataYF(
        ticker_name=ticker,
        start_date=start_date,
        collection_name="tickers",
        field_mapping=field_mapping_tickers,
        calculate_indicators=True
    )


# 2.2 Coleccion "indices_bursatiles"
field_mapping_ibursatil = {
        "Close": "Close"
    }
tickers_ibursatil = ["^GSPC", "^NDX", "^STOXX50E", "^N225", "000001.SS"]
    
for ticker in tickers_ibursatil:
    result = getDataYF(
        ticker_name=ticker,
        start_date=start_date,
        collection_name="indices_bursatiles",
        field_mapping=field_mapping_ibursatil,
        calculate_indicators=False
    )


# 2.3 Coleccion "indicadores_economicos"
field_mapping_ieconomico = {
        "Close": "Close"
    }
tickers_ieconomicos = ["^IRX", "^TNX", "^VIX", "BZ=F", "GLD"]
    
for ticker in tickers_ieconomicos:
    result = getDataYF(
        ticker_name=ticker,
        start_date=start_date,
        collection_name="indicadores_economicos",
        field_mapping=field_mapping_ieconomico,
        calculate_indicators=False
    )

# 2.4 Cerramos conexion con MongoDB
closeMongoClient()


