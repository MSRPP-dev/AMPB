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
GetData de Alpha Vantage v1.2
  - Ingestador automatico de datos en MongoDB desde Alpha Vantage
  - Exportado a GetDataAV.py
------------------------------------------------------------------------------------------------'''

# %%
# Importar librerías
from alpha_vantage.econindicators import EconIndicators
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import pymongo
from pymongo import MongoClient
import os
from ampblib import AMPBConfig, getMongoCollection, closeMongoClient

# %%
# 1. El ingestor de Alpha Vantage para "Indicadores económicos" a MongoDB
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
    

def getDataAV(ticker_name, start_date, collection_name):
    current_date = date.today().strftime('%Y-%m-%d')
   
    print(f"{AMPBConfig.COLOR_VALUE}-------------------------------------------------------------------{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_MSG}Ingestando datos de Alpha Vantage para el ticker '{ticker_name}' en '{collection_name}'{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_VALUE} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{AMPBConfig.COLOR_RESET}")
    print(f"{AMPBConfig.COLOR_VALUE}-------------------------------------------------------------------{AMPBConfig.COLOR_RESET}")

    try:
        # Conectar a MongoDB y configurar índices
        collection = getMongoCollection(collection_name)
        setupCollectionIndexes(collection_name)
        
        # Encontrar la última fecha en la base de datos
        last_date_in_db = getLastDateInCollection(collection, ticker_name)
        
        if last_date_in_db:
            print(f"Última fecha en MongoDB: {last_date_in_db}")
            filter_from_date = (datetime.strptime(last_date_in_db, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Filtrando datos desde: {filter_from_date}")
        else:
            filter_from_date = start_date
            print(f"No hay datos previos, filtrando desde: {filter_from_date}")
        
        # Descargar datos de Alpha Vantage según el ticker_name
        print("Descargando datos de Alpha Vantage...")
        
        try:
            indicator = EconIndicators(AMPBConfig.AV_API_KEY, output_format="pandas")
            
            # Mapeo de ticker_name a función de Alpha Vantage
            if ticker_name == "CPI":
                av_data, *_ = indicator.get_cpi(interval="monthly")
            elif ticker_name == "GDP_Real":
                av_data, *_ = indicator.get_real_gdp(interval="quarterly")
            elif ticker_name == "GDP_per_Capita":
                av_data, *_ = indicator.get_real_gdp_per_capita()            
            else:
                print(f" Ticker '{ticker_name}' no soportado. Opciones: CPI, GDP_Real, GDP_per_Capita")
                return {
                    "ticker": ticker_name,
                    "collection": collection_name,
                    "status": "error",
                    "message": f"Ticker '{ticker_name}' no soportado",
                    "records_updated": 0,
                    "records_inserted": 0,
                    "records_skipped": 0
                }
            
            if av_data is None or av_data.empty:
                print(f" No se encontraron datos para {ticker_name} en Alpha Vantage")
                return {
                    "ticker": ticker_name,
                    "collection": collection_name,
                    "status": "no_data",
                    "message": "No se encontraron datos en Alpha Vantage",
                    "records_updated": 0,
                    "records_inserted": 0,
                    "records_skipped": 0
                }
            
            # Procesar datos siguiendo tu patrón
            # Renombrar columna 'value' a 'Value'
            if 'value' in av_data.columns:
                av_data.rename(inplace=True, columns={"value": "Value"})
            
            # Alpha Vantage devuelve datos con 'date' en el índice como strings
            if 'date' in av_data.columns:
                # Si 'date' es una columna, usarla como índice
                av_data.set_index("date", inplace=True)
            
            # Convertir el índice a datetime, manejando diferentes formatos
            try:
                if not isinstance(av_data.index, pd.DatetimeIndex):
                    # Resetear índice para ver los datos originales
                    av_data_reset = av_data.reset_index()
                    print(f"Estructura de datos recibidos: {av_data_reset.columns.tolist()}")
                    print(f"Primeras fechas: {av_data_reset.head(3)}")
                    
                    # Alpha Vantage típicamente usa 'date' como nombre de índice
                    if 'date' in av_data_reset.columns:
                        av_data_reset['date'] = pd.to_datetime(av_data_reset['date'])
                        av_data = av_data_reset.set_index('date')
                    elif 'index' in av_data_reset.columns:
                        av_data_reset['index'] = pd.to_datetime(av_data_reset['index'])
                        av_data = av_data_reset.set_index('index')
                    else:
                        # Intentar convertir el índice directamente
                        av_data.index = pd.to_datetime(av_data.index)
                        
            except Exception as date_error:
                print(f" Error procesando fechas: {date_error}")
                # Intentar con el índice original
                av_data.index = pd.to_datetime(av_data.index, errors='coerce')
            
            # Ordenar por fecha
            av_data.sort_index(inplace=True)
            
            # Verificar que las fechas sean correctas
            if len(av_data) > 0:
                print(f"Datos descargados: {len(av_data)} registros")
                print(f"Rango de fechas: {av_data.index.min().strftime('%Y-%m-%d')} hasta {av_data.index.max().strftime('%Y-%m-%d')}")
                print(f"Primeras 3 fechas: {av_data.index[:3].strftime('%Y-%m-%d').tolist()}")
                print(f"Últimas 3 fechas: {av_data.index[-3:].strftime('%Y-%m-%d').tolist()}")
            else:
                print(" No se encontraron datos después del procesamiento de fechas")
            
        except Exception as e:
            print(f" Error descargando de Alpha Vantage: {str(e)}")
            return {
                "ticker": ticker_name,
                "collection": collection_name,
                "status": "error",
                "message": f"Error descargando de Alpha Vantage: {str(e)}",
                "records_updated": 0,
                "records_inserted": 0,
                "records_skipped": 0
            }
        
        # Filtrar datos desde la fecha de inicio o última fecha + 1
        print(f"Filtrando datos desde: {filter_from_date}")
        print(f"Tipo de índice: {type(av_data.index)}")
        
        # Convertir filter_from_date a datetime para comparación
        filter_date = pd.to_datetime(filter_from_date)
        av_data_filtered = av_data[av_data.index >= filter_date].copy()
        
        print(f"Registros antes del filtrado: {len(av_data)}")
        print(f"Registros después del filtrado: {len(av_data_filtered)}")
        
        if av_data_filtered.empty:
            print(f" No hay datos nuevos después de filtrar desde {filter_from_date}")
            # Mostrar información adicional para debugging
            if len(av_data) > 0:
                print(f"Fecha mínima disponible: {av_data.index.min().strftime('%Y-%m-%d')}")
                print(f"Fecha máxima disponible: {av_data.index.max().strftime('%Y-%m-%d')}")
            return {
                "ticker": ticker_name,
                "collection": collection_name,
                "status": "no_new_data",
                "message": f"No hay datos nuevos después de {filter_from_date}",
                "records_updated": 0,
                "records_inserted": 0,
                "records_skipped": 0
            }
        
        # Obtener fechas disponibles
        available_dates = set(av_data_filtered.index.strftime('%Y-%m-%d').tolist())
        today = date.today().strftime('%Y-%m-%d')
        update_today = today in available_dates
        
        # Consultar fechas existentes en el rango
        if available_dates:
            range_start = min(available_dates)
            range_end = max(available_dates)
            existing_dates = getExistingDatesInRange(collection, ticker_name, range_start, range_end)
        else:
            existing_dates = set()
        
        missing_dates = available_dates - existing_dates
        
        if update_today:
            missing_dates.discard(today)  # Procesaremos por separado
        
        print(f"Fechas disponibles en Alpha Vantage: {len(available_dates)}")
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
            filtered_data = av_data_filtered[av_data_filtered.index.strftime('%Y-%m-%d').isin(missing_dates_list)].copy()
            
            # Preparar documentos para inserción
            documents_to_insert = []
            for date_idx, row in filtered_data.iterrows():
                # Buscar el valor en la columna Value o en la primera columna numérica
                value = None
                if 'Value' in row.index and not pd.isna(row['Value']):
                    value = float(row['Value'])
                else:
                    # Buscar la primera columna numérica
                    for col in row.index:
                        if pd.api.types.is_numeric_dtype(type(row[col])) and not pd.isna(row[col]):
                            value = float(row[col])
                            break
                
                doc = {
                    "Date": date_idx.strftime('%Y-%m-%d'),
                    "Name": ticker_name,
                    "Value": value
                }
                documents_to_insert.append(doc)
            
            # Insertar en lotes con manejo robusto de duplicados
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
        if update_today and today in available_dates:
            print(f"Actualizando datos del día actual ({today})...")
            
            today_data = av_data_filtered[av_data_filtered.index.strftime('%Y-%m-%d') == today]
            
            if not today_data.empty:
                row = today_data.iloc[0]
                
                # Buscar el valor en la columna Value o en la primera columna numérica
                value = None
                if 'Value' in row.index and not pd.isna(row['Value']):
                    value = float(row['Value'])
                else:
                    # Buscar la primera columna numérica
                    for col in row.index:
                        if pd.api.types.is_numeric_dtype(type(row[col])) and not pd.isna(row[col]):
                            value = float(row[col])
                            break
                
                update_doc = {"Value": value}
                
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
            "filter_from_date": filter_from_date,
            "end_date": current_date,
            "records_inserted": records_inserted,
            "records_updated": records_updated,
            "records_skipped": records_skipped,
            "total_processed": total_processed
        }
        
    except Exception as e:
        print(f"{AMPBConfig.COLOR_INFO}Error en ingestDataAV para '{ticker_name}' en '{collection_name}': {str(e)}{AMPBConfig.COLOR_RESET}\n")
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

# 2.1 Coleccion "indicadores_economicos"
tickers_ieconomicos = ["CPI", "GDP_Real", "GDP_per_Capita"]
    
for ticker in tickers_ieconomicos:
    result = getDataAV(
        ticker_name=ticker,
        start_date=start_date,
        collection_name="indicadores_economicos",
    )

# 2.2 Cerramos conexion con MongoDB
closeMongoClient()


