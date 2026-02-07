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
SA/Analyzer FinBERT v1.3
  - Analiza la coleccion MongoDB y asigna el valor entre [0,1].
  - Exportado a SA_Analyzer.py
------------------------------------------------------------------------------------------------'''

# %%
# Importar librerías
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
from ampblib import AMPBConfig, getMongoCollection, closeMongoClient
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')

# %%
# 1. CONFIGURACÓN DE MODELO FINBERT
MODEL_NAME = "ProsusAI/finbert"
tokenizer = None
model = None
sentiment_analyzer_finbert = None
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1  # Usar GPU si está disponible
    sentiment_analyzer_finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
    print("FinBERT cargado correctamente.")
except Exception as e:
    print(f"Error cargando FinBERT: {e}. El análisis de sentimiento no funcionará.")

# %%
# 2. FUNCIÓN DE ANALSIS DE SENTIMIENTO DE DOCUMENTOS MONGODB
def analizeSentiment(limit=0):
    start_time = time.time()
    
    # Obtener conexión a MongoDB
    collection = getMongoCollection("market_sentiment")
    
    try:
        # Contar documentos totales
        total_docs = collection.count_documents({})
        
        # Filtrar documentos que no han sido analizados (sentiment_algorithm es None o no existe)
        query = {"$or": [{"sentiment_algorithm": "none"}, {"sentiment_algorithm": {"$exists": False}}]}
        
        # Contar documentos pendientes
        pending_count = collection.count_documents(query)
        
        # Aplicar límite si se especifica
        print(f"{AMPBConfig.COLOR_VALUE}-------------------------------------------------------------------{AMPBConfig.COLOR_RESET}")
        print(f"{AMPBConfig.COLOR_MSG}Análisis de sentimiento (FinBERT) de noticias en la base de datos{AMPBConfig.COLOR_RESET}")
        if limit > 0:
            pending_docs = collection.find(query).limit(limit)
            docs_to_process = min(limit, pending_count)
            print(f"{AMPBConfig.COLOR_VALUE} Modo limitado: procesando máximo {limit} documentos de {pending_count} pendientes{AMPBConfig.COLOR_RESET}")
        else:
            pending_docs = collection.find(query)
            docs_to_process = pending_count
            print(f"{AMPBConfig.COLOR_VALUE} Modo completo: procesando todos los {pending_count} documentos pendientes{AMPBConfig.COLOR_RESET}")
        
        print(f"{AMPBConfig.COLOR_VALUE} Total de documentos en la colección: {total_docs}{AMPBConfig.COLOR_RESET}")
        print(f"{AMPBConfig.COLOR_VALUE} Documentos a procesar en esta ejecución: {docs_to_process}{AMPBConfig.COLOR_RESET}")
        print(f"{AMPBConfig.COLOR_VALUE} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{AMPBConfig.COLOR_RESET}")
        print(f"{AMPBConfig.COLOR_VALUE}-------------------------------------------------------------------{AMPBConfig.COLOR_RESET}")
        
        # Contador de documentos analizados
        analyzed_count = 0
        
        # Procesar cada documento pendiente
        for doc in pending_docs:
            # Obtener el texto a analizar
            text = doc.get('texto', '')
            
            if text and sentiment_analyzer_finbert:
                try:
                    # Analizar sentimiento con FinBERT
                    result = sentiment_analyzer_finbert(text)
                    sentiment_value = result[0]
                    
                    # Mapear valores de sentimiento
                    # FinBERT devuelve: positive, negative, neutral
                    # Convertimos a un valor continuo de 0 a 1
                    label = sentiment_value['label']
                    score = sentiment_value['score']
                    
                    if label == 'positive':
                        value = 0.5 + (score * 0.5)  # Valor positivo (0.5 a 1.0)
                    elif label == 'negative':
                        value = 0.5 - (score * 0.5)  # Valor negativo (0.0 a 0.5)
                    else:  # neutral
                        value = 0.5  # Valor neutro exactamente en el medio
                    
                    # Actualizar documento en MongoDB
                    collection.update_one(
                        {"_id": doc["_id"]},
                        {"$set": {
                            "value": value,
                            "sentiment_algorithm": "finbert_v1",
                            "processed_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }}
                    )
                    
                    analyzed_count += 1
                    
                    # Mostrar progreso cada 100 documentos (o cada 10 si el límite es pequeño)
                    progress_interval = 10 if limit > 0 and limit <= 100 else 100
                    if analyzed_count % progress_interval == 0:
                        progress_pct = (analyzed_count / docs_to_process) * 100 if docs_to_process > 0 else 0
                        print(f" Procesados {analyzed_count}/{docs_to_process} documentos ({progress_pct:.1f}%)...")
                        
                except Exception as e:
                    print(f" Error analizando documento {doc.get('_id')}: {str(e)}")
            else:
                # Si no hay texto para analizar, marcar como procesado pero con valor nulo
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {
                        "value": 0.5,  # Valor neutro cuando no hay texto
                        "sentiment_algorithm": "finbert_v1",
                        "processed_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "processing_note": "No text available for analysis"
                    }}
                )
                analyzed_count += 1
        
        # Estadísticas finales
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Obtener conteos actualizados
        total_docs_after = collection.count_documents({})
        analyzed_docs = collection.count_documents({"sentiment_algorithm": "finbert_v1"})
        remaining_pending = collection.count_documents(query)
        
        print(f" Tiempo total de ejecución: {execution_time:.2f} segundos")
        if analyzed_count > 0:
            print(f" Velocidad promedio: {analyzed_count/execution_time:.1f} documentos/segundo")
        print(f" Total de documentos en la colección: {total_docs_after}")
        print(f" Documentos analizados en esta ejecución: {analyzed_count}")
        print(f" Total de documentos con análisis finbert_v1: {analyzed_docs}")
        print(f" Documentos pendientes de análisis: {remaining_pending}")
        
        if limit > 0:
            print(f" Modo limitado: Se procesaron {analyzed_count} de un máximo de {limit}")            
        print("")       

        return {
            "status": "success",
            "processed_count": analyzed_count,
            "total_analyzed": analyzed_docs,
            "remaining_pending": remaining_pending,
            "execution_time": execution_time,
            "limit_applied": limit
        }
        
    except Exception as e:
        print(f" Error durante el procesamiento: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "processed_count": analyzed_count if 'analyzed_count' in locals() else 0
        }
    finally:
        # Cerrar la conexión a MongoDB
        closeMongoClient()

# %%
# 3. EJECUCIÓN
ret = analizeSentiment(10)


