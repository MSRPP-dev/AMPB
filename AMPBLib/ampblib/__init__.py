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

------------------------------------------------
AMPBLib - Funciones auxiliares 4.3.0

TODO: anadir intervalo de confianza a printMetricsRegresssion()?
------------------------------------------------'''

# Importar librerías (optimizado)
import os
import sys
from urllib.parse import quote_plus
from pymongo import MongoClient
import numpy as np
import pandas as pd
import matplotlib

# -----------------------------------------------------------------------------
# Parámetros globales e enformación general para todo el proyecto
# -----------------------------------------------------------------------------
class AMPBConfig:
    SEED = 42              # Semilla para intentar que los modelos sean deterministas
    SCORING_WEIGHTS = {'regression': 0.45, 'classification': 0.55} # Parámetros de puntuación
    BASELINE_R2 = 0.811    # Baseline R2 de predictor base de regresión
    BASELINE_MAE = 3.29    # Baseline MAE de predictor base de regresión
    BASELINE_RMSE = 4.57   # Baseline RMSE de predictor base de regresión
    BASELINE_ACC = 0.517   # Baseline Accuracy de predictor base de clasificación
    BASELINE_F1 = 0.55     # Baseline F1-Score de predictor base de clasificación
    BASELINE_ROCAUC = 0.51 # Baseline ROC-AUC de predictor base de clasificación

    OUTPUT_DIR = "model_results" # Directorio de salida
    INTERACTIVE = True   # Indica si estamos en Jupyter Notebook o en command line
    MONGODB_CLIENT = None  # Almacena la conexion con MongoDB
    MONGODB_DEBUG = False  # Para depurar las conexiones con MongoDB
    
    PROYECTO = "Análisis de modelos predictivos en bolsa: NVIDIA"
    AUTOR = "MSRPP"
    UNIVERSIDAD = "Copyright (C) 2024-2025 MegaStorm Systems"
    COLOR_HEADER = "\033[96m"   # Cian claro
    COLOR_KEY    = "\033[93m"   # Amarillo
    COLOR_VALUE  = "\033[92m"   # Verde
    COLOR_INFO   = "\033[1;91m" # Rojo
    COLOR_MSG    =" \033[94m"   # Azul
    COLOR_RESET  = "\033[0m"    # Reset   
        
    # ---Credenciales (actualizar)---
    DB_USER = "admin"    # Usuario de MongoDB si no encuentra la variable de entorno "MONGO_AMPB_USER"
    DB_PASSWORD = "UEM"  # Password de MongoDB si no encuentra la variable de entorno "MONGO_AMPB_PASS"
    AV_API_KEY = "INTRODUCIR_KEY"   # Keys para Alpha Vantage, hay un limite de 25/peticiones al dia por key/IP si no encuentra la variable de entorno "AV_API_KEY"
    REDDIT_CLIENT_ID = "INTRODUCIR_CLIENTID"              # Reddit clientid si no encuentra la variable de entorno "REDDIT_CLIENT_ID"
    REDDIT_CLIENT_SECRET = "INTRODUCIR_CLIENTSECRET"  # Reddit clientsecret si no encuentra la variable de entorno "REDDIT_CLIENT_SECRET" 
    REDDIT_USERNAME = "INTRODUCIR_USER"                  # Reddit username si no encuentra la variable de entorno "REDDIT_USER"
    REDDIT_PASSWORD = "INTRODUCIR_PASSWORD"                        # Reddit password si no encuentra la variable de entorno "REDDIT_PASSWORD"
    # ---Fin de credenciales---
    
    @staticmethod
    def printHeader(title, testsize, optimize, backtesting, transform, exogscaling, exogsetid, fixedorder=None, fixedseasonalorder=None):
        C_H, C_K, C_V, C_R = (
            AMPBConfig.COLOR_HEADER,
            AMPBConfig.COLOR_KEY,
            AMPBConfig.COLOR_VALUE,
            AMPBConfig.COLOR_RESET,
        )
        def line(txt=""):
            print(f"{C_H}{txt}{C_R}")
        # Cabecera
        line("=" * 80)
        line(AMPBConfig.PROYECTO)
        line(AMPBConfig.UNIVERSIDAD)
        line(f"{title} - {AMPBConfig.AUTOR}")
        line("-" * 80)
        # Detalles
        details = [
            ("Días utilizados como test", testsize),
            ("Optimización automática", "activada" if optimize else "desactivada"),
            ("Back-testing", "activado" if backtesting else "desactivado"),
            ("Transformación", transform),
            ("Escalado de exógenas", exogscaling),
            ("Conjunto de exógenas (ID)", exogsetid),            
        ]
        for label, value in details:
            print(f"{C_K}  • {label}:{C_R} {C_V}{value}{C_R}")
        print(f"{C_K}  • Ejecución interactiva (Jupyter){C_R}") if AMPBConfig.INTERACTIVE else print(f"{C_K}  • Ejecución desde línea de comandos{C_R}")
        # Pie de cabecera
        line("=" * 80)
         
# Detectar si el entorno es interactivo 
if 'ipykernel_launcher' not in sys.argv[0]:
    matplotlib.use('Agg') # Cambiamos backend de Matplotlib
    AMPBConfig.INTERACTIVE=False
    
    
# -----------------------------------------------------------------------------
# Configuración y acceso a MongoDB
# -----------------------------------------------------------------------------
MONGODB_CONFIG = {
    "host": "financedb",
    "port": 27017,
    "database_name": "AMPB",
    "user": AMPBConfig.DB_USER,  
    "password": AMPBConfig.DB_PASSWORD, 
}
def getMongoClient():
    if AMPBConfig.MONGODB_CLIENT:
        return AMPBConfig.MONGODB_CLIENT

    if(AMPBConfig.MONGODB_DEBUG):
        print(f"{AMPBConfig.COLOR_HEADER}Creando nueva conexión a MongoDB...{AMPBConfig.COLOR_RESET}")
    # Prioridad 1: Variables de entorno del Sistema Operativo
    user_env = os.getenv('MONGO_AMPB_USER')
    pass_env = os.getenv('MONGO_AMPB_PASS')
    # Prioridad 2: Valores hardcodeados en el script (si las variables de entorno no están)
    user = user_env or MONGODB_CONFIG.get('user')
    password = pass_env or MONGODB_CONFIG.get('password')
    # Construir la cadena de conexión dinámicamente
    host = MONGODB_CONFIG['host']
    port = MONGODB_CONFIG['port']
    
    if user and password:
        user_safe = quote_plus(user)
        password_safe = quote_plus(password)
        connection_string = f"mongodb://{user_safe}:{password_safe}@{host}:{port}/"
        if(AMPBConfig.MONGODB_DEBUG):
            print(f"{AMPBConfig.COLOR_HEADER}Conectando con el usuario: '{user}' (obtenido de {'entorno' if user_env else 'script'}){AMPBConfig.COLOR_RESET}")
    else:
        # Conexión sin autenticación
        connection_string = f"mongodb://{host}:{port}/"
        if(AMPBConfig.MONGODB_DEBUG):
            print(f"{AMPBConfig.COLOR_HEADER}Conectando sin autenticación.{AMPBConfig.COLOR_RESET}")

    try:
        # Guardamos el cliente en la variable global para reutilizarlo
        AMPBConfig.MONGODB_CLIENT = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        # Forzamos una conexión para verificar que las credenciales son válidas
        AMPBConfig.MONGODB_CLIENT.server_info() 
        if(AMPBConfig.MONGODB_DEBUG):
            print(f"{AMPBConfig.COLOR_HEADER}Conexión a MongoDB exitosa.{AMPBConfig.COLOR_RESET}\n")
        return AMPBConfig.MONGODB_CLIENT
    except Exception as e:
        print(f"{AMPBConfig.COLOR_INFO}Error al conectar a MongoDB: {e}{AMPBConfig.COLOR_RESET}\n")
        AMPBConfig.MONGODB_CLIENT = None # Reseteamos el cliente en caso de error
        raise
        
def getMongoCollection(collection_name):
    client = getMongoClient()
    db = client[MONGODB_CONFIG["database_name"]]
    return db[collection_name]    

def deleteMongoCollection(collection_name):
    try:
        collection = getMongoCollection(collection_name)
        resultado = collection.delete_many({})
        print(f"{AMPBConfig.COLOR_HEADER}Se eliminaron {resultado.deleted_count} documentos de la colección '{collection_name}'.{AMPBConfig.COLOR_RESET}")
    except Exception as e:
        print(f"{AMPBConfig.COLOR_HEADER}Error al eliminar documentos de '{collection_name}': {e}{AMPBConfig.COLOR_RESET}")
        
def closeMongoClient():
    if AMPBConfig.MONGODB_CLIENT:
        AMPBConfig.MONGODB_CLIENT.close()
        AMPBConfig.MONGODB_CLIENT = None
        if(AMPBConfig.MONGODB_DEBUG):
            print(f"\n{AMPBConfig.COLOR_HEADER}Conexión a MongoDB cerrada.{AMPBConfig.COLOR_RESET}\n")

        
# -----------------------------------------------------------------------------
# Conjuntos de datos - Devuelve la lista de features
#  - Numero con los ids (del 1 al 6) de los conjuntos de datos a utilizaar.
# -----------------------------------------------------------------------------
def getExogVars(code: int) -> list:
    groups = {
        1: ['High', 'Low', 'Open', 'Volume'], # Directos
        2: ['SMA_200', 'SMA_50', 'SMA_20', 'EMA_20', 'ATR_14', 'BB_upper', 'BB_lower', 'Range', 'MACD', 'MACD_signal', 'Chaikin_Osc', 'RSI_14', 'OC_Change'], # IndicadoresTecnicos
        3: ['Google', 'Amazon', 'Apple', 'Meta', 'Microsoft', 'Tesla', 'AMD', 'Intel'], # BigTech
        4: ['SP500', 'NASDAQ100', 'EuroStoxx50', 'Nikkei225', 'ShanghaiComposite'], # IndicesBursatiles
        5: ['Treasury_3M', 'Treasury_10Y', 'VIX', 'Brent_Oil', 'Gold', 'CPI', 'GDP_Real', 'GDP_per_Capita'], # IndicadoresEconomicos
        6: ['googletrends_NVDA', 'av_nvidia'] # AnalisisSentimiento
    }
    # Extraer dígitos únicos del número
    digits = set(int(d) for d in str(code) if d.isdigit())
    # Verificar si hay dígitos no válidos
    invalid_digits = [d for d in digits if d not in groups]
    if invalid_digits:
        raise ValueError(f"Dígitos inválidos en el código: {invalid_digits}. Solo se permiten valores entre 1 y 6.")
    # Recopilar las columnas correspondientes a los grupos válidos
    selected_columns = []
    for d in sorted(digits):
        selected_columns.extend(groups[d])
    return selected_columns


# -----------------------------------------------------------------------------
# Reinicia todas las semillas aleatorias
# -----------------------------------------------------------------------------
def resetRandomSeeds(tensorflow_deterministic):    
    import random    
    np.random.seed(AMPBConfig.SEED)
    random.seed(AMPBConfig.SEED)
    os.environ['PYTHONHASHSEED'] = str(AMPBConfig.SEED) 
    if tensorflow_deterministic:
        import tensorflow as tf
        tf.random.set_seed(AMPBConfig.SEED)
        tf.keras.utils.set_random_seed(AMPBConfig.SEED)
        tf.config.experimental.enable_op_determinism()
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        
    
# -----------------------------------------------------------------------------
# Configura TensorFlow para usar CPU o GPU
# -----------------------------------------------------------------------------
def setupTensorflowDevice(use_gpu=True):
    import tensorflow as tf
    if not use_gpu:
        # Forzar CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.keras.backend.set_floatx('float32')
        print(f"{AMPBConfig.COLOR_INFO}Configurando TensorFlow para CPU{AMPBConfig.COLOR_RESET}")
    else:
        from tensorflow.keras import mixed_precision
        # Configurar GPU si está disponible
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Habilitar crecimiento dinámico de memoria
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                # Configurar mixed precision para acelerar entrenamiento en GPU
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                print(f"{AMPBConfig.COLOR_INFO}Configurando TensorFlow para GPU ({len(gpus)} disponible/s){AMPBConfig.COLOR_RESET}")
            except RuntimeError as e:
                print(f"Error configurando GPU, usando CPU: {e}")
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            tf.keras.backend.set_floatx('float32')
            print(f"{AMPBConfig.COLOR_INFO}No hay GPUs disponibles, usando CPU{AMPBConfig.COLOR_RESET}")
            
            
# -----------------------------------------------------------------------------
# Crear el titulo del modelo y su hash
# -----------------------------------------------------------------------------
def createModelIdentity(model_name, model_version, model_params, transformation, exog_scaling, exog_set_id):
    import hashlib
    transform_suffix = f" ({transformation})" if transformation and transformation != "None" else ""
    exog_suffix = f" + {exog_set_id}" if exog_set_id and exog_set_id != "None" else ""
    scaling_suffix = f" ({exog_scaling})" if exog_scaling != "None" else ""
    model_title = f'{model_name}{model_version} {model_params}{transform_suffix}{scaling_suffix}{exog_suffix}'
    model_hash = hashlib.sha1(model_title.encode()).hexdigest()[:8]
    return model_title, model_hash
 
    
# -----------------------------------------------------------------------------
# Crear secuencias para LSTM y Transformers
# -----------------------------------------------------------------------------
def createSequences(X, y, sequence_length, include_current_day=True):    
    X_seq, y_seq = [], []
    start_idx = sequence_length - 1 if include_current_day else sequence_length
    
    for i in range(start_idx, len(X)):
        if include_current_day:
            X_seq.append(X[i-sequence_length+1:i+1])  # incluye día i
        else:
            X_seq.append(X[i-sequence_length:i])      # NO incluye día i
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


# -----------------------------------------------------------------------------
# Calcular split de validación consistente
# -----------------------------------------------------------------------------
def calculateValidationSplit(total_length, sequence_length, target_val_size):
    min_val_size = sequence_length + 1 # Solo hacemos predicion en un paso, si no, incrementar el valor
    # Calculamos el split necesario para alcanzar el target
    required_split = target_val_size / total_length
    val_size = max(int(total_length * required_split), min_val_size)
    return val_size

    
# -----------------------------------------------------------------------------
# Crear un dataset de tf.data (más eficient)
# -----------------------------------------------------------------------------
def createTFDataset(X, y, sequence_length, batch_size, shuffle=True):
    import tensorflow as tf
    
    # Crear secuencias (operación costosa que vamos a cachear)
    X_seq, y_seq = createSequences(X, y, sequence_length)
    
    # Crear el dataset desde los arrays de numpy
    dataset = tf.data.Dataset.from_tensor_slices((X_seq, y_seq))
            
    # Cachear el dataset en memoria
    dataset = dataset.cache()
    
    # Barajar (si es para entrenamiento)
    if shuffle:
        # Usamos el tamaño completo del dataset para un barajado perfecto
        dataset = dataset.shuffle(buffer_size=len(X_seq), seed=AMPBConfig.SEED)
    
    # Crear lotes
    dataset = dataset.batch(batch_size, drop_remainder=shuffle)
    
    # Prefetch: Permite que la CPU prepare el siguiente lote mientras la GPU entrena
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset   


# -----------------------------------------------------------------------------
# Cross validation para series temporales que respeta el orden temporal.
# Usa ventana deslizante.
# -----------------------------------------------------------------------------
def createTimeSeriesCV(X, y, n_splits, test_size, sequence_length):
    total_samples = len(X)
    
    # Gap temporal para evitar data leakage
    temporal_gap = max(1, sequence_length // 2)
    
    # Tamaño fijo de entrenamiento (60% del total)
    train_window_size = int(total_samples * 0.6)
    train_window_size = max(train_window_size, sequence_length + 100)
    
    # Verificar que tenemos suficientes datos
    min_required = train_window_size + temporal_gap + test_size
    if total_samples < min_required:
        print(f"Datos insuficientes para CV. Necesarios: {min_required}, Disponibles: {total_samples}")
        return []
    
    # Calcular posiciones de inicio válidas
    max_start_pos = total_samples - train_window_size - temporal_gap - test_size
    if max_start_pos <= 0:
        print(f"Datos insuficientes para CV deslizante")
        return []
    
    # Distribuir los folds uniformemente a lo largo del tiempo
    cv_folds = []
    if n_splits == 1:
        start_positions = [max_start_pos]
    else:
        step = max_start_pos // (n_splits - 1)
        start_positions = [i * step for i in range(n_splits)]
    
    for i, start_pos in enumerate(start_positions):
        train_start = start_pos
        train_end = start_pos + train_window_size
        val_start = train_end + temporal_gap
        val_end = val_start + test_size
        
        # Verificar que no nos pasemos del total
        if val_end > total_samples:
            continue
            
        train_indices = list(range(train_start, train_end))
        val_indices = list(range(val_start, val_end))
        
        # Verificar tamaños mínimos
        if len(train_indices) >= sequence_length + 50 and len(val_indices) >= sequence_length + 5:
            cv_folds.append((train_indices, val_indices))
    
    # Información de debug con fechas
    if len(cv_folds) > 0:
        print(f"  Ventana deslizante: {len(cv_folds)} folds creados")
        print(f"  Tamaño entrenamiento fijo: {train_window_size} muestras por fold")
        print(f"  Tamaño validación fijo: {test_size} muestras por fold")
        print(f"  Gap temporal: {temporal_gap} muestras")
        
        # Mostrar fechas para cada fold
        print(f"  Detalle de folds:")
        for i, (train_idx, val_idx) in enumerate(cv_folds):
            train_start_date = y.index[train_idx[0]].strftime('%Y-%m-%d')
            train_end_date = y.index[train_idx[-1]].strftime('%Y-%m-%d')
            val_start_date = y.index[val_idx[0]].strftime('%Y-%m-%d')
            val_end_date = y.index[val_idx[-1]].strftime('%Y-%m-%d')
            
            print(f"    Fold {i+1}: Train[{train_start_date} → {train_end_date}] → Val[{val_start_date} → {val_end_date}]")
    else:
        print(f"  No se pudieron crear folds válidos")
    print("")
    return cv_folds
    
    
# -----------------------------------------------------------------------------
# Imprime metricas para un predictor de Regression
#  - Dashboard y lo guarda en un PNG externo ({create_image}_metrics_regression.png)
#  - Metricas en texto
# -----------------------------------------------------------------------------
def printMetricsRegression(y_true, y_pred, model_name, create_image=None):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    # Asegurar que son tablas Numpy
    y_true = np.array(y_true)
    y_predicted = np.array(y_pred)

    # Detectar valores problemáticos en y_pred
    if np.any(np.isnan(y_predicted)) or np.any(np.isinf(y_predicted)) or np.any(y_predicted == None):
        print(f"{AMPBConfig.COLOR_INFO}Error: y_pred contiene valores NaN, infinitos o nulos. No se pueden calcular las métricas.{AMPBConfig.COLOR_RESET}")
        return None, None, None
    
    # Calculamos las metricas
    mae_value = mean_absolute_error(y_true, y_predicted)
    mse_value = mean_squared_error(y_true, y_predicted)
    rmse_value = np.sqrt(mse_value) # Calculate RMSE using numpy's sqrt
    r2_value = r2_score(y_true, y_predicted)
 
    # Dashboard    
    r2_value_perc = r2_value * 100 # Convertir R² a porcentaje para el gauge    
    gauge_text_color = "#2A3F5F"
    if r2_value_perc < 0:
        gauge_text_color = "darkred"
        gauge_bar_color = "darkred"
        gauge_level = "Muy bajo"   
        if r2_value_perc < -99:
            r2_value_perc = -99 # no puede ser mas bajo que esto
    elif r2_value_perc < 55:
        gauge_bar_color = "red"
        gauge_level = "Bajo"
    elif r2_value_perc < 75: 
        gauge_bar_color = "orange"
        gauge_level = "Medio"
    else: 
        gauge_bar_color = "green"
        gauge_level = "Alto"
    # Crear la figura con subplots (1 fila, 3 columnas)
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        column_widths=[0.33, 0.33, 0.33] # Ajusta el ancho relativo si es necesario
    )
    # Tarjeta 1: Gauge para R²
    fig.add_trace(go.Indicator(
        mode = "gauge+number", 
        value = r2_value_perc,
        number = {'suffix': "%", 'valueformat': '.2f', 'font': {'size': 45, 'color': f"{gauge_text_color}"}}, 
        title = {'text': f"<b>R² ({gauge_level})</b><br><span style='font-size:0.8em;color:gray'>Coef. Determinación</span>", 'font': {"size": 16}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': gauge_bar_color, 'thickness': 0.3}, # Color dinámico de la barra
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 55], 'color': 'rgba(255, 0, 0, 0.1)'}, # Rojo muy claro
                {'range': [55, 75], 'color': 'rgba(255, 165, 0, 0.1)'}, # Naranja muy claro
                {'range': [75, 100], 'color': 'rgba(0, 128, 0, 0.1)'} # Verde muy claro
            ],
        }),
        row=1, col=1
    )
    # Tarjeta 2: Número para MAE
    fig.add_trace(go.Indicator(
        mode = "number",  
        value = mae_value,
        number = {'valueformat': '.2f', 'font': {'size': 55}},  
        title = {"text": "<b>MAE</b><br><span style='font-size:0.8em;color:gray'>Error Absoluto Medio</span>", 'font': {"size": 16}},
        ),
        row=1, col=2
    )
    # Tarjeta 3: Número para RMSE
    fig.add_trace(go.Indicator(
        mode = "number",  
        value = rmse_value,
        number = {'valueformat': '.2f','font': {'size': 55}},  
        title = {"text": "<b>RMSE</b><br><span style='font-size:0.8em;color:gray'>Raíz Error Cuadrático Medio</span>", 'font': {"size": 16}},
        # domain = {'x': [0.7, 0.9], 'y': [0.2, 0.8]}
        ),
        row=1, col=3
    )
    # Ajustes de Layout General
    fig.update_layout(
        # paper_bgcolor = "lightsteelblue", # Color de fondo general 
        height=350, # Altura total del gráfico
        margin=dict(l=30, r=30, t=100, b=20), # Márgenes
        title={
            'text': f"<b>Métricas de regresión - {model_name}</b>",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20}
        },
        template='plotly_white' 
    )
    # Mostrar el gráfico
    if AMPBConfig.INTERACTIVE:
        fig.show()
    # Guardar a PNG si usamos create_image (se recomienda un hash unico)
    if create_image:
        fig.write_image(f"{create_image}_metrics_regression.png", 
            width=1200,   
            height=350,   
            scale=1)      
    
    # Metricas en texto y los devuelve
    print(f"{AMPBConfig.COLOR_INFO}Métricas de regresión - {model_name}:{AMPBConfig.COLOR_RESET}")
    print(f' MAE (Mean Absolute Error): {mae_value:.4f}')
    print(f' MSE (Mean Squared Error): {mse_value:.4f}')
    print(f' RMSE (Root Mean Squared Error): {rmse_value:.4f}')
    print(f' R² (R-squared): {r2_value:.4f}\n')
    return r2_value, mae_value, rmse_value


# -----------------------------------------------------------------------------
# Imprime historia de entrenamiento de modelo Regression
#  - Historial
#  - Peso de cada feature    
# -----------------------------------------------------------------------------
def printTrainingRegression(history, modelo, X_test, features, scaler, y_test, test_pred):
    plt.figure(figsize=(14, 4))

    # 1.Gráfica de la métrica de error
    if 'mae' in history.history: 
        plt.subplot(1, 2, 1)
        plt.plot(history.history['mae'], label='Training MAE', color='green')
        plt.plot(history.history['val_mae'], label='Validation MAE', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.title('Model MAE')
        plt.legend()
        plt.grid()

    # 2.Gráfica de la pérdida (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='green')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid()    
    plt.tight_layout()
    plt.show()
    
    # 3.Calcular Permutation Importance
    baseline_mse = mean_squared_error(y_test, test_pred)
    importances = {}
    n_repeats = 10  # Número de repeticiones para estabilizar la estimación

    # Se hace una copia de X_test para no modificar los datos originales
    X_test_copy = X_test.copy()

    # Iteramos sobre cada feature
    for i, feature in enumerate(features):
        mse_diffs = []
        for _ in range(n_repeats):
            # Copiamos los datos del test
            X_permuted = X_test_copy.copy()
            # Permutamos aleatoriamente la columna i (la feature actual)
            np.random.shuffle(X_permuted[:, i])
            
            # Predecimos con la data permutada
            y_pred_perm = modelo.predict(X_permuted, verbose=0)
            y_pred_perm_inv = scaler.inverse_transform(y_pred_perm)
            
            # Calculamos el MSE de la predicción permutada
            mse_perm = mean_squared_error(y_test, y_pred_perm_inv)
            # La importancia es el incremento en el error
            mse_diffs.append(mse_perm - baseline_mse)
        
        # Promediamos el incremento en MSE para la feature i
        importances[feature] = np.mean(mse_diffs)
    
    # Mostrar las importancias ordenadas (mayor impacto primero)
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    # Suponiendo que 'sorted_importances' es una lista de tuplas (feature, importance)
    features_sorted = [feat for feat, imp in sorted_importances]
    importances_sorted = [imp for feat, imp in sorted_importances]

    plt.figure(figsize=(10, 6))
    # Usamos una gráfica de barras horizontales para mayor legibilidad
    plt.barh(features_sorted, importances_sorted, color='skyblue')
    plt.xlabel("Incremento en MSE al permutar")
    plt.title("Permutation Importances")
    plt.gca().invert_yaxis()  # Se invierte el eje Y para que la feature más importante aparezca arriba
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("\nPermutation Importances (Incremento en MSE al permutar la feature):")
    for feat, imp in sorted_importances:
        print(f"{feat}: {imp:.4f}")


# -----------------------------------------------------------------------------
# Mostrar estadísticas de Cross Validation.
# -----------------------------------------------------------------------------
def printMetricsCV(cv_results, cv_elapsed, total_folds_attempted):
    # Calcular estadísticas de CV si tenemos resultados
    if len(cv_results) > 0:
        cv_df = pd.DataFrame(cv_results)
        
        # Estadísticas resumen 
        cv_mean_r2 = cv_df['r2'].mean()
        cv_std_r2 = cv_df['r2'].std()
        cv_mean_mae = cv_df['mae'].mean()
        cv_std_mae = cv_df['mae'].std()
        cv_mean_rmse = cv_df['rmse'].mean()
        cv_std_rmse = cv_df['rmse'].std()
        cv_mean_accuracy = cv_df['accuracy'].mean()
        cv_std_accuracy = cv_df['accuracy'].std()
        cv_mean_f1_score = cv_df['f1_score'].mean()
        cv_std_f1_score = cv_df['f1_score'].std()
        cv_mean_roc_auc = cv_df['roc_auc'].mean()
        cv_std_roc_auc = cv_df['roc_auc'].std()
        
        print(f"\n{AMPBConfig.COLOR_INFO}Resultados Cross Validation:{AMPBConfig.COLOR_RESET}")
        print(f"  Folds completados: {AMPBConfig.COLOR_VALUE}{len(cv_results)}/{total_folds_attempted}{AMPBConfig.COLOR_RESET}")
        print(f"  Tiempo total: {AMPBConfig.COLOR_VALUE}{cv_elapsed:.1f}s{AMPBConfig.COLOR_RESET}")
        
        # Métricas de regresión
        print(f"  Métricas de regresión:")
        print(f"    R² promedio: {AMPBConfig.COLOR_VALUE}{cv_mean_r2:.4f} ± {cv_std_r2:.4f}{AMPBConfig.COLOR_RESET}")
        print(f"    MAE promedio: {AMPBConfig.COLOR_VALUE}{cv_mean_mae:.2f} ± {cv_std_mae:.2f}{AMPBConfig.COLOR_RESET}")
        print(f"    RMSE promedio: {AMPBConfig.COLOR_VALUE}{cv_mean_rmse:.2f} ± {cv_std_rmse:.2f}{AMPBConfig.COLOR_RESET}")
        
        # Métricas direccionales
        print(f"  Métricas direccionales:")
        print(f"    Accuracy promedio: {AMPBConfig.COLOR_VALUE}{cv_mean_accuracy:.1%} ± {cv_std_accuracy:.1%}{AMPBConfig.COLOR_RESET}")
        print(f"    F1-Score promedio: {AMPBConfig.COLOR_VALUE}{cv_mean_f1_score:.3f} ± {cv_std_f1_score:.3f}{AMPBConfig.COLOR_RESET}")
        print(f"    ROC-AUC promedio: {AMPBConfig.COLOR_VALUE}{cv_mean_roc_auc:.3f} ± {cv_std_roc_auc:.3f}{AMPBConfig.COLOR_RESET}")
        
        # Análisis de estabilidad mejorado
        print(f"\n  Análisis de estabilidad:")
        
        # Detectar si el modelo es fundamentalmente malo
        is_model_bad = cv_mean_r2 < 0.0
        
        if is_model_bad:
            print(f"    {AMPBConfig.COLOR_MSG}   Advertencia: R² promedio negativo ({cv_mean_r2:.4f}){AMPBConfig.COLOR_RESET}")
            print(f"    {AMPBConfig.COLOR_MSG}   El modelo es peor que predecir la media constante{AMPBConfig.COLOR_RESET}")
            
            # Para modelos malos, usamos MAE como métrica principal de estabilidad
            mae_cv = cv_std_mae / cv_mean_mae if cv_mean_mae != 0 else float('inf')
            print(f"    Coeficiente de variación MAE: {AMPBConfig.COLOR_VALUE}{mae_cv:.3f}{AMPBConfig.COLOR_RESET}")
            
            # Evaluación basada en MAE cuando R² es negativo
            if mae_cv < 0.3:
                stability_assessment = "Consistentemente malo"
                stability_color = AMPBConfig.COLOR_MSG
            elif mae_cv < 0.6:
                stability_assessment = "Malo e inestable"  
                stability_color = AMPBConfig.COLOR_MSG
            else:
                stability_assessment = "Muy malo e impredecible"
                stability_color = AMPBConfig.COLOR_MSG
                
        else:
            # Análisis normal para modelos con R² positivo
            r2_cv = cv_std_r2 / cv_mean_r2 if cv_mean_r2 != 0 else float('inf')
            mae_cv = cv_std_mae / cv_mean_mae if cv_mean_mae != 0 else float('inf')
            acc_cv = cv_std_accuracy / cv_mean_accuracy if cv_mean_accuracy != 0 else float('inf')
            
            print(f"    Coeficiente de variación R²: {AMPBConfig.COLOR_VALUE}{r2_cv:.3f}{AMPBConfig.COLOR_RESET}")
            print(f"    Coeficiente de variación MAE: {AMPBConfig.COLOR_VALUE}{mae_cv:.3f}{AMPBConfig.COLOR_RESET}")
            print(f"    Coeficiente de variación Accuracy: {AMPBConfig.COLOR_VALUE}{acc_cv:.3f}{AMPBConfig.COLOR_RESET}")
            
            # Evaluación de estabilidad combinando múltiples métricas
            combined_cv = (r2_cv + mae_cv + acc_cv) / 3
            
            if combined_cv < 0.15:
                stability_assessment = "Muy estable"
                stability_color = AMPBConfig.COLOR_VALUE
            elif combined_cv < 0.25:
                stability_assessment = "Estable"
                stability_color = AMPBConfig.COLOR_VALUE
            elif combined_cv < 0.35:
                stability_assessment = "Moderadamente estable"
                stability_color = AMPBConfig.COLOR_VALUE
            else:
                stability_assessment = "Inestable"
                stability_color = AMPBConfig.COLOR_MSG
                
        print(f"    Evaluación de estabilidad: {stability_color}{stability_assessment}{AMPBConfig.COLOR_RESET}")
        
        # Retornar métricas promedio para uso posterior
        return {
            'cv_mean_r2': cv_mean_r2,
            'cv_mean_mae': cv_mean_mae,
            'cv_mean_rmse': cv_mean_rmse,
            'cv_mean_accuracy': cv_mean_accuracy,
            'cv_mean_f1_score': cv_mean_f1_score,
            'cv_mean_roc_auc': cv_mean_roc_auc,
            'cv_std_r2': cv_std_r2,
            'cv_std_mae': cv_std_mae,
            'cv_std_rmse': cv_std_rmse,
            'cv_std_accuracy': cv_std_accuracy,
            'cv_std_f1_score': cv_std_f1_score,
            'cv_std_roc_auc': cv_std_roc_auc,
            'stability_assessment': stability_assessment,
            'is_model_bad': is_model_bad
        }
        
    else:
        print(f"\n{AMPBConfig.COLOR_MSG}Advertencia: No se pudieron completar folds válidos para CV{AMPBConfig.COLOR_RESET}")
        # Retornar valores por defecto
        return {
            'cv_mean_r2': 0.0,
            'cv_mean_mae': float('inf'),
            'cv_mean_rmse': float('inf'),
            'cv_mean_accuracy': 0.0,
            'cv_mean_f1_score': 0.0,
            'cv_mean_roc_auc': 0.5,
            'cv_std_r2': 0.0,
            'cv_std_mae': 0.0,
            'cv_std_rmse': 0.0,
            'cv_std_accuracy': 0.0,
            'cv_std_f1_score': 0.0,
            'cv_std_roc_auc': 0.0,
            'stability_assessment': 'No evaluada',
            'is_model_bad': True
        }
        
    
# -----------------------------------------------------------------------------
# Analiza los resultados del entrenamiento y detecta problemas   
# -----------------------------------------------------------------------------
def analyzeTrainingResults(history, model, X_val, y_val):
    import matplotlib.pyplot as plt
    import numpy as np

    hist = history.history

    # Mostrar gráficas de entrenamiento
    plt.figure(figsize=(14, 4))
    
    # 1. Gráfica de MAE
    if 'mae' in hist: 
        plt.subplot(1, 2, 1)
        plt.plot(hist['mae'], label='Training MAE', color='green')
        if 'val_mae' in hist:
            plt.plot(hist['val_mae'], label='Validation MAE', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.title('Model MAE')
        plt.legend()
        plt.grid()
    
    # 2. Gráfica de Loss
    plt.subplot(1, 2, 2)
    plt.plot(hist['loss'], label='Training Loss', color='green')
    if 'val_loss' in hist:
        plt.plot(hist['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')  # usas Huber; más genérico que "MSE Loss"
    plt.title('Model Loss')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    if AMPBConfig.INTERACTIVE:
        plt.show()

    # Detección simple de overfitting y pocas épocas
    final_train_loss = hist['loss'][-1]
    final_val_loss = hist['val_loss'][-1] if 'val_loss' in hist else np.nan
    
    print(f"{AMPBConfig.COLOR_INFO}Análisis del entrenamiento:{AMPBConfig.COLOR_RESET}")
    
    # Detectar overfitting
    if 'val_loss' in hist and final_val_loss > final_train_loss * 1.1:  # 10% más alto
        gap = ((final_val_loss - final_train_loss) / final_train_loss) * 100
        print(f"  Posible overfitting detectado: val_loss({final_val_loss:.6f}) {gap:.1f}% mayor que train_loss ({final_train_loss:.6f})")
    else:
        print(f"  Sin overfitting aparente")
    
    # Detectar si hacen falta más épocas
    if 'val_loss' in hist and len(hist['val_loss']) >= 5:
        last_5_val = hist['val_loss'][-5:]
        if all(last_5_val[i] >= last_5_val[i+1] for i in range(4)):  # Últimas 5 siguen bajando
            print(f"  Posiblemente se necesiten más épocas (val_loss sigue bajando)")
        elif hist['val_loss'][-1] < min(hist['val_loss'][:-5]):
            print(f"  Posiblemente se necesiten más épocas (nuevo mínimo reciente)")
        else:
            print(f"  Entrenamiento parece completo")
    
    # Resumen de validación (mejores épocas y métricas)
    if 'val_loss' in hist:
        val_loss_list = hist['val_loss']
        best_val_loss = float(np.min(val_loss_list))
        best_epoch_loss = int(np.argmin(val_loss_list))  # 0-based
        if 'val_mae' in hist:
            val_mae_list = hist['val_mae']
            best_val_mae_overall = float(np.min(val_mae_list))
            best_epoch_mae = int(np.argmin(val_mae_list))  # 0-based
            best_val_mae_at_loss = float(val_mae_list[best_epoch_loss])

            print(f"  Mejor val_loss: {best_val_loss:.6f} (época {best_epoch_loss+1}) - val_mae en esa época: {best_val_mae_at_loss:.6f}")
            print(f"  Mejor val_mae global: {best_val_mae_overall:.6f} (época {best_epoch_mae+1})")
        else:
            print(f"  Mejor val_loss: {best_val_loss:.6f} (época {best_epoch_loss+1})")
    else:
        print("Aviso: no se encontraron métricas de validación en 'history'.")


# -----------------------------------------------------------------------------
# Imprime metricas para un predictor de Classificacion binario
#  - Dashboard y lo guarda en un PNG ({create_image}_metrics_classification.png)
#  - Metricas en texto Distribución de Scores por Resultado
# -----------------------------------------------------------------------------  
def printMetricsClassification(y_true, y_pred_input, model_name, threshold_for_metrics=0.5, create_image=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from scipy.stats import gaussian_kde
    # 1.Ajustes iniciales
    class_labels = ['Baja', 'Sube/Igual'] 
    y_true = np.array(y_true) # Convertir a numpy arrays por si acaso
    y_pred_input = np.array(y_pred_input)        
    is_scores = len(np.unique(y_pred_input)) > 2 # Asumir scores si hay más de 2 valores únicos
    if is_scores:
        y_pred_scores = y_pred_input # Ya son scores/probabilidades
        y_pred_binarized = (y_pred_scores >= threshold_for_metrics).astype(int)
        print(f"INFO: Interpretando 'y_pred_input' como scores/probabilidades. Usando umbral {threshold_for_metrics} para métricas binarias.")
    else:
        y_pred_scores = y_pred_input # No hay scores disponibles
        y_pred_binarized = y_pred_input.astype(int) # Ya son binarias
        print("INFO: Interpretando 'y_pred_input' como clases binarias. Umbral fijado a 0.5.")
        # Forzar umbral a 0.5 si los datos son binarios para evitar errores lógicos más adelante
        threshold_for_metrics = 0.5
    plot_score_dist = 1# is_scores and y_pred_scores is not None 
        
    # 2.Calculamos las metricas generales
    accuracy = accuracy_score(y_true, y_pred_binarized)
    precision = precision_score(y_true, y_pred_binarized, zero_division=0, pos_label=1)
    recall = recall_score(y_true, y_pred_binarized, zero_division=0, pos_label=1)
    f1 = f1_score(y_true, y_pred_binarized, zero_division=0, pos_label=1)
    cm = confusion_matrix(y_true, y_pred_binarized)
    roc_auc = 0 
    plot_roc = False 
    try:
        # Asegurarse que hay más de una clase en y_true para calcular AUC y ROC
        if len(np.unique(y_true)) > 1:
             roc_auc = roc_auc_score(y_true, y_pred_scores)
             fpr, tpr, thresholds = roc_curve(y_true, y_pred_scores)
             plot_roc = True
        else:
             print("Advertencia: ROC-AUC y Curva ROC no se pueden calcular porque solo hay una clase presente en y_true.")
             roc_auc = 0 
             fpr, tpr = [0], [0] 
    except ValueError as e:
        print(f"Error calculando ROC-AUC o Curva ROC: {e}")
        roc_auc = 0
        fpr, tpr = [0], [0]    
    # Matriz de confusion
    cm = confusion_matrix(y_true, y_pred_binarized)

    # 3A. Dashboard
    specs=[
        [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
        [{'type': 'xy'},        {'type': 'heatmap'},   {'type': 'xy'}]
    ]
    fig = make_subplots(
        rows=2, cols=3,
        specs=specs,
        subplot_titles=(
            "<span style='font-size:16'><b>Accuracy</b></span><br><span style='font-size:13;color:gray'>Exactitud</span>", 
            "<span style='font-size:16'><b>F1-Score</b></span><br><span style='font-size:13;color:gray'>Balance de precisión y sensibilidad</span>", 
            "<span style='font-size:16'><b>ROC-AUC</b></span><br><span style='font-size:13;color:gray'>Valor AUC</span>", 
            "<span style='font-size:16'><b>Distribución de scores</b></span>", 
            "<span style='font-size:16'><b>Matriz de confusión</b></span>", 
            "<span style='font-size:16'><b>Curva ROC</b></span>",             
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
        column_widths=[0.33, 0.34, 0.33],
        row_heights = [0.45, 0.55]
    )

    # 3B. Fila 1 (3 gauges)
    # ***Gauge 1: Accuracy (Posición 1, 1)***
    acc_perc = accuracy * 100
    gauge_color_acc = "red" if acc_perc < 55 else ("orange" if acc_perc < 75 else "green")
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = acc_perc,
        number = {'suffix': "%", 'valueformat': '.2f', 'font': {'size': 45}},        
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': gauge_color_acc, 'thickness': 0.3},
            'bgcolor': "white",
            'steps': [
                    {'range': [0, 55], 'color': 'rgba(255, 0, 0, 0.1)'}, # Rojo muy claro
                    {'range': [55, 75], 'color': 'rgba(255, 165, 0, 0.1)'}, # Naranja muy claro
                    {'range': [75, 100], 'color': 'rgba(0, 128, 0, 0.1)'} # Verde muy claro
                ],
        }),
        row=1, col=1
    )
    # ***Gauge 2: F1-Score (Posición 1, 2)***
    f1_gauge_val = f1 * 100
    gauge_f1_text_color = "#2A3F5F"
    if f1 < 0.10:
        gauge_color_f1 = "red"
        gauge_f1_text_color = "darkred"
    elif f1 < 0.55:
        gauge_color_f1 = "red"
    elif f1 < 0.75:
        gauge_color_f1 = "orange" 
    else:
        gauge_color_f1 = "green"
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = f1,
        number = {'valueformat': '.2f', 'font': {'size': 45, 'color': f"{gauge_f1_text_color}"}},         
        gauge = {
            'axis': {'range': [0, 1], 'tickvals': [0, 0.25, 0.5, 0.75, 1], 'tickwidth': 1}, # Eje 0-1
            'bar': {'color': gauge_color_f1, 'thickness': 0.3},
            'bgcolor': "white", 
            'steps': [
                    {'range': [0, 0.55], 'color': 'rgba(255, 0, 0, 0.1)'}, # Rojo muy claro
                    {'range': [0.55, 0.75], 'color': 'rgba(255, 165, 0, 0.1)'}, # Naranja muy claro
                    {'range': [0.75, 1], 'color': 'rgba(0, 128, 0, 0.1)'} # Verde muy claro
                ],
        }),
        row=1, col=2
    )
    # ***Gauge 3: ROC AUC (Posición 1, 3)***
    gauge_color_auc = "red" if roc_auc < 0.6 else ("orange" if roc_auc < 0.75 else "green")
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = roc_auc,
        number = {'valueformat': '.2f', 'font': {'size': 45}},        
        gauge = {
            'axis': {'range': [0, 1], 'tickvals': [0, 0.25, 0.5, 0.75, 1], 'tickwidth': 1}, # Eje 0-1
            'bar': {'color': gauge_color_auc, 'thickness': 0.3},
            'bgcolor': "white", 
            'steps': [
                    {'range': [0, 0.55], 'color': 'rgba(255, 0, 0, 0.1)'}, # Rojo muy claro
                    {'range': [0.55, 0.75], 'color': 'rgba(255, 165, 0, 0.1)'}, # Naranja muy claro
                    {'range': [0.75, 1], 'color': 'rgba(0, 128, 0, 0.1)'} # Verde muy claro
                ],
            'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': 0.5} # Marca el 0.5 (aleatorio)
        }),
        row=1, col=3
    )

    # 3C. Fila 2 (1 grafica, 1 heatmap y 1 grafica)
    # ***Distribucion de scores (Posición 2, 1)***
    if plot_score_dist:
        # Separar scores por clase real
        scores_clase_0 = y_pred_scores[y_true == 0]
        scores_clase_1 = y_pred_scores[y_true == 1]
        
        # Rango y título del eje X
        score_min = y_pred_scores.min(); score_max = y_pred_scores.max()
        x_range = np.linspace(0.0, 1.0, 500); x_axis_title = "Score de predicción (Probabilidad)"
        x_lim = [0.0, 1.0] # Límites específicos para probs
        
        # Variables para almacenar valores KDE y máximo para ylim
        kde_values = []
        max_density = 0

        # Calcular y añadir KDE para Clase 0
        y_values_0 = None
        if len(scores_clase_0) >= 2 and len(np.unique(scores_clase_0)) >= 2:
            if np.var(scores_clase_0) > 1e-8:
                try:
                    kde_0 = gaussian_kde(scores_clase_0)
                    y_values_0 = kde_0(x_range)
                    kde_values.append(y_values_0)
                    max_density = max(max_density, np.max(y_values_0))
                    fig.add_trace(go.Scatter(
                        x=x_range, y=y_values_0, mode='lines',
                        name=f'{class_labels[0]}', # Nombre simple para posible leyenda
                        line=dict(color='#FF5733', width=2),
                        fill='tozeroy', fillcolor=f'rgba({int('#FF5733'[1:3], 16)}, {int('#FF5733'[3:5], 16)}, {int('#FF5733'[5:7], 16)}, 0.35)',
                        hovertemplate = (f"<b>{class_labels[0]}</b><br>" + f"F.Normalizada: %{{y:.2f}}" +  "<extra></extra>") 
                    ), row=2, col=1)
                except (np.linalg.LinAlgError, ValueError): pass # Ignorar error KDE

        # Calcular y añadir KDE para Clase 1
        y_values_1 = None
        if len(scores_clase_1) >= 2 and len(np.unique(scores_clase_1)) >= 2:
            if np.var(scores_clase_1) > 1e-8:
                try:
                    kde_1 = gaussian_kde(scores_clase_1)
                    y_values_1 = kde_1(x_range)
                    kde_values.append(y_values_1)
                    max_density = max(max_density, np.max(y_values_1))
                    fig.add_trace(go.Scatter(
                        x=x_range, y=y_values_1, mode='lines',
                        name=f'{class_labels[1]}',
                        line=dict(color='#3498DB', width=2),
                        fill='tozeroy', fillcolor=f'rgba({int('#3498DB'[1:3], 16)}, {int('#3498DB'[3:5], 16)}, {int('#3498DB'[5:7], 16)}, 0.35)',
                        hovertemplate = (f"<b>{class_labels[1]}</b><br>" + f"F.Normalizada: %{{y:.2f}}" +  "<extra></extra>") 
                    ), row=2, col=1)
                except (np.linalg.LinAlgError, ValueError): pass

        # Añadir Rug Plot (marcadores '|' abajo)
        fig.add_trace(go.Scatter(
            x=scores_clase_0, y=[0] * len(scores_clase_0), # Posición Y negativa
            mode='markers', name=f'{class_labels[0]} Points',
            marker=dict(color='#FF5733', symbol='line-ns-open', size=10, line=dict(width=2)), # Marcador '|'
            hoverinfo='skip'
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=scores_clase_1, y=[0] * len(scores_clase_1),
            mode='markers', name=f'{class_labels[1]} Points',
            marker=dict(color='#3498DB', symbol='line-ns-open', size=10, line=dict(width=2)),
            hoverinfo='skip'
        ), row=2, col=1)

        # Añadir línea de umbral usando add_shape
        fig.add_shape(type="line",
                      xref="x4", yref="paper", x0=threshold_for_metrics, y0=0, x1=threshold_for_metrics, y1=max_density,
                      line=dict(color='#2ECC71', width=2, dash="dash"),
                      row=2, col=1)

        # Configurar ejes para (2,1)
        fig.update_xaxes(title_text=x_axis_title, title_font_size=13, range=x_lim, row=2, col=1)
        fig.update_yaxes(title_text="Frecuencia normalizada", title_font_size=13, showticklabels=False, row=2, col=1)

    else: 
        fig.add_trace(go.Scatter(
        x=[], 
        y=[], 
        mode='markers', 
        name='Placeholder',
        hoverinfo='none', 
        ), row=2, col=1)
        fig.update_xaxes(
        visible=False,    
        showgrid=False,
        zeroline=False,
        row=2, col=1
        )
        fig.update_yaxes(
            visible=False, 
            showgrid=False,
            zeroline=False,            
            scaleanchor="x6",
            scaleratio=1,
            row=2, col=1
        )

    # ***Matriz de Confusión (Centrada) (Posición 2, 2)***
    fig.add_trace(go.Heatmap(                   
                       z=cm,
                       x=class_labels,
                       y=class_labels,
                       colorscale='Reds', # GnBu
                       showscale=False,
                       text=cm,
                       texttemplate="%{text}",
                       textfont={"size":11},
                       hoverinfo='skip',
        ),
        row=2, col=2
    )
    fig.update_xaxes(title_text="<b>Tendencia predicha</b>", row=2, col=2, title_font_size=13)
    fig.update_yaxes(title_text="<b>Tendencia real</b>", autorange="reversed", row=2, col=2, title_font_size=13,  tickangle=270) 
    
    # ***Curva ROC-AUC (Posición 2, 3)***
    if plot_roc:
        # Añadir curva ROC y línea base
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.2f})', 
                                 line=dict(color='darkorange', width=2.5),
                                 hovertemplate="FP: %{x:.2f}<br>VP: %{y:.2f}<extra></extra>"
                                ), row=2, col=3)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatorio', 
                                 line=dict(color='navy', width=1, dash='dash'),
                                 hoverinfo="skip"
                                ), row=2, col=3)

        # Actualizar ejes para ROC
        fig.update_xaxes(
            title_text="Tasa Falsos Positivos (FP)",
            range=[0, 1],
            dtick=0.2,
            row=2, col=3
        )
        fig.update_yaxes(
            title_text="Tasa Verdaderos Positivos (VP)",
            range=[0, 1],
            dtick=0.2, 
            scaleanchor="x3",  
            scaleratio=1,      
            row=2, col=3
        )        
    else:
        fig.add_trace(go.Scatter(
        x=[], 
        y=[], 
        mode='markers',
        name='Placeholder',
        hoverinfo='none', 
        ), row=2, col=3) 
        fig.update_xaxes(
        visible=False,    
        showgrid=False,
        zeroline=False,
        row=2, col=3
        )
        fig.update_yaxes(
            visible=False, 
            showgrid=False,
            zeroline=False,            
            scaleanchor="x6",
            scaleratio=1,
            row=2, col=3
        )
    
    # 4. Añadir Anotaciones para Leyendas
    # Leyenda para subplot (2,1)
    if plot_score_dist:
        fig.add_annotation(
            row=2, col=1,
            showarrow=False,
            text=f"<span style='color:#FF5733; font-size:1.2em;'>──</span> Baja ({len(scores_clase_0)})<br><span style='color:#3498DB; font-size:1.2em;'>──</span> Sube/Igual ({len(scores_clase_1)})<br><span style='color:#2ECC71; font-size:1.2em;'>- -</span> Umbral ({threshold_for_metrics})",
            align='left',
            xref='x4 domain',
            yref='y4 domain',
            x=0.98,          
            y=0.2,           
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor='darkgrey',
            borderwidth=1,
            xanchor='right',       
            yanchor='middle' 
            )
    else:
        fig.add_annotation(
                row=2, col=1,
                showarrow=False, 
                text="<b>No disponible</b>",
                font=dict(size=16, color="grey"), 
                xref="x4 domain", 
                yref="y4 domain"
            )      

    # Leyenda para subplot (2,3)
    if plot_roc:
        fig.add_annotation(
            row=2, col=3,
            showarrow=False,
            text="<span style='color:navy; font-size:1.2em;'>- -</span> Aleatorio<br><span style='color:darkorange; font-size:1.2em;'>──</span> ROC",
            align='left',
            xref='x6 domain',
            yref='y6 domain',
            x=0.98,          
            y=0.1,           
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor='darkgrey',
            borderwidth=1,
            xanchor='right',       
            yanchor='middle' 
        )
    else:
        fig.add_annotation(
                row=2, col=3,
                showarrow=False, 
                text="<b>No disponible</b>",
                font=dict(size=16, color="grey"), 
                xref="x6 domain", 
                yref="y6 domain"
            )      

    # 5. Actualizar Layout General
    fig.update_layout(
        title_text=f"<b>Métricas de clasificación - {model_name}</b>",
        title_font_size=20,
        title_x=0.5, title_y=0.98, height=800, #width=1000,
        #margin=dict(l=30, r=30, t=80, b=20), 
        template='plotly_white',
        showlegend=False,
        hovermode='x unified'
    )

    # 6. Mostrar
    if AMPBConfig.INTERACTIVE:
        fig.show()
    # Guardar a PNG si usamos create_image
    if create_image:
        fig.write_image(f"{create_image}_metrics_classification.png", 
            width=1200,   
            height=606,   
            scale=1)   

    # 7.Metricas y los devuelve
    print(f"{AMPBConfig.COLOR_INFO}Métricas de clasificación - {model_name}:{AMPBConfig.COLOR_RESET}")
    print(f' Accuracy (Exactitud): {accuracy:.4f}')
    print(f' Precision (Precision): {precision:.4f}')
    print(f' Recall (Sensibilidad): {recall:.4f}')
    print(f' F1-Score: {f1:.4f}')
    print(f' ROC-AUC: {roc_auc:.4f}')
    return accuracy, f1, roc_auc


# -----------------------------------------------------------------------------
# Imprime historia de entrenamiento de modelo clasificacion
#  - Historial
#  - Peso de cada feature    
# -----------------------------------------------------------------------------
def printTrainingClassification(history, modelo, X, y, features):
    # Evolucion de entrenamiento, validacion y perdida
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1) 
    plt.figure(figsize=(14, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)  
    plt.plot(epochs, acc, color='orange', label='Training Accuracy')
    plt.plot(epochs, val_acc, color='green', label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)  
    plt.plot(epochs, loss, color='orange', label='Training Loss')
    plt.plot(epochs, val_loss, color='green', label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Calcula la importancia de las características eliminándolas una por una y midiendo la caída en precisión.
    baseline_accuracy = accuracy_score(y, (modelo.predict(X) > 0.5).astype(int).ravel())
    importancias = {}
    
    for i in range(X.shape[2]):  # Iterar sobre las características
        X_perturbado = np.copy(X)  # Copia profunda para evitar modificar X original
        X_perturbado[:, :, i] = 0  # Perturbar la característica i (poner a 0)
        y_pred = (modelo.predict(X_perturbado, verbose=0) > 0.5).astype(int).ravel() # Predecir con la característica eliminada        
        accuracy_perturbado = accuracy_score(y, y_pred)
        importancias[features[i]] = baseline_accuracy - accuracy_perturbado  # Disminución de precisión
        #print(f"Feature: {features[i]} | Accuracy Perturbado: {accuracy_perturbado} | Importancia: {importancias[features[i]]}")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(importancias.keys(), importancias.values(), color='royalblue', alpha=0.75)    
    for bar in bars:
        if bar.get_height() < 0:
            bar.set_color('red') # Cambiar el color de las barras negativas a rojo
    plt.axhline(0, color='black', linewidth=0.8)  # Línea base en 0
    plt.title('Feature Importance (Sensitivity Analysis)', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Model Performance', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# -----------------------------------------------------------------------------
# Funciones auxiliares para obtener Google Trends
# -----------------------------------------------------------------------------
def googletrends_normalize(series):
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)

def googletrends_process_file(filepath):
    print(f"    Procesando archivo individual: {filepath}")
    try:
        df_temp = pd.read_csv(filepath, header=None, names=['date_str', 'value_raw'])
        if df_temp.empty:
            return pd.DataFrame(columns=['value']).set_index(pd.DatetimeIndex([]))

        df_temp['date'] = pd.to_datetime(df_temp['date_str'])
        df_temp['value'] = df_temp['value_raw'].replace({'<1': 0.1}) 
        df_temp['value'] = pd.to_numeric(df_temp['value'], errors='coerce')
        df_temp['value'] = df_temp['value'].clip(lower=0, upper=100)
        df_temp.dropna(subset=['value'], inplace=True)
        
        if df_temp.empty:
            return pd.DataFrame(columns=['value']).set_index(pd.DatetimeIndex([]))

        df_temp = df_temp[['date', 'value']].set_index('date')
        
        if df_temp.index.has_duplicates:
            df_temp = df_temp.groupby(df_temp.index).mean()
        
        return df_temp.sort_index()
        
    except Exception as e:
        print(f"      Error procesando {filepath}: {e}")
        return pd.DataFrame(columns=['value']).set_index(pd.DatetimeIndex([]))

def googletrends_add(
    source_df, 
    base_dir_google_trend, 
    ticker,
    overall_monthly_file_name,
    clip_rescaled_upper_limit=150): 

    if not isinstance(source_df.index, pd.DatetimeIndex):
        print("Error: El índice de source_df debe ser un DateTimeIndex.")
        return source_df # Devolver el original si hay error

    ticker_dir_path = os.path.join(base_dir_google_trend, ticker)
    new_column_name = f'googletrends_{ticker}'

    df_final_with_trends = source_df.copy()

    if not os.path.isdir(ticker_dir_path):
        print(f"Directorio no encontrado para el ticker {ticker}: {ticker_dir_path}")
        df_final_with_trends[new_column_name] = pd.NA 
        return df_final_with_trends

    print(f"\nProcesando Google Trends para: {ticker}")

    all_hg_data = []
    for filename in os.listdir(ticker_dir_path):
        if filename.endswith(".csv") and filename != overall_monthly_file_name:
            filepath = os.path.join(ticker_dir_path, filename)
            df_hg_single = googletrends_process_file(filepath)
            if not df_hg_single.empty:
                all_hg_data.append(df_hg_single)
    
    if not all_hg_data:
        print(f"  No se encontraron datos de alta granularidad válidos para {ticker}.")
        df_final_with_trends[new_column_name] = pd.NA
        return df_final_with_trends

    df_hg_raw_combined = pd.concat(all_hg_data)
    if df_hg_raw_combined.index.has_duplicates:
        df_hg_raw_combined = df_hg_raw_combined.groupby(df_hg_raw_combined.index).mean()
    df_hg_raw_combined = df_hg_raw_combined.sort_index()

    if df_hg_raw_combined.empty:
        print(f"  Datos de alta granularidad combinados vacíos para {ticker}.")
        df_final_with_trends[new_column_name] = pd.NA
        return df_final_with_trends

    min_hg_date = df_hg_raw_combined.index.min()
    max_hg_date = df_hg_raw_combined.index.max()
    daily_index_hg = pd.date_range(start=min_hg_date, end=max_hg_date + pd.Timedelta(days=6), freq='D')
    df_hg_daily_ffilled = df_hg_raw_combined.reindex(daily_index_hg).ffill()
    df_hg_daily_ffilled = df_hg_daily_ffilled.loc[min_hg_date:max_hg_date] 
    
    if df_hg_daily_ffilled.empty:
        print(f"  Datos de alta granularidad con ffill diario vacíos para {ticker}.")
        df_final_with_trends[new_column_name] = pd.NA
        return df_final_with_trends
        
    overall_filepath = os.path.join(ticker_dir_path, overall_monthly_file_name)
    df_to_join = pd.DataFrame() 
    use_simple_normalization = True

    if not os.path.exists(overall_filepath):
        print(f"  Archivo 'overall' mensual no encontrado. Usando normalización Min-Max simple.")
    else:
        df_overall_monthly = googletrends_process_file(overall_filepath)
        if df_overall_monthly.empty:
            print(f"  Archivo 'overall' mensual vacío. Usando normalización Min-Max simple.")
        else:
            use_simple_normalization = False
            print("  Reescalando datos de alta granularidad usando el archivo 'overall' mensual...")
            df_rescaled_values_list = []
            df_overall_monthly.index = df_overall_monthly.index.to_period('M').to_timestamp()

            for month_period in pd.unique(df_hg_daily_ffilled.index.to_period('M')):
                month_start_date = month_period.start_time
                current_month_hg_data_series = df_hg_daily_ffilled.loc[
                    (df_hg_daily_ffilled.index >= month_start_date) &
                    (df_hg_daily_ffilled.index <= month_period.end_time), 'value'
                ]
                if current_month_hg_data_series.empty: continue
                v_hg_avg_month = current_month_hg_data_series.mean()
                v_overall_month = pd.NA
                try: v_overall_month = df_overall_monthly.loc[month_start_date, 'value']
                except KeyError: factor_month = 1.0
                if pd.isna(v_overall_month): factor_month = 1.0
                elif pd.isna(v_hg_avg_month) or v_hg_avg_month < 1e-6:
                    if v_overall_month < 1e-6: factor_month = 1.0
                    else: 
                        if v_hg_avg_month !=0: factor_month = v_overall_month / v_hg_avg_month
                        else: factor_month = 1.0 
                elif v_overall_month == 0 and not (pd.isna(v_hg_avg_month) or v_hg_avg_month == 0):
                    factor_month = 0.0
                else: factor_month = v_overall_month / v_hg_avg_month
                
                rescaled_month_data = current_month_hg_data_series * factor_month
                df_rescaled_values_list.append(rescaled_month_data)

            if not df_rescaled_values_list:
                 print(f"  No se pudieron reescalar datos para {ticker}.")
                 use_simple_normalization = True
            else:
                df_high_granularity_rescaled = pd.concat(df_rescaled_values_list)
                if df_high_granularity_rescaled.empty:
                    print(f"  Datos reescalados resultaron vacíos.")
                    use_simple_normalization = True
                else:
                    df_high_granularity_rescaled_clipped = df_high_granularity_rescaled.clip(lower=0, upper=clip_rescaled_upper_limit)
                    print(f"  Valores reescalados (acotados a {clip_rescaled_upper_limit}) - Min: {df_high_granularity_rescaled_clipped.min():.2f}, Max: {df_high_granularity_rescaled_clipped.max():.2f}")
                    
                    df_normalized_trends = googletrends_normalize(df_high_granularity_rescaled_clipped)
                    df_to_join = pd.DataFrame({new_column_name: df_normalized_trends})

    if use_simple_normalization:
        print(f"  Realizando normalización Min-Max simple final para {ticker}.")
        if not df_hg_daily_ffilled.empty:
            df_normalized_trends = googletrends_normalize(df_hg_daily_ffilled['value'])
            df_to_join = pd.DataFrame({new_column_name: df_normalized_trends})
        else:
            df_final_with_trends[new_column_name] = pd.NA
            return df_final_with_trends

    if not df_to_join.empty:
        df_to_join.index.name = None 
        df_final_with_trends = df_final_with_trends.join(df_to_join, how='left')

        if new_column_name in df_final_with_trends.columns:
            df_final_with_trends[new_column_name] = df_final_with_trends[new_column_name].ffill()
            df_final_with_trends[new_column_name] = df_final_with_trends[new_column_name].fillna(0.5) 
        return df_final_with_trends
    else:
        print(f"  No hay datos de Google Trends para unir para el ticker {ticker}.")
        df_final_with_trends[new_column_name] = pd.NA
        return df_final_with_trends
        
# -----------------------------------------------------------------------------
# Obtiene los datos directos para un ticker en unas fechas dadas v2.0
#  - Intenta cargarlo de CSV con los datos guardados previamente y si no, de Yahoo Finance
#  - Obtiene datos indirectos
#  - Crea las variables calculadas (SMA, RSI, etc) 
#  - Obtiene tendencia del ticker de Google Trends
# -----------------------------------------------------------------------------
# Calcula todos los indicadores técnicos
def calculateTechnicalIndicators(data):    
    import talib
    # SMA (Simple Moving Average) de 20, 50 y 200 dias:
    data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20) # Mide la tendencia a corto plazo. Útil para detectar cambios rápidos en la tendencia. Común en estrategias de trading activo
    data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50) # Más estable que el SMA 20. Se usa para confirmar tendencias sin tanta volatilidad. Popular entre traders y analistas técnicos.
    data['SMA_200'] = talib.SMA(data['Close'], timeperiod=200) # Indicador clave para tendencias a largo plazo. Muy utilizado por inversores para evaluar si una acción está en tendencia alcista o bajista. Suele actuar como un fuerte nivel de soporte o resistencia.
    # EMA (Exponential Moving Average):
    data['EMA_20'] = talib.EMA(data['Close'], timeperiod=20)  # EMA de 20 días
    # RSI (Relative Strength Index):
    data['RSI_14'] = talib.RSI(data['Close'], timeperiod=14)  # RSI de 14 días
    # MACD (Moving Average Convergence Divergence):
    data['MACD'], data['MACD_signal'], _ = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    # Volatilidad
    data["ATR_14"] = talib.ATR(data["High"], data["Low"], data["Close"], timeperiod=14)
    # Bollinger Bands 
    upper, middle, lower = talib.BBANDS(data["Close"], timeperiod=20, nbdevup=2, nbdevdn=2)
    data["BB_upper"], data["BB_lower"] = upper, lower
    # Rango relativo diario (High / Low)
    data['Range'] = data['High'] / data['Low'] - 1
    # Cambio porcentual entre Open y Close
    data['OC_Change'] = ((data['Close'] - data['Open']) / data['Open']) * 100
    # Oscilador de Chaikin
    data["Chaikin_Osc"] = talib.ADOSC(data["High"], data["Low"], data["Close"], data["Volume"], fastperiod=3, slowperiod=10)    
    # En la v2.0, no calculamos estos valores ya que vamos al cierre del dia
    # Crear la columna 'Close_last_day' que representa el precio de cierre del día anterior
    #data['Close_last_day'] = data['Close'].shift(1)
    # Crear la columna 'Close_next_day' que representa el precio de cierre del día siguiente
    #data['Close_next_day'] = data['Close'].shift(-1)
    # Crear la columna Trend para comparar cierre actual vs. cierre anterior
    data['Trend'] = np.where(data['Close'].diff().fillna(0) >= 0, 1, 0)  # 1 si sube o se mantiene, 0 si baja    
    return data

def getTickerData(ticker, start_date, end_date, nombre_archivo):    
    if os.path.exists(nombre_archivo):
        print(f"Cargando los datos del archivo {nombre_archivo}. No se descargan nuevos datos.")
        return pd.read_csv(nombre_archivo, parse_dates=['Date'])        
    else:
        import yfinance as yf        
        from alpha_vantage.econindicators import EconIndicators
        print(f"Descargando datos directos para {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
                
        # Convertir MultiIndex a columnas planas
        data.columns = [col[0] for col in data.columns]  # Extraer solo el primer nivel

        # Agregar columna Ticker
        data["Ticker"] = ticker
        # Reorganizar columnas
        column_order = ["Ticker"] + data.columns.drop("Ticker").tolist()
        data = data[column_order]
        
        # Anadimos indicadores
        data = calculateTechnicalIndicators(data)

        # Extraer año, mes y día. Los modelos estadisticos no necesitan estos valores pero si los LSTM y Transformers.
        data.reset_index(inplace=True) # Aseguramos que date es una columna
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data["Month_sin"] = np.sin(2 * np.pi * data["Month"] / 12)
        data["Month_cos"] = np.cos(2 * np.pi * data["Month"] / 12)
        data['Day'] = data['Date'].dt.day

        # Datos indirectos de Yahoo Finance
        # Definir los tickers para cada uno de los indicadores:
        tickers = {
                    "Google": "GOOG",       # Google
                    "Amazon": "AMZN",       # Amazon
                    "Apple": "AAPL",        # Apple
                    "Meta": "META",         # Meta
                    "Microsoft": "MSFT",    # Microsoft
                    "NVIDIA": "NVDA",       # Nvidia
                    "Tesla": "TSLA",        # Tesla
                    "AMD": "AMD",           # AMD
                    "Intel": "INTC",        # Intel
                    #"TSMC": "TSMC34.SA",    # TSMC (solo tiene datos desde el 2019, lo descartamos)
                    "Treasury_3M": "^IRX",      # Rendimiento del Tesoro a 3 meses
                    "Treasury_10Y": "^TNX",     # Rendimiento del Tesoro a 10 años
                    "VIX": "^VIX",               # Volatilidad del S&P 500    
                    "Brent_Oil": "BZ=F",             # Petróleo Brent
                    "Gold": "GLD",               # Precio del oro (ETF)
                    "SP500": "^GSPC",           # Índice S&P 500
                    "NASDAQ100": "^NDX",         # Índice Nasdaq 100    
                    "EuroStoxx50": "^STOXX50E",  # Índice Euro Stoxx 50 (Europa)
                    "Nikkei225": "^N225",        # Índice Nikkei 225 (Bolsa de Tokio)
                    "ShanghaiComposite": "000001.SS" # Índice Shanghai Composite (Bolsa de China)
        }
        # Crear un diccionario para almacenar los dataframes
        data_dict = {}
        for nombre, ticker_i in tickers.items():
            if ticker != ticker_i:
                print(f"Descargando datos de {nombre} ({ticker_i})...")
                data_i = yf.download(ticker_i, start=start_date, end=end_date)    
                data_dict[nombre] = data_i[['Close']].rename(columns={'Close': nombre}) # Extraer la columna 'Close' y renombrarla con el nombre del indicador
            else:
                print(f"Omitiendo a {ticker} ya que es nuestro objetivo")
        # Unificamos y eliminamos multindex
        df_unificado = pd.concat(data_dict.values(), axis=1)
        df_unificado = df_unificado.droplevel("Ticker", axis=0) if isinstance(df_unificado.index, pd.MultiIndex) else df_unificado
        df_unificado.columns = df_unificado.columns.get_level_values(0) 
        df_unificado.reset_index(inplace=True)
        # Forward Fill para rellenar datos hacia el futuro
        columns_forward_fill = ['SP500', 'NASDAQ100', 'EuroStoxx50', 'Nikkei225', 'ShanghaiComposite', 'Brent_Oil', 'Treasury_3M', 'Treasury_10Y']
        df_unificado[columns_forward_fill] = df_unificado[columns_forward_fill].ffill()
        # Hacemos el merge
        data = pd.merge(
                        data,
                        df_unificado,
                        left_on='Date',
                        right_on='Date', 
                        how='left'
                       )

        # Datos indirectos de Alpha Vantage
        debug_alpha_vantage = 0
        if debug_alpha_vantage == 0:            
            api_key = "2CU8VJCQBQJ2E7GI" # "8PR01WCUEP6P8ZZV" "6M4B2XG63YC1BUPW" # Limite de 25 al dia
            # Inicializar el indicador económico
            indicator = EconIndicators(api_key, output_format="pandas")
            # Descargar CPI (Inflación)
            cpi_data, _ = indicator.get_cpi(interval="monthly")  # Mensual
            cpi_data.rename(inplace=True, columns={"value": "CPI"})
            cpi_data.set_index("date", inplace=True)
            cpi_data.sort_index(inplace=True)
            # Descargar PIB Real
            gdp_data, _ = indicator.get_real_gdp(interval="quarterly")  # Trimestral
            gdp_data.rename(inplace=True, columns={"value": "GDP_Real"})
            gdp_data.set_index("date", inplace=True)
            gdp_data.sort_index(inplace=True)
            # Descargar PIB per Capita 
            gdp_per_capita_data, _ = indicator.get_real_gdp_per_capita()  # Trimestral
            gdp_per_capita_data.rename(inplace=True, columns={"value": "GDP_per_Capita"})
            gdp_per_capita_data.set_index("date", inplace=True)
            gdp_per_capita_data.sort_index(inplace=True)
            # Los unimos
            list_of_dfs = [cpi_data, gdp_data, gdp_per_capita_data]
            merged_macro_data = pd.concat(list_of_dfs, axis=1, join='outer')
            merged_macro_data.reset_index(inplace=True)  # Asegura que 'date' esté como columna normal
            #merged_macro_data.to_csv("alpha.csv", index=False)
        else:
            merged_macro_data = pd.read_csv("alpha.csv", parse_dates=['date'])
        # Aseguramos formato de la fecha y la ponemos como indice
        merged_macro_data['date'] = pd.to_datetime(merged_macro_data['date'])
        merged_macro_data.set_index('date', inplace=True)        
        # Eliminamos valores anteriores a nuestro start date        
        merged_macro_data = merged_macro_data[start_date:]            
        # Creamos fechas para todos los dias
        merged_macro_data_index = pd.date_range(start=start_date, end=end_date, freq='D')
        merged_macro_data = merged_macro_data.reindex(merged_macro_data_index)
        # Forward Fill para rellenar datos hacia el futuro
        columns_forward_fill = ['CPI', 'GDP_Real', 'GDP_per_Capita']
        merged_macro_data[columns_forward_fill] = merged_macro_data[columns_forward_fill].ffill()
        # Hacemos el merge
        data.set_index('Date', inplace=True)
        data = pd.merge(
                        data,
                        merged_macro_data,
                        left_index=True,
                        right_index=True, 
                        how='left'
                        )
       
        # En la v2.0, dejamos valores nulos, cada modelo los tratara de una forma
        #data.dropna(inplace=True)
        
        # Google Trends (ficheros con datos en /GoogleTrends/ticker/)
        data_final = googletrends_add(
            source_df = data, 
            base_dir_google_trend = "./GoogleTrends", 
            ticker = ticker, 
            overall_monthly_file_name = "overall-monthly.csv",
            clip_rescaled_upper_limit = 100
        )
        
        # Guardar CSV        
        data_final.reset_index(inplace=True)  # Dejamos 'Date' como una columna normal
        data_final.to_csv(nombre_archivo, index=False)
        print(f"Datos guardados en {nombre_archivo}.")
        return data_final
        

# -----------------------------------------------------------------------------
# Genera gráficos comparando precios reales y predichos, con opciones para mostrar rangos diarios, predicción del día siguiente y gráfico de error.
#   Guarda en dahsboard creado a un PNG externo (z_prediction_graph.png).
#   ticker_name (str): Nombre del ticker (ej. 'NVIDIA').
#   model_name (str): Nombre del modelo usado para la predicción.
#   test_data (pd.DataFrame): DataFrame con los datos de prueba. Debe contener las columnas 'Date', 'Low', 'High'.
#   y_true (np.array): Valores reales del precio de cierre.
#   y_pred (np.array): Valores predichos del precio de cierre.
#   next_day_date (pd.Timestamp, opcional): Fecha del día siguiente a predecir. Requerido si show_next_day=True. Default None.
#   next_day_forecast (float, opcional): Valor predicho para el día siguiente. Requerido si show_next_day=True. Default None.
#   show_range (bool, opcional): Si es True, muestra el rango High-Low. Default False.
#   show_next_day (bool, opcional): Si es True, muestra la predicción del día siguiente. Default False.
#   show_error (bool, opcional): Si es True, muestra el subgráfico de error. Default False.
#   show_table (integer, opcional): Numero de ultimos dias a mostrar en una tabla. Default 0.
#   create_image: Una cadena que indica el prefijo de la imagen a crear.
# -----------------------------------------------------------------------------
def printPredictionGraph(
    ticker_name,
    model_name,
    test_data,
    y_true,
    y_pred,
    # Parámetros opcionales para el forecast del día siguiente
    next_day_date=None,
    next_day_forecast=None,
    # Parámetros booleanos para controlar la visualización
    show_range=False,
    show_next_day=False,
    show_error=False,
    show_table=False,
    create_image=None
):    
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    
    if 'Date' not in test_data.columns:
        raise ValueError("El DataFrame test_data debe contener la columna 'Date'.")
    if show_range and ('Low' not in test_data.columns or 'High' not in test_data.columns):
        print("Advertencia: show_range=True pero faltan columnas 'Low' o 'High' en test_data. Omitiendo el rango.")
        show_range = False
    if show_next_day and (next_day_date is None or next_day_forecast is None):
        print("Advertencia: show_next_day=True pero no se proporcionó next_day_date o next_day_forecast. Omitiendo el siguiente dia.")
        show_next_day = False

    # Convertir todo a numpy arrays para evitar problemas de tipo
    y_true_np = np.array(y_true).flatten()
    y_pred_np = np.array(y_pred).flatten()
    
    # Asegurar que las longitudes coincidan
    min_len = min(len(y_true_np), len(y_pred_np), len(test_data))
    y_true_np = y_true_np[:min_len]
    y_pred_np = y_pred_np[:min_len]
    test_data_subset = test_data.iloc[:min_len].copy()

    # Aseguramos que las fechas sean un objeto DatetimeIndex de pandas para un manejo correcto
    dates = pd.to_datetime(test_data_subset['Date'].values)

    # Configurar el estilo y contexto
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_context("talk") # 'talk' es bueno para presentaciones, 'notebook' para exploración

    # Creación dinámica de figura y ejes
    if show_error:
        # Crear 2 subplots si se muestra el error
        fig, axes = plt.subplots(
            2, 1,
            sharex=True, 
            gridspec_kw={'height_ratios': [4, 1]},
            figsize=(18, 10)
        )
        ax_price = axes[0]
        ax_error = axes[1]    
    else:
        # Crear 1 solo subplot si no se muestra el error
        fig, ax_price = plt.subplots(
            1, 1,            
            figsize=(18, 10)
        )
        ax_error = None # No existe el eje de error
    
     
    ax_price.set_title(f'Precio de cierre {ticker_name} - {model_name}', fontsize=18, pad=15)

    # Gráfico 1: precios y predicción
    ax_price.plot(dates, y_true_np, label='Precio real', marker='.', linestyle='-', color='navy', linewidth=1.5)
    # Opcional: banda de high-low
    if show_range:
        ax_price.fill_between(
            dates, test_data_subset['Low'].values, test_data_subset['High'].values,
            color='skyblue', alpha=0.2, label='Rango diario (bajo-alto)'
        )
    ax_price.plot(dates, y_pred_np, label=f'Predicción {model_name}', marker='.', linestyle='-', color='#FF9F45', linewidth=1.5) # Naranja-amarillo
    # Opcional: predicción siguiente día
    if show_next_day:
        next_day_date_dt = pd.to_datetime(next_day_date)
        ax_price.scatter(
            next_day_date_dt, next_day_forecast,
            label=f'Predicción próximo día ({next_day_date_dt.strftime('%Y-%m-%d')})',
            color='green', 
            marker='o', 
            s=100,
            zorder=5      # Asegurar que esté encima de las líneas
        )
    # Configuración del eje Y del gráfico de precios
    ax_price.set_ylabel('Precio de cierre (USD)', fontsize=14)
    ax_price.tick_params(axis='y', labelsize=14)
    ax_price.legend(fontsize=14)
    ax_price.grid(axis='y', linestyle='--', alpha=0.7)
    # Ocultar etiquetas X si hay gráfico de error debajo
    if show_error:
         ax_price.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Gráfico 2: Error de predicción (opcional)
    if show_error:
        # Calcular error como numpy array
        error_prediccion = y_pred_np - y_true_np
        
        # Usar approach más simple y robusto: convertir fechas a números para matplotlib
        dates_numeric = mdates.date2num(dates)
        
        # Crear barras individuales para evitar problemas con arrays
        for i, (date_num, error_val) in enumerate(zip(dates_numeric, error_prediccion)):
            color = 'lightcoral' if error_val > 0 else 'lightgreen'
            ax_error.bar(
                date_num, error_val,
                width=0.8,  # Ancho en días 
                color=color,
                alpha=0.8,
                linewidth=0  # Sin bordes
            )
        
        # Añadir label manualmente
        ax_error.bar([], [], color='gray', alpha=0.8, label='Error (Pred - Real)')
        ax_error.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7) 
        ax_error.set_ylabel('Error (USD)', fontsize=14)
        ax_error.tick_params(axis='y', labelsize=14)
        ax_error.grid(axis='y', linestyle='--', alpha=0.7)
        axis_to_format_x = ax_error
    else:
        # Si no hay gráfico de error, configurar el eje X en el primer grafico
        axis_to_format_x = ax_price
        # Definir error_prediccion para la tabla
        error_prediccion = y_pred_np - y_true_np

    # Formateo del eje X (se aplica al eje inferior o al único eje existente)
    major_locator = mdates.AutoDateLocator(minticks=5, maxticks=12) 
    major_formatter = mdates.DateFormatter('%Y-%m-%d') # Formato YYYY-MM-DD
    axis_to_format_x.xaxis.set_major_locator(major_locator)
    axis_to_format_x.xaxis.set_major_formatter(major_formatter)
    axis_to_format_x.tick_params(axis='x', labelsize=14, rotation=30, which='major') 
    fig.autofmt_xdate(rotation=30, ha='right')

    # Ajuste final del layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])     
    # Guardar a PNG si usamos create_image 
    if create_image:
        plt.savefig(f"{create_image}_prediction_graph.png", dpi=66.65)
    if AMPBConfig.INTERACTIVE:
        plt.show()    
            
    # Imprimir una tabla con los últimos n días
    if show_table > 0:
        print(f"{AMPBConfig.COLOR_INFO}Precio real vs predicción de {ticker_name} - {model_name} (últimos {show_table} días):{AMPBConfig.COLOR_RESET}")
        # Convertir a arrays simples y aplanar antes de redondear
        y_true_clean = np.asarray(y_true_np).flatten().astype(float)
        y_pred_clean = np.asarray(y_pred_np).flatten().astype(float)
        error_clean = np.asarray(error_prediccion).flatten().astype(float)        
        tabla_resultados = pd.DataFrame({
            'Precio real': np.around(y_true_clean, 2),
            'Predicción': np.around(y_pred_clean, 2),
            'Error (Pred - Real)': np.around(error_clean, 2),
        }, index=dates)
        tabla_resultados.index = tabla_resultados.index.strftime('%Y-%m-%d')
        print(tabla_resultados.tail(show_table).to_markdown())
        if show_next_day:
            next_day_date_dt = pd.to_datetime(next_day_date)
            print(f"{AMPBConfig.COLOR_VALUE}\n - Predicción para {next_day_date_dt.strftime('%Y-%m-%d')}: {next_day_forecast:.2f}{AMPBConfig.COLOR_RESET}")
    print("")
    

# -----------------------------------------------------------------------------
# Evalúa métricas y genera gráficos de forma unificada.
# -----------------------------------------------------------------------------
def generateEvaluation(y_test_original, forecast_original, df_test, model_title, model_hash, next_day_date, next_day_forecast, validation_type):
    # Métricas de regresión
    r2, mae, rmse = printMetricsRegression(y_test_original, forecast_original, model_title, create_image=model_hash)
    
    # Preparar datos para gráfico
    df_test_plot = df_test.copy()
    df_test_plot.reset_index(inplace=True)
    
    # Asegurar que next_day_forecast es escalar
    if isinstance(next_day_forecast, (pd.Series, np.ndarray)):
        next_day_forecast = next_day_forecast.item()
    
    # Gráfico
    printPredictionGraph(
        "NVIDIA", model_title, df_test_plot,
        y_test_original, forecast_original,
        show_error=True, show_table=5, show_next_day=True,
        next_day_date=next_day_date,
        next_day_forecast=next_day_forecast,
        create_image=model_hash
    )
    
    # Tendencias
    trend_predict = (forecast_original.diff() >= 0).astype(int).iloc[1:]
    trend_real = df_test['Trend'].iloc[1:]
    
    # Métricas de clasificación
    accuracy, f1_score, roc_auc = printMetricsClassification(trend_real, trend_predict, model_title, create_image=model_hash)
    
    # Tabla comparativa
    df_comparacion = pd.DataFrame({
        'Precio real': y_test_original.iloc[1:].round(2),
        'Tendencia real': trend_real.round(2),
        f'Predicción ({validation_type})': forecast_original.iloc[1:].round(2),
        'Tendencia predicha': trend_predict.round(2)
    }, index=y_test_original.index[1:])
    
    df_comparacion.index = df_comparacion.index.strftime('%Y-%m-%d')
    print(f"\nDatos de tendencia en Test ({validation_type} - últimos 15 días):")
    print(df_comparacion.tail(15).to_markdown())
    
    return r2, mae, rmse, accuracy, f1_score, roc_auc      
  
    
# -----------------------------------------------------------------------------
# Crea el informa final del modelo: crea una imagen con las imagenes apiladas y despues las borra y un CSV con las metricas
# Aqui utilizamos Índice Relativo al Baseline (IRB) comparando contra el modelo base (con sus indices a 1)
# -----------------------------------------------------------------------------    
def createReport(model_name, model_test, model_name_suffix, model_title, model_hash, r2, mae, rmse, accuracy, f1_score, roc_auc):
    from PIL import Image
    try:
        # Cálculo de IRB 
        norm_mae  = AMPBConfig.BASELINE_MAE / mae
        norm_rmse = AMPBConfig.BASELINE_RMSE / rmse
        norm_r2   = r2 / AMPBConfig.BASELINE_R2
        irb_regr  = np.mean([norm_mae, norm_rmse, norm_r2])

        norm_acc     = accuracy / AMPBConfig.BASELINE_ACC
        norm_f1      = f1_score / AMPBConfig.BASELINE_F1
        norm_roc_auc = roc_auc / AMPBConfig.BASELINE_ROCAUC
        irb_clas     = np.mean([norm_acc, norm_f1, norm_roc_auc])

        irb_total = (AMPBConfig.SCORING_WEIGHTS['regression'] * irb_regr + AMPBConfig.SCORING_WEIGHTS['classification'] * irb_clas)
        
    except Exception as e:
        print(f"  ERROR al calcular índices IRB para {model_title}: {e}")
        return

    # Datos redondeados para guardar
    results_data = {
        'model_title': [model_title],
        'model_hash': [model_hash],
        'r2': [round(r2, 4)],
        'mae': [round(mae, 4)],
        'rmse': [round(rmse, 4)],
        'accuracy': [round(accuracy, 4)],
        'f1_score': [round(f1_score, 4)],
        'roc_auc': [round(roc_auc, 4)],
        'irb_regr': [round(irb_regr, 4)],
        'irb_clas': [round(irb_clas, 4)],
        'irb_total': [round(irb_total, 4)]
    }
    results_df = pd.DataFrame(results_data)

    # Rutas y nombres de archivo
    base_path = os.path.join(AMPBConfig.OUTPUT_DIR, model_name, model_test)
    os.makedirs(base_path, exist_ok=True)

    filename_base = f"{model_name}_{model_test}_{model_name_suffix}"
    csv_filename = os.path.join(base_path, f"{filename_base}.csv")

    try:
        results_df.to_csv(csv_filename, index=False, sep=';', decimal=',')
    except Exception as e:
        print(f"  ERROR al guardar el CSV para {model_hash}: {e}")
        return

    # Imagen apilada
    base_names = [
        "_metrics_regression.png",
        "_prediction_graph.png",
        "_metrics_classification.png"
    ]
    img_files = [f"{model_hash}{name}" for name in base_names]

    try:
        imgs = []
        existing_files = []
        for f in img_files:
            if os.path.exists(f):
                imgs.append(Image.open(f))
                existing_files.append(f)
            else:
                print(f"  ADVERTENCIA: No se encontró el archivo de imagen '{f}'. Se omitirá.")

        if not imgs:
            print("  ERROR: No se encontraron imágenes para apilar. Abortando la creación del informe.")
            return

        # Lienzo
        widths = [img.width for img in imgs]
        heights = [img.height for img in imgs]
        canvas_w = max(widths)
        canvas_h = sum(heights)

        merged = Image.new("RGB", (canvas_w, canvas_h), color="white")
        y_offset = 0
        for img in imgs:
            merged.paste(img, (0, y_offset))
            y_offset += img.height

        png_filename = os.path.join(base_path, f"{filename_base}.png")
        merged.save(png_filename, format="PNG")

        for img, path in zip(imgs, existing_files):
            img.close()
            try:
                os.remove(path)
            except OSError as e:
                print(f"  ERROR: No se pudo eliminar el archivo temporal '{path}': {e}")

    except Exception as e:
        print(f"  ERROR: Ocurrió un problema al crear el informe apilado. Causa: {e}")

    print(f"\n{AMPBConfig.COLOR_MSG}Informe completo para '{model_title}' generado exitosamente ('{png_filename}').{AMPBConfig.COLOR_RESET}\n")

    
# -----------------------------------------------------------------------------
# Pipeline de des-transformación: escalado → transformación → original
# -----------------------------------------------------------------------------  
# -----------------------------------------------------------------------------
# Función genérica para deshacer transformaciones de Close
# -----------------------------------------------------------------------------  
def undoPredictionTransformation(transformed_preds, transformation, params, reference_value=None):
    from sklearn.preprocessing import PowerTransformer
    if transformation == "None":
        return transformed_preds
    lambda_param, shift = params['lambda'], params['shift']
    try:
        if transformation == "Log":
            # Para Log: original = exp(transformed) - shift
            return np.exp(transformed_preds) - shift
        
        elif transformation == "RetLog":
            if reference_value is None:
                raise ValueError("Para deshacer 'RetLog' se necesita un 'reference_value' (último valor original conocido).")
            log_reference = np.log(reference_value)
            log_prices = log_reference + np.cumsum(transformed_preds)
            return np.exp(log_prices)

        elif transformation == "YeoJohnson":            
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            pt.lambdas_ = np.array([lambda_param])
            # Reshape para que sea una matriz de (n_samples, 1)
            values_reshaped = np.array(transformed_preds).reshape(-1, 1)
            return pt.inverse_transform(values_reshaped).flatten()
            
    except Exception as e:
        print(f"\nError al deshacer la transformación '{transformation}': {e}. Devolviendo predicción sin transformar.")
        return transformed_preds
        
    return transformed_preds
    
    
def applyTransformation(data, transform_type): # FUNCIÓN AUXILIAR 
    from sklearn.preprocessing import PowerTransformer
    shift_value = 0
    lambda_param = None
    
    if transform_type == "Log":
        # Asegurar que todos los valores son positivos antes de aplicar log
        if isinstance(data, pd.Series):
            if (data <= 0).any():
                shift_value = abs(data.min()) + 1e-9
                data[:] = np.log(data + shift_value)
            else:
                data[:] = np.log(data)
        else:  # DataFrame
            min_val = data.min().min()
            if min_val <= 0:
                shift_value = abs(min_val) + 1e-9
                data[:] = np.log(data + shift_value)
            else:
                data[:] = np.log(data)
                
    elif transform_type == "RetLog":
        if isinstance(data, pd.Series):
            log_data = np.log(data)
            returns = log_data.diff()
            # Eliminar el primer NaN generado por diff()
            data_clean = returns.dropna()
            # Actualizar el índice del data original
            data.iloc[:] = np.nan  # Limpiar
            data.loc[data_clean.index] = data_clean.values
            # Eliminar filas con NaN
            data.dropna(inplace=True)
        else:  # DataFrame
            log_data = np.log(data)
            returns = log_data.diff()
            # Eliminar filas con NaN
            data_clean = returns.dropna()
            data.iloc[:] = np.nan
            data.loc[data_clean.index] = data_clean.values
            data.dropna(inplace=True)
            
    elif transform_type == "YeoJohnson":
        # PowerTransformer es robusto y maneja datos positivos/negativos
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        
        if isinstance(data, pd.Series):
            transformed = pt.fit_transform(data.values.reshape(-1, 1)).flatten()
            data[:] = transformed
            lambda_param = pt.lambdas_[0]
        else:  # DataFrame
            transformed = pt.fit_transform(data.values)
            data[:] = transformed
            lambda_param = pt.lambdas_  # Array de lambdas para cada columna            
    return lambda_param, shift_value

def applyTransformationWithParams(data, transform_type, lambda_param, shift_value): # FUNCIÓN AUXILIAR
    from sklearn.preprocessing import PowerTransformer
    if transform_type == "Log":
        if isinstance(data, pd.Series):
            data[:] = np.log(data + shift_value)
        else:  # DataFrame
            data[:] = np.log(data + shift_value)
            
    elif transform_type == "RetLog":
        if isinstance(data, pd.Series):
            log_data = np.log(data)
            returns = log_data.diff()
            # Manejar NaN apropiadamente
            data_clean = returns.dropna()
            data.iloc[:] = np.nan
            data.loc[data_clean.index] = data_clean.values
            data.dropna(inplace=True)
        else:  # DataFrame
            log_data = np.log(data)
            returns = log_data.diff()
            data_clean = returns.dropna()
            data.iloc[:] = np.nan
            data.loc[data_clean.index] = data_clean.values
            data.dropna(inplace=True)
            
    elif transform_type == "YeoJohnson":
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        
        if isinstance(data, pd.Series):
            pt.lambdas_ = np.array([lambda_param])
            transformed = pt.transform(data.values.reshape(-1, 1)).flatten()
            data[:] = transformed
        else:  # DataFrame
            pt.lambdas_ = lambda_param if isinstance(lambda_param, np.ndarray) else np.array([lambda_param] * len(data.columns))
            transformed = pt.transform(data.values)
            data[:] = transformed
                        
# Si la predicción es inválida (NaN, inf), aborta
def sanitizePrediction(prediction, position=None):
    # Convertir Series/array a escalar si es necesario
    if isinstance(prediction, (pd.Series, np.ndarray)):
        prediction = prediction.item() 
    pos_info = f" en posición {position}" if position is not None else ""
    
    # Validaciones
    if pd.isna(prediction):
        raise ValueError(f"Aborta ejecución: predicción inválida (NaN){pos_info}.")
    if np.isinf(prediction):
        raise ValueError(f"Aborta ejecución: predicción infinita{pos_info}.")
    if prediction <= 0:
        raise ValueError(f"Aborta ejecución: predicción no positiva{pos_info}. Valor: {prediction}")
    
    return prediction
    
# Pipeline de des-transformación: escalado → transformación → original
def reverseTransformPredictions(predictions, reference_value, y_scaler, transformation, params_close, max_limit=None):
    #print(f"\n --- DEBUG: Iniciando reversión de transformaciones ---")
    #print(f" Transformación: {transformation}")
    #print(f" Predicciones originales (min/max): {predictions.min():.6f} / {predictions.max():.6f}")
    # 1. Des-escalar (si existe y_scaler)
    if y_scaler is not None:
        #print(" Aplicando des-escalado...")
        predictions_transformed = y_scaler.inverse_transform(predictions.values.reshape(-1, 1)).flatten()
        predictions_transformed = pd.Series(predictions_transformed, index=predictions.index)
        #print(f" Después del des-escalado (min/max): {predictions_transformed.min():.6f} / {predictions_transformed.max():.6f}")
    else:
        #print(" Sin escalado aplicado")
        predictions_transformed = predictions
    # 2. Des-transformar
    #print(f" Aplicando des-transformación '{transformation}'...")
    #if transformation == "RetLog":
        #print(f" Valor de referencia: {reference_value}")
        #print(f" Parámetros: {params_close}")
    predictions_original = undoPredictionTransformation(predictions_transformed, transformation, params_close, reference_value)
    predictions_original = pd.Series(predictions_original, index=predictions_transformed.index)
    #print(f" Después de des-transformación (min/max): {predictions_original.min():.6f} / {predictions_original.max():.6f}")
    
    # 3. Sanitizar
    #print(" Sanitizando predicciones...")
    sanitized = []
    for i, pred in enumerate(predictions_original):        
        date_info = predictions_original.index[i] if hasattr(predictions_original.index[i], 'strftime') else predictions_original.index[i]
        position_info = f"{i} (fecha: {date_info})"
        # Puede fallar y abortar
        sanitized.append(sanitizePrediction(pred, position=position_info))        
    sanitized_series = pd.Series(sanitized, index=predictions_original.index)
    
    # 4. Aplicar límite máximo (si se proporciona)
    if max_limit is not None:
        #print(f" Aplicando límite máximo: {max_limit:.0f}")
        capped_values = []
        capped_count = 0        
        for i, pred in enumerate(sanitized_series):
            if pred > max_limit:
                print(f"  Predicción {pred:.2f} > límite máximo {max_limit:.0f}, ajustada")
                capped_values.append(max_limit)
                capped_count += 1
            else:
                capped_values.append(pred)
        
        if capped_count > 0:
            print(f"  Total de {capped_count} predicciones limitadas al máximo")
        final_result = pd.Series(capped_values, index=sanitized_series.index)
    else:
        final_result = sanitized_series
    
    #print(f"Predicciones finales (min/max): {final_result.min():.6f} / {final_result.max():.6f}")
    #print("--- FIN DEBUG reversión ---\n")    
    return final_result   
    
    
# -----------------------------------------------------------------------------
# Actualiza las variables exógenas para el día t usando datos disponibles hasta el día t-1.
#  Ahora: Copia todas las exógenas de t-1, excepto 'Open' que será el Close de t-1.
#  Futuro: Esta función se puede extender para actualizar con datos intraday del día t, permitiendo re-ejecutar el modelo durante el día con mayor precisión.
# -----------------------------------------------------------------------------   
def updateNextDayExog(history_X,
                      feature_original_close,
                      transformation,
                      params_exog,
                      exog_scaler,
                      prev_open_original=None
                      ):
    if history_X is None or history_X.empty:
        return None

    X_next = history_X.iloc[-1:].copy()
    if 'Open' not in X_next.columns:
        return X_next

    open_idx = X_next.columns.get_loc('Open')

    # Transformación de 'Open' con sus propios parámetros
    if transformation == "None":
        open_transformed = float(feature_original_close)
    elif transformation == "Log":
        shf = params_exog['shifts'].get('Open', 0.0)
        open_transformed = float(np.log(feature_original_close + shf))
    elif transformation == "YeoJohnson":
        lam = params_exog['lambdas'].get('Open', None)
        # PowerTransformer.transform requiere array 2D
        pt_val = np.array([[feature_original_close]], dtype=float)
        from sklearn.preprocessing import PowerTransformer
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        # fijar lambda de Open aprendido en train
        pt.lambdas_ = np.array([lam]) if lam is not None else np.array([0.0])
        open_transformed = float(pt.transform(pt_val).ravel()[0])
    elif transformation == "RetLog":
        # Necesitamos Open_{t-1} en original para calcular diff(log)
        if prev_open_original is None or prev_open_original <= 0 or feature_original_close <= 0:
            # Sin referencia fiable, no tocamos Open para evitar incoherencias
            raise ValueError(f"Aborta ejecución : prev_open_original NaN o <=0")
        open_transformed = float(np.log(feature_original_close) - np.log(prev_open_original))
    else:
        # Transformación desconocida: error
        raise ValueError(f"Aborta ejecución : transformación desconocida")

    # Escalado de 'Open'
    if exog_scaler is None:
        open_scaled = open_transformed
    elif hasattr(exog_scaler, 'mean_') and hasattr(exog_scaler, 'scale_'):
        # StandardScaler
        open_scaled = (open_transformed - exog_scaler.mean_[open_idx]) / exog_scaler.scale_[open_idx]
    elif hasattr(exog_scaler, 'data_min_') and hasattr(exog_scaler, 'data_range_'):
        # MinMaxScaler
        data_rng = exog_scaler.data_range_[open_idx]
        fr_min, fr_max = exog_scaler.feature_range
        if data_rng == 0:
            open_scaled = fr_min
        else:
            x01 = (open_transformed - exog_scaler.data_min_[open_idx]) / data_rng
            open_scaled = x01 * (fr_max - fr_min) + fr_min
    else:
        # Fallback: transformar fila completa
        #tmp_row = X_next.values.copy()
        #tmp_row[0, open_idx] = open_transformed
        #open_scaled = exog_scaler.transform(tmp_row)[0, open_idx]
        raise ValueError(f"Aborta ejecución : escalado desconocido")

    X_next.iloc[0, open_idx] = open_scaled
    #print(f"updateNextDayExog(): valor re-transformado: {open_scaled:.6f}") # Debug
    return X_next


# -----------------------------------------------------------------------------
# Análisis completo de calidad de datos y detección de outliers después de transformaciones.
# -----------------------------------------------------------------------------
def analyzeDataQuality(X_train, X_test, X_train_original, X_test_original, 
                       params_exog, scaler, transformation, exog_scaling):
    print(f"\n === ANÁLISIS DE CALIDAD DE DATOS ===")
    
    # 1. COMPARACIÓN TRAIN VS TEST POR VARIABLE
    print(f"\n COMPARACIÓN TRAIN VS TEST:")
    print(f"{'Variable':<15} {'Train >3σ':<10} {'Test >3σ':<10} {'% Train':<8} {'% Test':<8}")
    print(f"{'-'*60}")

    outlier_threshold = 3
    total_train_outliers = 0
    total_test_outliers = 0
    problematic_vars = []

    for col in X_train.columns:
        train_data = X_train[col]
        test_data = X_test[col]
        
        train_outliers = np.abs(train_data) > outlier_threshold
        test_outliers = np.abs(test_data) > outlier_threshold
        
        train_count = train_outliers.sum()
        test_count = test_outliers.sum()
        
        train_pct = (train_count / len(train_data)) * 100
        test_pct = (test_count / len(test_data)) * 100
        
        total_train_outliers += train_count
        total_test_outliers += test_count
        
        # Marcar variables problemáticas
        if test_pct > 10:
            flag = "Alerta!!!"
            problematic_vars.append(col)
        elif test_pct > 5:
            flag = "Regular!"
        else:
            flag = "OK"
        
        print(f"{col:<15} {train_count:<10} {test_count:<10} {train_pct:<7.1f}% {test_pct:<7.1f}% {flag}")

    print(f"{'-'*60}")
    print(f"{'TOTAL':<15} {total_train_outliers:<10} {total_test_outliers:<10}")

    # 2. ANÁLISIS DETALLADO DE VARIABLES PROBLEMÁTICAS
    if problematic_vars:
        print(f"\n ANÁLISIS DETALLADO DE VARIABLES PROBLEMÁTICAS:")
        
        for var in problematic_vars[:3]:  # Limitar a las 3 más problemáticas
            if var in X_train.columns:
                train_var = X_train[var]
                test_var = X_test[var]
                
                print(f"\n ESTADÍSTICAS {var}:")
                print(f"                Train       Test")
                print(f"   Min:      {train_var.min():8.3f}   {test_var.min():8.3f}")
                print(f"   Max:      {train_var.max():8.3f}   {test_var.max():8.3f}")
                print(f"   Media:    {train_var.mean():8.3f}   {test_var.mean():8.3f}")
                print(f"   Std:      {train_var.std():8.3f}   {test_var.std():8.3f}")
                
                # Verificar distribución de outliers
                for threshold in [3, 4, 5]:
                    train_over = np.abs(train_var) > threshold
                    test_over = np.abs(test_var) > threshold
                    print(f"   >{threshold}σ:     {train_over.sum():8d}   {test_over.sum():8d}")

    # 3. VERIFICAR PARÁMETROS DE TRANSFORMACIÓN
    if transformation != "None" and params_exog:
        print(f"\n VERIFICACIÓN DE PARÁMETROS DE TRANSFORMACIÓN:")
        print(f"   Transformación aplicada: {transformation}")
        
        # Mostrar parámetros de variables problemáticas
        for var in problematic_vars[:3]:
            if var in params_exog.get('lambdas', {}):
                var_lambda = params_exog['lambdas'][var]
                var_shift = params_exog['shifts'][var]
                print(f"   {var} - λ: {var_lambda}, shift: {var_shift}")

    # 4. VERIFICAR ESCALADO
    if exog_scaling != "None" and scaler is not None:
        print(f"\n VERIFICACIÓN DE ESCALADO:")
        print(f"   Escalado aplicado: {exog_scaling}")
        
        # Verificar parámetros del scaler para variables problemáticas
        for var in problematic_vars[:3]:
            if var in X_train.columns:
                var_idx = list(X_train.columns).index(var)
                scaler_mean = scaler.mean_[var_idx]
                scaler_scale = scaler.scale_[var_idx]
                
                print(f"   {var} - mean: {scaler_mean:.6f}, scale: {scaler_scale:.6f}")

    # 5. COMPARACIÓN TEMPORAL (usando datos originales)
    if len(problematic_vars) > 0:
        print(f"\n COMPARACIÓN TEMPORAL (datos originales):")
        
        for var in problematic_vars[:3]:
            if var in X_train_original.columns and var in X_test_original.columns:
                train_orig_mean = X_train_original[var].mean()
                test_orig_mean = X_test_original[var].mean()
                
                if train_orig_mean != 0:
                    ratio_means = test_orig_mean / train_orig_mean
                    
                    print(f"   {var}:")
                    print(f"     Train medio: {train_orig_mean:,.3f}")
                    print(f"     Test medio: {test_orig_mean:,.3f}")
                    print(f"     Ratio test/train: {ratio_means:.2f}x")
                    
                    if ratio_means > 2 or ratio_means < 0.5:
                        print(f"      Cambio significativo entre períodos")

    # 6. DIAGNÓSTICOS Y RECOMENDACIONES
    print(f"\n DIAGNÓSTICOS:")
    
    if total_test_outliers > total_train_outliers * 3:
        print(f"    PROBLEMA SEVERO: Demasiados outliers en test ({total_test_outliers} vs {total_train_outliers})")
        print(f"    POSIBLES CAUSAS:")
        print(f"      • Parámetros de transformación no aplicados correctamente al test")
        print(f"      • Parámetros de escalado calculados incorrectamente")
        print(f"      • Data leakage en cálculo de parámetros")
        print(f"      • Distribución muy diferente entre períodos train y test")
    elif total_test_outliers > total_train_outliers * 2:
        print(f"    PROBLEMA MODERADO: Más outliers en test de lo esperado")
        print(f"    Revisar variables: {', '.join(problematic_vars)}")
    else:
        print(f"    CALIDAD ACEPTABLE: Distribución de outliers normal")
    
    print(f"\n RECOMENDACIONES:")
    if problematic_vars:
        print(f"   1. Revisar transformación de variables: {', '.join(problematic_vars)}")
        print(f"   2. Considerar usar transformaciones más robustas para estas variables")
    else:
        print(f"   1. Calidad de datos satisfactoria")
        print(f"   2. Continuar con el entrenamiento del modelo")
    
    print(f"\n === FIN ANÁLISIS ===\n")
    
    return {
        'total_train_outliers': total_train_outliers,
        'total_test_outliers': total_test_outliers,
        'problematic_vars': problematic_vars,
        'quality_score': 'good' if len(problematic_vars) == 0 else 'moderate' if len(problematic_vars) <= 2 else 'poor'
    }    


# -----------------------------------------------------------------------------    
# Comprueba el porcentaje de registros eliminados por transformaciones y escalador y aborta si inferior al threshold
# -----------------------------------------------------------------------------
def checkDataLoss(original_train_size, original_test_size, final_train_size, final_test_size, max_loss_threshold=0.10):
    # Calcular pérdidas
    train_loss = original_train_size - final_train_size
    test_loss = original_test_size - final_test_size
    
    # Calcular porcentajes
    train_loss_pct = (train_loss / original_train_size) * 100 if original_train_size > 0 else 0
    test_loss_pct = (test_loss / original_test_size) * 100 if original_test_size > 0 else 0
    
    print(f"{AMPBConfig.COLOR_INFO}\nVerificación de pérdida de datos:{AMPBConfig.COLOR_RESET}")
    print(f"  Train: {AMPBConfig.COLOR_VALUE}{original_train_size} → {final_train_size}{AMPBConfig.COLOR_RESET} (perdidos: {train_loss}, {train_loss_pct:.1f}%)")
    print(f"  Test:  {AMPBConfig.COLOR_VALUE}{original_test_size} → {final_test_size}{AMPBConfig.COLOR_RESET} (perdidos: {test_loss}, {test_loss_pct:.1f}%)")
    
    # Verificar umbrales
    max_loss_pct = max_loss_threshold * 100
    
    if train_loss_pct > max_loss_pct or test_loss_pct > max_loss_pct:
        print(f"\n  Error critico: pérdida de datos excesiva")
        print(f"    Umbral máximo permitido: {AMPBConfig.COLOR_VALUE}{max_loss_pct:.1f}%{AMPBConfig.COLOR_RESET}")
        print(f"    Pérdida detectada:")
        if train_loss_pct > max_loss_pct:
            print(f"     - Train: {AMPBConfig.COLOR_VALUE}{train_loss_pct:.1f}% > {max_loss_pct:.1f}%{AMPBConfig.COLOR_RESET}")
        if test_loss_pct > max_loss_pct:
            print(f"     - Test: {AMPBConfig.COLOR_VALUE}{test_loss_pct:.1f}% > {max_loss_pct:.1f}%{AMPBConfig.COLOR_RESET}")
        raise ValueError(f"Aborta ejecución: pérdida de datos excesiva: {train_loss_pct:.1f}% > {max_loss_pct:.1f}% | {test_loss_pct:.1f}% > {max_loss_pct:.1f}%")        
    print()

    
# -----------------------------------------------------------------------------
# Procesa datos completos: transformaciones, escalado, alineación y análisis de calidad.
#    Args:
#        y_train, y_test: Series de variable objetivo (se modifican in-place)
#        X_train, X_test: DataFrames de variables exógenas (se modifican in-place)
#        y_train_original, y_test_original, X_train_original, X_test_original: Versiones originales
#        df_test: DataFrame completo de test para alineación
#        exog_vars: Lista de nombres de variables exógenas
#        transformation: Tipo de transformación ("None", "Log", "RetLog", "YeoJohnson")
#        exog_scaling: Tipo de escalado ("None", "Standard", "MinMax")
#        winsorization_value: Aplicar winsorización si >0 (buen valor 0.005)
#        analyze: Si ejecutar análisis de calidad de datos (default: False)
#    Returns:
#        dict: {
#            'params_close': Parámetros de transformación de Close,
#            'y_scaler': Scaler de variable objetivo,
#            'df_test_aligned': DataFrame de test alineado,
#            'prediction_max_limit': Límite máximo para predicciones,
#            'quality_results': Resultados de análisis de calidad (si analyze=True)
#        }
# -----------------------------------------------------------------------------   
def processData(y_train, y_test, X_train, X_test, 
                y_train_original, y_test_original, X_train_original, X_test_original,
                df_test, exog_vars, transformation, exog_scaling, 
                winsorization_value, analyze=False):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    # Verificar si tenemos variables exógenas
    has_exog_vars = (X_train is not None and X_test is not None and exog_vars is not None and len(exog_vars) > 0)

    # A. TRANSFORMACIONES
    print(f"{AMPBConfig.COLOR_INFO}Aplicando transformación '{transformation}'...{AMPBConfig.COLOR_RESET}")

    # Guardar tamaños originales para verificación
    original_train_size = len(y_train)
    original_test_size = len(y_test)

    # Inicializar parámetros
    params_close = {'lambda': None, 'shift': 0}
    params_exog = {'lambdas': {}, 'shifts': {}}
    if transformation != "None":
        # PASO 1: Calcular parámetros y aplicar transformación con datos de entrenamiento
        params_close['lambda'], params_close['shift'] = applyTransformation(y_train, transformation)
        
        # Solo procesar variables exógenas si existen
        exog_vars_neg = []
        if has_exog_vars:
            for var in exog_vars:
                if transformation in ["RetLog", "Log"]:
                    has_negative = (X_train[var] < 0).any() or (X_test[var] < 0).any()
                    if has_negative:
                        # No transformar variables con valores negativos
                        exog_vars_neg.append(var)
                        params_exog['lambdas'][var] = None
                        params_exog['shifts'][var] = 0
                    else:
                        # Transformar variables siempre positivas
                        lambda_var, shift_var = applyTransformation(X_train[var], transformation)
                        params_exog['lambdas'][var] = lambda_var
                        params_exog['shifts'][var] = shift_var
                else:
                    # Para otras transformaciones, aplicar normalmente
                    lambda_var, shift_var = applyTransformation(X_train[var], transformation)
                    params_exog['lambdas'][var] = lambda_var
                    params_exog['shifts'][var] = shift_var
        
        print(f"  Parámetros calculados (solo con entrenamiento):")
        print(f"    Close: λ={AMPBConfig.COLOR_VALUE}{params_close['lambda']}{AMPBConfig.COLOR_RESET}, shift={AMPBConfig.COLOR_VALUE}{params_close['shift']:.6f}{AMPBConfig.COLOR_RESET}")
        if has_exog_vars and transformation in ["RetLog", "Log"] and exog_vars_neg:
            print(f"  Variables no transformadas por valores negativos: {AMPBConfig.COLOR_KEY}{exog_vars_neg}{AMPBConfig.COLOR_RESET}")
        
        # PASO 2: Aplicar transformación al test usando parámetros del train
        # Siempre usa applyTransformationWithParams para test (sin calcular nuevos parámetros)
        applyTransformationWithParams(y_test, transformation, params_close['lambda'], params_close['shift'])
        
        if has_exog_vars:
            for var in exog_vars:
                if transformation in ["RetLog", "Log"] and var in exog_vars_neg:
                    # No transformar variables con valores negativos
                    continue
                else:
                    # Aplicar transformación con parámetros del train
                    applyTransformationWithParams(X_test[var], transformation, params_exog['lambdas'][var], params_exog['shifts'][var])
        
        
        # Winsorización (solo si hay variables exógenas y está habilitada)
        if has_exog_vars and winsorization_value > 0:
            print(f"  Aplicando winsorización... [{winsorization_value},{1-winsorization_value}]")
            total_changes_train = 0
            total_changes_test = 0
            for col in X_train.columns:
                # Aprender límites solo de train
                lower_limit = X_train[col].quantile(winsorization_value)  # 0.5% inferior default
                upper_limit = X_train[col].quantile(1 - winsorization_value)  # 99.5% superior default
                
                # Aplicar a train
                original_values_train = X_train[col].copy()
                X_train[col] = X_train[col].clip(lower=lower_limit, upper=upper_limit)
                changes_train = (X_train[col] != original_values_train).sum()
                
                # Aplicar a test
                original_values_test = X_test[col].copy()
                X_test[col] = X_test[col].clip(lower=lower_limit, upper=upper_limit)
                changes_test = (X_test[col] != original_values_test).sum()
                
                if changes_train > 0 or changes_test > 0:
                    print(f"    {col}: train={AMPBConfig.COLOR_VALUE}{changes_train}{AMPBConfig.COLOR_RESET}, test={AMPBConfig.COLOR_VALUE}{changes_test}{AMPBConfig.COLOR_RESET} valores modificados")
                total_changes_train += changes_train
                total_changes_test += changes_test
            
            if total_changes_train == 0 and total_changes_test == 0:
                print("    No se modificó ningún valor.")
            else:
                print(f"    Total: train={AMPBConfig.COLOR_VALUE}{total_changes_train}{AMPBConfig.COLOR_RESET}, test={AMPBConfig.COLOR_VALUE}{total_changes_test}{AMPBConfig.COLOR_RESET} valores winzorizados.")
     
        # Limpiar infinitos y NaN después de transformaciones
        y_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        y_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        if has_exog_vars:
            X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Eliminar filas con NaN (mantener consistencia entre todas las series)
        def clean_nan_indices(combined_data, series_to_clean):
            combined_clean = combined_data.dropna()
            indices_to_drop = combined_data.index.difference(combined_clean.index)
            if len(indices_to_drop) > 0:
                for serie in series_to_clean:
                    serie.drop(indices_to_drop, inplace=True, errors='ignore')
        if has_exog_vars:
            # Con variables exógenas: combinar y + X para detectar NaN
            train_combined = pd.concat([y_train.to_frame('target'), X_train], axis=1)
            test_combined = pd.concat([y_test.to_frame('target'), X_test], axis=1)
            clean_nan_indices(train_combined, [y_train, X_train, y_train_original, X_train_original])
            clean_nan_indices(test_combined, [y_test, X_test, y_test_original, X_test_original])
        else:
            # Sin variables exógenas: combinar y + y_original para detectar NaN
            train_combined = pd.concat([y_train.to_frame('target'), y_train_original.to_frame('target_original')], axis=1)
            test_combined = pd.concat([y_test.to_frame('target'), y_test_original.to_frame('target_original')], axis=1)
            clean_nan_indices(train_combined, [y_train, y_train_original])
            clean_nan_indices(test_combined, [y_test, y_test_original])
        
    print(f"  Transformación completada:")
    if has_exog_vars:
        print(f"    X_train ({AMPBConfig.COLOR_VALUE}{len(X_train)}{AMPBConfig.COLOR_RESET}) / y_train ({AMPBConfig.COLOR_VALUE}{len(y_train)}{AMPBConfig.COLOR_RESET})")
        print(f"    X_test ({AMPBConfig.COLOR_VALUE}{len(X_test)}{AMPBConfig.COLOR_RESET}) / y_test ({AMPBConfig.COLOR_VALUE}{len(y_test)}{AMPBConfig.COLOR_RESET})")
    else:
        print(f"    y_train ({AMPBConfig.COLOR_VALUE}{len(y_train)}{AMPBConfig.COLOR_RESET}) / y_test ({AMPBConfig.COLOR_VALUE}{len(y_test)}{AMPBConfig.COLOR_RESET})")
        print(f"    Sin variables exógenas")
    
    # ESCALADO
    print(f"\n{AMPBConfig.COLOR_INFO}Aplicando escalado '{exog_scaling}'...{AMPBConfig.COLOR_RESET}")
    scaler = None
    y_scaler = None

    if exog_scaling is not None and exog_scaling != "None":
        if has_exog_vars:
            print(f"  A las variables exógenas y objetivo...")
        else:
            print(f"  Solo a la variable objetivo...")

        if exog_scaling == "Standard":
            if has_exog_vars:
                scaler = StandardScaler() 
            y_scaler = StandardScaler()
        elif exog_scaling == "MinMax":
            if has_exog_vars:
                scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
        
        # Escalado de variables exógenas: solo si existen
        if has_exog_vars and scaler is not None:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Mantener como DataFrame con índices y columnas correctos
            X_train[:] = X_train_scaled
            X_test[:] = X_test_scaled
        
        # Escalado de variable objetivo: fit con train, transform con test
        if y_scaler is not None:
            y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
            
            # Mantener como Series con índices correctos
            y_train[:] = y_train_scaled
            y_test[:] = y_test_scaled
    else:
        print("  No se aplica escalado.")

    scaler_columns = list(X_train.columns) if (scaler and has_exog_vars) else None
    print(f"  Variables preparadas.")

    # ALINEACIÓN
    print(f"\n{AMPBConfig.COLOR_INFO}Alineando datos de evaluación...{AMPBConfig.COLOR_RESET}")
    df_test_aligned = df_test.loc[y_test_original.index]

    if len(df_test_aligned) != len(y_test_original):
        print(f"  Advertencia: alineación de datos necesaria")
        print(f"    Registros originales en test: {AMPBConfig.COLOR_VALUE}{len(df_test)}{AMPBConfig.COLOR_RESET}")
        print(f"    Registros después de limpieza: {AMPBConfig.COLOR_VALUE}{len(y_test_original)}{AMPBConfig.COLOR_RESET}")
    elif len(df_test) != len(df_test_aligned):
        print(f"  Info: se alinearon los datos de evaluación")
        print(f"    {AMPBConfig.COLOR_VALUE}{len(df_test)} → {len(df_test_aligned)}{AMPBConfig.COLOR_RESET} registros")
    else: print("  No es necesaria la alineación")

    # LÍMITE MÁXIMO PARA PREDICCIONES
    print(f"\n{AMPBConfig.COLOR_INFO}Calculando límite máximo para predicciones...{AMPBConfig.COLOR_RESET}")
    max_historical_df = pd.concat([y_train_original, y_test_original])
    max_historical = max_historical_df.max()    
    volatility = max_historical_df.std()
    prediction_max_limit = max_historical + (3 * volatility)
    print(f"  Límite máximo: {prediction_max_limit:.2f}")

    # LÍMITE MÁXIMO DE PERDIDA
    checkDataLoss(original_train_size, original_test_size, len(y_train), len(y_test), 0.10)

    # ANÁLISIS DE CALIDAD (OPCIONAL)
    quality_results = None
    if analyze and has_exog_vars:
        print(f"\n{AMPBConfig.COLOR_INFO}Ejecutando análisis de calidad de datos...{AMPBConfig.COLOR_RESET}")
        quality_results = analyzeDataQuality(
            X_train, X_test, X_train_original, X_test_original,
            params_exog, scaler, transformation, exog_scaling
        )
        print(f"Calidad del dataset: {AMPBConfig.COLOR_VALUE}{quality_results['quality_score']}{AMPBConfig.COLOR_RESET}")
    elif analyze and not has_exog_vars:
        print(f"\n{AMPBConfig.COLOR_INFO}Análisis de calidad omitido (sin variables exógenas){AMPBConfig.COLOR_RESET}")
    
    # Retornar todo lo necesario
    return {
        'params_close': params_close,
        'params_exog': params_exog,
        'y_scaler': y_scaler,
        'exog_scaler': scaler,
        'df_test_aligned': df_test_aligned,
        'prediction_max_limit': prediction_max_limit,
        'quality_results': quality_results
    }