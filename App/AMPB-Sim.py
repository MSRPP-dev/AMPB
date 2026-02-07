# ------------------------------------------------
# Análisis de modelos predictivos en bolsa: NVIDIA
# MSRPP
# Copyright (C) 2024-2025 MegaStorm Systems
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
#
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.
#
# ------------------------------------------------------------------------------------------------
# Aplicacion AMPB Simulador
#  - Puerto 443 con cifrado seguro SSL
#  - Login seguro que evita fuerza bruta e inyeccion de codigo
# ------------------------------------------------------------------------------------------------

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, Regexp
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import escape
import os, json, logging, ssl, atexit, random
from datetime import datetime, timedelta
import pandas as pd
from ampblib import AMPBConfig, getMongoClient, getMongoCollection, closeMongoClient

# ================================================================================================
# CONFIGURACIÓN DE FECHAS DE SIMULACIÓN
# ================================================================================================
FECHA_INICIAL_SIMULACION = "2025-04-02"  # Fecha de inicio de la simulación
FECHA_FINAL_SIMULACION = "2025-05-23"    # Fecha final de la simulación

app = Flask(__name__)
app.secret_key = os.environ['FLASK_SECRET_KEY']
csrf = CSRFProtect(app) 

# Configurar logging a stdout
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Directorio base de las predicciones
MODEL_PREDS_DIR = os.path.join(os.getcwd(), "model_preds")
os.makedirs(MODEL_PREDS_DIR, exist_ok=True)

# Configuración de tickers disponibles con modelos específicos y datasets
TICKERS_CONFIG = [
    {
        'ticker': 'NVDA',
        'name': 'NVIDIA Corporation',
        'image': 'nvda-logo.png',
        'general_models': [
            {'display_name': 'SARIMAX v2.4', 'model_file': 'nvda_sarimax_v24.pkl', 'dataset_number': 14}
        ],
        'regression_models': [
            {'display_name': 'XGBoost v1.3', 'model_file': 'nvda_xgboost_v13_regr.pkl', 'dataset_number': 16}
        ],
        'classification_models': [
            {'display_name': 'Transformer v3.4', 'model_file': 'nvda_transformer_v34_clas.pkl', 'dataset_number': 123456}
        ]
    },
    {
        'ticker': 'TSLA',
        'name': 'Tesla Inc.',
        'image': 'tsla-logo.png',
        'general_models': [],
        'regression_models': [],
        'classification_models': []
    },
    {
        'ticker': 'AAPL',
        'name': 'Apple Inc.',
        'image': 'aapl-logo.png',
        'general_models': [],
        'regression_models': [],
        'classification_models': []
    }
]
    
# Hacer AMPBConfig disponible globalmente para las templates
@app.context_processor
def inject_ampb_config():
    return dict(AMPBConfig=AMPBConfig)

# Inicializar conexiones MongoDB
users_collection = getMongoCollection('app_usuarios')
transactions_collection = getMongoCollection('app_transacciones')

# Asegurar que el usuario admin existe en la base de datos
def ensure_admin_user():    
    admin_user = users_collection.find_one({'username': 'admin'})
    
    if not admin_user:
        # Crear usuario admin con contraseña por defecto
        admin_doc = {
            'username': 'admin',
            'password_hash': generate_password_hash('UEMRPP'),
            'primer_login': datetime.now(),
            'ultimo_login': datetime.now(),
            'simulaciones': {},
            'predicciones': {},
            'is_admin': True
        }
        users_collection.insert_one(admin_doc)
        logger.info(" Usuario admin creado con contraseña por defecto: UEMRPP")

# Inicializar usuarios al arrancar la aplicación
ensure_admin_user()

# Formulario de login con validación
class LoginForm(FlaskForm):
    username = StringField(
        'Usuario',
        validators=[
            DataRequired(),
            Length(min=3, max=25),
            Regexp(r'^[A-Za-z0-9_]+$', message="Solo letras, números y guion bajo")
        ]
    )
    password = PasswordField(
        'Contraseña',
        validators=[DataRequired(), Length(min=6, max=128)]
    )
    submit = SubmitField('Entrar')

# Verifica si un ticker tiene al menos un modelo configurado
def has_models(ticker_config):
    return (len(ticker_config.get('general_models', [])) > 0 or 
            len(ticker_config.get('regression_models', [])) > 0 or 
            len(ticker_config.get('classification_models', [])) > 0)

# Actualizar información de login del usuario
def update_user_login(username):    
    current_datetime = datetime.now()
    
    # Actualizar último login (el usuario ya debe existir en la BD)
    users_collection.update_one(
        {'username': username},
        {'$set': {'ultimo_login': current_datetime}}
    )

#   Busca los siguientes 'count' días hábiles (con datos) a partir de 'from_date_str' para un ticker.
def find_next_valid_days(ticker, from_date_str, count=1, max_attempts=30):
    results = []
    from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
    max_date = datetime.strptime(FECHA_FINAL_SIMULACION, '%Y-%m-%d')
    attempts = 0
    next_date = from_date + timedelta(days=1)

    while next_date <= max_date and attempts < max_attempts and len(results) < count:
        next_date_str = next_date.strftime('%Y-%m-%d')
        try:
            data = getData(ticker, next_date_str)
            if data and data[-1]['date'] == next_date_str:
                results.append(next_date_str)
        except Exception as e:
            logger.error(f"Error verificando datos para {next_date_str}: {e}")

        next_date += timedelta(days=1)
        attempts += 1

    return results

# Ruta de login
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    ip = request.remote_addr

    if form.validate_on_submit():
        # Escapar la entrada para evitar inyección en logs o templates
        username = escape(form.username.data)
        password = form.password.data

        # Buscar usuario en la base de datos
        user = users_collection.find_one({'username': username})
        
        if user and check_password_hash(user['password_hash'], password):
            # Login correcto
            session['user'] = username
            update_user_login(username)
            logger.info(f" Login correcto: usuario={username} ip={ip}")
            return redirect(url_for('home'))
        else:
            # Login fallido
            logger.warning(f" Login fallido: usuario={username} ip={ip}")
            flash('Usuario o contraseña incorrectos', 'error')

    return render_template('login.html', form=form)
    

# Ruta de logout
@app.route('/logout')
def logout():
    user = session.pop('user', None)
    if user:
        logger.info(f"User logged out: {user}")
    return redirect(url_for('login'))

# Página principal con tickers
@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    # Agregar información sobre disponibilidad de modelos
    tickers_with_availability = []
    for ticker in TICKERS_CONFIG:
        ticker_copy = ticker.copy()
        ticker_copy['has_models'] = has_models(ticker)
        tickers_with_availability.append(ticker_copy)
    
    return render_template('home.html', username=session['user'], tickers=tickers_with_availability)

# Página de configuración
@app.route('/config')
def config():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    current_user = session['user']
    is_admin = users_collection.find_one({'username': current_user, 'is_admin': True}) is not None
    
    # Obtener información de usuarios desde MongoDB
    usuarios_info = []
    
    if is_admin:
        # Admin ve todos los usuarios
        users_cursor = users_collection.find({})
    else:
        # Usuario normal solo ve su información
        users_cursor = users_collection.find({'username': current_user})
    
    for user in users_cursor:
        # Calcular estadísticas de transacciones
        user_transactions = list(transactions_collection.find({'usuario': user['username']}))
        total_transacciones = len(user_transactions)
        
        # Calcular ganancias/pérdidas por ticker
        ticker_stats = {}
        
        for trans in user_transactions:
            ticker = trans['ticker']
            if ticker not in ticker_stats:
                ticker_stats[ticker] = {'compras': 0, 'ventas': 0, 'balance': 0}
            
            cantidad = trans['cantidad']
            precio = trans['precio_unitario']
            valor = cantidad * precio
            
            if trans['tipo'] == 'compra':
                ticker_stats[ticker]['compras'] += valor
                ticker_stats[ticker]['balance'] -= valor
            else:  # venta
                ticker_stats[ticker]['ventas'] += valor
                ticker_stats[ticker]['balance'] += valor
        
        # Calcular total general sumando todos los balances por ticker
        total_general = sum(stats['balance'] for stats in ticker_stats.values())
        
        usuarios_info.append({
            'username': user['username'],
            'primer_login': user.get('primer_login', 'N/A'),
            'ultimo_login': user.get('ultimo_login', 'N/A'),
            'simulaciones': user.get('simulaciones', {}),
            'total_transacciones': total_transacciones,
            'ticker_stats': ticker_stats,
            'total_general': total_general,
            'is_admin': user.get('is_admin', False)
        })
    
    return render_template('config.html', 
                         username=session['user'], 
                         tickers=TICKERS_CONFIG,
                         usuarios_info=usuarios_info,
                         is_admin=is_admin)

# Ruta para página de trading
@app.route('/trading/<ticker>')
def trading(ticker):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    current_user = session['user']
    
    # Buscar el ticker en la configuración
    ticker_info = None
    for t in TICKERS_CONFIG:
        if t['ticker'] == ticker:
            ticker_info = t
            break
    
    if not ticker_info or not has_models(ticker_info):
        flash(f'Ticker {ticker} no disponible', 'error')
        return redirect(url_for('home'))
    
    # Obtener fecha actual de simulación del usuario
    user_doc = users_collection.find_one({'username': current_user})
    current_simulation_date = user_doc.get('simulaciones', {}).get(ticker)
    
    # Si no existe fecha, establecer fecha inicial
    if not current_simulation_date:
        users_collection.update_one(
            {'username': current_user},
            {'$set': {f'simulaciones.{ticker}': FECHA_INICIAL_SIMULACION}}
        )
        current_simulation_date = FECHA_INICIAL_SIMULACION
    
    return render_template('trading.html', 
                         ticker=ticker, 
                         ticker_info=ticker_info, 
                         username=session['user'],
                         current_date=current_simulation_date,
                         fecha_inicial=FECHA_INICIAL_SIMULACION,
                         fecha_final=FECHA_FINAL_SIMULACION)

# Ruta para aumentar un dia el ticker
@app.route('/next_day/<ticker>')
def next_day(ticker):
    if 'user' not in session:
        return redirect(url_for('login'))

    current_user = session['user']
    user_doc = users_collection.find_one({'username': current_user})
    current_date_str = user_doc.get('simulaciones', {}).get(ticker, FECHA_INICIAL_SIMULACION)

    next_days = find_next_valid_days(ticker, current_date_str, count=1)

    if next_days:
        next_date_str = next_days[0]
        users_collection.update_one(
            {'username': current_user},
            {'$set': {f'simulaciones.{ticker}': next_date_str}}
        )
        flash(f'Fecha avanzada a {next_date_str} (siguiente día hábil)', 'info')
    else:
        flash('No se encontraron más días hábiles en el período disponible', 'warning')

    return redirect(url_for('trading', ticker=ticker))


# Función temporal que leerá CSV de NVIDIA
#    Parámetro: max_date (string YYYY-MM-DD)
#    Retorna: DataFrame con datos hasta max_date
# El problema es que los datos de Google Trends requieren pasos manuales, mejorar esto esta previsto para futuras lineas de actuacion
def getData(ticker, max_date):
    nombre_archivo = "NVDA_2015-01-05_2025-05-23_SA.csv"
    datos = pd.read_csv(nombre_archivo)
    
    # Convertir a datetime y filtrar
    datos['Date'] = pd.to_datetime(datos['Date'])
    max_date = pd.to_datetime(max_date)  # Convertir el parámetro a datetime
    
    # Filtrar datos hasta max_date (incluida)
    datos_filtrados = datos[datos['Date'] <= max_date]
    
    # Convertir DataFrame a lista de diccionarios para JSON
    result = []
    for _, row in datos_filtrados.iterrows():
        result.append({
            'date': row['Date'].strftime('%Y-%m-%d'),
            'close': float(row['Close'])
        })
    
    return result
    
    
@app.route('/get_chart_data/<ticker>')
def get_chart_data(ticker):
    if 'user' not in session:
        return jsonify({'error': 'No autorizado'}), 401

    current_user = session['user']
    user_doc = users_collection.find_one({'username': current_user})
    max_date = user_doc.get('simulaciones', {}).get(ticker, FECHA_INICIAL_SIMULACION)

    data = getData(ticker, max_date)

    start_date_param = request.args.get('start_date')
    if start_date_param:
        start_date = start_date_param
    else:
        current_date = datetime.strptime(max_date, '%Y-%m-%d')
        default_start = current_date - timedelta(weeks=4)
        start_date = default_start.strftime('%Y-%m-%d')

    if data:
        min_date_in_data = min(d['date'] for d in data)
        if start_date < min_date_in_data:
            start_date = min_date_in_data

    filtered_data = [d for d in data if d['date'] >= start_date]

    user_transactions = list(transactions_collection.find({
        'usuario': current_user,
        'ticker': ticker,
        'fecha_simulacion': {
            '$gte': start_date,
            '$lte': max_date
        }
    }).sort('fecha_simulacion', 1))

    buy_transactions = []
    sell_transactions = []
    for trans in user_transactions:
        transaction_data = {
            'x': trans['fecha_simulacion'],
            'y': trans['precio_unitario'],
            'cantidad': trans['cantidad'],
            'fecha_real': trans['fecha_simulacion']
        }
        if trans['tipo'] == 'compra':
            buy_transactions.append(transaction_data)
        else:
            sell_transactions.append(transaction_data)

    user_predictions = user_doc.get('predicciones', {}).get(ticker, [])
    prediction_data = []

    max_date_obj = datetime.strptime(max_date, '%Y-%m-%d')
    tomorrow = max_date_obj + timedelta(days=1)
    tomorrow_str = tomorrow.strftime('%Y-%m-%d')

    # Obtener la siguiente fecha válida (podemos pedir n días si queremos)
    next_valid_dates = find_next_valid_days(ticker, max_date, count=1)

    # Límite superior para incluir predicciones: último día válido futuro si existe; si no, mañana
    max_pred_date = next_valid_dates[-1] if next_valid_dates else tomorrow_str

    for pred in user_predictions:
        fecha_prediccion = pred.get('fecha_prediccion', pred.get('fecha'))
        fecha_generacion = pred.get('fecha_generacion', pred.get('fecha'))
        # Incluir predicciones visibles hasta el último día válido futuro mostrado en la gráfica
        if start_date <= fecha_prediccion <= max_pred_date:
            prediction_data.append({
                'fecha_prediccion': fecha_prediccion,
                'fecha_generacion': fecha_generacion,
                'y': pred['prediccion'],
                'modelo': pred['modelo'],
                'anchor_y': pred.get('anchor_y'),
                'is_up': pred.get('is_up')
            })

    return jsonify({
        'data': filtered_data,
        'min_date': min_date_in_data if data else start_date,
        'max_date': max_date,
        'buy_transactions': buy_transactions,
        'sell_transactions': sell_transactions,
        'predictions': prediction_data,
        'next_valid_dates': next_valid_dates
    })



@app.route('/get_position_data/<ticker>')
def get_position_data(ticker):
    if 'user' not in session:
        return jsonify({'error': 'No autorizado'}), 401
    
    current_user = session['user']
    
    # Obtener transacciones del usuario para este ticker
    user_transactions = list(transactions_collection.find({
        'usuario': current_user, 
        'ticker': ticker
    }).sort('fecha', 1))  # Orden cronológico
    
    # Variables para cálculos
    shares_owned = 0
    weighted_avg_cost = 0.0
    total_invested = 0.0  # Dinero gastado en compras
    total_received = 0.0  # Dinero recibido por ventas
    realized_gains = 0.0  # Ganancias de ventas ejecutadas
    
    # Procesar transacciones en orden cronológico
    for trans in user_transactions:
        quantity = trans['cantidad']
        price = trans['precio_unitario']
        
        if trans['tipo'] == 'compra':
            # Actualizar costo promedio ponderado
            if shares_owned > 0:
                total_cost_before = shares_owned * weighted_avg_cost
                total_cost_new = quantity * price
                weighted_avg_cost = (total_cost_before + total_cost_new) / (shares_owned + quantity)
            else:
                weighted_avg_cost = price
            
            shares_owned += quantity
            total_invested += quantity * price
            
        elif trans['tipo'] == 'venta':
            # Calcular ganancia realizada de esta venta
            if shares_owned > 0:
                gain_loss = (price - weighted_avg_cost) * quantity
                realized_gains += gain_loss
                shares_owned -= quantity
                total_received += quantity * price
    
    # Si no hay acciones, el costo promedio debería ser 0
    if shares_owned == 0:
        weighted_avg_cost = 0.0
    
    # Obtener precio actual
    user_doc = users_collection.find_one({'username': current_user})
    current_date = user_doc.get('simulaciones', {}).get(ticker, FECHA_INICIAL_SIMULACION)
    data = getData(ticker, current_date)
    current_price = data[-1]['close'] if data else 0
    
    # Calcular métricas finales
    position_value = shares_owned * current_price
    unrealized_gains = shares_owned * (current_price - weighted_avg_cost) if shares_owned > 0 else 0
    net_cash_flow = total_received - total_invested  # Negativo = más invertido, Positivo = más retirado
    
    return jsonify({
        'shares': shares_owned,
        'current_price': current_price,
        'position_value': position_value,
        'avg_cost': weighted_avg_cost,
        'realized_gains': realized_gains,
        'unrealized_gains': unrealized_gains,
        'net_cash_flow': net_cash_flow,
        'total_invested': total_invested,
        'total_received': total_received
    })

@app.route('/execute_trade/<ticker>', methods=['POST'])
def execute_trade(ticker):
    if 'user' not in session:
        return jsonify({'error': 'No autorizado'}), 401
    
    current_user = session['user']
    trade_type = request.json.get('type')  # 'compra' o 'venta'
    quantity = int(request.json.get('quantity', 0))
    
    if quantity <= 0:
        return jsonify({'error': 'Cantidad debe ser mayor a 0'}), 400
    
    # Obtener precio actual
    user_doc = users_collection.find_one({'username': current_user})
    current_date = user_doc.get('simulaciones', {}).get(ticker, FECHA_INICIAL_SIMULACION)
    data = getData(ticker, current_date)
    current_price = data[-1]['close'] if data else 0
    
    # Para ventas, verificar que tenga suficientes acciones
    if trade_type == 'venta':
        user_transactions = list(transactions_collection.find({
            'usuario': current_user, 
            'ticker': ticker
        }))
        
        total_shares = 0
        for trans in user_transactions:
            if trans['tipo'] == 'compra':
                total_shares += trans['cantidad']
            elif trans['tipo'] == 'venta':
                total_shares -= trans['cantidad']
        
        if quantity > total_shares:
            return jsonify({'error': f'Solo tienes {total_shares} acciones disponibles'}), 400
    
    # Registrar transacción
    transaction = {
        'usuario': current_user,
        'ticker': ticker,
        'tipo': trade_type,
        'cantidad': quantity,
        'precio_unitario': current_price,
        'fecha': datetime.now(),
        'fecha_simulacion': current_date
    }
    
    transactions_collection.insert_one(transaction)
    
    # Calcular nuevo balance
    total_cost = quantity * current_price
    action_text = 'Comprado' if trade_type == 'compra' else 'Vendido'
    
    return jsonify({
        'success': True,
        'message': f'{action_text} {quantity} acciones de {ticker} a ${current_price:.2f} c/u',
        'total_cost': total_cost
    })

@app.route('/get_transaction_history/<ticker>')
def get_transaction_history(ticker):
    if 'user' not in session:
        return jsonify({'error': 'No autorizado'}), 401
    
    current_user = session['user']
    
    # Obtener transacciones del usuario para este ticker, ordenadas por fecha
    user_transactions = list(transactions_collection.find({
        'usuario': current_user, 
        'ticker': ticker
    }).sort('fecha', -1))  # Más recientes primero para mostrar
    
    # Procesar en orden cronológico para calcular correctamente
    transactions_chrono = sorted(user_transactions, key=lambda x: x['fecha'])
    
    # Variables para tracking del costo promedio ponderado
    shares_owned = 0
    weighted_avg_cost = 0.0
    transaction_gains = {}  # Diccionario para almacenar ganancias por transacción
    
    # Calcular ganancias/pérdidas usando costo promedio ponderado
    for trans in transactions_chrono:
        trans_id = str(trans['_id'])
        quantity = trans['cantidad']
        price = trans['precio_unitario']
        
        if trans['tipo'] == 'compra':
            # Actualizar costo promedio ponderado al comprar
            if shares_owned > 0:
                total_cost_before = shares_owned * weighted_avg_cost
                total_cost_new = quantity * price
                weighted_avg_cost = (total_cost_before + total_cost_new) / (shares_owned + quantity)
            else:
                weighted_avg_cost = price
            
            shares_owned += quantity
            transaction_gains[trans_id] = 0  # Las compras no tienen ganancia
            
        elif trans['tipo'] == 'venta':
            # Calcular ganancia/pérdida al vender usando costo promedio actual
            if shares_owned > 0:
                gain_loss = (price - weighted_avg_cost) * quantity
                transaction_gains[trans_id] = gain_loss
                shares_owned -= quantity
            else:
                transaction_gains[trans_id] = 0  # No debería pasar, pero por seguridad
    
    # Formatear transacciones para el frontend (orden inverso = más recientes primero)
    formatted_transactions = []
    for trans in user_transactions:
        trans_id = str(trans['_id'])
        quantity = trans['cantidad']
        price = trans['precio_unitario']
        total_value = quantity * price
        gain_loss = transaction_gains.get(trans_id, 0)
        
        formatted_transactions.append({
            'fecha': trans['fecha_simulacion'],  # Fecha de simulación (Y-M-D)
            'tipo': trans['tipo'],   
            'cantidad': quantity,
            'precio': price,
            'total': total_value,
            'gain_loss': gain_loss  # 0 para compras, valor real para ventas
        })
    
    return jsonify({'transactions': formatted_transactions})

# ================================================================================================
# RUTAS PARA PREDICCIONES
# ================================================================================================

# Carga la predicción para 'target_date_str' desde un CSV:
#    model_preds/<TICKER>/<model_file_sin_ext>.csv  con columnas: date,prediction 
def load_prediction_from_csv(ticker, model_file, target_date_str):
    base = os.path.splitext(model_file)[0] + ".csv"
    csv_path = os.path.join(MODEL_PREDS_DIR, ticker.upper(), base)
    if not os.path.exists(csv_path):
        logger.error(f"Fichero {csv_path}")
        return None  # No hay CSV -> sin predicción disponible
    try:
        df = pd.read_csv(csv_path)
        # normalizar columna fecha
        if 'date' not in df.columns or 'prediction' not in df.columns:
            return None
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        row = df.loc[df['date'] == target_date_str]
        if row.empty:
            return None
        return float(row.iloc[0]['prediction'])
    except Exception as e:
        logger.error(f"Error leyendo CSV de predicción {csv_path}: {e}")
        return None
        
 # Generar predicción usando modelos seleccionados
@app.route('/generate_prediction/<ticker>', methods=['POST'])
def generate_prediction(ticker):   
    if 'user' not in session:
        return jsonify({'error': 'No autorizado'}), 401
    
    current_user = session['user']
    selected_models = request.json.get('models', [])
    
    if not selected_models:
        return jsonify({'error': 'Debe seleccionar al menos un modelo'}), 400
    
    # Obtener fecha actual de simulación
    user_doc = users_collection.find_one({'username': current_user})
    current_date = user_doc.get('simulaciones', {}).get(ticker, FECHA_INICIAL_SIMULACION)
    
    # Buscar el siguiente día hábil con datos
    current_date_obj = datetime.strptime(current_date, '%Y-%m-%d')
    max_date = datetime.strptime(FECHA_FINAL_SIMULACION, '%Y-%m-%d')
    prediction_date = current_date_obj + timedelta(days=1)
    
    # Buscar un día con datos (máximo 30 intentos)
    attempts = 0
    prediction_date_str = None
    while prediction_date <= max_date and attempts < 30:
        test_date_str = prediction_date.strftime('%Y-%m-%d')
        try:
            # Verificar si hay datos para esta fecha
            test_data = getData(ticker, test_date_str)
            if test_data and len(test_data) > 0:
                # Verificar si el último dato corresponde a la fecha buscada
                last_date = test_data[-1]['date']
                if last_date == test_date_str:
                    prediction_date_str = test_date_str
                    break
        except Exception as e:
            logger.error(f"Error verificando datos para predicción en {test_date_str}: {e}")
        
        prediction_date += timedelta(days=1)
        attempts += 1
    
    # Si no se encontró un día hábil futuro
    if not prediction_date_str:
        return jsonify({
            'error': 'No hay días hábiles futuros disponibles para generar predicciones'
        }), 400
    
    # Buscar ticker config
    ticker_config = None
    for t in TICKERS_CONFIG:
        if t['ticker'] == ticker:
            ticker_config = t
            break
    
    if not ticker_config:
        return jsonify({'error': 'Ticker no encontrado'}), 400
    
    # Obtener precio actual para generar predicción relativa
    data = getData(ticker, current_date)
    current_price = data[-1]['close'] if data else 100
    
    generated_predictions = []
    
    # Generar predicciones para cada modelo seleccionado
    for model_name in selected_models:
        # Buscar el modelo en la configuración
        model_info = None
        for category in ['general_models', 'regression_models', 'classification_models']:
            for model in ticker_config.get(category, []):
                if model['display_name'] == model_name:
                    model_info = model
                    break
            if model_info:
                break
        
        if not model_info:
            continue
        
        # TODO (futuro): aquí se debería cargar el modelo real y generar la predicción online.
        # Ahora: cargar la predicción diaria precalculada desde CSV de /model_preds/<TICKER>/
        prediction_value = load_prediction_from_csv(ticker, model_info['model_file'], prediction_date_str)
        if prediction_value is None:
            logger.warning(f"Sin predicción para {ticker} {model_name} en {prediction_date_str}.")
            continue
        
        prediction = {
            'fecha_generacion': current_date,  # Fecha en que se hace la predicción
            'fecha_prediccion': prediction_date_str,  # Fecha para la que se predice (día siguiente)
            'modelo': model_name,
            'prediccion': round(prediction_value, 2),
            'dataset_number': model_info['dataset_number'],
            'model_file': model_info['model_file'],
            'anchor_y': float(current_price), 
            'is_up': bool(prediction_value > current_price),
            'generated_at': datetime.now().isoformat()
        }        
        generated_predictions.append(prediction)
    
    # Obtener predicciones existentes
    existing_predictions = user_doc.get('predicciones', {}).get(ticker, [])
    
    # Filtrar predicciones existentes para esta fecha y modelos
    filtered_predictions = []
    for pred in existing_predictions:
        # Mantener predicciones que no sean para esta fecha de predicción o no sean de los modelos seleccionados
        pred_fecha = pred.get('fecha_prediccion', pred.get('fecha'))  # Compatibilidad con formato anterior
        if pred_fecha != prediction_date_str or pred['modelo'] not in selected_models:
            filtered_predictions.append(pred)
    
    # Añadir nuevas predicciones
    filtered_predictions.extend(generated_predictions)
    
    # Actualizar en la base de datos
    users_collection.update_one(
        {'username': current_user},
        {'$set': {f'predicciones.{ticker}': filtered_predictions}}
    )
    
    logger.info(f"Usuario {current_user} generó {len(generated_predictions)} predicciones para {ticker} desde {current_date} hacia {prediction_date_str}")
    
    return jsonify({
        'success': True,
        'message': f'Se generaron {len(generated_predictions)} predicciones para {prediction_date_str}',
        'predictions': generated_predictions
    })

# Borrar todas las predicciones para un ticker
@app.route('/delete_predictions/<ticker>', methods=['POST'])
def delete_predictions(ticker):    
    if 'user' not in session:
        return jsonify({'error': 'No autorizado'}), 401
    
    current_user = session['user']
    
    # Borrar predicciones del usuario para este ticker
    users_collection.update_one(
        {'username': current_user},
        {'$unset': {f'predicciones.{ticker}': ''}}
    )
    
    logger.info(f"Usuario {current_user} borró todas las predicciones para {ticker}")
    
    return jsonify({
        'success': True,
        'message': f'Se borraron todas las predicciones para {ticker}'
    })

# Obtener modelos disponibles para un ticker
@app.route('/get_available_models/<ticker>')
def get_available_models(ticker):
    if 'user' not in session:
        return jsonify({'error': 'No autorizado'}), 401
    
    # Buscar ticker config
    ticker_config = None
    for t in TICKERS_CONFIG:
        if t['ticker'] == ticker:
            ticker_config = t
            break
    
    if not ticker_config:
        return jsonify({'error': 'Ticker no encontrado'}), 400
    
    models = {
        'general_models': ticker_config.get('general_models', []),
        'regression_models': ticker_config.get('regression_models', []),
        'classification_models': ticker_config.get('classification_models', [])
    }
    
    return jsonify(models)

# ================================================================================================
# RUTAS DE ADMINISTRACIÓN  
# ================================================================================================
    
# Ruta para reiniciar simulaciones y predicciones de un ticker
@app.route('/reset_simulation/<ticker>')
def reset_simulation(ticker):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    current_user = session['user']
    is_admin = users_collection.find_one({'username': current_user, 'is_admin': True}) is not None
    
    if is_admin:
        # Admin puede resetear para todos los usuarios
        users_collection.update_many(
            {},
            {'$unset': {f'simulaciones.{ticker}': '', f'predicciones.{ticker}': ''}}
        )
        flash(f'Fecha de simulación y predicciones de {ticker} reiniciadas para todos los usuarios', 'info')
    else:
        # Usuario normal solo puede resetear su propia simulación
        users_collection.update_one(
            {'username': current_user},
            {'$unset': {f'simulaciones.{ticker}': '', f'predicciones.{ticker}': ''}}
        )
        flash(f'Tu fecha de simulación y predicciones de {ticker} han sido reiniciadas', 'info')
    
    return redirect(url_for('config'))

# Ruta para reiniciar transacciones de un ticker
@app.route('/reset_transactions/<ticker>')
def reset_transactions(ticker):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    current_user = session['user']
    is_admin = users_collection.find_one({'username': current_user, 'is_admin': True}) is not None
    
    if is_admin:
        # Admin puede borrar todas las transacciones del ticker
        result = transactions_collection.delete_many({'ticker': ticker})
        flash(f'Se eliminaron {result.deleted_count} transacciones de {ticker} (todos los usuarios)', 'info')
    else:
        # Usuario normal solo puede borrar sus propias transacciones
        result = transactions_collection.delete_many({'ticker': ticker, 'usuario': current_user})
        flash(f'Se eliminaron {result.deleted_count} de tus transacciones de {ticker}', 'info')
    
    return redirect(url_for('config'))

# Ruta para crear nuevo usuario (solo admin)
@app.route('/create_user', methods=['POST'])
def create_user():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    current_user = session['user']
    is_admin = users_collection.find_one({'username': current_user, 'is_admin': True}) is not None
    
    if not is_admin:
        flash('No tienes permisos para crear usuarios', 'error')
        return redirect(url_for('config'))
    
    new_username = escape(request.form.get('username', '').strip())
    new_password = request.form.get('password', '').strip()
    
    if not new_username or not new_password:
        flash('Usuario y contraseña son requeridos', 'error')
        return redirect(url_for('config'))
    
    if len(new_username) < 3 or len(new_password) < 6:
        flash('Usuario mínimo 3 caracteres, contraseña mínimo 6 caracteres', 'error')
        return redirect(url_for('config'))
    
    # Verificar si el usuario ya existe
    existing_user = users_collection.find_one({'username': new_username})
    if existing_user:
        flash(f'El usuario {new_username} ya existe', 'error')
        return redirect(url_for('config'))
    
    # Crear nuevo usuario
    user_doc = {
        'username': new_username,
        'password_hash': generate_password_hash(new_password),
        'primer_login': datetime.now(),
        'ultimo_login': datetime.now(),
        'simulaciones': {},
        'predicciones': {},
        'is_admin': False
    }
    users_collection.insert_one(user_doc)
    
    flash(f'Usuario {new_username} creado exitosamente', 'info')
    logger.info(f"Admin {current_user} creó usuario: {new_username}")
    return redirect(url_for('config'))

# Ruta para eliminar usuario (solo admin)
@app.route('/delete_user/<username>')
def delete_user(username):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    current_user = session['user']
    is_admin = users_collection.find_one({'username': current_user, 'is_admin': True}) is not None
    
    if not is_admin:
        flash('No tienes permisos para eliminar usuarios', 'error')
        return redirect(url_for('config'))
    
    if username == 'admin':
        flash('No se puede eliminar el usuario admin', 'error')
        return redirect(url_for('config'))
    
    if username == current_user:
        flash('No puedes eliminarte a ti mismo', 'error')
        return redirect(url_for('config'))
    
    # Eliminar usuario y sus transacciones
    user_result = users_collection.delete_one({'username': username})
    trans_result = transactions_collection.delete_many({'usuario': username})
    
    if user_result.deleted_count > 0:
        flash(f'Usuario {username} eliminado exitosamente (y {trans_result.deleted_count} transacciones)', 'info')
        logger.info(f"Admin {current_user} eliminó usuario: {username}")
    else:
        flash(f'Usuario {username} no encontrado', 'error')
    
    return redirect(url_for('config'))

# Ruta para cambiar contraseña
@app.route('/change_password', methods=['POST'])
def change_password():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    current_user = session['user']
    current_password = request.form.get('current_password', '')
    new_password = request.form.get('new_password', '')
    confirm_password = request.form.get('confirm_password', '')
    
    if not current_password or not new_password or not confirm_password:
        flash('Todos los campos son requeridos', 'error')
        return redirect(request.referrer or url_for('home'))
    
    if new_password != confirm_password:
        flash('Las contraseñas nuevas no coinciden', 'error')
        return redirect(request.referrer or url_for('home'))
    
    if len(new_password) < 6:
        flash('La contraseña debe tener al menos 6 caracteres', 'error')
        return redirect(request.referrer or url_for('home'))
    
    # Verificar contraseña actual
    user = users_collection.find_one({'username': current_user})
    if not user or not check_password_hash(user['password_hash'], current_password):
        flash('Contraseña actual incorrecta', 'error')
        return redirect(request.referrer or url_for('home'))
    
    # Actualizar contraseña
    users_collection.update_one(
        {'username': current_user},
        {'$set': {'password_hash': generate_password_hash(new_password)}}
    )
    
    flash('Contraseña cambiada exitosamente', 'info')
    logger.info(f"Usuario {current_user} cambió su contraseña")
    return redirect(request.referrer or url_for('home'))

# Registrar función de cierre
@atexit.register
def cleanup():
    closeMongoClient()
    logger.info("MongoDB connection closed")

if __name__ == '__main__':
    # SSL setup
    cert_path = '/app/ssl_certs/cert.pem'
    key_path = '/app/ssl_certs/key.pem'
    password = os.environ.get('SSL_KEY_PASSWORD')

    ssl_context = None
    if os.path.exists(cert_path) and os.path.exists(key_path):
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path, password=password)
            print(f" SSL cargado desde {cert_path} / {key_path}")
        except Exception as e:
            print(f" Error cargando SSL: {e}")
            ssl_context = None

    if ssl_context:
        print(f" Iniciando Flask con SSL en puerto 443")
        app.run(host='0.0.0.0', port=443, ssl_context=ssl_context)
    else:
        print(f"️ SSL no disponible, iniciando en puerto 5000 sin cifrado")
        app.run(host='0.0.0.0', port=5000)