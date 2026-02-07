/**
 * AMPB Simulador - Utilidades JavaScript
 * Funciones auxiliares para la aplicación de trading
 */

// ================================================================================================
// CONSTANTES Y CONFIGURACIÓN
// ================================================================================================
const AMPB_CONFIG = {
    CHART_COLORS: {
        price: '#1e3a8a',
        buy: '#28a745',
        sell: '#ff9800',
        predictions: ['#e74c3c', '#9b59b6', '#f39c12', '#2ecc71', '#3498db']
    },
    ANIMATION_DURATION: 300,
    DEBOUNCE_DELAY: 500,
    AUTO_REFRESH_INTERVAL: 30000 // 30 segundos
};

// ================================================================================================
// FUNCIONES DE VALIDACIÓN
// ================================================================================================

/**
 * Valida que una cantidad sea un número válido y positivo
 * @param {string|number} value - Valor a validar
 * @param {number} min - Valor mínimo permitido
 * @param {number} max - Valor máximo permitido
 * @returns {number|null} - Número validado o null si es inválido
 */
function validateQuantity(value, min = 1, max = Infinity) {
    const num = parseFloat(value);
    if (isNaN(num) || num < min || num > max) {
        return null;
    }
    return Math.floor(num); // Asegurar que sea entero
}

/**
 * Valida que un precio sea válido
 * @param {string|number} value - Precio a validar
 * @returns {number|null} - Precio validado o null si es inválido
 */
function validatePrice(value) {
    const price = parseFloat(value);
    if (isNaN(price) || price <= 0) {
        return null;
    }
    return Math.round(price * 100) / 100; // Redondear a 2 decimales
}

/**
 * Valida formato de fecha YYYY-MM-DD
 * @param {string} dateString - Fecha a validar
 * @returns {boolean} - True si la fecha es válida
 */
function validateDateFormat(dateString) {
    const regex = /^\d{4}-\d{2}-\d{2}$/;
    if (!regex.test(dateString)) return false;
    
    const date = new Date(dateString);
    return date instanceof Date && !isNaN(date);
}

// ================================================================================================
// FUNCIONES DE FORMATEO
// ================================================================================================

/**
 * Formatea un número como moneda USD
 * @param {number} amount - Cantidad a formatear
 * @param {boolean} showSign - Mostrar signo + para números positivos
 * @returns {string} - Cantidad formateada
 */
function formatCurrency(amount, showSign = false) {
    const formatted = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(Math.abs(amount));
    
    if (showSign && amount > 0) {
        return '+' + formatted;
    } else if (amount < 0) {
        return '-' + formatted;
    }
    return formatted;
}

/**
 * Formatea un número con separadores de miles
 * @param {number} num - Número a formatear
 * @param {number} decimals - Número de decimales
 * @returns {string} - Número formateado
 */
function formatNumber(num, decimals = 0) {
    return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    }).format(num);
}

/**
 * Formatea fecha para mostrar
 * @param {string} dateString - Fecha en formato YYYY-MM-DD
 * @param {boolean} showTime - Incluir hora si está disponible
 * @returns {string} - Fecha formateada
 */
function formatDisplayDate(dateString, showTime = false) {
    const date = new Date(dateString);
    
    if (showTime) {
        return date.toLocaleDateString('es-ES', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
    
    return date.toLocaleDateString('es-ES', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit'
    });
}

/**
 * Calcula porcentaje de cambio entre dos valores
 * @param {number} oldValue - Valor anterior
 * @param {number} newValue - Valor nuevo
 * @returns {number} - Porcentaje de cambio
 */
function calculatePercentageChange(oldValue, newValue) {
    if (oldValue === 0) return 0;
    return ((newValue - oldValue) / oldValue) * 100;
}

// ================================================================================================
// FUNCIONES DE INTERACCIÓN CON API
// ================================================================================================

/**
 * Realiza una petición POST con manejo de errores
 * @param {string} url - URL de la petición
 * @param {object} data - Datos a enviar
 * @param {object} options - Opciones adicionales
 * @returns {Promise} - Promesa con la respuesta
 */
async function apiPost(url, data = {}, options = {}) {
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken(),
                ...options.headers
            },
            body: JSON.stringify(data),
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API POST Error:', error);
        throw error;
    }
}

/**
 * Realiza una petición GET con manejo de errores
 * @param {string} url - URL de la petición
 * @param {object} options - Opciones adicionales
 * @returns {Promise} - Promesa con la respuesta
 */
async function apiGet(url, options = {}) {
    try {
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'X-CSRFToken': getCSRFToken(),
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API GET Error:', error);
        throw error;
    }
}

// ================================================================================================
// FUNCIONES DE UI
// ================================================================================================

/**
 * Muestra un spinner de carga en un botón
 * @param {HTMLElement} button - Elemento del botón
 * @param {string} loadingText - Texto durante la carga
 * @returns {Function} - Función para restablecer el botón
 */
function showButtonLoading(button, loadingText = 'Cargando...') {
    const originalHTML = button.innerHTML;
    const originalDisabled = button.disabled;
    
    button.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${loadingText}`;
    button.disabled = true;
    
    return function resetButton() {
        button.innerHTML = originalHTML;
        button.disabled = originalDisabled;
    };
}

/**
 * Anima un elemento con efecto de pulso
 * @param {HTMLElement} element - Elemento a animar
 * @param {string} color - Color del pulso
 */
function pulseElement(element, color = '#C00') {
    element.style.transition = 'box-shadow 0.3s ease';
    element.style.boxShadow = `0 0 20px ${color}`;
    
    setTimeout(() => {
        element.style.boxShadow = '';
    }, 300);
}

/**
 * Actualiza un elemento con efecto de fade
 * @param {HTMLElement} element - Elemento a actualizar
 * @param {string} newContent - Nuevo contenido
 * @param {Function} callback - Función a ejecutar después del fade
 */
function fadeUpdateElement(element, newContent, callback = null) {
    element.style.transition = 'opacity 0.3s ease';
    element.style.opacity = '0.5';
    
    setTimeout(() => {
        element.innerHTML = newContent;
        element.style.opacity = '1';
        if (callback) callback();
    }, 150);
}

/**
 * Resalta cambios en un valor numérico
 * @param {HTMLElement} element - Elemento que contiene el valor
 * @param {number} oldValue - Valor anterior
 * @param {number} newValue - Valor nuevo
 * @param {Function} formatter - Función de formateo
 */
function highlightValueChange(element, oldValue, newValue, formatter = formatCurrency) {
    const isIncrease = newValue > oldValue;
    const isDecrease = newValue < oldValue;
    
    // Actualizar contenido
    element.textContent = formatter(newValue);
    
    // Aplicar clase de color
    element.classList.remove('positive', 'negative');
    if (newValue > 0) {
        element.classList.add('positive');
    } else if (newValue < 0) {
        element.classList.add('negative');
    }
    
    // Efecto de resaltado temporal si hay cambio
    if (isIncrease || isDecrease) {
        const color = isIncrease ? '#28a745' : '#dc3545';
        pulseElement(element, color);
    }
}

// ================================================================================================
// FUNCIONES DE ALMACENAMIENTO LOCAL
// ================================================================================================

/**
 * Guarda configuración del usuario en localStorage
 * @param {string} key - Clave de configuración
 * @param {any} value - Valor a guardar
 */
function saveUserSetting(key, value) {
    try {
        const settings = getUserSettings();
        settings[key] = value;
        localStorage.setItem('ampb_user_settings', JSON.stringify(settings));
    } catch (error) {
        console.warn('Error saving user setting:', error);
    }
}

/**
 * Obtiene configuración del usuario
 * @param {string} key - Clave específica (opcional)
 * @returns {any} - Configuración completa o valor específico
 */
function getUserSetting(key = null) {
    try {
        const settings = getUserSettings();
        return key ? settings[key] : settings;
    } catch (error) {
        console.warn('Error getting user setting:', error);
        return key ? null : {};
    }
}

/**
 * Obtiene todas las configuraciones del usuario
 * @returns {object} - Objeto con configuraciones
 */
function getUserSettings() {
    try {
        const settings = localStorage.getItem('ampb_user_settings');
        return settings ? JSON.parse(settings) : {};
    } catch (error) {
        return {};
    }
}

// ================================================================================================
// FUNCIONES DE ESTADÍSTICAS
// ================================================================================================

/**
 * Calcula estadísticas básicas de un array de números
 * @param {number[]} values - Array de valores
 * @returns {object} - Objeto con estadísticas
 */
function calculateStats(values) {
    if (!values || values.length === 0) {
        return { min: 0, max: 0, avg: 0, sum: 0, count: 0 };
    }
    
    const sum = values.reduce((a, b) => a + b, 0);
    const avg = sum / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    return { min, max, avg, sum, count: values.length };
}

/**
 * Calcula la desviación estándar de un array de números
 * @param {number[]} values - Array de valores
 * @returns {number} - Desviación estándar
 */
function calculateStandardDeviation(values) {
    if (!values || values.length <= 1) return 0;
    
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(value => Math.pow(value - avg, 2));
    const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
    
    return Math.sqrt(avgSquaredDiff);
}

// ================================================================================================
// FUNCIONES DE EXPORTACIÓN DE DATOS
// ================================================================================================

/**
 * Convierte datos a CSV
 * @param {Array} data - Array de objetos con datos
 * @param {string} filename - Nombre del archivo
 */
function exportToCSV(data, filename = 'export.csv') {
    if (!data || data.length === 0) {
        showAlert('No hay datos para exportar', 'warning');
        return;
    }
    
    // Obtener headers
    const headers = Object.keys(data[0]);
    
    // Crear contenido CSV
    const csvContent = [
        headers.join(','),
        ...data.map(row => 
            headers.map(header => {
                const value = row[header];
                // Escapar valores que contengan comas o comillas
                if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
                    return `"${value.replace(/"/g, '""')}"`;
                }
                return value;
            }).join(',')
        )
    ].join('\n');
    
    // Crear y descargar archivo
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    
    if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}

// ================================================================================================
// INICIALIZACIÓN
// ================================================================================================

/**
 * Inicializa utilidades AMPB cuando se carga el DOM
 */
document.addEventListener('DOMContentLoaded', function() {
    console.log('AMPB Utils loaded successfully');
    
    // Configurar tooltips si hay algún elemento con data-tooltip
    initializeTooltips();
    
    // Configurar validaciones automáticas para inputs numéricos
    initializeNumericInputs();
});

/**
 * Inicializa tooltips básicos
 */
function initializeTooltips() {
    document.querySelectorAll('[data-tooltip]').forEach(element => {
        element.addEventListener('mouseenter', function() {
            // Aquí podrías implementar tooltips personalizados
            this.title = this.getAttribute('data-tooltip');
        });
    });
}

/**
 * Inicializa validaciones para inputs numéricos
 */
function initializeNumericInputs() {
    document.querySelectorAll('input[type="number"]').forEach(input => {
        input.addEventListener('input', function() {
            const min = parseFloat(this.min) || 0;
            const max = parseFloat(this.max) || Infinity;
            const value = parseFloat(this.value);
            
            if (!isNaN(value)) {
                if (value < min) this.value = min;
                if (value > max) this.value = max;
            }
        });
        
        // Prevenir valores no numéricos
        input.addEventListener('keydown', function(e) {
            // Permitir: backspace, delete, tab, escape, enter
            if ([46, 8, 9, 27, 13].indexOf(e.keyCode) !== -1 ||
                // Permitir Ctrl+A, Ctrl+C, Ctrl+V, Ctrl+X
                (e.keyCode === 65 && e.ctrlKey === true) ||
                (e.keyCode === 67 && e.ctrlKey === true) ||
                (e.keyCode === 86 && e.ctrlKey === true) ||
                (e.keyCode === 88 && e.ctrlKey === true) ||
                // Permitir home, end, left, right
                (e.keyCode >= 35 && e.keyCode <= 39)) {
                return;
            }
            // Asegurar que sea un número
            if ((e.shiftKey || (e.keyCode < 48 || e.keyCode > 57)) && (e.keyCode < 96 || e.keyCode > 105)) {
                e.preventDefault();
            }
        });
    });
}

// Exportar funciones principales al scope global
window.AMPB = {
    validateQuantity,
    validatePrice,
    validateDateFormat,
    formatCurrency,
    formatNumber,
    formatDisplayDate,
    calculatePercentageChange,
    apiPost,
    apiGet,
    showButtonLoading,
    pulseElement,
    fadeUpdateElement,
    highlightValueChange,
    saveUserSetting,
    getUserSetting,
    calculateStats,
    calculateStandardDeviation,
    exportToCSV
};