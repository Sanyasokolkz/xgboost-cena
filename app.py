from flask import Flask, request, jsonify
import re
import pickle
import pandas as pd
import numpy as np
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Загружаем модель при старте сервера
try:
    model_path = os.path.join(os.path.dirname(__file__), 'xgboost_token_model.pkl')
    with open(model_path, 'rb') as f:
        model_artifacts = pickle.load(f)
    print("✅ Модель загружена успешно!")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    model_artifacts = None

def parse_token_data(text):
    """
    Парсит текстовые данные токена в структурированный формат
    Оптимизирован для нового формата данных
    """
    try:
        token_data = {}
        
        # Market Cap - "MC: $136.8K"
        mcap_match = re.search(r'MC:\s*\$?([0-9,.]+)([KMB]?)', text, re.IGNORECASE)
        if mcap_match:
            value_str = mcap_match.group(1).replace(',', '')
            value = float(value_str)
            unit = mcap_match.group(2).upper()
            if unit == 'K':
                value *= 1000
            elif unit == 'M':
                value *= 1000000
            elif unit == 'B':
                value *= 1000000000
            token_data['market_cap'] = value
        else:
            token_data['market_cap'] = 0
        
        # Liquidity - "Liq: $42.4K"
        liq_match = re.search(r'Liq:\s*\$?([0-9,.]+)([KMB]?)', text, re.IGNORECASE)
        if liq_match:
            value_str = liq_match.group(1).replace(',', '')
            value = float(value_str)
            unit = liq_match.group(2).upper()
            if unit == 'K':
                value *= 1000
            elif unit == 'M':
                value *= 1000000
            elif unit == 'B':
                value *= 1000000000
            token_data['liquidity'] = value
        else:
            token_data['liquidity'] = 0
        
        # Volume - берем 1min volume: "Volume: $12,129.12"
        vol_1min_match = re.search(r'1 min:.*?Volume:\s*\$?([0-9,.]+)', text, re.DOTALL)
        if vol_1min_match:
            value_str = vol_1min_match.group(1).replace(',', '')
            token_data['volume_1min'] = float(value_str)
        else:
            # Если нет 1min, берем 5min
            vol_5min_match = re.search(r'5 min:.*?Volume:\s*\$?([0-9,.]+)', text, re.DOTALL)
            if vol_5min_match:
                value_str = vol_5min_match.group(1).replace(',', '')
                token_data['volume_1min'] = float(value_str)
            else:
                token_data['volume_1min'] = 0
        
        # Last Volume - нет в новом формате, ставим значения по умолчанию
        token_data['last_volume'] = token_data['volume_1min']  # Используем текущий объем
        token_data['last_volume_multiplier'] = 1.0
        
        # Token Age - "Token age: 25m"
        age_match = re.search(r'Token age:\s*(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+)s\s*)?', text, re.IGNORECASE)
        if age_match:
            hours = int(age_match.group(1)) if age_match.group(1) else 0
            minutes = int(age_match.group(2)) if age_match.group(2) else 0
            seconds = int(age_match.group(3)) if age_match.group(3) else 0
            total_minutes = hours * 60 + minutes + seconds / 60
            token_data['token_age_numeric'] = total_minutes
        else:
            token_data['token_age_numeric'] = 0
        
        # Держатели по цветам - "🟢: 8 | 🔵: 5 | 🟡: 12 | ⭕️: 42"
        holders_match = re.search(r'🟢:\s*(\d+)\s*\|\s*🔵:\s*(\d+)\s*\|\s*🟡:\s*(\d+)\s*\|\s*⭕️:\s*(\d+)', text)
        if holders_match:
            token_data['green_holders'] = int(holders_match.group(1))
            token_data['blue_holders'] = int(holders_match.group(2))
            token_data['yellow_holders'] = int(holders_match.group(3))
            token_data['circle_holders'] = int(holders_match.group(4))
        else:
            token_data['green_holders'] = 0
            token_data['blue_holders'] = 0
            token_data['yellow_holders'] = 0
            token_data['circle_holders'] = 0
        
        # Специальные держатели - "🤡: 0 | 🌞: 0 | 🌗: 0 | 🌚: 3"
        special_holders_match = re.search(r'🤡:\s*(\d+)\s*\|\s*🌞:\s*(\d+)\s*\|\s*🌗:\s*(\d+)\s*\|\s*🌚:\s*(\d+)', text)
        if special_holders_match:
            token_data['clown_holders'] = int(special_holders_match.group(1))
            token_data['sun_holders'] = int(special_holders_match.group(2))
            token_data['half_moon_holders'] = int(special_holders_match.group(3))
            token_data['dark_moon_holders'] = int(special_holders_match.group(4))
        else:
            token_data['clown_holders'] = 0
            token_data['sun_holders'] = 0
            token_data['half_moon_holders'] = 0
            token_data['dark_moon_holders'] = 0
        
        # Total holders - "Total: 168"
        total_holders_match = re.search(r'Total:\s*(\d+)', text)
        if total_holders_match:
            token_data['total_holders'] = int(total_holders_match.group(1))
        else:
            token_data['total_holders'] = (
                token_data['green_holders'] + token_data['blue_holders'] + 
                token_data['yellow_holders'] + token_data['circle_holders']
            )
        
        # Top 10 percent - "Top 10: 23%"
        top10_match = re.search(r'Top 10:\s*([0-9.]+)%', text)
        if top10_match:
            token_data['top10_percent'] = float(top10_match.group(1))
        else:
            token_data['top10_percent'] = 50.0
        
        # Current/Initial percentages - "Current/Initial: 16.76% / 98.87%"
        current_initial_match = re.search(r'Current/Initial:\s*([0-9.]+)%\s*/\s*([0-9.]+)%', text)
        if current_initial_match:
            token_data['total_now_percent'] = float(current_initial_match.group(1))
            token_data['total_percent'] = float(current_initial_match.group(2))
        else:
            token_data['total_percent'] = 100.0
            token_data['total_now_percent'] = 50.0
        
        # Dev holds - "Dev current balance: 0%"
        dev_match = re.search(r'Dev current balance:\s*([0-9.]+)%', text)
        if dev_match:
            token_data['dev_holds_percent'] = float(dev_match.group(1))
        else:
            token_data['dev_holds_percent'] = 0.0
        
        # Поля которых нет в новом формате - заполняем значениями по умолчанию
        token_data['insiders_count'] = 0
        token_data['insiders_percent'] = 0.0
        token_data['snipers_count'] = 0
        token_data['bundle_total'] = 0
        token_data['bundle_supply_percent'] = 0.0
        
        # Создаем дополнительные признаки для модели
        token_data['volume_to_liquidity'] = float(
            np.log1p(token_data['volume_1min']) / np.log1p(token_data['liquidity'] + 1)
            if token_data['liquidity'] > 0 else 0
        )
        
        # Логарифмические признаки
        token_data['log_market_cap'] = float(np.log1p(token_data['market_cap']))
        token_data['log_liquidity'] = float(np.log1p(token_data['liquidity']))
        token_data['log_volume_1min'] = float(np.log1p(token_data['volume_1min']))
        token_data['log_last_volume'] = float(np.log1p(token_data['last_volume']))
        
        # Концентрация держателей
        token_data['holder_concentration'] = int(
            token_data['green_holders'] + token_data['blue_holders'] + 
            token_data['yellow_holders'] + token_data['circle_holders']
        )
        
        # Риск-индикаторы
        token_data['total_risk_percent'] = float(
            token_data['dev_holds_percent'] + token_data['insiders_percent']
        )
        
        return convert_to_json_serializable(token_data)
        
    except Exception as e:
        logging.error(f"Ошибка парсинга: {e}")
        return {}

def convert_to_json_serializable(obj):
    """Конвертирует numpy типы в стандартные Python типы для JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

def predict_token_success(token_data):
    """
    Предсказывает успешность токена (с исправлением дрифта данных)
    """
    if model_artifacts is None:
        return {'error': 'Модель не загружена'}
    
    try:
        # Создаем DataFrame
        df_new = pd.DataFrame([token_data])
        
        # Добавляем недостающие столбцы
        for col in model_artifacts['feature_names']:
            if col not in df_new.columns:
                df_new[col] = 0
        
        # Берем только нужные признаки в правильном порядке
        df_new = df_new[model_artifacts['feature_names']]
        
        # Применяем импутер
        df_imputed = pd.DataFrame(
            model_artifacts['imputer'].transform(df_new), 
            columns=model_artifacts['feature_names']
        )
        
        # Получаем исходные предсказания модели
        raw_probability = float(model_artifacts['model'].predict_proba(df_imputed)[0, 1])
        
        # 🔄 ИСПРАВЛЕНИЕ ДРИФТА: Инвертируем вероятность
        probability = 1.0 - raw_probability
        prediction = int(probability >= 0.5)
        
        # Определяем уровень уверенности
        confidence_score = abs(probability - 0.5) * 2
        if confidence_score > 0.8:
            confidence_level = "Очень высокая"
        elif confidence_score > 0.6:
            confidence_level = "Высокая"
        elif confidence_score > 0.4:
            confidence_level = "Средняя"
        else:
            confidence_level = "Низкая"
        
        # Определяем рекомендацию
        if probability >= 0.7:
            recommendation = "ПОКУПАТЬ"
            risk_level = "Низкий"
        elif probability >= 0.5:
            recommendation = "ИЗУЧИТЬ"
            risk_level = "Средний"
        else:
            recommendation = "ПРОПУСТИТЬ"
            risk_level = "Высокий"
        
        result = {
            'prediction': 'ДА' if prediction == 1 else 'НЕТ',
            'probability': round(float(probability), 4),
            'probability_percent': f"{probability*100:.1f}%",
            'confidence_level': confidence_level,
            'confidence_score': round(float(confidence_score), 4),
            'recommendation': recommendation,
            'risk_level': risk_level,
            'threshold_conservative': 'ДА' if probability >= 0.7 else 'НЕТ',
            'threshold_optimal': 'ДА' if probability >= 0.5 else 'НЕТ',
            'threshold_aggressive': 'ДА' if probability >= 0.3 else 'НЕТ',
            
            # Информация о коррекции для отладки
            'model_info': {
                'raw_probability': round(raw_probability, 4),
                'corrected_probability': round(probability, 4),
                'drift_correction': True,
                'data_format': 'new_format_optimized'
            }
        }
        
        return convert_to_json_serializable(result)
        
    except Exception as e:
        logging.error(f"Ошибка предсказания: {e}")
        return {'error': f'Ошибка предсказания: {str(e)}'}

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка работоспособности API"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_artifacts is not None,
        'version': '2.1',
        'optimized_for': 'new_data_format',
        'drift_correction': 'enabled',
        'environment': os.environ.get('RAILWAY_ENVIRONMENT', 'unknown')
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Основной эндпоинт для предсказания"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Необходимо передать поле "text" с данными токена'}), 400
        
        # Парсим текст
        token_data = parse_token_data(data['text'])
        
        if not token_data:
            return jsonify({'error': 'Не удалось распарсить данные токена'}), 400
        
        # Получаем предсказание
        result = predict_token_success(token_data)
        
        # Добавляем распарсенные данные для отладки
        if data.get('include_parsed_data', False):
            result['parsed_data'] = convert_to_json_serializable(token_data)
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Ошибка API: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/test', methods=['GET'])
def test():
    """Тестовый эндпоинт с данными нового формата"""
    test_text = """🎲 $PVE | President vs Elon

3nuogKUQuxfxjCRud7Bpm5a9Q7eT7mxpFGNe9WeNbonk

⏳ Token age:  25m  | 👁 14
├ MC: $136.8K
├ Liq: $42.4K / SOL pooled: 111.02
└ ATH: $134.6K (-4% / 4s)

1 min:
├ Volume: $12,129.12
├ Buy volume ($): $6,446.81
├ Sell volume ($): $5,682.31
├ Buys: 165
└ Sells: 177

5 min:
├ Volume: $71,175.80
├ Buy volume ($): $46,124.03
├ Sell volume ($): $25,051.77
├ Buys: 585
└ Sells: 448

🎯 First 70 buyers:
🌚🌚🌚🟡🟡🟡🟡🟡🟡🟡
🟡🟢🟡⭕️⭕️⭕️⭕️⭕️⭕️⭕️
⭕️🟢🔵🔵🟢🟢⭕️🟡🟡⭕️
⭕️⭕️⭕️⭕️⭕️⭕️⭕️⭕️⭕️⭕️
⭕️⭕️⭕️⭕️⭕️⭕️⭕️⭕️🟢🟢
🔵⭕️⭕️⭕️🟢🔵⭕️⭕️⭕️⭕️
🔵⭕️⭕️⭕️🟢⭕️⭕️⭕️⭕️🟡

├ 🟢: 8 | 🔵: 5 | 🟡: 12 | ⭕️: 42
├ 🤡: 0 | 🌞: 0 | 🌗: 0 | 🌚: 3
├ Current/Initial: 16.76% / 98.87%

👥 Holders:
├ Total: 168
├ Freshies: 8.8% 1D | 87% 7D
├ Top 10: 23%
💰 Top 10 Holding (%)
15.82 | 2.48 | 2.42 | 2.42 | 2.41 | 2.38 | 2.31 | 2.29 | 2.26 | 2.22

😎 Dev
├ Dev current balance: 0%
└ Dev SOL balance: 0 SOL

🔒 Security:
├ NoMint: 🟢
├ Blacklist: 🟢
├ Burnt: 🟢
├ Dev Sold: 🟢
└ Dex Paid: 🔴"""
    
    token_data = parse_token_data(test_text)
    result = predict_token_success(token_data)
    result['parsed_data'] = convert_to_json_serializable(token_data)
    
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('RAILWAY_ENVIRONMENT') != 'production'
    
    print("🚀 Token Prediction API v2.1")
    print(f"📍 Порт: {port}")
    print(f"🔄 Оптимизировано для нового формата данных")
    print(f"🧪 Тест: /test")
    print(f"❤️  Статус: /health")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
