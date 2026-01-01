#!/usr/bin/env python3
"""
AutoFormulaOptimizer - REST API Server
支持用 curl -s 遠端執行优化器

使用示例：
curl -s -X POST http://localhost:5000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "data_url": "https://...",
    "iterations": [50, 100, 150]
  }' | jq
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import traceback
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(__file__))
from auto_formula_optimizer import AutoFormulaOptimizer

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# 全域优化器存储
optimizers = {}

def generate_synthetic_data(num_bars=500, seed=42):
    """生成演示数据"""
    np.random.seed(seed)
    
    close_prices = np.zeros(num_bars)
    close_prices[0] = 100
    
    for i in range(1, num_bars):
        change = np.random.normal(0.0005, 0.02)
        close_prices[i] = close_prices[i-1] * (1 + change)
    
    open_prices = close_prices * (1 + np.random.normal(0, 0.005, num_bars))
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.01, num_bars)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.01, num_bars)))
    volumes = np.random.randint(1000000, 10000000, num_bars)
    
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    return df

def identify_reversals(data, lookback=5):
    """識別反轉点"""
    labels = np.zeros(len(data))
    close_prices = data['close'].values
    
    for i in range(lookback, len(data) - lookback):
        if i > 10 and close_prices[i] == np.min(close_prices[i-10:i]):
            future_high = np.max(close_prices[i:i+lookback])
            if (future_high - close_prices[i]) / close_prices[i] > 0.02:
                labels[i] = 1
        
        if i > 10 and close_prices[i] == np.max(close_prices[i-10:i]):
            future_low = np.min(close_prices[i:i+lookback])
            if (close_prices[i] - future_low) / close_prices[i] > 0.02:
                labels[i] = 1
    
    return labels

@app.route('/api/health', methods=['GET'])
def health_check():
    """\u5065康\u68c4\u67e5"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'AutoFormulaOptimizer API Server'
    })

@app.route('/api/optimize', methods=['POST'])
def optimize():
    """
    \u9060\u7aef\u4f18\u5316 API \u7aef\u70b9
    
    Request JSON:
    {
        "data_num_bars": 500,  # \u6578\u636e\u6570\u91cf (\u9ed8\u8a8d 500)
        "iterations": [50, 100, 150],  # \u4e09\u4e2a\u9636\u6bb5\u7684\u8fed\u4ee3\u6b21\u6570
        "session_id": "optional_session_id"  # \u4f1a\u8bdd ID \u53ef\u9078
    }
    """
    try:
        payload = request.get_json()
        
        if not payload:
            return jsonify({'error': 'Empty request body'}), 400
        
        # \u53d6\u53d6\u6570\u636e\u914d\u7f6e
        num_bars = payload.get('data_num_bars', 500)
        iterations = payload.get('iterations', [50, 100, 150])
        session_id = payload.get('session_id', f"session_{datetime.now().timestamp()}")
        
        # \u9a57\u8b49\u53c3\u6578
        if not isinstance(iterations, list) or len(iterations) != 3:
            return jsonify({'error': 'iterations must be a list of 3 integers'}), 400
        
        if num_bars < 100:
            return jsonify({'error': 'data_num_bars must be >= 100'}), 400
        
        # \u751f\u6210\u6570\u636e
        print(f'[{session_id}] \u6b63\u5728\u751f\u6210 {num_bars} \u6839 K\u7dda\u6570\u636e...')
        df = generate_synthetic_data(num_bars=num_bars)
        target = identify_reversals(df, lookback=5)
        
        # \u521d\u59cb\u5316\u4f18\u5316\u5668
        print(f'[{session_id}] \u521d\u59cb\u5316\u4f18\u5316\u5668...')
        optimizer = AutoFormulaOptimizer(df, target)
        optimizers[session_id] = optimizer
        
        # \u4e09\u9636\u6bb5\u4f18\u5316
        results = {}
        stage_names = ['simple', 'composite', 'advanced']
        
        for i, (stage_name, iter_count) in enumerate(zip(stage_names, iterations)):
            print(f'[{session_id}] \u6267\u884c\u7b2c {i+1} \u9636\u6bb5: {stage_name} ({iter_count} \u6b21\u8fed\u4ee3)...')
            
            if stage_name == 'simple':
                result = optimizer.optimize_formula_simple(iterations=iter_count)
            elif stage_name == 'composite':
                result = optimizer.optimize_formula_composite(iterations=iter_count)
            else:  # advanced
                result = optimizer.optimize_formula_advanced(iterations=iter_count)
            
            results[stage_name] = {
                'formula': result['formula'],
                'params': {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v 
                          for k, v in result['params'].items()},
                'metrics': {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v 
                           for k, v in result['metrics'].items()}
            }
        
        # \u4f1a\u8bdd\u8b66\u7269\u8a08\u8f2f
        best_formula = optimizer.get_best_formula()
        pine_script = optimizer.export_to_pinescript(results['composite']['params'], 'composite')
        
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            'data_info': {
                'num_bars': len(df),
                'num_reversals': int(np.sum(target)),
                'reversal_ratio': float(np.sum(target) / len(df))
            },
            'results': results,
            'best_formula': best_formula,
            'best_score': float(optimizer.best_score),
            'pinescript': pine_script
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f'[ERROR] {traceback.format_exc()}')
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/optimize/lightweight', methods=['POST'])
def optimize_lightweight():
    """
    \u8f7b\u91cf\u7ea7\u4f18\u5316 API (\u53ea\u8fd0\u6c47\u7b80\u5355\u516c\u5f0f)
    
    Request JSON:
    {
        "data_num_bars": 500,
        "iterations": 50
    }
    """
    try:
        payload = request.get_json()
        
        num_bars = payload.get('data_num_bars', 500)
        iterations = payload.get('iterations', 50)
        
        df = generate_synthetic_data(num_bars=num_bars)
        target = identify_reversals(df, lookback=5)
        
        optimizer = AutoFormulaOptimizer(df, target)
        result = optimizer.optimize_formula_simple(iterations=iterations)
        
        return jsonify({
            'status': 'success',
            'formula': result['formula'],
            'score': float(result['metrics']['score']),
            'accuracy': float(result['metrics']['accuracy']),
            'params': {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v 
                      for k, v in result['params'].items()}
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """
    \u53d6\u5f97\u4f1a\u8bdd\u4fe1\u606f
    """
    if session_id not in optimizers:
        return jsonify({'error': 'Session not found'}), 404
    
    optimizer = optimizers[session_id]
    history = optimizer.get_history()
    
    return jsonify({
        'session_id': session_id,
        'best_formula': optimizer.get_best_formula(),
        'best_score': float(optimizer.best_score),
        'history': history.to_dict(orient='records')
    }), 200

@app.route('/api/pinescript/<session_id>', methods=['GET'])
def get_pinescript(session_id):
    """
    \u53d6\u5f97 Pine Script \u4ee3\u78bc
    """
    if session_id not in optimizers:
        return jsonify({'error': 'Session not found'}), 404
    
    optimizer = optimizers[session_id]
    history = optimizer.get_history()
    
    if len(history) == 0:
        return jsonify({'error': 'No optimization results'}), 400
    
    best_record = history.iloc[-1]
    best_params = json.loads(best_record['formula'])  # \u9700\u8981\u81ea\u5b9a\u4e49 to_dict
    
    pine_script = optimizer.export_to_pinescript({}, 'composite')
    
    return pine_script, 200, {'Content-Type': 'text/plain; charset=utf-8'}

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': str(error)}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"""
    \n{'='*60}
    AutoFormulaOptimizer REST API Server
    {'='*60}
    
    \u76d1\u542c\u4f4d\u7f6e: http://0.0.0.0:{port}
    \u8c03\u8a66\u6a21\u5f0f: {debug}
    
    \u53ef\u7528 API:\n
    1. \u5065\u5eb7\u68c4\u67e5:
       curl -s http://localhost:{port}/api/health | jq
    
    2. \u5168\u9762\u4f18\u5316 (3\u9636\u6bb5):
       curl -s -X POST http://localhost:{port}/api/optimize \\
         -H "Content-Type: application/json" \\
         -d '{{
           "data_num_bars": 500,
           "iterations": [50, 100, 150]
         }}' | jq
    
    3. \u7b80\u6613\u4f18\u5316 (\u4ec5\u7b80\u5355\u516c\u5f0f):
       curl -s -X POST http://localhost:{port}/api/optimize/lightweight \\
         -H "Content-Type: application/json" \\
         -d '{{
           "data_num_bars": 500,
           "iterations": 50
         }}' | jq
    
    4. \u67e5\u770b\u6c47\u60a0\u7b26\u5e8a:\n       curl -s http://localhost:{port}/api/session/SESSION_ID | jq
    
    5. \u4e0b\u8f09 Pine Script:\n       curl -s http://localhost:{port}/api/pinescript/SESSION_ID
    
    {'='*60}\n
    """)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
