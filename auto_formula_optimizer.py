import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize, differential_evolution
import talib
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class AutoFormulaOptimizer:
    """
    自動公式生成與優化系統
    支持符號化公式迭代、差分進化算法參數調整
    
    三層優化架構：
    1. 簡單公式：(OPEN * a) / (CLOSE * b + c)
    2. 複合公式：價格比率 + 技術指標
    3. 高級公式：多層次多指標組合
    """
    
    def __init__(self, data: pd.DataFrame, target: np.ndarray):
        """
        Args:
            data: OHLCV K線數據 (需包含 open, high, low, close, volume)
            target: 反轉點標籤 (0/1 array)
        """
        self.data = data
        self.target = target
        self.history = []
        self.best_formula = None
        self.best_score = -np.inf
        self.iteration = 0
        
        self._precompute_indicators()
    
    def _precompute_indicators(self):
        """預先計算所有可能需要的技術指標"""
        close = self.data['close'].values
        
        self.indicators = {
            'close': close,
            'open': self.data['open'].values,
            'high': self.data['high'].values,
            'low': self.data['low'].values,
            'volume': self.data['volume'].values,
        }
        
        # 多個週期的 RSI
        for period in [7, 14, 21]:
            try:
                self.indicators[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)
            except:
                self.indicators[f'rsi_{period}'] = np.zeros_like(close)
        
        # 多個週期的 MACD
        for fast, slow in [(12, 26), (5, 35)]:
            try:
                macd, signal, hist = talib.MACD(close, fastperiod=fast, slowperiod=slow)
                self.indicators[f'macd_hist_{fast}_{slow}'] = hist
            except:
                self.indicators[f'macd_hist_{fast}_{slow}'] = np.zeros_like(close)
        
        # 多個週期的 KD
        for period in [9, 14]:
            try:
                slowk, slowd = talib.STOCH(self.data['high'].values, 
                                           self.data['low'].values, 
                                           close, fastk_period=period)
                self.indicators[f'kd_k_{period}'] = slowk if slowk is not None else np.zeros_like(close)
            except:
                self.indicators[f'kd_k_{period}'] = np.zeros_like(close)
    
    def _build_formula_string(self, formula_type: str, params: Dict) -> str:
        """構建人類可讀的公式字符串"""
        if formula_type == 'simple':
            a, b, c = params['a'], params['b'], params['c']
            return f"(OPEN * {a:.4f}) / (CLOSE * {b:.4f} + {c:.8f})"
        
        elif formula_type == 'composite':
            a, b, c, d = params['a'], params['b'], params['c'], params['d']
            rsi_period = int(params.get('rsi_period', 14))
            return f"(OPEN * {a:.4f} + CLOSE * {b:.4f}) / (HIGH - LOW + {c:.8f}) + RSI[{rsi_period}] * {d:.4f}"
        
        elif formula_type == 'advanced':
            return self._build_advanced_formula(params)
        
        return ""
    
    def _build_advanced_formula(self, params: Dict) -> str:
        """構建複雜多層公式"""
        components = []
        
        open_w = params.get('open_weight', 0.4)
        close_w = params.get('close_weight', 0.3)
        price_ratio = f"(OPEN * {open_w:.4f} + CLOSE * {close_w:.4f}) / (HIGH - LOW + 1e-6)"
        components.append(price_ratio)
        
        rsi_period = int(params.get('rsi_period', 14))
        rsi_w = params.get('rsi_weight', 0.2)
        momentum = f"RSI[{rsi_period}] * {rsi_w:.4f}"
        components.append(momentum)
        
        macd_w = params.get('macd_weight', 0.1)
        components.append(f"MACD_HIST * {macd_w:.4f}")
        
        formula = " + ".join(components)
        return formula
    
    def _evaluate_formula(self, signal: np.ndarray, 
                         threshold: float = 0.5) -> Dict[str, float]:
        """評估公式的交易績效"""
        signal_normalized = (signal - np.nanmin(signal)) / (np.nanmax(signal) - np.nanmin(signal) + 1e-6)
        predictions = signal_normalized > threshold
        
        valid_idx = ~np.isnan(signal)
        predictions = predictions[valid_idx]
        actual = self.target[valid_idx]
        
        if len(predictions) == 0:
            return {'score': -np.inf, 'accuracy': 0}
        
        TP = np.sum(predictions & actual)
        FP = np.sum(predictions & ~actual)
        FN = np.sum(~predictions & actual)
        TN = np.sum(~predictions & ~actual)
        
        accuracy = (TP + TN) / (TP + FP + FN + TN + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        signal_returns = np.diff(signal_normalized[valid_idx])
        sharpe_ratio = np.mean(signal_returns) / (np.std(signal_returns) + 1e-6)
        
        win_rate = np.sum(signal_returns > 0) / len(signal_returns)
        
        composite_score = (
            accuracy * 0.25 +
            precision * 0.2 +
            recall * 0.2 +
            f1_score * 0.15 +
            (0.5 + sharpe_ratio / 10) * 0.1 +
            win_rate * 0.1
        )
        
        return {
            'score': composite_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate
        }
    
    def _apply_formula_simple(self, params: Dict) -> np.ndarray:
        """執行簡單公式"""
        a, b, c = params['a'], params['b'], params['c']
        open_prices = self.indicators['open']
        close_prices = self.indicators['close']
        
        signal = (open_prices * a) / (close_prices * b + c)
        return signal
    
    def _apply_formula_composite(self, params: Dict) -> np.ndarray:
        """執行複合公式"""
        a, b, c, d = params['a'], params['b'], params['c'], params['d']
        open_p = self.indicators['open']
        close_p = self.indicators['close']
        high_p = self.indicators['high']
        low_p = self.indicators['low']
        
        rsi_period = int(params.get('rsi_period', 14))
        rsi_key = f'rsi_{rsi_period}'
        rsi = self.indicators.get(rsi_key, np.zeros_like(open_p))
        
        price_ratio = (open_p * a + close_p * b) / (high_p - low_p + c)
        signal = price_ratio + rsi * d
        
        return signal
    
    def _apply_formula_advanced(self, params: Dict) -> np.ndarray:
        """執行高級多層公式"""
        open_w = params.get('open_weight', 0.4)
        close_w = params.get('close_weight', 0.3)
        price_comp = (self.indicators['open'] * open_w + 
                     self.indicators['close'] * close_w) / \
                    (self.indicators['high'] - self.indicators['low'] + 1e-6)
        
        rsi_period = int(params.get('rsi_period', 14))
        rsi_key = f'rsi_{rsi_period}'
        rsi_w = params.get('rsi_weight', 0.2)
        momentum_comp = self.indicators.get(rsi_key, np.zeros_like(self.indicators['open'])) * rsi_w
        
        fast = int(params.get('macd_fast', 12))
        slow = int(params.get('macd_slow', 26))
        macd_key = f'macd_hist_{fast}_{slow}'
        macd_w = params.get('macd_weight', 0.1)
        macd_comp = self.indicators.get(macd_key, np.zeros_like(self.indicators['open'])) * macd_w
        
        signal = price_comp + momentum_comp + macd_comp
        return signal
    
    def _objective_function(self, params: np.ndarray, formula_type: str) -> float:
        """適應度函數（用於最小化）"""
        params_dict = self._params_array_to_dict(params, formula_type)
        
        if formula_type == 'simple':
            signal = self._apply_formula_simple(params_dict)
        elif formula_type == 'composite':
            signal = self._apply_formula_composite(params_dict)
        elif formula_type == 'advanced':
            signal = self._apply_formula_advanced(params_dict)
        else:
            return 0
        
        metrics = self._evaluate_formula(signal, threshold=0.5)
        return -metrics['score']
    
    def _params_array_to_dict(self, params: np.ndarray, formula_type: str) -> Dict:
        """將參數陣列轉換為字典"""
        if formula_type == 'simple':
            return {'a': params[0], 'b': params[1], 'c': params[2]}
        elif formula_type == 'composite':
            return {
                'a': params[0], 'b': params[1], 'c': params[2], 'd': params[3],
                'rsi_period': int(np.clip(params[4], 7, 30))
            }
        elif formula_type == 'advanced':
            return {
                'open_weight': params[0],
                'close_weight': params[1],
                'rsi_period': int(np.clip(params[2], 7, 30)),
                'rsi_weight': params[3],
                'macd_weight': params[4],
                'macd_fast': int(np.clip(params[5], 5, 20)),
                'macd_slow': int(np.clip(params[6], 20, 40))
            }
    
    def optimize_formula_simple(self, iterations: int = 50) -> Dict:
        """優化簡單公式"""
        print(f"\n{'='*60}")
        print(f"開始優化簡單公式（迭代數：{iterations}）")
        print(f"公式範本：(OPEN * a) / (CLOSE * b + c)")
        print(f"{'='*60}\n")
        
        bounds = [
            (0.1, 10),
            (0.1, 10),
            (1e-6, 0.1)
        ]
        
        result = differential_evolution(
            lambda params: self._objective_function(params, 'simple'),
            bounds,
            maxiter=iterations,
            popsize=15,
            seed=42,
            workers=1,
            updating='deferred',
            callback=self._callback
        )
        
        best_params = result.x
        params_dict = self._params_array_to_dict(best_params, 'simple')
        signal = self._apply_formula_simple(params_dict)
        metrics = self._evaluate_formula(signal)
        
        formula_str = self._build_formula_string('simple', params_dict)
        
        self.history.append({
            'iteration': len(self.history),
            'formula_type': 'simple',
            'formula': formula_str,
            'params': params_dict,
            'metrics': metrics
        })
        
        if metrics['score'] > self.best_score:
            self.best_score = metrics['score']
            self.best_formula = formula_str
        
        self._print_results(formula_str, metrics)
        
        return {
            'params': params_dict,
            'metrics': metrics,
            'formula': formula_str
        }
    
    def optimize_formula_composite(self, iterations: int = 100) -> Dict:
        """優化複合公式"""
        print(f"\n{'='*60}")
        print(f"開始優化複合公式（迭代數：{iterations}）")
        print(f"公式範本：(OPEN*a + CLOSE*b)/(HIGH-LOW+c) + RSI[period]*d")
        print(f"{'='*60}\n")
        
        bounds = [
            (0.1, 5),
            (0.1, 5),
            (1e-6, 0.1),
            (0.1, 5),
            (7, 30)
        ]
        
        result = differential_evolution(
            lambda params: self._objective_function(params, 'composite'),
            bounds,
            maxiter=iterations,
            popsize=20,
            seed=42,
            workers=1,
            callback=self._callback
        )
        
        best_params = result.x
        params_dict = self._params_array_to_dict(best_params, 'composite')
        signal = self._apply_formula_composite(params_dict)
        metrics = self._evaluate_formula(signal)
        
        formula_str = self._build_formula_string('composite', params_dict)
        
        self.history.append({
            'iteration': len(self.history),
            'formula_type': 'composite',
            'formula': formula_str,
            'params': params_dict,
            'metrics': metrics
        })
        
        if metrics['score'] > self.best_score:
            self.best_score = metrics['score']
            self.best_formula = formula_str
        
        self._print_results(formula_str, metrics)
        
        return {
            'params': params_dict,
            'metrics': metrics,
            'formula': formula_str
        }
    
    def optimize_formula_advanced(self, iterations: int = 150) -> Dict:
        """優化高級公式"""
        print(f"\n{'='*60}")
        print(f"開始優化高級公式（迭代數：{iterations}）")
        print(f"公式範本：價格比率 + RSI + MACD")
        print(f"{'='*60}\n")
        
        bounds = [
            (0.1, 2),
            (0.1, 2),
            (7, 30),
            (0.1, 2),
            (0.05, 0.5),
            (5, 20),
            (20, 40)
        ]
        
        result = differential_evolution(
            lambda params: self._objective_function(params, 'advanced'),
            bounds,
            maxiter=iterations,
            popsize=25,
            seed=42,
            workers=1,
            callback=self._callback
        )
        
        best_params = result.x
        params_dict = self._params_array_to_dict(best_params, 'advanced')
        signal = self._apply_formula_advanced(params_dict)
        metrics = self._evaluate_formula(signal)
        
        formula_str = self._build_formula_string('advanced', params_dict)
        
        self.history.append({
            'iteration': len(self.history),
            'formula_type': 'advanced',
            'formula': formula_str,
            'params': params_dict,
            'metrics': metrics
        })
        
        if metrics['score'] > self.best_score:
            self.best_score = metrics['score']
            self.best_formula = formula_str
        
        self._print_results(formula_str, metrics)
        
        return {
            'params': params_dict,
            'metrics': metrics,
            'formula': formula_str
        }
    
    def _callback(self, xk, convergence=None):
        """進度回調函數"""
        self.iteration += 1
        if self.iteration % 10 == 0:
            print(f"  迭代 {self.iteration}... 正在優化中...")
    
    def _print_results(self, formula: str, metrics: Dict):
        """打印結果摘要"""
        print(f"\n✓ 最佳公式: {formula}")
        print(f"  綜合得分: {metrics['score']:.6f}")
        print(f"  準確率: {metrics['accuracy']:.4f}")
        print(f"  精度: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Sharpe 比率: {metrics['sharpe_ratio']:.4f}")
        print(f"  勝率: {metrics['win_rate']:.4f}\n")
    
    def get_history(self) -> pd.DataFrame:
        """返回優化歷史記錄"""
        history_data = []
        for record in self.history:
            history_data.append({
                'iteration': record['iteration'],
                'formula_type': record['formula_type'],
                'formula': record['formula'],
                'score': record['metrics']['score'],
                'accuracy': record['metrics']['accuracy'],
                'precision': record['metrics']['precision'],
                'recall': record['metrics']['recall'],
                'f1_score': record['metrics']['f1_score'],
                'sharpe_ratio': record['metrics']['sharpe_ratio'],
                'win_rate': record['metrics']['win_rate']
            })
        return pd.DataFrame(history_data)
    
    def get_best_formula(self) -> str:
        """返回全局最佳公式"""
        return self.best_formula
    
    def export_to_pinescript(self, params: Dict, formula_type: str = 'composite') -> str:
        """將最佳公式轉換為 Pine Script"""
        if formula_type == 'simple':
            pine_code = f"""
//@version=5
indicator("Auto-Optimized Simple Reversal Indicator", overlay=false)

// 參數 (由 AI 自動優化)
a = input.float({params.get('a', 0.5)}, title="Open Weight", step=0.001)
b = input.float({params.get('b', 0.3)}, title="Close Weight", step=0.001)
c = input.float({params.get('c', 0.01)}, title="Smooth Factor", step=0.0001)

// 計算指標
indicator_value = (open * a) / (close * b + c)

// 標準化
normalized = (indicator_value - ta.lowest(indicator_value, 20)) / 
             (ta.highest(indicator_value, 20) - ta.lowest(indicator_value, 20) + 0.0001)

// 繪製
plot(normalized, title="Reversal Signal", color=color.blue, linewidth=2)
hline(0.5, "Threshold", color=color.gray, linestyle=hline.style_dashed)

// 信號
is_signal = ta.crossover(normalized, 0.5) or ta.crossunder(normalized, 0.5)
plotshape(is_signal, title="Signal", style=shape.circle, location=location.bottom,
          color=is_signal ? color.green : color.red, size=size.small)
"""
        
        elif formula_type == 'composite':
            rsi_period = int(params.get('rsi_period', 14))
            pine_code = f"""
//@version=5
indicator("Auto-Optimized Composite Reversal Indicator", overlay=false)

// 參數 (由 AI 自動優化)
a = input.float({params.get('a', 0.5)}, title="Open Weight", step=0.001)
b = input.float({params.get('b', 0.3)}, title="Close Weight", step=0.001)
c = input.float({params.get('c', 0.01)}, title="Smooth Factor", step=0.0001)
d = input.float({params.get('d', 0.5)}, title="RSI Weight", step=0.001)
rsi_period = input.int({rsi_period}, title="RSI Period", minval=5, maxval=50)

// 計算指標
rsi_value = ta.rsi(close, rsi_period)
indicator_value = ((open * a + close * b) / (high - low + c)) + rsi_value * d

// 標準化
normalized = (indicator_value - ta.lowest(indicator_value, 20)) / 
             (ta.highest(indicator_value, 20) - ta.lowest(indicator_value, 20) + 0.0001)

// 繪製
plot(normalized, title="Reversal Signal", color=color.blue, linewidth=2)
hline(0.5, "Threshold", color=color.gray, linestyle=hline.style_dashed)

// 信號
is_signal = ta.crossover(normalized, 0.5) or ta.crossunder(normalized, 0.5)
plotshape(is_signal, title="Signal", style=shape.circle, location=location.bottom,
          color=is_signal ? color.green : color.red, size=size.small)
"""
        
        else:
            pine_code = "// Advanced formula - 請手動構建"
        
        return pine_code.strip()
