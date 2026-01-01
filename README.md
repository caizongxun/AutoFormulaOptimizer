# AutoFormulaOptimizer

一個基於差分進化演算法和貝葉斯優化的自動交易指標公式優化系統。

## 項目概述

AutoFormulaOptimizer 使用 AI 自動化地生成和優化交易反轉指標公式。系統通過對比數百種不同的數學組合，
自動找到最能預測市場反轉點的公式。每次迭代都會改進公式的準確率。

### 核心特性

- **三層級公式優化**
  - 第1層：簡單公式 `(OPEN * a) / (CLOSE * b + c)`
  - 第2層：複合公式 `(OPEN*a + CLOSE*b)/(HIGH-LOW+c) + RSI[p]*d`
  - 第3層：高級公式 - 多層次、多指標組合

- **自動參數搜索**
  - 使用差分進化演算法（Differential Evolution）
  - 支持邊界約束和多目標優化
  - 智慧搜索，避免盲目嘗試

- **多維度評估**
  - 準確率（Accuracy）
  - 精度（Precision）和召回率（Recall）
  - Sharpe 比率
  - 勝率（Win Rate）
  - 綜合評分加權

- **一鍵輸出**
  - 自動生成 Pine Script 代碼
  - 直接複製到 TradingView 使用
  - 完整優化歷史記錄

## 安裝

### 環境要求

```bash
Python >= 3.8
```

### 依賴安裝

```bash
pip install numpy pandas scipy scikit-learn TA-Lib
```

如果 TA-Lib 安裝有困難，可以改用：

```bash
pip install pandas-ta
```

## 快速開始

### 1. 基本使用

```python
import pandas as pd
import numpy as np
from auto_formula_optimizer import AutoFormulaOptimizer

# 加載 OHLCV 數據
df = pd.read_csv('your_data.csv')
# 準備反轉點標籤（0/1）
target = np.array([...])  

# 初始化優化器
optimizer = AutoFormulaOptimizer(df, target)

# 執行優化（三個階段）
result_simple = optimizer.optimize_formula_simple(iterations=50)
result_composite = optimizer.optimize_formula_composite(iterations=100)
result_advanced = optimizer.optimize_formula_advanced(iterations=150)

# 查看結果
print(f"最佳公式: {optimizer.get_best_formula()}")
```

### 2. 數據格式

OHLCV 數據應包含以下列：

| 列名 | 說明 |
|------|------|
| `open` | 開盤價 |
| `high` | 最高價 |
| `low` | 最低價 |
| `close` | 收盤價 |
| `volume` | 成交量 |

### 3. 反轉點標籤

反轉點標籤是一個 0/1 數組，表示每個時間點是否為反轉點。

**生成方法（示例）**：

```python
import numpy as np

def find_reversals(data, lookback=5):
    labels = np.zeros(len(data))
    close = data['close'].values
    
    for i in range(lookback, len(data) - lookback):
        # 識別底部反轉
        if close[i] == np.min(close[i-10:i]):
            future_high = np.max(close[i:i+lookback])
            if (future_high - close[i]) / close[i] > 0.02:
                labels[i] = 1
        
        # 識別頂部反轉
        if close[i] == np.max(close[i-10:i]):
            future_low = np.min(close[i:i+lookback])
            if (close[i] - future_low) / close[i] > 0.02:
                labels[i] = 1
    
    return labels

target = find_reversals(df)
```

## 詳細文檔

### 主要方法

#### 1. `optimize_formula_simple(iterations=50)`

優化簡單公式：`(OPEN * a) / (CLOSE * b + c)`

**參數**：
- `iterations`: 優化迭代次數（推薦 50-100）

**返回值**：
```python
{
    'params': {'a': 3.245, 'b': 2.156, 'c': 0.012345},
    'metrics': {
        'score': 0.627843,
        'accuracy': 0.6245,
        'precision': 0.6789,
        'recall': 0.5912,
        'sharpe_ratio': 1.2345
    },
    'formula': '(OPEN * 3.2450) / (CLOSE * 2.1560 + 0.01234500)'
}
```

#### 2. `optimize_formula_composite(iterations=100)`

優化複合公式：`(OPEN*a + CLOSE*b)/(HIGH-LOW+c) + RSI[period]*d`

自動調整：
- 開盤價權重 (a)
- 收盤價權重 (b)
- 平滑因子 (c)
- RSI 權重 (d)
- RSI 週期 (7-30)

#### 3. `optimize_formula_advanced(iterations=150)`

優化高級公式，結合多個技術指標：
- 價格比率成分
- RSI 動量成分
- MACD 趨勢成分

### 輔助方法

#### `get_history()`

返回 Pandas DataFrame，包含所有優化迭代的記錄

#### `get_best_formula()`

返回全局最佳公式字符串

#### `export_to_pinescript(params, formula_type='composite')`

將優化結果轉換為 Pine Script 代碼

## 優化流程說明

### 演算法細節

系統使用**差分進化演算法（Differential Evolution）**進行全局最優化：

1. **初始化**：隨機生成初始參數種群（通常 15-25 個個體）

2. **變異**：每一代中，算法基於當前種群的最優解生成新的候選解
   ```
   新參數 = 當前最優參數 + 變異係數 × (隨機參數1 - 隨機參數2)
   ```

3. **評估**：計算新參數對應的公式準確率
   ```
   適應度 = 準確率×0.25 + 精度×0.2 + 召回率×0.2 + F1×0.15 + Sharpe×0.1 + 勝率×0.1
   ```

4. **選擇**：保留更優的參數，淘汰劣質解

5. **重複**：迭代直到收斂或達到指定次數

### 參數空間

| 公式類型 | 參數 | 搜索範圍 |
|---------|------|----------|
| 簡單 | a | [0.1, 10] |
| 簡單 | b | [0.1, 10] |
| 簡單 | c | [1e-6, 0.1] |
| 複合 | rsi_period | [7, 30] |
| 複合 | macd_fast | [5, 20] |
| 複合 | macd_slow | [20, 40] |

## 使用範例

完整的工作流程見 `example_usage.py`：

```bash
python example_usage.py
```

輸出示例：

```
============================================================
開始優化簡單公式（迭代數：50）
公式範本：(OPEN * a) / (CLOSE * b + c)
============================================================

  迭代 10... 正在優化中...
  迭代 20... 正在優化中...
  迭代 30... 正在優化中...
  迭代 40... 正在優化中...
  迭代 50... 正在優化中...

✓ 最佳公式: (OPEN * 3.2450) / (CLOSE * 2.1560 + 0.01234500)
  綜合得分: 0.627843
  準確率: 0.6245
  精度: 0.6789
  召回率: 0.5912
  F1-Score: 0.6300
  Sharpe 比率: 1.2345
  勝率: 0.6234
```

## Pine Script 集成

優化完成後，系統可自動生成 Pine Script 代碼：

```python
pinescript = optimizer.export_to_pinescript(result_composite['params'], 'composite')
print(pinescript)
```

生成的代碼可直接複製到 TradingView：

```pinescript
//@version=5
indicator("Auto-Optimized Composite Reversal Indicator", overlay=false)

a = input.float(1.456, title="Open Weight", step=0.001)
b = input.float(0.789, title="Close Weight", step=0.001)
c = input.float(0.008765, title="Smooth Factor", step=0.0001)
d = input.float(0.345, title="RSI Weight", step=0.001)
rsi_period = input.int(14, title="RSI Period", minval=5, maxval=50)

rsi_value = ta.rsi(close, rsi_period)
indicator_value = ((open * a + close * b) / (high - low + c)) + rsi_value * d

normalized = (indicator_value - ta.lowest(indicator_value, 20)) / 
             (ta.highest(indicator_value, 20) - ta.lowest(indicator_value, 20) + 0.0001)

plot(normalized, title="Reversal Signal", color=color.blue, linewidth=2)
hline(0.5, "Threshold", color=color.gray, linestyle=hline.style_dashed)
```

## 性能考慮

### 優化時間

| 階段 | 迭代數 | 估計時間 | CPU 資源 |
|------|--------|---------|----------|
| 簡單公式 | 50 | 5-10 秒 | 1 核心 |
| 複合公式 | 100 | 20-30 秒 | 1 核心 |
| 高級公式 | 150 | 40-60 秒 | 1-2 核心 |
| **總計** | **300** | **1-2 分鐘** | **1-2 核心** |

### 數據要求

- **最少數據量**：500 根 K 線（約 1-2 週日線）
- **推薦數據量**：2,000-10,000 根 K 線（3-12 個月）
- **最少反轉點**：50+ 個標籤點

### 避免過擬合

1. 使用足夠多的數據（> 1 年）
2. 定期進行 Walk-Forward 測試
3. 定期重新優化以適應市場變化
4. 在測試集上驗證公式

## 常見問題

### Q1: 優化後的公式在實盤中效果不好

**A**：可能原因：
- 訓練數據不足或偏差
- 過擬合（公式過度適應歷史數據）
- 市場環境變化

**解決方案**：
- 增加訓練數據
- 定期重新優化
- 使用 Walk-Forward 測試驗證
- 降低對單個公式的依賴，組合多個指標

### Q2: 如何調整優化的激進程度

**A**：
- 增加 `iterations` 參數可以進行更深入的搜索
- 調整 `popsize` 可以改變種群大小

### Q3: 支持哪些技術指標

**A**：當前支持：
- RSI (多個週期)
- MACD (多個參數組合)
- KD / Stochastic (多個週期)
- 基本 OHLCV 數據

可擴展支持其他指標（修改 `_precompute_indicators()` 方法）

## 擴展和自訂

### 添加新的技術指標

在 `_precompute_indicators()` 中添加：

```python
for period in [20, 30]:
    self.indicators[f'bb_upper_{period}'] = calculate_bollinger_upper(close, period)
    self.indicators[f'bb_lower_{period}'] = calculate_bollinger_lower(close, period)
```

### 自訂評估指標

修改 `_evaluate_formula()` 中的權重：

```python
composite_score = (
    accuracy * 0.25 +
    precision * 0.2 +
    recall * 0.2 +
    f1_score * 0.15 +
    (0.5 + sharpe_ratio / 10) * 0.1 +
    win_rate * 0.1
)
```

## 許可證

MIT

## 作者

AutoFormulaOptimizer Team

## 貢獻

歡迎提交 Issue 和 Pull Request！

## 更新日誌

### v1.0.0 (2026-01-01)
- 初始發佈
- 支持三層級公式優化
- 完整 Pine Script 導出
- 詳細文檔和示例
