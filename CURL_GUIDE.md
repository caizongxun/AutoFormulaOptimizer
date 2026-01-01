# curl 遠端執行完全指南

使用 `curl -s` 從命令列遠端優化交易指標公式。

## 快速開始

### 1. 啟動 API 伺服器

```bash
# 本地執行
pip install flask
python server.py

# 或用 Docker
docker build -t auto-formula-optimizer .
docker run -p 5000:5000 auto-formula-optimizer

# 或部署到雲端 (Heroku/Railway/Render)
```

### 2. 一行指令優化公式

```bash
curl -s -X POST http://localhost:5000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "data_num_bars": 500,
    "iterations": [50, 100, 150]
  }' | jq
```

**輸出**（自動格式化的 JSON）：
```json
{
  "status": "success",
  "best_formula": "(OPEN * 3.245) / (CLOSE * 2.156 + 0.012345)",
  "best_score": 0.627843,
  "results": {
    "simple": {...},
    "composite": {...},
    "advanced": {...}
  }
}
```

---

## 詳細 API 文檔

### 端點 1: 健康檢查 ✅

```bash
curl -s http://localhost:5000/api/health | jq
```

**用途**：驗證伺服器是否正常運行

**回應**：
```json
{
  "status": "healthy",
  "timestamp": "2026-01-01T09:30:00.000000",
  "service": "AutoFormulaOptimizer API Server"
}
```

---

### 端點 2: 完全優化 (推薦)

```bash
curl -s -X POST http://localhost:5000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "data_num_bars": 1000,
    "iterations": [50, 100, 150],
    "session_id": "my_experiment_001"
  }' | jq '.best_formula'
```

**請求參數**：

| 參數 | 類型 | 預設值 | 說明 |
|------|------|-------|------|
| `data_num_bars` | int | 500 | K 線數量 (最少 100) |
| `iterations` | array | [50,100,150] | 三階段的迭代次數 [簡單, 複合, 高級] |
| `session_id` | string | auto | 會話標識符 (可選) |

**回應內容**：

```json
{
  "status": "success",
  "timestamp": "2026-01-01T09:30:00",
  "session_id": "my_experiment_001",
  "data_info": {
    "num_bars": 1000,
    "num_reversals": 45,
    "reversal_ratio": 0.045
  },
  "results": {
    "simple": {
      "formula": "(OPEN * 3.245) / (CLOSE * 2.156 + 0.012345)",
      "score": 0.627843,
      "accuracy": 0.6245
    },
    "composite": {
      "formula": "(OPEN * 1.456 + CLOSE * 0.789) / (HIGH - LOW + 0.008765) + RSI[14] * 0.345",
      "score": 0.689234,
      "accuracy": 0.6923
    },
    "advanced": {...}
  },
  "best_formula": "(OPEN * 1.456 + CLOSE * 0.789) / (HIGH - LOW + 0.008765) + RSI[14] * 0.345",
  "best_score": 0.689234,
  "pinescript": "//@version=5\nindicator(\"Auto-Optimized...\")"
}
```

---

### 端點 3: 輕量級優化

只優化簡單公式，快速得到結果：

```bash
curl -s -X POST http://localhost:5000/api/optimize/lightweight \
  -H "Content-Type: application/json" \
  -d '{
    "data_num_bars": 500,
    "iterations": 50
  }' | jq
```

**耗時**：~1 分鐘

**回應**：
```json
{
  "status": "success",
  "formula": "(OPEN * 3.245) / (CLOSE * 2.156 + 0.012345)",
  "score": 0.627843,
  "accuracy": 0.6245
}
```

---

### 端點 4: 查詢會話

```bash
curl -s http://localhost:5000/api/session/my_experiment_001 | jq '.history'
```

**用途**：查看優化歷史記錄

**回應**：
```json
{
  "session_id": "my_experiment_001",
  "best_formula": "...",
  "best_score": 0.689234,
  "history": [
    {
      "iteration": 0,
      "formula_type": "simple",
      "score": 0.627843,
      "accuracy": 0.6245
    },
    {...}
  ]
}
```

---

### 端點 5: 下載 Pine Script

```bash
curl -s http://localhost:5000/api/pinescript/my_experiment_001 > optimized.pine
```

**用途**：直接取得可複製到 TradingView 的程式碼

---

## 實戰範例

### 範例 1: 最簡單 - 一行指令優化

```bash
#!/bin/bash
# 一行優化，只取最佳公式
curl -s -X POST http://localhost:5000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{"data_num_bars": 500, "iterations": [30, 50, 70]}' | jq -r '.best_formula'
```

**輸出**：
```
(OPEN * 1.456 + CLOSE * 0.789) / (HIGH - LOW + 0.008765) + RSI[14] * 0.345
```

---

### 範例 2: 完整工作流

```bash
#!/bin/bash

echo "🚀 開始優化交易公式..."

# 1. 發送優化請求
RESPONSE=$(curl -s -X POST http://localhost:5000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "data_num_bars": 1000,
    "iterations": [50, 100, 150],
    "session_id": "trading_001"
  }')

echo "📊 優化完成"

# 2. 提取結果
BEST_FORMULA=$(echo $RESPONSE | jq -r '.best_formula')
BEST_SCORE=$(echo $RESPONSE | jq -r '.best_score')
SESSION_ID=$(echo $RESPONSE | jq -r '.session_id')

echo "🎯 最佳公式: $BEST_FORMULA"
echo "📈 得分: $BEST_SCORE"

# 3. 保存結果
echo $RESPONSE | jq . > result_${SESSION_ID}.json

# 4. 下載 Pine Script
curl -s http://localhost:5000/api/pinescript/${SESSION_ID} > indicator_${SESSION_ID}.pine

echo "✅ 結果已保存到:"
echo "   - result_${SESSION_ID}.json"
echo "   - indicator_${SESSION_ID}.pine"
```

---

### 範例 3: 批量優化 (迴圈)

```bash
#!/bin/bash

# 用不同的迭代次數試試
for iterations in "30,50,70" "50,100,150" "100,200,300"; do
  echo "嘗試迭代配置: $iterations"
  
  RESPONSE=$(curl -s -X POST http://localhost:5000/api/optimize \
    -H "Content-Type: application/json" \
    -d "{
      \"data_num_bars\": 1000,
      \"iterations\": [$(echo $iterations | tr ',' ' ')]
    }")
  
  SCORE=$(echo $RESPONSE | jq -r '.best_score')
  FORMULA=$(echo $RESPONSE | jq -r '.best_formula')
  
  echo "  得分: $SCORE"
  echo "  公式: $FORMULA"
  echo ""
done
```

---

### 範例 4: 與 jq 組合過濾

```bash
# 只取得指標的準確率
curl -s -X POST http://localhost:5000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{"data_num_bars": 500}' | jq '.results | map_values(.metrics.accuracy)'

# 輸出:
# {
#   "simple": 0.6245,
#   "composite": 0.6923,
#   "advanced": 0.7012
# }
```

---

### 範例 5: 保存為 CSV

```bash
#!/bin/bash

curl -s -X POST http://localhost:5000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{"data_num_bars": 500}' | \
jq -r '.results[] | [.formula, .metrics.score, .metrics.accuracy] | @csv' > formulas.csv

echo "結果已保存到 formulas.csv"
```

---

## curl 命令行技巧

### 技巧 1: 無聲模式

```bash
# -s: 無聲模式 (不顯示進度條)
curl -s http://localhost:5000/api/health
```

### 技巧 2: 美化 JSON 輸出

```bash
# 使用 jq 美化輸出
curl -s http://localhost:5000/api/health | jq .

# 或用 python
curl -s http://localhost:5000/api/health | python -m json.tool
```

### 技巧 3: 保存到檔案

```bash
# -o: 輸出到檔案
curl -s -X POST ... -d '...' -o result.json

# -w: 顯示額外信息
curl -s -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n" http://localhost:5000/api/health
```

### 技巧 4: 設置超時

```bash
# --max-time: 最大超時秒數
curl -s --max-time 300 -X POST http://localhost:5000/api/optimize -d '...'
```

### 技巧 5: 重試

```bash
# 使用 --retry 自動重試
curl -s --retry 3 --retry-delay 2 http://localhost:5000/api/health
```

---

## 自動化部署

### 使用 GitHub Actions

```yaml
name: Auto Formula Optimization

on:
  schedule:
    - cron: '0 0 * * *'  # 每天午夜運行

jobs:
  optimize:
    runs-on: ubuntu-latest
    steps:
      - name: Optimize formulas
        run: |
          curl -s -X POST http://${{ secrets.API_HOST }}/api/optimize \
            -H "Content-Type: application/json" \
            -d '{"data_num_bars": 1000, "iterations": [50, 100, 150]}' > result.json
          
          # 上傳結果到 GitHub
          git config user.name "AutoBot"
          git config user.email "bot@example.com"
          git add result.json
          git commit -m "Auto formula optimization result"
          git push
```

---

## 常見問題

### Q1: 連接拒絕 (Connection refused)

```bash
# 檢查伺服器是否運行
curl -s http://localhost:5000/api/health

# 如果失敗，啟動伺服器
python server.py
```

### Q2: 超時 (timeout)

```bash
# 增加超時時間
curl -s --max-time 600 -X POST http://localhost:5000/api/optimize -d '{"iterations": [150, 200, 250]}'
```

### Q3: 修改 POST 為 GET

```bash
# curl 預設是 GET，除非指定 -X POST
# 對於 GET 請求，參數在 URL 中：
curl -s "http://localhost:5000/api/session/my_session"
```

### Q4: 處理特殊字符

```bash
# 使用 -d @file.json 從檔案讀取
echo '{"data_num_bars": 500}' > payload.json
curl -s -X POST http://localhost:5000/api/optimize -d @payload.json
```

---

## 部署到雲端

### Heroku

```bash
heroku create my-formula-optimizer
git push heroku main
curl -s https://my-formula-optimizer.herokuapp.com/api/health
```

### Railway

```bash
railway login
railway link
railway up
```

### Render

```bash
# 連接 GitHub repo
# 自動部署到 Render
# curl https://my-api.onrender.com/api/health
```

---

## 效能優化

### 並行請求

```bash
#!/bin/bash

# 同時執行多個優化任務
for i in {1..5}; do
  curl -s -X POST http://localhost:5000/api/optimize \
    -H "Content-Type: application/json" \
    -d "{\"data_num_bars\": $((300 + i*100)), \"iterations\": [30, 50, 70]}" &
done

wait  # 等待所有背景任務完成
echo "所有優化完成"
```

---

## 總結

使用 `curl -s` 遠端優化交易公式的流程：

```bash
# 1. 啟動伺服器
python server.py

# 2. 一行指令優化
curl -s -X POST http://localhost:5000/api/optimize \
  -H "Content-Type: application/json" \
  -d '{"data_num_bars": 500}' | jq '.best_formula'

# 3. 複製公式到 TradingView
# 完成！🚀
```
