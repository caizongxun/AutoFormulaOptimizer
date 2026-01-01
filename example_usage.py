import numpy as np
import pandas as pd
from auto_formula_optimizer import AutoFormulaOptimizer
import talib

def generate_synthetic_ohlcv_data(num_bars=500, seed=42):
    """
    生成演示 OHLCV 數據
    實際使用時应載入真實歷史數據
    """
    np.random.seed(seed)
    
    close_prices = np.zeros(num_bars)
    close_prices[0] = 100
    
    # 模擬平穩隨機步途
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

def generate_reversal_labels(data, lookback=5):
    """
    生成反轉點標笤
    定義：最低點之後的 5 根 K 線中，低點後追紀貨 > 2% 為上漫反轉
    最高點之後的 5 根 K 線中，高點後下跌 > 2% 為下轐反轉
    """
    labels = np.zeros(len(data))
    close_prices = data['close'].values
    
    for i in range(lookback, len(data) - lookback):
        # 上漫反轉：低點探底
        if i > 10 and close_prices[i] == np.min(close_prices[i-10:i]):
            future_high = np.max(close_prices[i:i+lookback])
            if (future_high - close_prices[i]) / close_prices[i] > 0.02:
                labels[i] = 1
        
        # 下轐反轉：高點壳顶
        if i > 10 and close_prices[i] == np.max(close_prices[i-10:i]):
            future_low = np.min(close_prices[i:i+lookback])
            if (close_prices[i] - future_low) / close_prices[i] > 0.02:
                labels[i] = 1
    
    return labels

def main():
    print("""
    ================================================================================
                    自動公式优化系統 - 演示例子
    ================================================================================
    """)
    
    # 步骤1: 準备数据
    print("\n[1] 正在生成演示数据...")
    df = generate_synthetic_ohlcv_data(num_bars=500)
    print(f"    数据形状: {df.shape}")
    print(f"    日期范围: {df.index[0]} - {df.index[-1]}")
    print(f"    价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # 步骤2: 生成反轉点标筤
    print("\n[2] 正在生成反轉点标筤...")
    target = generate_reversal_labels(df, lookback=5)
    num_reversals = np.sum(target)
    print(f"    找到 {num_reversals} 个反轉点 ({100*num_reversals/len(df):.1f}%)")
    
    # 步骤3: 初始化优化器
    print("\n[3] 初始化优化器...")
    optimizer = AutoFormulaOptimizer(df, target)
    print("    ✓ 优化器已準备就绪")
    
    # 步骤4: 优化简单公式
    print("\n[4] 第一阶段：优化简单公式")
    result_simple = optimizer.optimize_formula_simple(iterations=50)
    print(f"    简单公式优化完成")
    
    # 步骤5: 优化复合公式
    print("\n[5] 第二阶段：优化复合公式")
    result_composite = optimizer.optimize_formula_composite(iterations=100)
    print(f"    复合公式优化完成")
    
    # 步骤6: 优化高级公式
    print("\n[6] 第三阶段：优化高级公式")
    result_advanced = optimizer.optimize_formula_advanced(iterations=150)
    print(f"    高级公式优化完成")
    
    # 步骤7: 查看优化历程
    print("\n[7] 优化历程统计")
    history = optimizer.get_history()
    print("\n优化历史记录:")
    print(history.to_string())
    
    # 步骤8: 输出最佳结果
    print("\n[8] 最佳结果汇总")
    print(f"\n♥ 全局最佳公式:")
    print(f"  {optimizer.get_best_formula()}")
    print(f"\n综合得分: {optimizer.best_score:.6f}")
    
    # 步骤9: 导出 Pine Script
    print("\n[9] 导出 Pine Script 代码")
    pinescript = optimizer.export_to_pinescript(result_composite['params'], 'composite')
    print("\nPine Script 代码(会保存到 pinescript_output.pine文件):")
    print("-" * 60)
    print(pinescript)
    print("-" * 60)
    
    # 保存 Pine Script
    with open('pinescript_output.pine', 'w', encoding='utf-8') as f:
        f.write(pinescript)
    print("\n✓ Pine Script 代码已保存到 pinescript_output.pine")
    
    # 保存优化结果
    print("\n[10] 保存优化结果")
    history.to_csv('optimization_history.csv', index=False)
    print("    ✓ 优化历史已保存到 optimization_history.csv")
    
    # 打印推荐
    print("\n" + "="*80)
    print("推荐下一步:")
    print("="*80)
    print("""
    1. 将 pinescript_output.pine 中的代码复制到 TradingView 的 Pine Editor
    2. 添加到图表上返测高你寻找的反轉点
    3. 自定义參数以获得最优的交易信号
    4. 将优化历史 CSV 日文本罗上的改进您的整体策略
    """)

if __name__ == "__main__":
    main()
