# 反转黑白棋权重优化 - 快速开始

## ⚡ 快速入门

1️⃣ 测试所有组件
    `python test_components.py`

2️⃣ 快速演示
    `python demo_optimization.py`

3️⃣ 快速优化
    `python run_optimization.py --pop-size 15 --generations 20`

4️⃣ 完整优化
    `python run_optimization.py --pop-size 25 --generations 40`

## 📊 正在优化的参数

启发式权重 HURISTIC_WEIGHTS (12 个参数):
  begin:  [w1, w2, w3, w4]  → 游戏早期 (≤20 棋子)
  middle: [w1, w2, w3, w4]  → 游戏中期 (21-40 棋子)
  end:    [w1, w2, w3, w4]  → 游戏后期 (>40 棋子)
  
  w1 = 棋盘位置权重
  w2 = 稳定棋子权重
  w3 = 棋子数量权重
  w4 = 行动力权重

RWEIGHT_BOARD (64 个参数):
  8×8 位置权重矩阵
  负值 = 避免, 正值 = 偏好

总计: 76 个参数 (或使用 --symmetry 时为 22 个)

## ⚙️ 关键命令行选项

```bash
--pop-size N           种群大小 (默认值: 20)
--generations N        代数 (默认值: 30)
--games-per-eval N     每次适应度评估的游戏数 (默认值: 6)
--mutation-rate X      变异概率 (默认值: 0.15)
--crossover-rate X     交叉概率 (默认值: 0.7)
--symmetry             使用对称棋盘 (更快, 22 个参数)
--self-play-ratio X    自博弈与基线游戏的比例 (默认值: 0.3)
--search-depth N       优化期间的搜索深度 (默认值: 3)
--eval-depth N         最终评估的搜索深度 (默认值: 5)
--eval-games N         最终评估的游戏数 (默认值: 50)
--output-dir PATH      输出目录 (默认值: Project1/results)
```

## 📁 输出文件

results/run_YYYYMMDD_HHMMSS/
  ├─ optimized_weights.toml     ← 在 agent.py 中使用此文件
  ├─ best_genome.npy             (原始基因组向量)
  ├─ history.json                (进化历史)
  ├─ evaluation.json             (最终性能)
  ├─ fitness_evolution.png       (进度图表)
  └─ fitness_distribution.png    (种群多样性)

## 💻 使用优化后的权重

from agent import AI, COLOR_BLACK

### 加载优化后的配置

```python
agent = AI(
    chessboard_size=8,
    color=COLOR_BLACK,
    time_out=4.9,
    config_path='Project1/results/run_xxx/optimized_weights.toml'
)
```
### 正常使用
```python
candidate_list = agent.go(chessboard)
```

## 📚 文档文件

OVERVIEW.md                    完整概览和使用指南
README_OPTIMIZATION.md         详细文档
IMPLEMENTATION_SUMMARY.md      技术实现细节
QUICKSTART.md                  此文件

## 🎯 推荐工作流程

阶段 1: 探索 (带对称性，快速)
  `python run_optimization.py --symmetry --pop-size 15 --generations 20 --search-depth 3`

阶段 2: 微调 (全部参数)
  `python run_optimization.py --pop-size 25 --generations 40 --search-depth 4`

阶段 3: 最终评估 (深度搜索)
  `python run_optimization.py --pop-size 30 --generations 50 --search-depth 4 --eval-depth 6 --eval-games 100`

## 🎓 成功指标

✓ 胜率 >60% 对抗基线   = 良好提升
✓ 胜率 >70% 对抗基线   = 优秀!
✓ 收敛 <40 代         = 健康进化
✓ 稳定适应度曲线        = 不振荡
✓ 不同阶段权重          = 适当的区分

## 🐛 故障排除

❌ ImportError: tomli
   → pip install tomli

❌ 游戏太慢
   → 使用 --search-depth 2 或 3
   → 使用 --symmetry 标志
   → 将 --games-per-eval 减少到 4

❌ 没有提升
   → 将 --mutation-rate 增加到 0.2-0.3
   → 将 --pop-size 增加到 30
   → 更多 --generations (50+)

❌ 内存错误
   → 减少 --pop-size
   → 使用 --symmetry
   → 关闭其他应用程序

## 🚀 后台执行

对于长时间运行，使用 nohup:
```bash
nohup python run_optimization.py --pop-size 30 --generations 50 > optimization.log 2>&1 &

检查进度:
  tail -f optimization.log

如有需要，终止进程:
  ps aux | grep run_optimization
  kill <PID>
```
## 📞 需要帮助?

1. 阅读 OVERVIEW.md 获取详细指南
2. 运行 test_components.py 验证设置
3. 首先尝试 demo_optimization.py
4. 检查 results/*/evaluation.json 获取性能指标
5. 查看 results/*/fitness_evolution.png 了解进度