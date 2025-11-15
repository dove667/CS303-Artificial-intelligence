# 🎯 项目结构概览

## 📁 目录结构

```
Project1/
├── agent.py                      # 增强型AI智能体（包含配置加载）
├── game_wrapper.py               # 游戏基础设施
├── optimize_weights.py           # 遗传算法优化器
├── evaluate.py                   # 锦标赛与评估框架
├── run_optimization.py           # 主优化流程
├── demo_optimization.py          # 快速演示脚本
├── test_components.py            # 组件测试套件
│
├── config/
│   └── default_weights.toml      # 基线权重配置
│
├── results/                      # 输出目录（自动创建）
│   └── run_YYYYMMDD_HHMMSS/     # 每次运行生成带时间戳的文件夹
│       ├── optimized_weights.toml
│       ├── best_genome.npy
│       ├── history.json
│       ├── evaluation.json
│       ├── fitness_evolution.png
│       └── fitness_distribution.png
│
├── README_OPTIMIZATION.md        # 完整文档
└── IMPLEMENTATION_SUMMARY.md     # 此概览
```

## 🚀 使用命令

### 1️⃣ 测试设置
```bash
cd /Users/dove/Desktop/AI/Project1
/Users/dove/Desktop/AI/.venv/bin/python test_components.py
```

### 2️⃣ 快速演示（约5分钟）
```bash
/Users/dove/Desktop/AI/.venv/bin/python demo_optimization.py
```

### 3️⃣ 完整优化

**快速探索（带对称性）**
```bash
/Users/dove/Desktop/AI/.venv/bin/python run_optimization.py \
    --symmetry \
    --pop-size 15 \
    --generations 20 \
    --search-depth 3
```

**彻底优化**
```bash
/Users/dove/Desktop/AI/.venv/bin/python run_optimization.py \
    --pop-size 25 \
    --generations 40 \
    --search-depth 4 \
    --games-per-eval 8
```

**生产质量**
```bash
/Users/dove/Desktop/AI/.venv/bin/python run_optimization.py \
    --pop-size 30 \
    --generations 50 \
    --search-depth 4 \
    --eval-depth 6 \
    --eval-games 100
```

### 4️⃣ 使用优化后的权重

**在你的代码中：**
```python
from agent import AI, COLOR_BLACK

# 加载优化后的权重
agent = AI(
    chessboard_size=8, 
    color=COLOR_BLACK, 
    time_out=4.9,
    config_path='Project1/results/run_xxx/optimized_weights.toml'
)

# 正常使用
candidate_list = agent.go(chessboard)
```

**提交给评测系统：**
将 `optimized_weights.toml` 复制到您的提交文件，并在 `agent.py` 中加载它。

## 📊 优化内容

### 12 启发式权重
```python
HURISTIC_WEIGHTS = {
    'begin':  (w1, w2, w3, w4),   # 游戏早期
    'middle': (w1, w2, w3, w4),   # 游戏中期  
    'end':    (w1, w2, w3, w4)    # 游戏后期
}
```
- w1: 棋盘位置权重
- w2: 稳定棋子权重
- w3: 棋子数量权重
- w4: 行动力权重

### 64 棋盘位置权重
```python
RWEIGHT_BOARD = [
    [-8,  0, -4, -1, -1, -4,  0, -8],
    [ 0, -2, -4, -2, -2, -4, -2,  0],
    ...
]
```

## 🎓 工作原理

### 遗传算法流程
```
1. 初始化随机权重的配置种群
2. 对于每一代：
   a. 评估适应度（进行游戏，衡量胜率）
   b. 选择父代（锦标赛选择）
   c. 创建子代（交叉 + 变异）
   d. 替换旧种群（精英保留 + 新子代）
3. 返回最佳配置
```

### 适应度评估
```
- 对抗基线/种群进行 N 场游戏
- 胜 = 1.0, 平局 = 0.5, 负 = 0.0
- 适应度 = (胜利数 + 0.5*平局数) / 总游戏数
- 基线游戏 (70%) + 自博弈 (30%) 混合
```

### 关键优化
- **进化过程中降低搜索深度** (depth=3) 以提高速度
- **Numba JIT 编译**以实现快速走法生成
- **锦标赛选择**以平衡探索/利用
- **精英保留**以保持最佳解
- **对称性约束**选项以减少参数

## 📈 预期表现

### 优化时间线
- **第 1-10 代**：快速提升（0.3 → 0.5 胜率）
- **第 11-30 代**：稳步进展（0.5 → 0.65）
- **第 31-50 代**：微调（0.65 → 0.75）

### 资源要求
- **快速演示**：约 5 分钟，最低限度 CPU
- **快速优化**：约 30-60 分钟
- **彻底优化**：约 2-4 小时
- **生产运行**：约 4-8 小时

### 成功指标
- 胜率 >60% 对抗基线 = 良好提升
- 胜率 >70% 对抗基线 = 优秀
- 在 30-40 代内收敛 = 健康进化
- 多样化的阶段权重 = 适当的阶段区分

## 🔧 定制点

### 1. 适应度函数（`optimize_weights.py`）
修改 `fitness()` 以纳入其他目标：
- 游戏时长
- 走法稳定性
- 对抗多个对手的表现

### 2. 进化算子
- 改变变异策略（高斯 vs 均匀）
- 调整交叉点
- 实现自适应变异率

### 3. 搜索策略
- 对不同阶段使用不同的深度
- 实现抱负窗口
- 添加置换表

### 4. 并行执行
在 `fitness()` 中添加多进程：
```python
from multiprocessing import Pool
with Pool() as pool:
    fitness_scores = pool.map(self.fitness, population)
```

## 🐛 故障排除

### 问题：ImportError: No module named 'tomli'
```bash
/Users/dove/Desktop/AI/.venv/bin/pip install tomli
```

### 问题：游戏太慢
- 将 `--search-depth` 减少到 2-3
- 将 `--games-per-eval` 减少到 4
- 使用 `--symmetry` 标志

### 问题：适应度没有提升
- 检查基线是否过强/过弱
- 增加变异率 (0.2-0.3)
- 增加种群多样性
- 需要更多代

### 问题：内存错误
- 减少种群规模
- 减少并行游戏数量
- 使用对称性约束

## 📚 关键文件解释

### `agent.py` (增强型)
- 原始的 minimax 智能体 + 配置加载
- `load_weights_from_config()` 读取 TOML 文件
- 向后兼容硬编码的权重

### `game_wrapper.py`
- `OthelloGame`: 游戏规则和状态管理
- `play_game()`: 智能体之间的单场比赛
- `play_match()`: 轮换颜色的多场游戏

### `optimize_weights.py`
- `WeightGenome`: 将 76 个参数编码为基因组
- `OthelloWeightOptimization`: GA（遗传算法）实现
  - 以合理的边界初始化种群
  - 通过锦标赛进行适应度评估
  - 选择、交叉、变异算子

### `evaluate.py`
- `TournamentManager`: 循环赛
- `ELOCalculator`: 评分系统
- `evaluate_against_baseline()`: 正面比较

### `run_optimization.py`
- 完整优化流程的命令行界面 (CLI)
- 处理所有参数
- 保存结果并生成图表

## 🎯 下一步

1. **验证设置**
   ```bash
   /Users/dove/Desktop/AI/.venv/bin/python test_components.py
   ```

2. **快速测试**
   ```bash
   /Users/dove/Desktop/AI/.venv/bin/python demo_optimization.py
   ```

3. **隔夜运行**
   ```bash
   nohup /Users/dove/Desktop/AI/.venv/bin/python run_optimization.py \
       --generations 50 --pop-size 30 > optimization.log 2>&1 &
   ```

4. **评估结果**
   - 检查 `results/run_xxx/fitness_evolution.png`
   - 查看 `results/run_xxx/evaluation.json`
   - 比较胜率

5. **部署最佳权重**
   - 将 `optimized_weights.toml` 复制到提交文件
   - 更新 `agent.py` 以加载配置
   - 在评测平台上测试

## 💡 专业提示

1. **从小处着手**：首先使用演示来理解流程
2. **使用对称性**：对于初步探索，收敛更快
3. **监控进度**：每次运行后检查图表
4. **尝试不同的种子**：多次运行，选择最佳结果
5. **增量改进**：从良好的基线开始，进行微调
6. **集成**：组合多次运行的权重

## 🏆 成功标准

- [ ] 测试通过 (`test_components.py`)
- [ ] 演示成功运行
- [ ] 完整优化无错误完成
- [ ] 对抗基线胜率 >60%
- [ ] 适应度收敛（不振荡）
- [ ] 优化后的智能体在评测系统上表现良好

