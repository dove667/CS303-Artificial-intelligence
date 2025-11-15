# Reverse Othello Weight Optimization

This directory contains tools for optimizing agent weights using genetic algorithms.

## Files

- **agent.py**: Main AI agent with minimax search (original implementation with config loading capability)
- **game_wrapper.py**: Othello game manager for tournament play
- **optimize_weights.py**: Genetic algorithm implementation for weight optimization
- **evaluate.py**: Tournament framework and ELO rating system
- **run_optimization.py**: Main pipeline script for running optimization
- **config/**: Directory for weight configuration files

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy numba matplotlib tomli
```

### 2. Run Optimization

Basic usage with default parameters:

```bash
python run_optimization.py
```

With custom parameters:

```bash
python run_optimization.py \
    --pop-size 30 \
    --generations 50 \
    --games-per-eval 10 \
    --mutation-rate 0.2 \
    --search-depth 4 \
    --output-dir Project1/results
```

### 3. Use Optimized Weights

Load optimized weights in your agent:

```python
from agent import AI

# Create agent with custom weights
agent = AI(8, COLOR_BLACK, 4.9, config_path='Project1/results/run_xxx/optimized_weights.toml')
```

## Configuration Parameters

### Optimization Parameters

- `--pop-size`: Population size (default: 20)
- `--generations`: Number of generations (default: 30)
- `--games-per-eval`: Games per fitness evaluation (default: 6)
- `--mutation-rate`: Mutation probability (default: 0.15)
- `--crossover-rate`: Crossover probability (default: 0.7)
- `--symmetry`: Enforce board symmetry (reduces parameters from 76 to 22)
- `--self-play-ratio`: Ratio of self-play vs baseline games (default: 0.3)

### Search Parameters

- `--search-depth`: Search depth during optimization (default: 3, faster)
- `--eval-depth`: Search depth for final evaluation (default: 5, more accurate)
- `--eval-games`: Number of games for final evaluation (default: 50)

### Output

- `--output-dir`: Output directory (default: Project1/results)
- `--no-plot`: Skip generating plots

## Weight Parameters

### HURISTIC_WEIGHTS (12 parameters)

Dynamic weights for different game stages:

- **begin** (pieces ≤ 20): `[w1, w2, w3, w4]`
- **middle** (21 ≤ pieces ≤ 40): `[w1, w2, w3, w4]`
- **end** (pieces > 40): `[w1, w2, w3, w4]`

Where:
- `w1`: Board position weight coefficient
- `w2`: Stable disk coefficient
- `w3`: Piece count coefficient
- `w4`: Mobility coefficient

### RWEIGHT_BOARD (64 parameters)

8×8 position weight matrix. Negative values indicate positions to avoid.

With `--symmetry` flag, only 10 unique values are optimized (8-fold symmetry).

## Output Structure

```
Project1/results/run_YYYYMMDD_HHMMSS/
├── optimized_weights.toml    # Optimized weights in TOML format
├── best_genome.npy            # Raw genome vector
├── history.json               # Fitness history per generation
├── evaluation.json            # Final evaluation results
├── fitness_evolution.png      # Fitness over generations plot
└── fitness_distribution.png   # Fitness distribution plot
```

## Example: Tournament Evaluation

Compare multiple weight configurations:

```python
from agent import AI, COLOR_BLACK, COLOR_WHITE
from evaluate import TournamentManager

# Create tournament
tournament = TournamentManager(games_per_matchup=20)

# Add agents with different configurations
agent1 = AI(8, COLOR_BLACK, config_path='config/default_weights.toml')
agent2 = AI(8, COLOR_BLACK, config_path='results/run_xxx/optimized_weights.toml')
agent3 = AI(8, COLOR_BLACK, config_path='results/run_yyy/optimized_weights.toml')

tournament.add_agent("Default", agent1)
tournament.add_agent("Optimized_v1", agent2)
tournament.add_agent("Optimized_v2", agent3)

# Run tournament
results = tournament.run_tournament(verbose=True)

# Save results
tournament.save_results(Path('tournament_results.json'))
```

## Performance Tips

1. **Parallel Evaluation**: Modify `fitness()` in `optimize_weights.py` to use multiprocessing for parallel game execution

2. **Adaptive Depth**: Start with shallow search depth (3) and gradually increase in later generations

3. **Early Stopping**: Add convergence detection to stop when fitness plateaus

4. **Population Diversity**: Increase mutation rate if population converges too quickly

5. **Hybrid Approach**: 
   - First optimize with symmetry constraint (faster, 22 parameters)
   - Then fine-tune without constraint (full 76 parameters)

## Troubleshooting

### ImportError: No module named 'tomli'

Install tomli:
```bash
pip install tomli
```

### Games are too slow

- Reduce `--search-depth` (default 3 is already fast)
- Reduce `--games-per-eval`
- Use `--symmetry` to reduce parameter space

### Fitness not improving

- Increase `--mutation-rate`
- Increase `--pop-size`
- Increase `--self-play-ratio` for more diversity
- Check baseline agent is not too strong/weak

## Advanced: Custom Fitness Function

Modify `fitness()` in `optimize_weights.py` to include additional metrics:

```python
def fitness(self, genome: np.ndarray, population: Optional[List[np.ndarray]] = None) -> float:
    agent = self.create_agent_from_genome(genome)
    
    # Multi-objective fitness
    win_rate = self.evaluate_win_rate(agent)
    avg_game_length = self.evaluate_game_length(agent)
    stability = self.evaluate_move_consistency(agent)
    
    # Weighted combination
    fitness = 0.7 * win_rate + 0.2 * stability + 0.1 * (1 - avg_game_length / 60)
    return fitness
```
