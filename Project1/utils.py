import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from agent import AI
from game_wrapper import play_match

def save_weights_to_toml(config: dict, output_path: Path):
    """Save optimized weights to TOML format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def format_array(arr):
        """Format array for TOML (ensure proper integer formatting)."""
        if isinstance(arr, (list, tuple)):
            return '[' + ', '.join(str(int(x)) for x in arr) + ']'
        else:
            return '[' + ', '.join(str(int(x)) for x in arr.tolist()) + ']'
    
    with open(output_path, 'w') as f:
        f.write("# Optimized weight configuration for Reverse Othello Agent\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write HURISTIC_WEIGHTS
        hw = config['HURISTIC_WEIGHTS']
        f.write("[HURISTIC_WEIGHTS]\n")
        f.write(f"begin = {format_array(hw['begin'])}\n")
        f.write(f"middle = {format_array(hw['middle'])}\n")
        f.write(f"end = {format_array(hw['end'])}\n\n")
        
        # Write RWEIGHT_BOARD as separate rows
        board = config['RWEIGHT_BOARD']
        f.write("[RWEIGHT_BOARD]\n")
        for i, row in enumerate(board):
            f.write(f"row{i} = {format_array(row)}\n")
    
    print(f"Saved weights to: {output_path}")


def plot_fitness_distribution(history: list, output_path: Path):
    """Plot fitness distribution at key generations."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Select generations to plot (start, 1/3, 2/3, end)
    n_gens = len(history)
    indices = [0, n_gens // 3, 2 * n_gens // 3, n_gens - 1]
    
    for idx, gen_idx in enumerate(indices):
        if gen_idx < len(history):
            gen, scores = history[gen_idx]
            ax = axes[idx]
            ax.hist(scores, bins=15, edgecolor='black', alpha=0.7)
            ax.set_title(f'Generation {gen}')
            ax.set_xlabel('Fitness')
            ax.set_ylabel('Count')
            ax.set_xlim([0, 1])
    
    plt.suptitle('Fitness Distribution Evolution')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved distribution plot to: {output_path}")
    plt.close()


def plot_optimization_history(history: list, output_path: Path):
    """Plot fitness evolution over generations."""
    generations = [gen for gen, _ in history]
    
    # Extract statistics per generation
    best_fitness = [max(scores) for _, scores in history]
    avg_fitness = [np.mean(scores) for _, scores in history]
    worst_fitness = [min(scores) for _, scores in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, 'g-', label='Best', linewidth=2)
    plt.plot(generations, avg_fitness, 'b-', label='Average', linewidth=2)
    plt.plot(generations, worst_fitness, 'r-', label='Worst', linewidth=1)
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Win Rate)')
    plt.title('Genetic Algorithm Optimization Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def evaluate_against_baseline(agent: AI,
                              baseline: AI,
                              agent_name: str = "Optimized",
                              baseline_name: str = "Baseline",
                              num_games: int = 50,
                              board_size: int = 8,
                              verbose: bool = True) -> dict:
    """
    Evaluate an agent against a baseline opponent.
    
    Args:
        agent: Agent to evaluate
        baseline: Baseline opponent
        agent_name: Name for the evaluated agent
        baseline_name: Name for the baseline
        num_games: Number of games to play
        board_size: Board size
        verbose: Print progress
        
    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print(f"Evaluating {agent_name} vs {baseline_name}")
        print(f"Playing {num_games} games...")
    
    start_time = time.time()
    
    result = play_match(
        agent,
        baseline,
        num_games=num_games,
        board_size=board_size,
        reverse_mode=True,
        verbose=False
    )
    
    elapsed = time.time() - start_time
    
    win_rate = (result['agent1_wins'] + 0.5 * result['draws']) / result['total_games']
    
    if verbose:
        print(f"\nResults:")
        print(f"  {agent_name}: {result['agent1_wins']} wins ({result['agent1_wins']/num_games*100:.1f}%)")
        print(f"  {baseline_name}: {result['agent2_wins']} wins ({result['agent2_wins']/num_games*100:.1f}%)")
        print(f"  Draws: {result['draws']} ({result['draws']/num_games*100:.1f}%)")
        print(f"  Win rate: {win_rate*100:.1f}%")
        print(f"  Time: {elapsed:.1f}s ({elapsed/num_games:.2f}s per game)")
    
    return {'agent_wins': result['agent1_wins'],
            'baseline_wins': result['agent2_wins'],
            'draws': result['draws'],
            'total_games': result['total_games'],
            'win_rate': win_rate,
            'time_elapsed': elapsed,
            'time_per_game': elapsed / num_games}
