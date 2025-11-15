import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import json

from agent import AI, COLOR_BLACK, COLOR_WHITE
from optimize_weights import OthelloWeightOptimization, genome_to_config
from utils import evaluate_against_baseline, save_weights_to_toml, plot_optimization_history, plot_fitness_distribution


def main():
    parser = argparse.ArgumentParser(description='Optimize Reverse Othello agent weights')
    parser.add_argument('--pop-size', type=int, default=24, help='Population size')
    parser.add_argument('--generations', type=int, default=30, help='Number of generations')
    parser.add_argument('--games-per-eval', type=int, default=6, help='Games per fitness evaluation')
    parser.add_argument('--mutation-rate', type=float, default=0.15, help='Mutation rate')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--search-depth', type=int, default=5, help='Search depth during optimization')
    parser.add_argument('--eval-depth', type=int, default=5, help='Search depth for final evaluation')
    parser.add_argument('--eval-games', type=int, default=10, help='Games for final evaluation')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Reverse Othello Weight Optimization")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Population size: {args.pop_size}")
    print(f"  Generations: {args.generations}")
    print(f"  Games per evaluation: {args.games_per_eval}")
    print(f"  Mutation rate: {args.mutation_rate}")
    print(f"  Search depth (optimization): {args.search_depth}")
    print(f"  Search depth (evaluation): {args.eval_depth}")
    print(f"  Output directory: {run_dir}")
    print("=" * 70)
    
    # Create baseline agent (default weights)
    baseline_agent = AI(8, COLOR_WHITE, 4.9)
    baseline_agent.max_depth = args.search_depth
    
    # Initialize optimizer
    optimizer = OthelloWeightOptimization(
        pop_size=args.pop_size,
        mutation_rate=args.mutation_rate,
        games_per_evaluation=args.games_per_eval,
        baseline_agent=baseline_agent,
        search_depth_for_eval=args.search_depth
    )
    
    # Run optimization
    print("\nStarting optimization...")
    best_genome, history = optimizer.evolve(args.generations, verbose=True)
    
    # Convert best genome to config
    best_config = genome_to_config(best_genome)
    
    # Save weights
    weights_path = run_dir / "optimized_weights.toml"
    save_weights_to_toml(best_config, weights_path)
    
    # Save raw genome
    genome_path = run_dir / "best_genome.npy"
    np.save(genome_path, best_genome)
    print(f"Saved genome to: {genome_path}")
    
    # Save history
    history_path = run_dir / "history.json"
    history_data = {
        'generations': [gen for gen, _ in history],
        'fitness_scores': [scores for _, scores in history]
    }
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"Saved history to: {history_path}")
    
    # Plot results
    if not args.no_plot:
        print("\nGenerating plots...")
        plot_optimization_history(history, run_dir / "fitness_evolution.png")
        plot_fitness_distribution(history, run_dir / "fitness_distribution.png")
    
    # Final evaluation against baseline with full search depth
    print("\n" + "=" * 70)
    print("Final Evaluation (full search depth)")
    print("=" * 70)
    
    # Create optimized agent
    optimized_agent = AI(8, COLOR_BLACK, 4.9)
    optimized_agent.max_depth = args.eval_depth
    # Load optimized weights
    optimized_agent.load_weights_from_config(weights_path)
    
    # Create baseline for comparison
    baseline_eval = AI(8, COLOR_WHITE, 4.9)
    baseline_eval.max_depth = args.eval_depth
    
    eval_result = evaluate_against_baseline(
        optimized_agent,
        baseline_eval,
        agent_name="Optimized Agent",
        baseline_name="Baseline Agent",
        num_games=args.eval_games,
        verbose=True
    )
    
    # Save evaluation results
    eval_path = run_dir / "evaluation.json"
    with open(eval_path, 'w') as f:
        json.dump(eval_result, f, indent=2)
    print(f"\nSaved evaluation to: {eval_path}")
    
    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print(f"Results saved to: {run_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
