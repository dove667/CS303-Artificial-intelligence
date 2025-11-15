"""
Quick demo script to test the optimization pipeline on a small scale.
"""

from agent import AI, COLOR_BLACK, COLOR_WHITE
from optimize_weights import OthelloWeightOptimization, genome_to_config
from utils import evaluate_against_baseline
from pathlib import Path

def main():
    print("=" * 60)
    print("Quick Demo: Weight Optimization")
    print("=" * 60)
    print("\nThis is a quick demo with minimal parameters.")
    print("For full optimization, use run_optimization.py\n")
    
    # Create baseline agent
    baseline = AI(8, COLOR_WHITE, 4.9)
    baseline.max_depth = 3
    
    # Create optimizer with minimal settings
    optimizer = OthelloWeightOptimization(
        pop_size=10,              # Small population
        mutation_rate=0.2,
        games_per_evaluation=4,    # Few games per eval
        baseline_agent=baseline,
        search_depth_for_eval=3    # Shallow search
    )
    
    print("Running optimization...")
    print("  Population: 10")
    print("  Generations: 5")
    print("  Games per eval: 4")
    print("  Search depth: 3")
    print()
    
    # Run for just 5 generations
    best_genome, history = optimizer.evolve(num_generations=5, verbose=True)
    
    # Show results
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    
    # Show best weights
    best_config = genome_to_config(best_genome)
    print("\nBest Huristic Weights:")
    for stage, weights in best_config['HURISTIC_WEIGHTS'].items():
        print(f"  {stage}: {weights}")
    
    print("\nBest Board Weights (first row):")
    print(f"  {best_config['RWEIGHT_BOARD'][0]}")
    
    # Quick evaluation
    print("\n" + "=" * 60)
    print("Quick Evaluation vs Baseline")
    print("=" * 60)
    
    optimized = AI(8, COLOR_BLACK, 4.9)
    optimized.max_depth = 3
    
    # Apply optimized weights (monkey-patch for demo)
    import agent as agent_module
    agent_module.HURISTIC_WEIGHTS = best_config['HURISTIC_WEIGHTS']
    agent_module.RWEIGHT_BOARD = best_config['RWEIGHT_BOARD']
    
    result = evaluate_against_baseline(
        optimized,
        baseline,
        agent_name="Optimized",
        baseline_name="Baseline",
        num_games=10,
        verbose=True
    )
    
    print("\nFor full optimization, run:")
    print("  python run_optimization.py --generations 30 --pop-size 20")

if __name__ == "__main__":
    main()
