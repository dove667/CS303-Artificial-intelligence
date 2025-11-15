import numpy as np
import random
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from agent import AI, COLOR_BLACK, COLOR_WHITE, RWEIGHT_BOARD
from game_wrapper import play_match


@dataclass
class WeightGenome:
    """12个阶段性启发式权重的基因表示。

    索引布局:
        0-3   : 开局(begin)  [位置, 稳定子, 子数, 行动力]
        4-7   : 中局(middle) [位置, 稳定子, 子数, 行动力]
        8-11  : 终局(end)    [位置, 稳定子, 子数, 行动力]

    棋盘位置表 `RWEIGHT_BOARD` 固定，不在优化范围内。
    """
    weights: np.ndarray  # shape (12,)

    @classmethod
    def from_vector(cls, vector: np.ndarray):
        return cls(weights=vector[:12])

    def to_vector(self) -> np.ndarray:
        return self.weights.copy()

    def to_agent_params(self) -> dict:
        w_begin = tuple(self.weights[0:4].astype(int))
        w_middle = tuple(self.weights[4:8].astype(int))
        w_end = tuple(self.weights[8:12].astype(int))
        return {
            'HURISTIC_WEIGHTS': {
                'begin': w_begin,
                'middle': w_middle,
                'end': w_end
            },
            'RWEIGHT_BOARD': RWEIGHT_BOARD
        }


class OthelloWeightOptimization:
    """遗传算法：优化 12 个阶段性启发式评估权重。

    参数:
        pop_size: 种群规模
        mutation_rate: 基因变异率
        elite_ratio: 精英比例
        games_per_evaluation: 每次适应度评估的对局数
        baseline_agent: 用于对比评估的基线代理
        board_size: 棋盘大小
        time_limit: 每步时间限制
        search_depth_for_eval: 评估时的搜索深度
        stagnation_patience: 早停耐心代数
        adaptive_mutation_factor: 自适应变异因子
    """

    def __init__(self,
                 pop_size: int = 24,
                 mutation_rate: float = 0.12,
                 elite_ratio: float = 0.12,
                 games_per_evaluation: int = 6,
                 baseline_agent: Optional[AI] = None,
                 board_size: int = 8,
                 time_limit: float = 4.9,
                 search_depth_for_eval: int = 5,
                 stagnation_patience: int = 8,
                 adaptive_mutation_factor: float = 1.6,):

        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.base_mutation_rate = mutation_rate
        self.elite_count = max(1, int(pop_size * elite_ratio))
        self.games_per_evaluation = games_per_evaluation
        self.baseline_agent = baseline_agent
        self.board_size = board_size
        self.time_limit = time_limit
        self.search_depth_for_eval = search_depth_for_eval
        self.stagnation_patience = stagnation_patience
        self.adaptive_mutation_factor = adaptive_mutation_factor

        self.best_fitness_so_far = -1.0
        self.generations_without_improvement = 0
        self.fitness_cache = {}
        self.genome_size = 12
        self.huristic_bounds = (0, 10)

    def init_population(self) -> List[np.ndarray]:
        """
        初始化种群。
        """
        population = []
        for _ in range(self.pop_size):
            genome = np.random.randint(self.huristic_bounds[0],
                                       self.huristic_bounds[1] + 1,
                                       size=self.genome_size).astype(np.float32)
            population.append(genome)
        return population

    def create_agent_from_genome(self, genome: np.ndarray) -> AI:
        """
        从基因组创建一个AI。
        """
        weight_genome = WeightGenome.from_vector(genome)
        params = weight_genome.to_agent_params()
        agent = AI(self.board_size, COLOR_BLACK, self.time_limit)
        agent.huristic_weights = params['HURISTIC_WEIGHTS']
        agent.max_depth = self.search_depth_for_eval
        return agent

    def fitness(self, genome: np.ndarray) -> float:
        """
        计算基因组的适应度分数。
        """
        key = genome.tobytes()
        if key in self.fitness_cache:
            return self.fitness_cache[key]
        agent = self.create_agent_from_genome(genome)
        wins = draws = total_games = 0
        baseline_games = int(self.games_per_evaluation)
        if baseline_games > 0 and self.baseline_agent is not None:
            result = play_match(agent, self.baseline_agent, baseline_games, self.board_size, True, False)
            wins += result['agent1_wins']
            draws += result['draws']
            total_games += result['total_games']
        if total_games == 0:
            return 0.0
        score = (wins + 0.5 * draws) / total_games
        self.fitness_cache[key] = score
        return score

    def select(self, r, population: List[np.ndarray], fitness_scores: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        从种群中选择k个个体进行繁殖。
        """
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            probabilities = [1 / len(population)] * len(population)
        else:
            epsilon = 1e-2  # small constant to avoid zero probability
            probabilities = [score + epsilon / total_fitness for score in fitness_scores]
        selected = random.choices(population, weights=probabilities, k=r)
        return selected

    def recombine(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        基因重组，生成yi个子代。
        """
        p1 = random.randint(1, self.genome_size // 2)
        p2 = random.randint(p1 + 1, self.genome_size - 1)
        c1 = parent1.copy(); c2 = parent2.copy()
        c1[p1:p2] = parent2[p1:p2]; c2[p1:p2] = parent1[p1:p2]
        return c1, c2

    def mutate(self, genome: np.ndarray) -> np.ndarray:
        """
        基因突变，对基因进行小幅度扰动。
        """
        g = genome.copy()
        for i in range(self.genome_size):
            if random.random() < self.mutation_rate:
                scale = 1.5 if self.generations_without_improvement == 0 else 2.2
                g[i] = np.clip(g[i] + np.random.normal(0, scale), self.huristic_bounds[0], self.huristic_bounds[1])
        return g

    def reproduce(self, population_sorted: List[np.ndarray], fitness_sorted: List[float]) -> List[np.ndarray]:
        """
        生成新一代种群。
        """
        new_pop: List[np.ndarray] = []
        new_pop.extend(population_sorted[:self.elite_count])
        while len(new_pop) < self.pop_size:
            p1, p2 = self.select(2, population_sorted, fitness_sorted)
            c1, c2 = self.recombine(p1, p2)
            c1 = self.mutate(c1); c2 = self.mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < self.pop_size:
                new_pop.append(c2)
        return new_pop
    
    def evolve(self, num_generations: int, verbose: bool = True) -> Tuple[np.ndarray, List]:
        """
        进化算法主循环。
        """
        population = self.init_population()
        history: List[Tuple[int, List[float]]] = []
        for gen in range(num_generations):
            t0 = time.time()

            if verbose:
                print(f"\nGeneration {gen+1}/{num_generations}")

            scores = [self.fitness(g) for g in population]
            history.append((gen, scores.copy()))

            order = np.argsort(scores)[::-1] # 降序
            population_sorted = [population[i] for i in order]
            fitness_sorted = [scores[i] for i in order]
            best = fitness_sorted[0]; avg = float(np.mean(scores))

            if verbose:
                print(f"Best {best:.3f} | Avg {avg:.3f} | mutation {self.mutation_rate:.3f} | time {time.time()-t0:.1f}s")

            # 早停与自适应变异
            if best > self.best_fitness_so_far + 1e-6:
                self.best_fitness_so_far = best
                self.generations_without_improvement = 0
                self.mutation_rate = self.base_mutation_rate
            else:
                self.generations_without_improvement += 1
                if self.generations_without_improvement >= 2:
                    self.mutation_rate = min(0.6, self.mutation_rate * self.adaptive_mutation_factor)
            if self.generations_without_improvement >= self.stagnation_patience:
                if verbose:
                    print(f"早停: {self.stagnation_patience} 代无提升")
                break
            if gen == num_generations - 1:
                break

            new_pop = self.reproduce(population_sorted, fitness_sorted)
            population = new_pop

        final_scores = [self.fitness(g) for g in population]
        best_idx = int(np.argmax(final_scores))
        best_genome = population[best_idx]

        if verbose:
            print("\n" + "="*50)
            print(f"Optimization complete. Best {final_scores[best_idx]:.3f}")

        return best_genome, history


def genome_to_config(genome: np.ndarray) -> dict:
    """12 基因 -> Agent 参数配置。"""
    weight_genome = WeightGenome.from_vector(genome)
    return weight_genome.to_agent_params()
