import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class BuffonExperiment:
    def __init__(self, needle_length: float = 1.0, line_distance: float = 2.0):
        self.needle_length = needle_length
        self.line_distance = line_distance
        if needle_length > line_distance:
            print(f"警告: 针长({needle_length})超过线间距({line_distance}), 结果可能不准确")
    
    def simulate(self, num_trials: int) -> float:
        np.random.seed(42)
        y_center = np.random.uniform(0, self.line_distance/2, num_trials)
        theta = np.random.uniform(0, np.pi/2, num_trials)
        half_projection = (self.needle_length/2) * np.sin(theta)
        intersections = y_center <= half_projection
        num_hits = np.sum(intersections)
        return (2 * self.needle_length * num_trials) / (self.line_distance * num_hits) if num_hits != 0 else 0
    
    def run_experiment(self, trial_sizes: List[int]) -> Tuple[List[float], List[float]]:
        pi_estimates, errors = [], []
        for size in trial_sizes:
            est = self.simulate(size)
            err = abs(est - np.pi) if est != 0 else float('inf')
            pi_estimates.append(est)
            errors.append(err)
            print(f"次数:{size}, 估计值:{est:.8f}, 误差:{err:.8f}")
        return pi_estimates, errors
    
    def visualize_results(self, trial_sizes: List[int], pi_estimates: List[float]) -> None:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(trial_sizes, pi_estimates, 'o-', label='估计值')
        plt.axhline(np.pi, c='r', ls='--', label='真实值')
        plt.xscale('log'), plt.legend(), plt.grid(True)
        plt.xlabel('实验次数'), plt.ylabel('π估计值'), plt.title('估计值变化趋势')
        
        plt.subplot(1, 2, 2)
        errors = [abs(est - np.pi) for est in pi_estimates]
        plt.loglog(trial_sizes, errors, 'o-')
        plt.xlabel('实验次数'), plt.ylabel('绝对误差'), plt.title('误差变化趋势')
        plt.grid(True, which='both')
        
        plt.tight_layout(), plt.savefig('buffon_experiment_results.png'), plt.show()

def main():
    trial_sizes = [100, 1000, 10000, 100000, 1000000]
    experiment = BuffonExperiment()
    pi_estimates, _ = experiment.run_experiment(trial_sizes)
    experiment.visualize_results(trial_sizes, pi_estimates)

if __name__ == "__main__":
    main()
