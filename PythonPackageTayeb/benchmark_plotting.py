import os,sys

import json
import numpy as np
import matplotlib.pyplot as plt

#Local Imports
parent_dir = os.path.join(os.path.dirname(__file__), '..')
normalized_path = os.path.normpath(parent_dir)
sys.path.append(normalized_path)
from utils import *
from dgd_helpers import *

def load_benchmark_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_sim_data(file_path):
    data = sp.io.loadmat(file_path)
    sim_jfro = data["Jfro"] # Input Frobenius norm
    return sim_jfro

def plot_benchmark_comparisons(benchmarks):
    for benchmark in benchmarks:
        consensus_theta = np.array(benchmark['consensus_theta'])
        avg_diff_values = np.array(benchmark['avg_diff_values'])
        obj_values = np.array(benchmark['obj_values'])
        num_processors = benchmark['num_processors']

        sim_data_vals = load_sim_data(benchmark['file_path'])

        L = benchmark['L']
        q = benchmark['q']

        results = vector_to_tensor_array(consensus_theta, L, q)
        out_fields = results[0]
        out_couplings = results[1]

        norm_couplings = np.linalg.norm(out_couplings, ord='fro', axis=(2, 3))
        norm_couplings = norm_couplings + norm_couplings.T

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        im1 = ax1.imshow(sim_data_vals, cmap='viridis')
        ax1.set_title("Simulated Couplings")
        ax1.set_xticks(np.arange(L))
        ax1.set_yticks(np.arange(L))
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Position")

        im2 = ax2.imshow(norm_couplings, cmap='viridis')
        ax2.set_title("MPF Inferred Couplings")
        ax2.set_xticks(np.arange(L))
        ax2.set_yticks(np.arange(L))
        ax2.set_xlabel("Position")
        ax2.set_ylabel("Position")



        fig.suptitle(f"Couplings: L={L}, q={q}, Graph={benchmark['graph_structure']}, Num Processors={num_processors}",fontsize=20)

        plt.tight_layout()
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im2, cax=cbar_ax)
        plt.show()

        plt.figure(2)
        plt.plot(avg_diff_values[1:])
        plt.title(f"Average Diff Values: L={L}, q={q}, Graph={benchmark['graph_structure']}, Steps={benchmark['n_steps']}")
        plt.yscale('log')
        plt.show()

        plt.figure(3)
        plt.plot(np.trim_zeros(obj_values, 'b'))
        plt.plot(moving_average(np.trim_zeros(obj_values, 'b'), benchmark['n_steps']))
        plt.title(f"Objective Values: L={L}, q={q}, Graph={benchmark['graph_structure']}, Steps={benchmark['n_steps']}")
        plt.show()

if __name__ == "__main__":
    benchmark_file = 'benchmark_results.json'
    data = load_benchmark_data(benchmark_file)

    plot_benchmark_comparisons(data)