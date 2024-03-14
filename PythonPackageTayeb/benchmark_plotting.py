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


    for i, benchmark in enumerate(benchmarks):

        sim_data_vals = load_sim_data(benchmark['file_path'])
        benchmark['sim_data_vals'] = sim_data_vals

        graph_structure = benchmark['graph_structure']
        total_seqs = benchmark['total_seqs']
        consensus_theta = np.array(benchmark['consensus_theta'])
        avg_diff_values = np.array(benchmark['avg_diff_values'])
        obj_values = np.array(benchmark['obj_values'])
        num_processors = benchmark['num_processors']
        L = benchmark['L']
        q = benchmark['q']

        ### Select the set of items I want to plot -- Figure 1
        if set((6,6,'complete',8)) == set((L,q,graph_structure,num_processors)):
            print("L{}q{}, total_seq {},num_procs {}, graph {}".format(L,q,total_seqs,num_processors, graph_structure))
            
            results = vector_to_tensor_array(consensus_theta, L, q)
            out_fields = results[0]
            out_couplings = results[1]

            norm_couplings = np.linalg.norm(out_couplings, ord='fro', axis=(2, 3))
            norm_couplings = norm_couplings + norm_couplings.T
            benchmark['norm_couplings'] = norm_couplings
                
            if total_seqs==2048:
                im1_data = benchmark
            elif total_seqs == 4096:
                im2_data = benchmark
            elif total_seqs == 8192:
                im3_data = benchmark

        ### Select the set of items I want to plot -- Figure 2 Convergence
        if set((10,10,'complete',8)) == set((L,q,graph_structure,num_processors)):
            print("L{}q{}, total_seq {},num_procs {}, graph {}".format(L,q,total_seqs,num_processors, graph_structure))
            
            results = vector_to_tensor_array(consensus_theta, L, q)
            out_fields = results[0]
            out_couplings = results[1]

            norm_couplings = np.linalg.norm(out_couplings, ord='fro', axis=(2, 3))
            norm_couplings = norm_couplings + norm_couplings.T
            benchmark['norm_couplings'] = norm_couplings
                
            conv_data = benchmark

        ### Select the set of items I want to plot -- Figure 2 Convergence
        if set((10,10,'complete',8)) == set((L,q,graph_structure,num_processors)):
            print("L{}q{}, total_seq {},num_procs {}, graph {}".format(L,q,total_seqs,num_processors, graph_structure))
            
        continue
        


        results = vector_to_tensor_array(consensus_theta, L, q)
        out_fields = results[0]
        out_couplings = results[1]

        norm_couplings = np.linalg.norm(out_couplings, ord='fro', axis=(2, 3))
        norm_couplings = norm_couplings + norm_couplings.T

        fig = plt.figure(i*1+1)
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
        #plt.show()

        plt.figure(i*2+2)
        plt.plot(avg_diff_values[1:])
        plt.title(f"Average Diff Values: L={L}, q={q}, Graph={benchmark['graph_structure']}, Steps={benchmark['n_steps']}")
        plt.yscale('log')
        #plt.show()

        plt.figure(i*3+3)
        plt.plot(np.trim_zeros(obj_values, 'b'))
        plt.plot(moving_average(np.trim_zeros(obj_values, 'b'), benchmark['n_steps']))
        plt.title(f"Objective Values: L={L}, q={q}, Graph={benchmark['graph_structure']}, Steps={benchmark['n_steps']}")
        #plt.show()


    ### Figure 1 -- Effect of total sequence number
    fig1 = plt.figure(1, figsize=(15,6))
    ax1, ax2, ax3, ax4 = fig1.subplots(1,4)
    L = im1_data['L']

    im1 = ax1.imshow(im1_data['sim_data_vals'], cmap='viridis')
    ax1.set_title("Simulated",fontsize=20,fontweight='bold')
    ax1.set_xticks(np.arange(L))
    ax1.set_yticks(np.arange(L))
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Position")

    im2 = ax2.imshow(im1_data['norm_couplings'], cmap='viridis')
    ax2.set_title("MPF Seq={}".format(im1_data['total_seqs']),fontsize=20,fontweight='bold')
    ax2.set_xticks(np.arange(L))
    ax2.set_yticks(np.arange(L))
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Position")


    im3 = ax3.imshow(im2_data['norm_couplings'], cmap='viridis')
    ax3.set_title("MPF Seq={}".format(im2_data['total_seqs']),fontsize=20,fontweight='bold')
    ax3.set_xticks(np.arange(L))
    ax3.set_yticks(np.arange(L))
    ax3.set_xlabel("Position")
    ax3.set_ylabel("Position")

    im4 = ax4.imshow(im3_data['norm_couplings'], cmap='viridis')
    ax4.set_title("MPF Seq={}".format(im3_data['total_seqs']),fontsize=20,fontweight='bold')
    ax4.set_xticks(np.arange(L))
    ax4.set_yticks(np.arange(L))
    ax4.set_xlabel("Position")
    ax4.set_ylabel("Position")

    plt.tight_layout()
    fig1.subplots_adjust(right=0.9)
    cbar_ax = fig1.add_axes([0.92, 0.25, 0.02, 0.5])
    fig1.colorbar(im2, cax=cbar_ax)
    plt.savefig('../Figures/Figure1_TotalSeqComparison.png',dpi=300, bbox_inches='tight')
    plt.show()


    ## Convergence Figure
    fig2 = plt.figure(2, figsize=(15,5))
    ax1, ax2 = fig2.subplots(1,2)
    L = conv_data['L']
    
    ax1.plot(avg_diff_values[1:])
    ax1.set_yscale('log')
    ax1.set_ylabel("Average Objective Difference",fontsize=16)
    ax1.set_xlabel("Consensus Iterations",fontsize=16)

    ax2.plot(obj_values, 'b')
    ax2.set_xlabel("Total Iterations",fontsize=16)
    ax2.set_ylabel("Local Objective Value",fontsize=16)

    ax1.set_title(f"Overall Convergence",fontweight='bold',fontsize=20)
    ax2.set_title(f"Local Objective (Node 0)",fontweight='bold',fontsize=20)
    #plt.plot(moving_average(np.trim_zeros(obj_values, 'b'), benchmark['n_steps']))

    plt.savefig('../Figures/Figure2_ConvergenceL10q10.png',dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    #benchmark_file = 'benchmark_L5q4_L6q6_all.json'
    benchmark_file = 'benchmark_results.json'
    data = load_benchmark_data(benchmark_file)

    plot_benchmark_comparisons(data)