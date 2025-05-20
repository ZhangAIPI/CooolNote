import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

# Load CKA similarity matrices
deepseek_ckas = []

with open("./full_cka_weight_mat_list_qwen.pkl", "rb") as file:
    result = pkl.load(file)
    for layer_cmb in result:
        deepseek_ckas.append(layer_cmb[1])  # D^l
     

def compute_redundancy_from_matrices(matrices):
    scores = []
    for D in matrices:
        N = D.shape[0]
        off_diag_sum = np.sum(D) - np.trace(D)
        R_l = off_diag_sum / (N * (N - 1))
        scores.append(R_l)
    R = np.array(scores)
    R_min, R_max = R.min(), R.max()
    return (R - R_min) / (R_max - R_min)

# Use real data that was provided earlier
deepseek_scores = compute_redundancy_from_matrices(deepseek_ckas)


# Plot with actual number of layers per model
fig, axs = plt.subplots(1, 1, figsize=(10, 8))

colors = ['#D95F02']
titles = ['QwenMoE']
score_sets = [deepseek_scores]


axs.bar(np.arange(len(deepseek_scores)), deepseek_scores, color=colors[0], width=0.6)

axs.set_ylabel("Norm. Redundancy")
axs.set_ylim(0, 1.05)
axs.grid(axis="y", linestyle="--", alpha=0.5)



plt.tight_layout()

plt.savefig("qwen_x_layer_cmp_other.png")