import numpy as np
from scipy.sparse.csgraph import laplacian
from sklearn.preprocessing import normalize
import pickle as pkl
import os

def build_block_diag_knn_graph(full_results, num_layers, num_experts_per_layer, top_k=4):
    total_experts = num_layers * num_experts_per_layer
    full_graph = -np.ones((total_experts, total_experts))

    sims_list = []

    for layer in range(num_layers):
        sim = full_results[layer][1]
        sims_list.extend(sim.reshape(-1).tolist())
    
    # normalize sims_list
    # sims_list = normalize(sims_list)

    sims_list = np.array(sims_list)
    sims_list[sims_list==0] = 1
    
    # import pdb;pdb.set_trace()
    
    
    # 0-1 归一化
    sims_list = (sims_list - sims_list.min()) / (sims_list.max() - sims_list.min())

    sims_list = np.log(sims_list + 1e-6)

    sims_list = (sims_list - sims_list.min()) / (sims_list.max() - sims_list.min())
    # import pdb;pdb.set_trace()


    layer_redundancy = []
    for layer in range(num_layers):
        sim = sims_list[layer * num_experts_per_layer*num_experts_per_layer : (layer + 1) * num_experts_per_layer*num_experts_per_layer]
        layer_redundancy.append(sim.sum().item()-num_experts_per_layer)
        sim = sim.reshape(num_experts_per_layer, num_experts_per_layer)
        # sim = np.log(sim + 1e-6)
        full_graph[layer * num_experts_per_layer : (layer + 1) * num_experts_per_layer, layer * num_experts_per_layer : (layer + 1) * num_experts_per_layer] = sim
    print("layer_redundancy: {}".format(layer_redundancy))
    # import pdb;pdb.set_trace()

    
    return full_graph, sims_list






with open("./full_cka_weight_mat_list_qwen.pkl", 'rb') as f:
    full_results = pkl.load(f)



num_experts_ori = 60
num_layers = len(full_results)
top_k = 60
to_num_experts_list = [2,4]  # 每层期望剪掉多少 expert ⇒ 变成 total budget



from scipy.sparse.csgraph import laplacian
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans


for to_num_experts in to_num_experts_list:
    total_budget = num_layers * to_num_experts
    total_experts = num_experts_ori * num_layers
    num_clusters = total_experts - total_budget  # 聚类数 = 最终保留的 expert 数

    # Step 1: 构建图
    full_graph, sims_list = build_block_diag_knn_graph(full_results, num_layers, num_experts_ori, top_k=top_k)
    # sims_list = smooth(sims_list)

    


    # Step 2: 谱嵌入
    L = laplacian(full_graph, normed=False)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    Z = normalize(eigenvectors[:, 1:num_clusters])  # top eigenvectors

    # Step 3: 全局均衡聚类
    cluster_labels = KMeans(n_clusters=num_clusters, random_state=42).fit_predict(Z)

    print("num_clusters: {}".format(num_clusters))
    print("actural number of clusters: {}".format(len(set(cluster_labels))))
    # Step 4: 按层统计剪枝结果
    # import pdb;pdb.set_trace()

    # Step 4: 按层统计剪枝结果
    result_group = []
    total_pruned_experts = 0
    remaining_experts = []
    for layer in range(num_layers):
        layer_result = []
        layer_labels = cluster_labels[layer * num_experts_ori : (layer + 1) * num_experts_ori]
        

        cluster_to_indices = {}
        for idx, label in enumerate(layer_labels):
            cluster_to_indices.setdefault(label, []).append(idx)
        pruned = sum(len(v) - 1 for v in cluster_to_indices.values() if len(v) > 1)
        total_pruned_experts += pruned
        print(f"Layer {layer:2d} - Pruned Experts: {pruned:2d} / {num_experts_ori}")
        # remining experts 
        remaining_experts.append(num_experts_ori - pruned)

        # relabel the cluster labels for different MoE layers
        for key, value in cluster_to_indices.items():
            layer_result.append(value)
        result_group.append(layer_result)

    print("expect number of remaining experts: {}".format(num_clusters))
    print("remaining experts: {}".format(np.sum(remaining_experts)))    
    print("expected to prune: {}".format(total_budget))
    print("actual pruned experts: {}".format(total_pruned_experts))
    print("computed from the result_group:")
    num = 0.0
    for layer in range(num_layers):
        num += len(result_group[layer])
    print("num: {}".format(num))

    # save the result to /home/zhiyuan/patch_tst/global_prune/prune/deepseek/result/final_save
    with open("./result/fast_discover_result_{}.pkl".format(to_num_experts), 'wb') as f:
        pkl.dump(result_group, f)
    # import pdb;pdb.set_trace()