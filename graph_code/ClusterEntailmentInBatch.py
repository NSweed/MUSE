import os
import time
import argparse
import AbstractionWithNLI
from DAGUtils import CycleBreaker, LongestPathAlgorithms, DFSCycleBreaker, NetworkxGraphUtils, NHCycleBreaker,HierarchyFinder
import Serializer
import sys



def remove_double_edges(nodes_list,edge_to_weight):
    new_node_to_neighbors = {node:[] for node in nodes_list}

    for node1,node2 in edge_to_weight:
        if (node2,node1) in edge_to_weight:
            reg_weight = edge_to_weight[(node1,node2)]
            rev_weight = edge_to_weight[(node2, node1)]
            if reg_weight > rev_weight:
                new_node_to_neighbors[node1].append(node2)
        else:
            new_node_to_neighbors[node1].append(node2)
    return new_node_to_neighbors


def run_multiple_entailment_clusters(abs_with_nli, label_to_cluster_list, labels, args,f,job_id):
    label_to_longest_cluster_to_more_abs = {}
    label_to_top_order = {}
    label_to_node_to_level = {}
    label_to_node_to_highest = {}
    print(f"there are {len(labels)} to entail for this job" )
    print(f"indices are {labels}")
    for label in labels:
        f.write(f"starting with label {label}\n")
        x = time.time()
        cluster_list = label_to_cluster_list[label]
        longest_cluster_to_more_abs, cluster_to_more_abstract, top_order, nodouble_cluster_to_more_abstract, node_to_level, node_to_highest_in_path = \
            check_entailment_within_cluster(abs_with_nli, cluster_list, args)
        f.write(f"entailment checking took {time.time() - x} seconds\n")
        label_to_longest_cluster_to_more_abs[label]=  longest_cluster_to_more_abs
        label_to_top_order[label] = top_order
        label_to_node_to_highest[label] = node_to_highest_in_path
        label_to_node_to_level[label] = node_to_level
    #saving the dicts
    suffix = f"{args.purpose_type}_{args.entail_prefix}_{args.distance_threshold}_{args.num_of_points}_{args.entailment_threshold}_{args.nli_model}_{job_id}"
    partial_neighbor_dict_path = os.path.join("Clusters", "partial_neighbor_dicts", f"{suffix}")
    Serializer.Serializer.save_dict(label_to_longest_cluster_to_more_abs, partial_neighbor_dict_path)
    partial_top_order_dict_path = os.path.join("Clusters", "partial_top_order_dicts",f"{suffix}")
    Serializer.Serializer.save_dict(label_to_top_order, partial_top_order_dict_path)
    partial_node_level_dict_path = os.path.join("Clusters", "partial_node_level_dicts",f"{suffix}")
    Serializer.Serializer.save_dict(label_to_node_to_level, partial_node_level_dict_path)
    partial_node_to_highest_dict_path = os.path.join("Clusters", "partial_node_to_highest_dicts",f"{suffix}")
    Serializer.Serializer.save_dict(label_to_node_to_highest, partial_node_to_highest_dict_path)

def check_entailment_within_cluster(abs_with_nli,clusters_list, args):
    import torch
    # performing abstraction within a cluster  - in parallel
    print("i want to entail")
    abs_with_nli.update_clusters(clusters_list)
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        print(f"clearing the cache")
        torch.cuda.empty_cache()
    cluster_to_id = {x: i for i, x in enumerate(clusters_list)}
    print("starting to entail")
    cluster_to_more_abstract, cluster_to_less_abstract, edge_to_weight = abs_with_nli.find_abstractions_with_nli()
    print("finished entailing")
    # g = NetworkxGraphUtils.create_networkx_graph(cluster_to_more_abstract, cluster_to_id)
    nodouble_cluster_to_more_abstract = remove_double_edges(list(cluster_to_more_abstract.keys()), edge_to_weight)
    edges_fname = os.path.join("breaking_cycles_in_noisy_hierarchies", "data",
                               f"edges_{args.entail_prefix}_{args.distance_threshold}_{args.num_of_points}_{args.entailment_threshold}_{args.nli_model}_{args.job_id}")
    cycle_breaker = NHCycleBreaker(cluster_to_more_abstract, edges_fname)
    no_cycle_cluster_to_more_abs = cycle_breaker.break_cycles()
    print("broke some cycles")
    # cycle_breaker = DFSCycleBreaker(cluster_to_more_abstract)
    # edges_to_remove, _ = cycle_breaker.dfs_remove_back_edges()
    # no_cycle_cluster_to_more_abs = remove_edges_from_dict(cluster_to_more_abstract, edges_to_remove)
    # no_cycle_cluster_to_more_abs = CycleBreaker.break_cycles(cluster_to_more_abstract)

    # pruning out circles & keeping only longest
    LPA = LongestPathAlgorithms(clusters_list, no_cycle_cluster_to_more_abs)
    longest_cluster_to_more_abs, reverse_top_order = LPA.eliminate_nonlongest_edges()
    top_order = list(reversed(reverse_top_order))
    hfinder = HierarchyFinder(longest_cluster_to_more_abs, top_order)
    node_to_level, node_to_highest_in_path = hfinder.find_hierarchy_in_disconnected_graph()
    # the abstract graph for this cluster is formed../
    for cluster in clusters_list:
        if cluster not in cluster_to_more_abstract:
            cluster_to_more_abstract[cluster] = set()
    # if args.visualize_abs_clusters:
    #     visualize_and_write_abs_clusters(args, clusters_list, cluster_to_more_abstract, edge_to_weight,
    #                                      longest_cluster_to_more_abs, node_to_level, label)

    return longest_cluster_to_more_abs, cluster_to_more_abstract, top_order, nodouble_cluster_to_more_abstract, node_to_level,node_to_highest_in_path

def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", type=str, default="embeddings")
    parser.add_argument("--purpose_type", type=str, default="GPT3")
    parser.add_argument("--entail_prefix", type=str, default="The device is able to")
    parser.add_argument("--clustering_type", type=str, default="dbscan")
    parser.add_argument("--loose_clustering_type", type=str, default="dbscan")
    parser.add_argument("--cluster_vis_dir", type=str, default="cluster_visualization")
    parser.add_argument("--nli_model", type=str, default="deberta_v2")
    parser.add_argument("--patents_dir", type=str,
                        default=os.path.join("1_milion_gpt3_tagged_patents-20240207T140452Z-001","1_milion_gpt3_tagged_patents"))
    parser.add_argument("--num_of_points", type=int, default=1000)
    parser.add_argument("--embedding_batch_size", type=int, default=30)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--distance_threshold", type=float, default=None)
    parser.add_argument("--entailment_threshold", type=float, default=None)
    parser.add_argument("--min_samples", type=int, default=1)
    parser.add_argument("--num_clusters", type=int, default=200)
    parser.add_argument("--loose_num_clusters", type=int, default=200)
    parser.add_argument("--visualize_clusters", action="store_true")
    parser.add_argument("--visualize_abs_clusters", action="store_true")
    parser.add_argument("--remove_doubles_before", action="store_true")
    parser.add_argument("--clustering_only", action="store_true")
    parser.add_argument("--total_jobs", type=int, default=100)
    parser.add_argument("--job_id", type=int, default=0)
    return parser

if __name__ == "__main__":
    print("first things first")
    parser = add_args()
    args = parser.parse_args()
    suffix = f"{args.purpose_type}_{args.entail_prefix}_{args.distance_threshold}_{args.num_of_points}_{args.entailment_threshold}_{args.nli_model}"
    job_id = args.job_id
    print(f"here we go {job_id}")
    with open(os.path.join("sbatch_logs", f"log_{job_id}"), "w") as f:

        start_time = time.time()
        if args.nli_model == "v3_large":
            abs_with_NLI = AbstractionWithNLI.AbstractionWithClustersMNLI([], args.entail_prefix,
                                                                          threshold=args.entailment_threshold)

        else:
            abs_with_NLI = AbstractionWithNLI.AbstractionWithClusters([], args.entail_prefix,
                                                                      threshold=args.entailment_threshold)
        time_after_entailment_loading = time.time()
        f.write(f"Loading the entailment module took {time_after_entailment_loading - start_time} seconds\n")
        #loading the label_to_cluster_list dict:
        suffix = f"{args.purpose_type}_{args.entail_prefix}_{args.distance_threshold}_{args.num_of_points}_{args.entailment_threshold}_{args.nli_model}"
        labels_to_clusters_list_path = os.path.join("Clusters", "labels_to_cluster_list",
                                                    f"label_to_cluster_list_{suffix}")
        label_to_cluster_list = Serializer.Serializer.load_dict(labels_to_clusters_list_path)
        loading_dict_time = time.time()
        f.write(f"Loading the clusters list dict took {loading_dict_time - time_after_entailment_loading} seconds\n")
        #getting the current labels we wish to work on
        all_labels = list(sorted(label_to_cluster_list.keys()))
        labels_len = len(all_labels)
        num_labels_per_job = labels_len // args.total_jobs
        cur_labels = []
        if job_id == args.total_jobs - 1:
            cur_labels = all_labels[job_id*num_labels_per_job:]
            f.write(f"running from index {job_id*num_labels_per_job} to the end\n\n")
        else:
            cur_labels = all_labels[job_id*num_labels_per_job: (job_id+1)*num_labels_per_job]
            f.write(f"running from index {job_id * num_labels_per_job} to {(job_id+1)*num_labels_per_job}\n\n")
        #now, running entailment on the current labels:
        run_multiple_entailment_clusters(abs_with_NLI, label_to_cluster_list, cur_labels,args,f,job_id)



