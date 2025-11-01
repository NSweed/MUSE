#imports
import os.path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import pickle
from  PurposeReader import PurposeReader, MechanismReader
import torch
from sentence_transformers import CrossEncoder
import argparse
import SaveSentenceEmbeddings
from ClusterFactory import ClusterFactory
import ClusterVisualizer
import AbstractionWithNLI
from DAGUtils import CycleBreaker, LongestPathAlgorithms, DFSCycleBreaker, NetworkxGraphUtils, NHCycleBreaker,HierarchyFinder
import Serializer


class ClusterPoints:

    def __init__(self, cluster_func, points_to_cluster):
        self.cluster_func = cluster_func
        self.points_to_cluster = points_to_cluster
        self.create_index_to_point()

    def create_index_to_point(self):
        self.index_to_point = {i : x for i,x in enumerate(self.points_to_cluster)}

    def get_index_to_point(self):
        return self.index_to_point

    def cluster_points(self):
        """

        :return:  A mapping between the point embeddings to their cluster label
        """
        clusters = self.cluster_func.fit(self.points_to_cluster)
        index_to_label = {}
        for i,label in enumerate(clusters.labels_):
            index_to_label[i] = label

        #emb_to_label = {self.points_to_cluster[i] : clusters.labels_[i]  for i in range(len(self.points_to_cluster))}
        return index_to_label,clusters #

def get_clustering_function(args):
    if args.clustering_type == "dbscan":
        return DBSCAN(eps=args.eps, metric="cosine", min_samples=args.min_samples, n_jobs=-1)
    elif args.clustering_type == "kmeans":
        return KMeans(n_clusters=args.num_loose_clusters)
    elif args.clustering_type == "agglomerative":
        if args.distance_threshold:
            #print("hey")
            return AgglomerativeClustering(n_clusters=None,
                                           distance_threshold=args.distance_threshold, compute_full_tree = True,
                                           metric = "cosine", linkage="complete")

        else:
            return AgglomerativeClustering(n_clusters=args.num_clusters,
                                       distance_threshold=args.distance_threshold, metric = "cosine")
def get_loose_clustering_function(args):
    if args.loose_clustering_type == "dbscan":
        return DBSCAN(eps=args.eps, metric="cosine", min_samples=args.min_samples, n_jobs=-1)
    elif args.loose_clustering_type == "kmeans":
        return KMeans(n_clusters=args.loose_num_clusters)
    elif args.loose_clustering_type == "agglomerative":
        if args.distance_threshold:
            return AgglomerativeClustering(n_clusters=None,
                                           distance_threshold=args.distance_threshold, compute_full_tree = True,
                                           metric = "cosine", linkage="complete")

        else:
            return AgglomerativeClustering(n_clusters=args.num_clusters,
                                       distance_threshold=args.distance_threshold, metric = "cosine")

def get_clustering_kwargs(args):
    if not args.entailment_threshold:
        threshold = 0
    else:
        threshold  = args.entailment_threshold

    if args.clustering_type == "dbscan":
        return {"eps" : args.eps, "min_samples" : args.min_samples, "entailment_thresh" : threshold, "nli" : args.nli_model}
    elif args.clustering_type == "kmeans":
        return {"n_clusters" : args.num_clusters, "entailment_thresh" : threshold, "nli" : args.nli_model}
    elif args.clustering_type == "agglomerative":
        if args.distance_threshold:
            return {"n_clusters": 0, "distance_threshold": args.distance_threshold, "entailment_thresh" : threshold, "nli" : args.nli_model}
        return {"n_clusters" : args.num_clusters, "distance_threshold" : 1, "entailment_thresh" : threshold, "nli" : args.nli_model}


def get_clustering_figure_path(dir, type, size, longest,prefix,loose_label, **kwargs):
    longest_str = ""

    if longest:
        longest_str = "ol_"
    final_path = os.path.join(dir,f"graph_visulization_{longest_str}{type}_size_{size}_prefix_{prefix}")
    for arg in kwargs:
        cur_arg = kwargs[arg]
        if "." in str(cur_arg):
            cur_arg = str(cur_arg).replace(".", "")
        cur_arg = str(cur_arg).replace(" ","_")
        final_path += f"{arg}_{cur_arg}"
    final_path += f"_{loose_label}"
    final_path += ".html"
    return final_path

def get_clustering_information_path(dir, type, size, prefix,loose_label, **kwargs):
    final_path = os.path.join(dir,f"{type}_size_{size}_prefix_{prefix}_")
    for arg in kwargs:
        cur_arg = kwargs[arg]
        if "." in str(cur_arg):
            cur_arg = str(cur_arg).replace(".","")
        final_path += f"{arg}_{cur_arg}"
    final_path += f"_{loose_label}"
    return final_path

def get_abs_clustering_information_path(dir, type, size, longest, prefix, loose_label, **kwargs):
    longest_str = ""
    if longest:
        longest_str = "non_longest_"
    final_path = os.path.join(dir,f"abs_clustering_{longest_str}{type}_size_{size}__prefix_{prefix}_")
    for arg in kwargs:
        cur_arg = kwargs[arg]
        if "." in str(cur_arg):
            cur_arg = str(cur_arg).replace(".", "")
        final_path += f"{arg}_{cur_arg}"
    final_path += f"_{loose_label}"
    return final_path

def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", type=str, default="embeddings")
    parser.add_argument("--entail_prefix", type=str, default="The device is able to")
    parser.add_argument("--clustering_type", type=str, default="dbscan")
    parser.add_argument("--loose_clustering_type", type=str, default="dbscan")
    parser.add_argument("--cluster_vis_dir", type=str, default="cluster_visualization")
    parser.add_argument("--nli_model", type=str, default="deberta_v2")
    parser.add_argument("--patents_dir", type=str,
                        default=os.path.join("1_milion_gpt3_tagged_patents-20240207T140452Z-001","1_milion_gpt3_tagged_patents"))
    parser.add_argument("--purpose_type", type=str, default="GPT3")
    parser.add_argument("--CPC_path", type = str, default= os.path.join("bert_based_info","pid_to_purpose_sentences.json"))
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
    return parser


def remove_edges_from_dict(node_to_neighbors, edges_to_remove):
    new_node_to_neighbors = {}
    for node in node_to_neighbors:
        new_node_to_neighbors[node] = set()
        for neighbor in node_to_neighbors[node]:
            if (node, neighbor) not in edges_to_remove:
                new_node_to_neighbors[node].add(neighbor)
    return new_node_to_neighbors

def visualize_and_write_abs_clusters(args, clusters_list, cluster_to_more_abstract, edge_to_weight, longest_cluster_to_more_abs, node_to_level, label):
    cur_path_full = os.path.join(f"{args.cluster_vis_dir}","aggressive_clusters","full_abs_clusters")
    fig_path = get_clustering_figure_path(cur_path_full, clustering_type,
                                          num_of_points, False,args.entail_prefix,label, **get_clustering_kwargs(args))
    cur_path_longest = os.path.join(f"{args.cluster_vis_dir}", "aggressive_clusters", "only_longest")
    longest_fig_path = get_clustering_figure_path(cur_path_longest, clustering_type,
                                                  num_of_points, True, args.entail_prefix,label, **get_clustering_kwargs(args))


    abs_path = get_abs_clustering_information_path(cur_path_full,
                                                   clustering_type,
                                                   num_of_points, False,args.entail_prefix,label, **get_clustering_kwargs(args))
    longest_abs_path = get_abs_clustering_information_path(cur_path_longest
                                                           , clustering_type,
                                                           num_of_points,True,args.entail_prefix,label, **get_clustering_kwargs(args))

    # writing clusters before & after cutting cycles and keeping only the longest paths
    ClusterVisualizer.ClusterVisualizer.write_clusters_with_abstraction(clusters_list, cluster_to_more_abstract,
                                                                        abs_path)
    ClusterVisualizer.ClusterVisualizer.write_clusters_with_abstraction(clusters_list, longest_cluster_to_more_abs,
                                                                        longest_abs_path)

    # saving the graph html file before & after cutting cycles and keeping only the longest paths
    # ClusterVisualizer.ClusterVisualizer.draw_cluster_graph_with_plotly(clusters_list, cluster_to_more_abstract,
    #                                                                    fig_path,
    #                                                                    title="agglomerative_10000_0.15")
    # ClusterVisualizer.ClusterVisualizer.draw_cluster_graph_with_plotly(clusters_list, longest_cluster_to_more_abs,
    #                                                                    longest_fig_path,
    #                                                                    title="agglomerative_10000_0.15")
    #                                                                    title="agglomerative_10000_0.15")
    ClusterVisualizer.ClusterVisualizer.draw_colored_cluster_graph_with_pyvis(clusters_list,
                                                                              cluster_to_more_abstract,
                                                                              edge_to_weight,
                                                                              node_to_level,
                                                                              fig_path,
                                                                              title=f"cluster {label} full graph")
    ClusterVisualizer.ClusterVisualizer.draw_colored_cluster_graph_with_pyvis(clusters_list, longest_cluster_to_more_abs,
                                                                              edge_to_weight,node_to_level,

                                                                              longest_fig_path, title=f"cluster {label} graph")

def prepare_loose_clusters(index_to_label, index_to_sentence, index_to_pid, sentence_to_embedding):
    label_to_cluster_sentences = {}
    label_to_cluster_pids = {}
    label_to_cluster_embeddings = {}
    label_to_indices = {}
    labels = []
    for ind in index_to_label:
        label  = index_to_label[ind]
        if label not in label_to_cluster_sentences:
            label_to_cluster_sentences[label] = []
            label_to_cluster_embeddings[label] = []
            label_to_cluster_pids[label] = []
            label_to_indices[label] = []
            labels.append(label)
        cur_sentence = index_to_sentence[ind]
        label_to_cluster_sentences[label].append(cur_sentence)
        label_to_cluster_embeddings[label].append(sentence_to_embedding[ind])
        label_to_cluster_pids[label].append(index_to_pid[ind])
        label_to_indices[label].append(ind)
    return labels, label_to_cluster_sentences,label_to_cluster_pids, label_to_indices,label_to_cluster_embeddings





def check_entailment_within_cluster(abs_with_nli,clusters_list, args, label):
    # performing abstraction within a cluster  - in parallel
    print("i want to entail")
    abs_with_nli.update_clusters(clusters_list)
    cluster_to_id = {x: i for i, x in enumerate(clusters_list)}
    print("starting to entail")
    cluster_to_more_abstract, cluster_to_less_abstract, edge_to_weight = abs_with_NLI.find_abstractions_with_nli()
    print("finished entailing")
    # g = NetworkxGraphUtils.create_networkx_graph(cluster_to_more_abstract, cluster_to_id)
    nodouble_cluster_to_more_abstract = remove_double_edges(list(cluster_to_more_abstract.keys()), edge_to_weight)
    edges_fname = os.path.join("breaking_cycles_in_noisy_hierarchies", "data",
                               f"edges_{args.entail_prefix}_{args.distance_threshold}_{args.num_of_points}_{args.entailment_threshold}_{args.nli_model}")
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

    return longest_cluster_to_more_abs, cluster_to_more_abstract, top_order, nodouble_cluster_to_more_abstract, node_to_level,node_to_highest_in_path, label

def process_label(args):
    abs_with_nli,label, clusters_list, method_args = args
    longest_cluster_to_more_abs, cluster_to_more_abstract, top_order, nodouble_cluster_to_more_abstract, node_to_level,node_to_highest_in_path, label\
        = check_entailment_within_cluster(abs_with_nli,clusters_list, method_args, label)
    return longest_cluster_to_more_abs, cluster_to_more_abstract, top_order, nodouble_cluster_to_more_abstract, node_to_level,node_to_highest_in_path, label

def run_entailment_in_parallel(abs_with_nli,labels, label_to_clusters_list, args):
    import multiprocessing as mp
    print(f"number of cores is {mp.cpu_count()}")
    # This will hold the results in dictionary form
    label_to_longest_cluster_to_more_abs = {}
    label_to_cluster_to_more_abstract = {}
    label_to_top_order = {}
    label_to_nodouble_cluster_to_more_abstract = {}
    label_to_node_to_level = {}
    label_to_node_to_highest_in_path = {}

    # Create a pool of workers based on the number of CPU cores
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Prepare the input for each label
        inputs = [(abs_with_nli,label, label_to_clusters_list[label], args) for label in labels]
        print("before pool")
        # Run the check_entailment_within_cluster method in parallel
        results = pool.map(process_label, inputs)

        # Store the results in respective dictionaries
        for longest_cluster_to_more_abs, cluster_to_more_abstract, top_order, nodouble_cluster_to_more_abstract, node_to_level, node_to_highest_in_path, label in results:
            label_to_longest_cluster_to_more_abs[label] = longest_cluster_to_more_abs
            label_to_cluster_to_more_abstract[label] = cluster_to_more_abstract
            label_to_top_order[label] = top_order
            label_to_nodouble_cluster_to_more_abstract[label] = nodouble_cluster_to_more_abstract
            label_to_node_to_level[label] = node_to_level
            label_to_node_to_highest_in_path[label] = node_to_highest_in_path
        return label_to_longest_cluster_to_more_abs, label_to_cluster_to_more_abstract, label_to_top_order,\
               label_to_nodouble_cluster_to_more_abstract,label_to_node_to_level, label_to_node_to_highest_in_path



'''
The main function does the following:
    1. performs clustering (default: k-means) to create n loose clusters
    2. For each loose cluster:
        a. perform clustering (default: agglomerative clustering) to create aggressive clusters. 
        b.find abstraction relations between the clusters using NLI
        c. Prune the graph to eliminate circles, and keep only longest paths between nodes
    3. saving visualizations of both loose, aggressive clusters
    4. saving 3 dictionaries:
        a. label (loose cluster) to neighbor dict (aggressive cluster to neighbors)
        b. label (loose cluster) to reverse topological order (important for finding height)
        c. label (loose cluster) to index_to_cluster dict (maps index to cluster)
'''
if __name__ == "__main__":

    #parsing args
    parser = add_args()
    args = parser.parse_args()
    patents_dir = args.patents_dir
    embeddings_path = args.embeddings_path
    num_of_points = args.num_of_points
    batch_size = args.embedding_batch_size
    clustering_type = args.clustering_type
    mech_path = os.path.join("bert_based_info","pid_to_cpc_purpose_clusters_cache_file.json")


    #getting purpose sentences, and their embeddings
    purpose_path = ""
    if args.purpose_type == "CPC":
        purpose_path = args.CPC_path
    else:
        purpose_path = patents_dir
    PR = PurposeReader(args.purpose_type, purpose_path)
    purpose_dict = PR.create_purpose_dict()
    # mech_dict = MR.load_raw_mechanism_dict(os.path.join("bert_based_info","id_to_mech"))
    #mech_dict = MR.create_clustered_mechnisms_dict(mech_path)
    #reduce sentences to include num_of_points
    patent_ids = list(sorted(purpose_dict.keys()))

    chosen_patent_ids = patent_ids[:num_of_points]
    #getting the sentences correlated to the patent ids:
    sentences = []
    pids = []
    for pid in chosen_patent_ids:
        if args.purpose_type == "CPC":
            for sentence in purpose_dict[pid]:
                sentences.append(sentence)
                pids.append(pid)
        else:
            sentences.append(purpose_dict[pid])
            pids.append(pid)

    index_to_pid = {i:pid for i,pid in enumerate(pids)}

    # patent_id_to_purpose_mechanism = {}
    # for patent_id in chosen_patent_ids:
    #     purpose  = purpose_dict[patent_id]
    #     if patent_id in mech_dict:
    #         patent_id_to_purpose_mechanism[patent_id] = (purpose, mech_dict[patent_id])
    #     else:
    #         patent_id_to_purpose_mechanism[patent_id] = (purpose, [])
    #
    # sentences_mechanisms = [patent_id_to_purpose_mechanism[key] for key in chosen_patent_ids]
    #
    # sentences = [x[0] for x in sentences_mechanisms]
    # mechanisms = [x[1] for x in sentences_mechanisms]
    # if args.purpose_type == "CPC":  # this means each sentence might have multiple purposes
    #     new_sentences = []
    #     new_mechanisms = []
    #     for ind,lst in enumerate(sentences):
    #         mech = mechanisms[ind]
    #         for sent in lst:
    #             new_mechanisms.append(mech)
    #             new_sentences.append(sent)
    #     sentences = new_sentences
    #     mechanisms = new_mechanisms
    #sentence_to_patent_id = {purpose_dict[i]: i for i in chosen_patent_ids}
    #print(f"loose clustering {len(sentences)} sentences")
    #loading or saving embeddings
    print(f"num_of_points is {num_of_points}")
    embeddings_loader = SaveSentenceEmbeddings.SaveOrLoadEmbeddings(sentences, batch_size, args.purpose_type,embeddings_path)
    embeddings_dict, index_to_sentence, embeddings_to_cluster = embeddings_loader.save_or_load_embeddings(num_of_points)
    #index_to_patent_id = {ind : sentence_to_patent_id[index_to_sentence[ind]] for ind in index_to_sentence}
    #creating the loose clusters:
    loose_clustering_func = get_loose_clustering_function(args)
    loose_clusterer = ClusterPoints(loose_clustering_func, embeddings_to_cluster)
    loose_index_to_label, loose_clusters =  loose_clusterer.cluster_points()
    labels, label_to_cluster_sentences, label_to_cluster_pids,label_to_indices, label_to_cluster_embeddings = \
        prepare_loose_clusters(loose_index_to_label, index_to_sentence, index_to_pid, embeddings_dict)
    label_to_index_to_cluster = {}
    label_to_clusters_list = {}
    label_to_problem_to_mechanisms = {}
    all_sentences_num = []
    # clustering aggressively
    offset = 0
    for it, label in enumerate(labels):
        #getting the sentences and pids for this cluster:
        label_sentences = [index_to_sentence[ind] for ind in label_to_indices[label]]
        if 'a bicycle component operating device' in label_sentences:
            stop = 1
        label_pids = [index_to_pid[ind] for ind in label_to_indices[label]]
        # label_sentences = label_to_cluster_sentences[label]
        # label_pids = label_to_cluster_pids[label]
        print(f"starting to cluster aggressively cluster number {it}, with {len(label_sentences)} sentences")
        all_sentences_num.append(len(label_sentences))
        if len(label_sentences) == 1:
            print("found one with 1")
            continue
        cur_embeddings_to_cluster = label_to_cluster_embeddings[label]
        # getting the clustering function and aggressive clustering all points

        clustering_func = get_clustering_function(args)
        clusterer = ClusterPoints(clustering_func, cur_embeddings_to_cluster)
        index_to_label, clusters = clusterer.cluster_points()
        num_of_noise = len([x for x in list(index_to_label.values()) if x == -1])
        #print(f"there are {num_of_noise} noisy points")

        # creating new index_to_sentence

        cur_index_to_sentence = {i: sentence for i, sentence in enumerate(label_sentences)}
        cur_index_to_pids = {i: pid for i,pid in enumerate(label_pids)}
        clusters_list, max_count_id = ClusterFactory.create_problem_clusters(index_to_label,
                                                                    cur_index_to_sentence, cur_index_to_pids, embeddings_dict, offset)
        label_to_index_to_cluster[label] = {i+offset: x for i, x in enumerate(clusters_list)}
        offset += max_count_id
        if args.visualize_clusters:
            vis_path = get_clustering_information_path(os.path.join(f"{args.cluster_vis_dir}","loose_clusters"), clustering_type,
                                                       num_of_points, args.entail_prefix, label,
                                                       **get_clustering_kwargs(args))
            vis_condensed_path  = vis_path + "_condensed"
            ClusterVisualizer.ClusterVisualizer.write_clusters_to_file(clusters_list, vis_path)
            ClusterVisualizer.ClusterVisualizer.write_clusters_condensed(clusters_list, vis_condensed_path)

            label_to_clusters_list[label] = clusters_list
    #saving label to clusters list
    print("saving labels dict")
    suffix = f"{args.purpose_type}_{args.entail_prefix}_{args.distance_threshold}_{args.num_of_points}_{args.entailment_threshold}_{args.nli_model}"
    labels_to_clusters_list_path = os.path.join("Clusters", "labels_to_cluster_list", f"label_to_cluster_list_{suffix}")
    print(f"saving {len(label_to_clusters_list)} labels to {labels_to_clusters_list_path}")
    Serializer.Serializer.save_dict(label_to_clusters_list, labels_to_clusters_list_path)
    # #saving the problem to mechanism dict
    # labels_to_problem_to_mech_path = os.path.join("Clusters", "labels_to_problem_to_mech", f"label_problem_to_mech_{suffix}")

    print(f"the 20-largest clusters: {list(reversed(sorted(all_sentences_num)))[:20]}")
    print(f"the sum of sentences in the clusters: {sum(all_sentences_num)}")
    print(f"there are {len(patent_ids)} ids")
    print(f"there are {len(embeddings_to_cluster)} emnbeddings")
    print(f"there are {len(chosen_patent_ids)} chosen ids")
    print(f"there are {len(sentences)} sentences")
    # if args.nli_model == "v3_large":
    #     abs_with_NLI = AbstractionWithNLI.AbstractionWithClustersMNLI([], args.entail_prefix,
    #                                                                   threshold=args.entailment_threshold)
    #
    # else:
    #     abs_with_NLI = AbstractionWithNLI.AbstractionWithClusters([], args.entail_prefix,
    #                                                               threshold=args.entailment_threshold)
    # label_to_node_to_level[label] = node_to_level
    # label_to_neighbor_dict[label] = longest_cluster_to_more_abs
    # label_to_nodouble_neighbor_dict[label] = nodouble_cluster_to_more_abstract
    # label_to_full_neighbor_dict[label] = cluster_to_more_abstract
    # label_to_top_order[label] = top_order

    # label_to_longest_cluster_to_more_abs, label_to_cluster_to_more_abstract, label_to_top_order, label_to_nodouble_cluster_to_more_abstract, label_to_node_to_level, \
    #     label_to_node_to_highest_in_path = run_entailment_in_parallel(abs_with_NLI,labels, label_to_clusters_list, args)
    # suffix = f"{args.entail_prefix}_{args.distance_threshold}_{args.num_of_points}_{args.entailment_threshold}_{args.nli_model}"
    # reverse_top_order_path = os.path.join("Clusters","reverse_top_orders",f"label_to_top_order_{suffix}")
    # Serializer.Serializer.save_dict(label_to_top_order, reverse_top_order_path)
    # neighbor_dict_path = os.path.join("Clusters","neighbor_dict",f"label_to_neighbor_dict_{suffix}")
    # Serializer.Serializer.save_dict(label_to_longest_cluster_to_more_abs, neighbor_dict_path)
    # nodouble_neighbor_dict_path = os.path.join("Clusters","neighbor_dict",f"label_to_nodouble_neighbor_dict_{suffix}")
    # Serializer.Serializer.save_dict(label_to_nodouble_cluster_to_more_abstract, nodouble_neighbor_dict_path)
    # full_neighbor_dict_path = os.path.join("Clusters","neighbor_dict",f"label_to_full_neighbor_dict_{suffix}")
    # Serializer.Serializer.save_dict(label_to_cluster_to_more_abstract, full_neighbor_dict_path)
    cluster_save_path = os.path.join("Clusters","abs_clusters",f"label_to_index_to_cluster_{suffix}")
    print(f"saving to {cluster_save_path}")
    Serializer.Serializer.save_dict(label_to_index_to_cluster, cluster_save_path)
    # node_to_level_save_path = os.path.join("Clusters", "node_to_level", f"label_to_node_to_level_{suffix}")
    # Serializer.Serializer.save_dict(label_to_node_to_level, node_to_level_save_path)
    # node_to_highest_save_path = os.path.join("Clusters", "node_to_highest", f"label_to_node_to_highest_{suffix}")
    # Serializer.Serializer.save_dict(label_to_node_to_highest_in_path, node_to_highest_save_path)
