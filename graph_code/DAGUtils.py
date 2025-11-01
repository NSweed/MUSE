import os

import networkx as nx
import numpy as np
import pyparsing


class NetworkxGraphUtils:

    @staticmethod
    def create_networkx_graph(node_to_neighbors, node_to_id):
        G = nx.DiGraph()
        for node in node_to_neighbors:
            G.add_node(node_to_id[node])
        for node in node_to_neighbors:
            node_id = node_to_id[node]
            for neighbor in node_to_neighbors[node]:
                neighbor_id = node_to_id[neighbor]
                G.add_edge(node_id, neighbor_id)
        return G


class NHCycleBreaker:
    import os
    def __init__(self, node_to_neighbors, fname):
        self.node_to_neighbors = node_to_neighbors
        self.node_list = list(node_to_neighbors.keys())
        self.fname = fname
        self.write_edges_to_file()

    def write_edges_to_file(self):
        with open(self.fname, "w") as f:
            for node in self.node_to_neighbors:
                node_id = node.get_id()
                for neighbor in self.node_to_neighbors[node]:
                    neighbor_id = neighbor.get_id()
                    f.write(f"{node_id} {neighbor_id}\n")

    def break_cycles(self):
        #breaking cycles with outer package
        file_path = os.path.join(f"breaking_cycles_in_noisy_hierarchies","remove_cycle_edges_by_hierarchy.py")
        os.system(f"python  {file_path} -g"
                  f" \"{self.fname}\" -s trueskill")
        removed_edges_path = f"{self.fname}_removed_by_SA-G.edges"
        return self.remove_unnecessary_edges(removed_edges_path)

    def remove_unnecessary_edges(self, removed_path):
        removed_edges = set()
        with open(removed_path) as f:
            for line in f.readlines():
                ids = line.split(" ")
                removed_edges.add((int(ids[0]), int(ids[1])))
        new_node_to_neighbors = {}
        for node in self.node_to_neighbors:
            new_node_to_neighbors[node] = []
            node_id = node.get_id()
            for neighbor in self.node_to_neighbors[node]:
                neighbor_id = neighbor.get_id()
                if (node_id, neighbor_id) not in removed_edges:
                    new_node_to_neighbors[node].append(neighbor)
        return new_node_to_neighbors







class DFSCycleBreaker:

    def __init__(self, node_to_neighbors):
        self.node_to_neighbors = node_to_neighbors
        self.node_list = list(node_to_neighbors.keys())

    def dfs_visit_recursively(self, node, nodes_color, edges_to_be_removed, top_order):

        nodes_color[node] = 1
        nodes_order = list(self.node_to_neighbors[node])
        nodes_order = np.random.permutation(nodes_order)
        for child in nodes_order:
            if nodes_color[child] == 0:
                self.dfs_visit_recursively(child, nodes_color, edges_to_be_removed, top_order)
            elif nodes_color[child] == 1:
                edges_to_be_removed.append((node, child))

        nodes_color[node] = 2
        top_order.append(node)

    def dfs_remove_back_edges(self, nodetype=int):
        '''
        0: white, not visited
        1: grey, being visited
        2: black, already visited
        '''

        #g = nx.read_edgelist(graph_file, create_using=nx.DiGraph(), nodetype=nodetype)
        top_order = []
        nodes_stamp = {}
        edges_to_be_removed = []
        for node in self.node_list:
            nodes_stamp[node] = 0

        nodes_order = np.random.permutation(self.node_list)
        num_dfs = 0
        for node in nodes_order:

            if nodes_stamp[node] == 0:
                num_dfs += 1
                self.dfs_visit_recursively(node, nodes_stamp, edges_to_be_removed, top_order)


        return edges_to_be_removed, top_order

class CycleBreaker:

    @staticmethod
    def break_cycles(nodes_to_neighbors):
        node_to_reachables = {node : set() for node in nodes_to_neighbors}
        new_node_to_neighbors = {node : set() for node in nodes_to_neighbors}
        for u in nodes_to_neighbors:
            for v in nodes_to_neighbors[u]:
                if u not in node_to_reachables[v]:
                    new_node_to_neighbors[u].add(v)
                    node_to_reachables[u].add(v)
                    for node in node_to_reachables:
                        if u in node_to_reachables[node]:
                            node_to_reachables[node].add(v)
        return new_node_to_neighbors


class LongestPathAlgorithms:

    def __init__(self, nodes, node_to_neighbors):
        self.nodes = nodes
        self.node_to_neighbors = node_to_neighbors

    def topological_sort_util(self, node, node_to_visited, stack):
        node_to_visited[node] = True
        if node in self.node_to_neighbors:
            for neighbor in self.node_to_neighbors[node]:
                if not node_to_visited[neighbor]:
                    self.topological_sort_util(neighbor, node_to_visited, stack)
        stack.append(node)


    def topological_sort(self):
        stack  = []
        node_to_visited = {node : False for node in self.nodes}
        for node in self.nodes:
            if not node_to_visited[node]:
                self.topological_sort_util(node, node_to_visited, stack)
        return stack

    def calculate_reachables(self, reverse_topological_ordering):
        #for each node, calculate the nodes that are not DIRECTLY reachable from it.
        #For instnace, if x->y, y->z, we define the reachables of x to be z, + the reachables of y.
        node_to_reachables_from = {}
        for node in reverse_topological_ordering:
            reachable_nodes = set()
            node_to_reachables_from[node] = set()
            #adding the neighbor_dict neighbor_dict, + their reachables:
            for neighbor in self.node_to_neighbors[node]:
                for neighbors_neighbor in self.node_to_neighbors[neighbor]:
                    reachable_nodes.add(neighbors_neighbor)
                for neighbor_reachable in node_to_reachables_from[neighbor]:
                    reachable_nodes.add(neighbor_reachable)
            node_to_reachables_from[node] = reachable_nodes


        return node_to_reachables_from

    def eliminate_nonlongest_edges(self):
        reverse_top_order = self.topological_sort()
        node_to_reachable_from = self.calculate_reachables(reverse_top_order)
        new_node_to_neighbors = {}
        #going over all edges
        for u in self.node_to_neighbors:
            new_node_to_neighbors[u] = set()
            for v in self.node_to_neighbors[u]:
                #the edge is u,v. If v is reachble from u, no need for this edge
                if v not in node_to_reachable_from[u]:
                    new_node_to_neighbors[u].add(v)
        return new_node_to_neighbors, reverse_top_order



class NodeGetter:

    def __init__(self, node_to_neighbors, node_to_level, node_to_highest_in_path):
        self.node_to_neighbors = node_to_neighbors
        self.node_to_level = node_to_level
        self.node_to_heighest_in_path = node_to_highest_in_path

    def find_nodes(self,min_height, max_height, max_dist_from_highest, max_out_degree):
        level_to_nodes = {}
        nodes_to_return = []
        for node in self.node_to_level:
            out_degree = len(self.node_to_neighbors[node])
            if out_degree <= max_out_degree:

                height  = self.node_to_level[node]
                if height >= min_height and height <= max_height:
                    dist_from_highest = self.node_to_heighest_in_path[node] - height
                    if dist_from_highest <= max_dist_from_highest:
                        if height not in level_to_nodes:
                            level_to_nodes[height] = []
                        level_to_nodes[height].append(node)
                        nodes_to_return.append(node)
        return nodes_to_return, level_to_nodes

class NodeGetterForAbstraction(NodeGetter):

    def reverse_neighbor_dict(self):
        self.reverse_dict = {}
        for node in self.node_to_neighbors:
            for neighbor in self.node_to_neighbors[node]:
                if neighbor not in self.reverse_dict:
                    self.reverse_dict[neighbor] = []
                self.reverse_dict[neighbor].append(node)

    def find_abstract_nodes(self, cluster_to_label,min_connected_num):
        level_to_nodes = {}
        nodes_to_return = []
        node_to_clusters_connected = {}
        self.reverse_neighbor_dict()

        for node in self.node_to_neighbors:
            height = self.node_to_level[node]
            if height not in level_to_nodes:
                level_to_nodes[height] = []
            level_to_nodes[height].append(node)
            if node not in nodes_to_return:
                if node not in node_to_clusters_connected:
                    node_to_clusters_connected[node] = set()
                node_cluster = cluster_to_label[node]
                if node_cluster not in node_to_clusters_connected[node]:
                    node_to_clusters_connected[node].add(node_cluster)
                if len(node_to_clusters_connected[node]) >= min_connected_num:
                    nodes_to_return.append(node)
        return nodes_to_return, level_to_nodes










    # def get_node_by_hierarchy_range(self, min_level, max_level):
    #     level_to_nodes = {}
    #     for node in self.node_to_level:
    #         level = self.node_to_level[node]
    #         if level >= min_level and level <=max_level:
    #             if level not in level_to_nodes:
    #                 level_to_nodes[level] = []
    #             level_to_nodes[level].append(node)
    #     return level_to_nodes
    #
    # def get_leaf_nodes_within_range(self, node_to_level, min_level, max_level):
    #     level_to_nodes = {}
    #     nodes_to_return = []
    #     for node in node_to_level:
    #         if not self.node_to_neighbors[node]: #we have a leaf
    #             level = node_to_level[node]
    #             if min_level <= level and level <= max_level:
    #                 if level not in level_to_nodes:
    #                     level_to_nodes[level] = []
    #                 level_to_nodes[level].append(node)
    #                 nodes_to_return.append(node)
    #     return nodes_to_return, level_to_nodes
    #
    # """
    # This function is meant to extract nodes from the graph whose:
    # 1. level is at most max_level
    # 2. level is at least min level
    # 3. distance from the highest leaf node in its path is at most dist_from_leaf
    # """
    # def get_nodes_by_level_range_and_relative_height(self, node_to_level, node_to_longest, min_level,
    #                                                  max_level, dist_from_leaf):
    #
    #     level_to_nodes = {}
    #     nodes_to_return  = []
    #     for node in node_to_level:
    #         level = node_to_level[node]
    #         highest_in_path = node_to_longest[node]
    #         leaf_dist = highest_in_path - level
    #         if leaf_dist <= dist_from_leaf and level <= max_level and level >= min_level:
    #             if level not in level_to_nodes:
    #                 level_to_nodes[level] = []
    #             level_to_nodes[level].append(node)
    #             nodes_to_return.append(node)
    #     return nodes_to_return, level_to_nodes
    #
    # def get_nodes_by_level_range_and_out_degree(self, node_to_level, min_level,
    #                                                  max_level, max_out_degree):
    #
    #     level_to_nodes = {}
    #     nodes_to_return  = []
    #     for node in node_to_level:
    #         level = node_to_level[node]
    #         out_degree = len(self.node_to_neighbors[node])
    #         if out_degree <= max_out_degree and level <= max_level and level >= min_level:
    #             if level not in level_to_nodes:
    #                 level_to_nodes[level] = []
    #             level_to_nodes[level].append(node)
    #             nodes_to_return.append(node)
    #     return nodes_to_return, level_to_nodes



class HierarchyFinder:

    def __init__(self, node_to_neighbors, top_order):
        self.node_to_neighbors = node_to_neighbors
        self.top_order = top_order




    def find_hierarchy_in_single_graph_recursion(self, starting_node, cur_level):
        if not self.node_to_neighbors[starting_node]:
            node_level = cur_level
            if starting_node in self.node_to_level:
                node_level = max(cur_level, self.node_to_level[starting_node])
            self.node_to_level[starting_node] = node_level
            self.node_to_highest_in_path[starting_node] = node_level
            return node_level
        else:
            max_highest = 0
            for neighbor in self.node_to_neighbors[starting_node]:
                cur_highest = self.find_hierarchy_in_single_graph_recursion(neighbor, cur_level + 1)
                if cur_highest > max_highest:
                    max_highest = cur_highest
            self.node_to_highest_in_path[starting_node] = max_highest
            node_level = cur_level
            if starting_node in self.node_to_level:
                node_level = max(cur_level, self.node_to_level[starting_node])
            self.node_to_level[starting_node]=  node_level
            return max_highest



    def find_hierarchy_in_single_graph(self, starting_node, node_to_level):
        #node_to_level = {}
        nodes_visited = set()
        highest_level = 0
        nodes_to_visit = [starting_node]
        node_to_level[starting_node] = 0
        while nodes_to_visit:
            cur_node = nodes_to_visit.pop()
            nodes_visited.add(cur_node)
            if cur_node in self.node_to_neighbors:
                for neighbor in self.node_to_neighbors[cur_node]:
                    if neighbor not in nodes_visited:
                        cur_level = node_to_level[cur_node] + 1
                        node_to_level[neighbor] = cur_level
                        nodes_to_visit.append(neighbor)
                        if cur_level > highest_level:
                            highest_level = cur_level
        return node_to_level, highest_level

    def merge_dicts(self, d1, d2):
        new_d = {x:d1[x] for x in d1}
        for x in d2:
            if x in new_d:
                new_d[x] = max(new_d[x], d2[x])
            else:
                new_d[x] = d2[x]
        return new_d



    def find_hierarchy_in_disconnected_graph(self):
        starting_points = []
        self.node_to_level = {}
        self.node_to_highest_in_path = {}
        seen_points = set()
        for node in self.top_order:
            if node not in seen_points:
                starting_points.append(node)
            for neighbor in self.node_to_neighbors[node]:
                seen_points.add(neighbor)
        node_to_level = {}
        for point in starting_points:
            self.find_hierarchy_in_single_graph_recursion(point, 0)
            #node_to_level = self.merge_dicts(node_to_level, cur_node_to_level)
            # if cur_highest_level > highest_level:
            #     highest_level = cur_highest_level

        return self.node_to_level, self.node_to_highest_in_path
















# nodes = ["x", "y", "z", "t", "m", "a", "b", "c", "n"]
# node_to_neighbors = {"x": ["y", ], "y":["z","t"], "z":[], "t":[], "m":["n"], "a":["b"], "b":["m"], "c":["a"], "n":[] }
# starting_node = "x"
#
# LPA = LongestPathAlgorithms(nodes, node_to_neighbors)
# node_to_neighbors, rev_top_order = LPA.eliminate_nonlongest_edges()
# top_order = list(reversed(rev_top_order))
# HF = HierarchyFinder(node_to_neighbors, top_order)
# x,y = HF.find_hierarchy_in_disconnected_graph()
# level_to_nodes = HF.get_node_by_hierarchy_range(x, y-2, y)
# print(x)
# print(y)
# print(level_to_nodes)




from  ClusterVisualizer import ClusterVisualizer
# from ClusterFactory import Cluster
# nodes = ["x", "y", "z", "t"]
# cluster_nodes = [Cluster([x], None, i) for i,x in enumerate(nodes)]
# node_to_neighbors = {"x": ["y", "t"], "y":["z","t", "x"], "z":["t", "x"], "t":[]}
# node_to_parents = {"x": ["z" ,"y"], "y":["x"], "z":["y"], "t": ["z", "x", "y"]}
# cbreaker = DFSCycleBreaker(node_to_neighbors)
# edges_to_remove, top_order = cbreaker.dfs_remove_back_edges()
# nir = 1
# d = CycleBreaker.break_cycles(node_to_neighbors)
# DAG = LongestPathAlgorithms(nodes, node_to_neighbors, node_to_parents)
# order = DAG.topological_sort()
# print(order)
# new_node_to_neighbors = DAG.eliminate_nonlongest_edges()
# ClusterVisualizer.draw_regular_graph_with_plotly(nodes, node_to_neighbors, fig_path="figs/before_trial.html")
# ClusterVisualizer.draw_regular_graph_with_plotly(nodes, new_node_to_neighbors, fig_path="figs/after_trial.html")

# nodes = ["x", "y", "z", "a", "b","t","f","g", "l","m"]
# node_to_neighbors = {"x":["l"], "y":["z"], "z":["m"], "a":["b"], "b":["z"], "t":["f"],"f":["g"],"g":[], "l":["m"], "m":[]}
# hfinder = HierarchyFinder(node_to_neighbors, ["a","b","x","y","t","f","z","g"])
