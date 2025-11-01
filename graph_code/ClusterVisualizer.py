import networkx as nx
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objects as go
from pyvis.network import Network

def get_level_to_color():
    level_to_color = {}
    level_to_color[0] = "black"
    level_to_color[1] = "gray"
    level_to_color[2] = "silver"
    level_to_color[3] = "whitesmoke"
    level_to_color[4] = "linen"
    level_to_color[5] = "lavender"
    level_to_color[6] = "thistle"
    level_to_color[7] = "pink"
    level_to_color[8] = "orchid"
    level_to_color[9] = "magenta"
    level_to_color[10] = "blueviolet"
    level_to_color[11] = "purple"
    level_to_color[12] = "mediumblue"
    level_to_color[13] = "royalblue"
    level_to_color[14] = "deepskyeblue"
    level_to_color[15] = "cyan"
    level_to_color[16] = "skyblue"
    level_to_color[17] = "palegreen"
    level_to_color[18] = "lawngreen"
    level_to_color[19] = "limegreen"
    level_to_color[20] = "forestgreen"
    level_to_color[21] = "olive"
    level_to_color[22] = "gold"
    level_to_color[23] = "khaki"
    level_to_color[24] = "lemonchiffon"
    level_to_color[25] = "yellow"
    level_to_color[26] = "orange"
    level_to_color[27] = "salmon"
    level_to_color[28] = "tomato"
    level_to_color[29] = "red"
    return level_to_color

def reduce_level_to_color(level_to_color, new_size, orig_size = 30):
    import math
    if orig_size < new_size:
        factor = orig_size / new_size
    else:
        factor = orig_size // new_size
    new_level_to_color = {}
    for i in range(new_size):
        new_level_to_color[i] = level_to_color[math.floor(i*factor)]
    return new_level_to_color

class ClusterVisualizer:


    @staticmethod
    def write_clusters_condensed(clusters,path):
        with open(path, "w") as f:
            for i,cluster in enumerate(clusters):
                points = cluster.get_cluster_points()
                for point in points:
                    try:
                        f.write(f"{point}%%%")
                    except:
                        print(f"faulty point is {point}")
                f.write("\n")

    def write_clusters_to_file(clusters, path):
        with open(path, "w") as f:
            f.write("Full clusters information:\n")
            f.write(f"In total, there are {len(clusters)} clusters in the data\n")
            f.write("#########################\n####################\n\n")
            for i,cluster in enumerate(clusters):
                f.write(f"information about cluster number {i+1}:\n")
                points = cluster.get_cluster_points()
                f.write(f"The cluster has a total of {len(points)} points in it. \n")
                f.write("The sentences forming the cluster are:\n")
                for j,point in enumerate(points):
                    try:
                        f.write(f"{j+1}. {point}\n")
                    except:
                        print(point)

            f.write("#######################\n#######################\n")
    @staticmethod
    def write_interconnecting_clusters(cluster_to_neighbor, cluster_to_label, path):
        with open(path, "w") as f:
            for i,cluster in enumerate(cluster_to_neighbor):
                f.write(f"{i+1}. Cluster represented by {cluster.get_random_point()}, from loose cluster {cluster_to_label[cluster]}\n")
                f.write("The cluster is connected to the following clusters:\n")
                for j,neighbor in enumerate(cluster_to_neighbor[cluster]):
                    f.write(f"{j+1}. Cluster represented by {neighbor.get_random_point()},from loose cluster {cluster_to_label[neighbor]}\n ")
                f.write("_______________________________________________________________________\n\n")



    @staticmethod
    def write_clusters_with_abstraction(clusters, cluster_to_more_abstract, path):
        with open(path, "w") as f:
            f.write("Full clusters information:\n")
            f.write(f"In total, there are {len(clusters)} clusters in the data\n")
            f.write("#########################\n####################\n\n")
            for i,cluster in enumerate(clusters):
                f.write(f"information about cluster number {i+1}:\n")
                points = cluster.get_cluster_points()
                f.write(f"The cluster has a total of {len(points)} points in it. \n")
                f.write("The sentences forming the cluster are:\n")
                for j,point in enumerate(points):
                    try:
                        f.write(f"{j+1}. {point}\n")
                    except:
                        print(point)
                if cluster in cluster_to_more_abstract:
                    more_abstract = cluster_to_more_abstract[cluster]
                else:
                    more_abstract = []
                f.write("-----------------------------------\n")
                f.write(f"This cluster is connected to {len(more_abstract)} abstract clusters. Here are some representative sentences from them:\n ")
                for j, abs_cluster in enumerate(more_abstract):
                    f.write(f"{j+1}. {abs_cluster.get_random_point()}\n")
                f.write("#######################\n#######################\n")

    @staticmethod
    def create_networkx_graph(nodes, nodes_to_entailments):
        G = nx.DiGraph(directed=True)
        for node in nodes:
            # node_to_sentence[node] = node.get_random_point().encode("utf-8")
            G.add_node(node.get_id())
        for node in nodes_to_entailments:
            G.add_edges_from([(node.get_id(), adj_node.get_id()) for adj_node in nodes_to_entailments[node]])
        return G

    @staticmethod
    def create_networkx_regular_graph(nodes, nodes_to_entailments):
        G = nx.DiGraph(directed=True)
        for i,node in enumerate(nodes):
            # node_to_sentence[node] = node.get_random_point().encode("utf-8")
            G.add_node(node)
        for node in nodes_to_entailments:
            G.add_edges_from([(node, adj_node) for adj_node in nodes_to_entailments[node]])
        return G

    @staticmethod
    def draw_cluster_graph_with_plotly(nodes, nodes_to_entailments, fig_path, title = None):
        G = ClusterVisualizer.create_networkx_graph(nodes, nodes_to_entailments)
        random_points = [node.get_random_point() for node in nodes]
        edge_x = []
        edge_y = []
        pos = nx.layout.spring_layout(G)
        for node in G.nodes:
            G.nodes[node]['pos'] = list(pos[node])
        for edge in G.edges():
            x0, y0 = G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x = []
            node_y = []
            for node in G.nodes():
                x, y = G.nodes[node]['pos']
                node_x.append(x)
                node_y.append(y)
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                )


            #adding hover information
            nodes_text = []
            for i,node in enumerate(nodes):
                nodes_text.append(f"node {i} is : {random_points[i]}")
            node_trace.text = nodes_text

            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title=title,
                                titlefont_size=16,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                            )
            # for node in G.nodes():
            #     print(f"{node} node is {random_points[node]}")
            fig.write_html(fig_path)

    @staticmethod
    def create_pyvis_graph(nodes, nodes_to_entailments):



        G = nx.DiGraph(directed=True)
        for node in nodes:
            # node_to_sentence[node] = node.get_random_point().encode("utf-8")
            G.add_node(node.get_id())
        for node in nodes_to_entailments:
            G.add_edges_from([(node.get_id(), adj_node.get_id()) for adj_node in nodes_to_entailments[node]])
        return G

    @staticmethod
    def draw_colored_cluster_graph_with_pyvis(nodes, nodes_to_entailments, edge_to_weight,node_to_level, fig_path, title = None):
        net = Network(height="600px", width="100%", font_color="black", heading=title, directed=True)
        #net.show_buttons(filter_=["physics"])
        num_of_colors = max(node_to_level.values()) + 1
        level_to_color = get_level_to_color()
        reduced_level_to_color = reduce_level_to_color(level_to_color, num_of_colors)
        ids = []
        for node in nodes:
            ids.append(node.get_id())
            net.add_node(n_id=node.get_id(), title=node.get_random_point(), color=reduced_level_to_color[node_to_level[node.get_id()]])

        for node in nodes:
            for node2 in nodes_to_entailments[node]:
                if edge_to_weight:
                    net.add_edge(node.get_id(), node2.get_id(), title = str(edge_to_weight[(node,node2)]),arrowStrikethrough=False,
                                 physics = False)
                else:
                    net.add_edge(node.get_id(), node2.get_id(), title="no weight",
                                 arrowStrikethrough=False,
                                 physics=False)
        net.save_graph(fig_path)

    @staticmethod
    def draw_node_neighborhood_with_pyvis(nodes, node_to_neighbors, edge_to_type,start_node_id, title, fig_path):
        net = Network(height="600px", width="100%", font_color="black", heading=title, directed=True)
        ids = []
        for node in nodes:
            ids.append(node.get_id())
            if node.get_id() == start_node_id:
                net.add_node(n_id=node.get_id(), title=node.get_random_point(), shape = 'square')
            else:
                net.add_node(n_id=node.get_id(), title=node.get_random_point())

        for node in nodes:
            node_id = node.get_id()
            for node2 in node_to_neighbors[node]:
                node2_id = node2.get_id()
                edge = (node_id, node2_id)
                if edge in edge_to_type:
                    edge_type  = edge_to_type[edge]
                    if edge_type == 1: #verb node
                        net.add_edge(node.get_id(), node2.get_id(), title="abs with verb",
                                     arrowStrikethrough=False,
                                     physics=False, color = "blue")
                    elif edge_type == 2: #LLM node
                        net.add_edge(node.get_id(), node2.get_id(), title="abs with LLM",
                                     arrowStrikethrough=False,
                                     physics=False, color="red")


                else:
                    net.add_edge(node.get_id(), node2.get_id(), title="abs with entailment",
                                 arrowStrikethrough=False,
                                 physics=False, color = "green")
        net.save_graph(fig_path)


    @staticmethod
    def draw_colored_interconnecting_graph(cluster_to_neighbor, cluster_to_label, title, path ):
        net = Network(height="600px", width="100%", font_color="black", heading=title, directed=True)

        num_of_colors = max(cluster_to_label.values()) + 1
        label_to_color = get_level_to_color()
        reduced_level_to_color = reduce_level_to_color(label_to_color, num_of_colors)
        for node in cluster_to_neighbor:
            net.add_node(n_id=node.get_id(), title=node.get_random_point(),
                         color=reduced_level_to_color[cluster_to_label[node]])

        for node in cluster_to_neighbor:
            for node2 in cluster_to_neighbor[node]:
                net.add_edge(node.get_id(), node2.get_id(), arrowStrikethrough=False,  physics = False)
        net.save_graph(path)


    @staticmethod
    def draw_regular_graph_with_plotly(nodes, nodes_to_entailments, fig_path, title = None):
        G = ClusterVisualizer.create_networkx_regular_graph(nodes, nodes_to_entailments)
        random_points = [node  for node in nodes]
        edge_x = []
        edge_y = []
        pos = nx.layout.spring_layout(G)
        for node in G.nodes:
            G.nodes[node]['pos'] = list(pos[node])
        for edge in G.edges():
            x0, y0 = G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x = []
            node_y = []
            for node in G.nodes():
                x, y = G.nodes[node]['pos']
                node_x.append(x)
                node_y.append(y)
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                )
                # marker=dict(
                #     showscale=True,
                #     colorscale='YlGnBu',
                #     reversescale=True,
                #     color=[],
                #     size=10,
                #     colorbar=dict(
                #         thickness=15,
                #         title='Node Connections',
                #         xanchor='left',
                #         titleside='right'
                #     ),
            # line_width = 2)

            #adding hover information
            nodes_text = []
            for i,node in enumerate(nodes):
                nodes_text.append(f"node {i} is : {random_points[i]}")
            node_trace.text = nodes_text

            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title=title,
                                titlefont_size=16,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                            )
            # for node in G.nodes():
            #     print(f"{node} node is {random_points[node]}")
            fig.write_html(fig_path)


