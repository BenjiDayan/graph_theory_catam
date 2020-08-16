# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:05:51 2020
"""

import networkx as nx
from collections import defaultdict
from tqdm import tqdm

def greedy_color_vertices(graph, nodes):
    """Uses greedy algorithm to color graph nodes given ordering nodes"""
    node_colors = defaultdict(lambda: 0)
    colors_used = set()
    max_color = 0
    for node in nodes:
        neighbors = list(graph.neighbors(node))
        neighbor_colors = {node_colors[neighbor] for neighbor in neighbors}
        try:
            color = min(colors_used - neighbor_colors)
        except ValueError: # all colors_used are in neighbours
            max_color += 1
            colors_used.add(max_color)
            color = max_color
        node_colors[node] = color
    
    return node_colors

def get_greedy_chromatic_upper_bound(graph, ordering_func=None):
    """Runs greedy node coloring algorithm and returns the number of colors used
    ordering_func: optional func, ordering_func(nx.Graph) =
        some permutation of graph.nodes"""
    if ordering_func is None:
        ordering_func = lambda G: list(G.nodes)
    colors = greedy_color_vertices(graph, ordering_func(graph))
    return max(colors.values())
    
def greedy_find_clique_number(graph, progress=False, nodes=None, \
                              ordering_func=None, node_deletion=True):
    """Finds the order of graph's clique (largest complete subgraph)
    Very dumb algorithm just goes through each node, making sub-graphs
    on sub-graphs from each.
    
    progress: bool whether or not to display a tqdm progress bar for this graph.
    Want True for highest level graph but False for any induced sub-graphs
    
    nodes: list of nodes of graph - can be recursively passed to save recalculation.
    If None, is calculated via ordering_func, with default ordering if ordering_func=None
    
    node_deletion: whether to delete nodes as we go along - in theory good but
    we have to make graph copies which could add time
    """
    graph_order = graph.order()
    if graph_order == 0:
        return 0
    
    if nodes is None:
        nodes = list(graph.nodes) if ordering_func is None else ordering_func(graph)
        
    output=0
    node_index_iterator = tqdm(range(graph_order)) if progress else\
        range(graph_order)
    for i in node_index_iterator:
        node = nodes[i]
        neighbours = set(graph.neighbors(node))
        if node_deletion:
            output = max(output,
                1 + greedy_find_clique_number(graph.subgraph(neighbours).copy(),
                progress=False, nodes=[u for u in nodes[i+1:] if u in neighbours],
                node_deletion=node_deletion))
            graph.remove_node(node)
        else:
            output = max(output,
                1 + greedy_find_clique_number(graph.subgraph(neighbours),
                progress=False, nodes=[u for u in nodes if u in neighbours],
                node_deletion=node_deletion))
        
    return output

# This version has incorrect node ordering but somehow runs a little faster.
# Perhaps there's a bug in greedy_find_clique_number?
# Or perhaps the extra computations to pass on nodes isn't worth the effort.
def greedy_find_clique_number2(graph, progress=False, nodes=None, \
                               ordering_func=None, node_deletion=True):
    graph_order = graph.order()
    if graph_order == 0:
        return 0

    nodes = list(graph.nodes) if ordering_func is None else ordering_func(graph)
        
    output=0
    node_index_iterator = tqdm.tqdm(range(graph_order)) if progress else\
        range(graph_order)
    for i in node_index_iterator:
        node = nodes[i]
        neighbours = set(graph.neighbors(node))
        if node_deletion:
            output = max(output,
                1 + greedy_find_clique_number2(graph.subgraph(neighbours).copy(),
                progress=False, node_deletion=node_deletion))
            graph.remove_node(node)
        else:
            output = max(output,
                1 + greedy_find_clique_number2(graph.subgraph(neighbours),
                progress=False, node_deletion=node_deletion))
        
    return output

def greedy_find_clique(graph, progress=False, nodes=None, ordering_func=None,
                       node_deletion=True):
    """Finds the graph's clique (largest complete subgraph)
    """
    graph_order = graph.order()
    if graph_order == 0:
        return []

    if nodes is None:
        nodes = list(graph.nodes) if ordering_func is None else \ordering_func(graph)
        
    output=[]
    node_index_iterator = tqdm.tqdm(range(graph_order)) if progress else\
        range(graph_order)
    for i in node_index_iterator:
        node = nodes[i]
        neighbours = set(graph.neighbors(node))
        if node_deletion:
            subgraph_output = [node] + \
                greedy_find_clique(graph.subgraph(neighbours).copy(),
                progress=False, nodes=[u for u in nodes[i+1:] if u in neighbours],
                node_deletion=node_deletion)
            
            graph.remove_node(node)
        else:
            subgraph_output = [node] + \
            greedy_find_clique(graph.subgraph(neighbours),
                progress=False, nodes=[u for u in nodes if u in neighbours],
                node_deletion=node_deletion)
            
        output = output if len(output) >= len(subgraph_output) else subgraph_output
        
    return output

def clique_colouring_algorithm(G):
    """Uses a clique finding algorithm on G's complement to iteratively find largest
    independent sets, which are assigned the same colour then removed fromt the graph
    """
    G_complement = nx.algorithms.complement(G)
    colors = {}
    color=1
    while G_complement.order() > 0:
        clique = set(greedy_find_clique(G_complement.copy(), progress=False))
        colors[color] = clique
        color += 1
        G_complement.remove_nodes_from(clique)
        
    #{color: nodelist} and {node:color} dictionaries
    return colors, {u:color for color, nodelist in colors.items() for u in nodelist}

def greedy_complete_subgraph_find(graph, node=None):
    """Goes through graph.nodes, adding them to an accumulating complete subgraph,
    or discarding them if they don't fit. Returns the resultant clique's list of nodes.
    
    Currently only goes through graph.nodes in order. (random if graph is random)
    
    node: starting node for the snowball, optional"""
    nodes = list(graph.nodes)
    if node is None:
        node = nodes.pop(0)
    else:
        nodes.remove(node)
        
    complete_subgraph = [node]
    for v in nodes:
        for u in complete_subgraph:
            if not graph.has_edge(u, v):
                break
        else:
            complete_subgraph.append(v)
    
    return complete_subgraph
        
    

def greedy_algo_expected_remaining(n, p):
    """List of expected remaining nodes of G(n, p) at stage 1,2,3...
    of greedy clique finding algorithm"""
    results = [n-1]
    j = 1
    while results[-1] > 0:
        results.append(int(results[-1] - 1/p**j))
        j += 1
    return results