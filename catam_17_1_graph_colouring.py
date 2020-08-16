# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:32:54 2020

@author: benja
"""

import networkx as nx
import numpy as np
from collections import defaultdict
from functools import partial
import random
import matplotlib.pyplot as plt
import itertools
import tqdm
import time


def color_vertices(graph, nodes):
    """Uses greedy algorithm to colour graph nodes given ordering nodes"""
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
    """Runs greedy node coloring algorithm and returns the number of colours used
    ordering_func: optional func, ordering_func(nx.Graph) = some permutation of graph.nodes"""
    if ordering_func is None:
        ordering_func = random_ordering
    colors = color_vertices(graph, ordering_func(graph))
    return max(colors.values())
        

def assign_node_colors(graph, colors):
    """Assigns colors from {node:color} dict colors to nx.Graph graph"""
    for node, color in colors.items():
        graph.add_node(node, color=color)
        

def draw_colored_graph(graph):
    """Displays the graph, colouring nodes - nodes should have 'color' data"""
    node_colors = [x[1] for x in graph.nodes(data='color')]
    nx.draw(graph, with_labels=True, node_color=node_colors, font_weight='bold', font_color='white')
    
def draw_random_colored_graph(n, p):
    """Generates a random graph, then greedy colours its nodes with default
    order 1,2,...,n. Then displays the graph"""
    G = get_random_graph(n, p)
    assign_node_colors(G, color_vertices(G, G.nodes))
    plt.figure()
    draw_colored_graph(G)
    
def get_random_graph(n, p, k=None):
    """Random graph of n nodes and prob p for each edge"""
    nodes = set(range(1, n+1))
    edges = set(itertools.combinations(nodes, 2))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for edge in edges:
        if random.random() < p:
            G.add_edges_from([edge])
            
    if not k is None:
        for i, j in [[i, j]  for i, j in G.edges if (i-j) % k == 0]:
            G.remove_edge(i, j)
    
    return G

def get_random_graph_k(n, p, k):
    """Random graph of n nodes and p edge prob with constraint i not adjacent 
    to j if i-j = 0 mod k"""
    G = get_random_graph(n, p)
    for i, j in [[i, j]  for i, j in G.edges if (i-j) % k == 0]:
        G.remove_edge(i, j)
        
    return G

def incr_deg_ordering(G):
    """returns list of nodes ordered by increasing degree"""
    node_degrees = [[node, G.degree(node)] for node in G.nodes]
    node_degrees.sort(key=lambda x: x[1])
    return [x[0] for x in node_degrees]

def decr_deg_ordering(G):
    """returns list of nodes ordered by decreasing degree"""
    node_degrees = [[node, G.degree(node)] for node in G.nodes]
    node_degrees.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in node_degrees]

def min_sub_deg_ordering(G):
    node_degrees = [[node, G.degree(node)] for node in G.nodes]
    ordering = []
    
    # starting from vn, take vn := min deg v in G
    # vn-1 := min deg v in G - {vn}, 
    # vj := min deg v in G - {vj+1, ..., vn}
    for _ in range(len(node_degrees)):
        node_degrees.sort(key=lambda x: x[1])
        node = node_degrees.pop(0)[0]
        ordering.append(node)
        neighbors = list(G.neighbors(node))
        for node_degree in node_degrees:
            if node_degree[0] in neighbors:
                node_degree[1] -= 1
        
    #reverse the ordering to get [v1, v2, ..., vn]
    return list(reversed(ordering))

# min_sub_deg_ordering and max_sub_deg_ordering sound similar but not same, e.g.
# [1,2,3,4,5,6,7] nodes and [[1,4], [2,4], [3,4], [5,6], [6,7], [7,5]] edges

def max_sub_deg_ordering(G):
    node_degrees = [[node, G.degree(node)] for node in G.nodes]
    ordering = []
    
    # starting from v1, take v1 := max deg v in G
    # v2 := max deg v in G - {v1}, 
    # vj := min deg v in G - {v1, ..., vj-1}
    for _ in range(len(node_degrees)):
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        node = node_degrees.pop(0)[0]
        ordering.append(node)
        neighbors = list(G.neighbors(node))
        for node_degree in node_degrees:
            if node_degree[0] in neighbors:
                node_degree[1] -= 1
        
    return ordering
            
        
def random_ordering(G):
    return G.nodes # graph G was randomly formed so I won't actually randomly permute its nodes here



ordering_funcs = [incr_deg_ordering, decr_deg_ordering, min_sub_deg_ordering, max_sub_deg_ordering, random_ordering]

graphs = [get_random_graph(60, 0.5) for _ in range(10)]
graphs2 = [get_random_graph(60, 0.75, k=3) for _ in range(10)]

def plot_num_colors(graphs, ordering_funcs, ordering_func_names):
    """Draws a plot of number of colours used by the greedy node colouring algorithm for each graph in graphs
    ordering_funcs: list of funcs which take a nx.Graph and give an ordering of nodes for the greedy algorithm
    ordering_func_names: list of strings - label on legend for each ordering_func 
    """
    num_colors_used = {}
    plt.figure()
    for ordering_func, ordering_func_name in zip(ordering_funcs, ordering_func_names):
        num_colors_used[ordering_func_name] = []
        for graph in graphs:
            node_ordering = ordering_func(graph)
            colors = color_vertices(graph, node_ordering)
            num_colors_used[ordering_func_name].append(max(colors.values()))

    for k, v in num_colors_used.items():
        plt.plot(v, label=k)
    
    plt.legend()
    return num_colors_used

# num_colors_used = plot_num_colors(graphs, ordering_funcs, ['incr_deg', 'decr_deg', 'min_sub_deg', 'max_sub_deg', 'random'])
# num_colors_used2 = plot_num_colors(graphs2, ordering_funcs, ['incr_deg', 'decr_deg', 'min_sub_deg', 'max_sub_deg', 'random'])


import operator as op
from functools import reduce

def fact(n):
    """fact(4) = 4*3*2*1 = 4! = 24"""
    return reduce(op.mul, range(1, n+1))

def ncr(n, r):
    """n choose r e.g. ncr(n, 2) = n*(n-1)/2"""
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def expected_num_ks(n, p, s):
    """expected number of K_s complete subgraphs in randomly generated G(n, p)"""
    return ncr(n, s) * p ** ncr(s, 2)

def variance_num_ks(n, p, s):
    """varaince of number of K_s complete subgraphs in randomly generated G(n, p)"""
    output = 0
    for l in range(2, s+1):
        output += ncr(n-s, s-l) * ncr(s, l) * (p**(-ncr(l, 2)) - 1)
        
    output *= ncr(n, s) * p ** ncr(s, 2)
    return output

def rough_expected_node_num_cliques(n, p, j):
    """This function doesn't seem quite right as I thought it would agree with
    expected_num_ks, i.e. f(n, p, j) = g(pn, p, j-1) for f, g = rough,expect
    This func aims to estimate the expected number of K_j a randomly chosen node of G(n, p) would be in"""
    j = j-1
    return n**j * p**(j*(j+1)/2) / fact(j)

v = variance_num_ks(1000, 0.5, 13)
e = expected_num_ks(1000, 0.5, 13)
other_e = rough_expected_node_num_cliques(2000, 0.5, 14)
print(v/e**2)
print(e)
print(other_e)


    
def greedy_find_clique_number(graph, progress=False, nodes=None, ordering_func=None, node_deletion=True):
    """Finds the order of graph's clique (largest complete subgraph)
    Very dumb algorithm just goes through each node, making sub-graphs
    on sub-graphs from each.
    
    progress: Boolean whether or not to display a tqdm progress bar for this graph -
    generally want this True for highest level graph but False for any induced sub-graphs
    
    nodes: list of nodes of graph - can be recursively passed to save recalculation.
    If is None, is calculated via ordering_func, with default ordering if ordering_func=None
    
    node_deletion: whether to delete nodes as we go along - in theory good but we
    have to make graph copies which could add time
    """
    graph_order = graph.order()
    if graph_order == 0:
        return 0

    if nodes is None:
        nodes = list(graph.nodes) if ordering_func is None else ordering_func(graph)
        
    output=0
    node_index_iterator = tqdm.tqdm(range(graph_order)) if progress else range(graph_order)
    for i in node_index_iterator:
        node = nodes[i]
        neighbours = set(graph.neighbors(node))
        if node_deletion:
            output = max(output, 1 + greedy_find_clique_number(graph.subgraph(neighbours).copy(),
                progress=False, nodes=[u for u in nodes[i+1:] if u in neighbours], node_deletion=node_deletion))
            graph.remove_node(node)
        else:
            output = max(output, 1 + greedy_find_clique_number(graph.subgraph(neighbours),
                progress=False, nodes=[u for u in nodes if u in neighbours], node_deletion=node_deletion))
        
    return output

# This version has incorrect node ordering but somehow runs a little faster. Perhaps there's a bug in greedy_find_clique_number?
# Or perhaps the extra computations to pass on nodes isn't worth the effort.
def greedy_find_clique_number2(graph, progress=False, nodes=None, ordering_func=None, node_deletion=True):
    """Finds the order of graph's clique (largest complete subgraph)
    Very dumb algorithm just goes through each node, making sub-graphs
    on sub-graphs from each.
    
    progress: Boolean whether or not to display a tqdm progress bar for this graph -
    generally want this True for highest level graph but False for any induced sub-graphs
    
    nodes: list of nodes of graph - can be recursively passed to save recalculation.
    If is None, is calculated via ordering_func, with default ordering if ordering_func=None
    
    node_deletion: whether to delete nodes as we go along - in theory good but we
    have to make graph copies which could add time
    """
    graph_order = graph.order()
    if graph_order == 0:
        return 0

    nodes = list(graph.nodes) if ordering_func is None else ordering_func(graph)
        
    output=0
    node_index_iterator = tqdm.tqdm(range(graph_order)) if progress else range(graph_order)
    for i in node_index_iterator:
        node = nodes[i]
        neighbours = set(graph.neighbors(node))
        if node_deletion:
            output = max(output, 1 + greedy_find_clique_number2(graph.subgraph(neighbours).copy(),
                progress=False, node_deletion=node_deletion))
            graph.remove_node(node)
        else:
            output = max(output, 1 + greedy_find_clique_number2(graph.subgraph(neighbours),
                progress=False, node_deletion=node_deletion))
        
    return output

def greedy_find_clique(graph, progress=False, nodes=None, ordering_func=None, node_deletion=True):
    """Finds the graph's clique (largest complete subgraph)
    Very dumb algorithm just goes through each node, making sub-graphs
    on sub-graphs from each.
    
    progress: Boolean whether or not to display a tqdm progress bar for this graph -
    generally want this True for highest level graph but False for any induced sub-graphs
    
    nodes: list of nodes of graph - can be recursively passed to save recalculation.
    If is None, is calculated via ordering_func, with default ordering if ordering_func=None
    
    node_deletion: whether to delete nodes as we go along - in theory good but we
    have to make graph copies which could add time
    """
    graph_order = graph.order()
    if graph_order == 0:
        return []

    if nodes is None:
        nodes = list(graph.nodes) if ordering_func is None else ordering_func(graph)
        
    output=[]
    node_index_iterator = tqdm.tqdm(range(graph_order)) if progress else range(graph_order)
    for i in node_index_iterator:
        node = nodes[i]
        neighbours = set(graph.neighbors(node))
        if node_deletion:
            subgraph_output = [node] + \
                greedy_find_clique(graph.subgraph(neighbours).copy(),
                progress=False, nodes=[u for u in nodes[i+1:] if u in neighbours], node_deletion=node_deletion)
            
            graph.remove_node(node)
        else:
            subgraph_output = [node] + \
            greedy_find_clique(graph.subgraph(neighbours),
                progress=False, nodes=[u for u in nodes if u in neighbours], node_deletion=node_deletion)
            
        output = output if len(output) >= len(subgraph_output) else subgraph_output
        
    return output

def clique_colouring_algorithm(G):
    """Uses a clique finding algorithm on G's complement to iteratively find largest
    independent sets, which are assigned the same colour then removed fromt the graph"""
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
    

def get_complete_graph(n):
    """Returns a K_n"""
    G = nx.Graph()
    nodes = list(range(1, n+1))
    G.add_nodes_from(nodes)
    G.add_edges_from(list(itertools.combinations(nodes, 2)))
    return G
        
    

def greedy_complete_subgraph_find(graph, node=None):
    """Goes through graph.nodes, adding them to an accumulating complete subgraph,
    or discarding them if they don't fit. Returns the resultant clique's list of nodes.
    
    Currently only goes through graph.nodes in order. (This is random if graph is random)
    
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
    """List of expected remaining nodes of G(n, p) at stage 1,2,3... of greedy clique finding algorithm"""
    results = [n-1]
    j = 1
    while results[-1] > 0:
        results.append(int(results[-1] - 1/p**j))
        j += 1
    return results

def get_chromatic_number_bounds(graph, clique_finder=greedy_find_clique_number):
    """Gives lower and upper bounds on chromatic number of graph"""
    lb_greedy = len(greedy_complete_subgraph_find(graph))
    lb_clique = clique_finder(graph.copy())
    ub_clique = max(clique_colouring_algorithm(graph.copy())[0].keys())
    ub_greedy = max(color_vertices(graph, graph.nodes).values())
    
    return lb_greedy, lb_clique, ub_clique, ub_greedy

def q4_plot_chromatic_num_lower_upper_bounds(n, p, k=None, clique_finder=greedy_find_clique_number):
    print("Q4: generating random graphs")
    graphs = [get_random_graph(n, p, k) for _ in range(10)]
    print("Q4: getting upper bounds")
    upper_bounds_greedy = [max(color_vertices(graph, graph.nodes).values()) for graph in graphs]
    upper_bounds_clique = list(map(lambda graph: max(clique_colouring_algorithm(graph)[0].keys()), graphs))
    print("Q4: getting greedy lower bounds")
    lower_bounds_greedy = [greedy_complete_subgraph_find(graph) for graph in graphs]
    print("Q4: getting clique  lower bounds")
    lower_bounds_clique = [clique_finder(graph) for graph in tqdm.tqdm(graphs)]
    for greedy_row, lb, ub_greedy, ub_clique in zip(lower_bounds_greedy, lower_bounds_clique, upper_bounds_greedy, upper_bounds_clique):
        print("{} & {} & {} & {} & {}\\\\".format(len(greedy_row), lb, ub_clique, ub_greedy, " ".join(map(lambda x: str(x), greedy_row))))
        

def q5_plot_chromatic_num_bounds_by_prob(n, prange, pstep, k=None, clique_finder=greedy_find_clique_number):
    """Plots a graph of number of colours against edge probability, for each of the various lower/upper bounds
    of chromatic number"""
    probs = np.arange(prange[0], prange[1], pstep)
    print("generating random graphs")
    graphs = [[get_random_graph(n, p, k) for _ in range(10)] for p in probs]
    mean_bounds = []
    for Gs in tqdm.tqdm(graphs):
        bounds = np.array(  list(map(get_chromatic_number_bounds, Gs)) )
        mean_bounds.append(np.mean(bounds, axis=0))
        
    mean_bounds = np.array(mean_bounds)
    plt.figure()
    for i, label in zip(range(mean_bounds.shape[1]), ['lb_greedy', 'lb_clique', 'ub_clique', 'ub_greedy']):
        plt.plot(probs, mean_bounds[:, i], label=label)
    plt.legend()
    
    return probs, mean_bounds
        
        



def permutations_list(bar):
    if len(bar) == 1:
        return [bar]
    else:
        outputs = []
        for i in range(len(bar)):
            outputs +=  [[bar[i]] + thing for thing in permutations_list(bar[:i] + bar[i+1:])]
        return outputs
    
def permutations_generator(bar):
    if len(bar) == 1:
        yield bar
    else:
        for i in range(len(bar)):
            for thing in permutations_generator(bar[:i] + bar[i+1:]):
                yield [bar[i]] + thing 

def clique_finding_speed_comparison(n=60, p=0.5, node_deletion=True):
    """Times various clique finding algorithms and compares their outputs"""
    graphs = [get_random_graph(n, p) for _ in range(10)]
    ordering_funcs = [None, incr_deg_ordering, decr_deg_ordering]
    node_deletions= [True, False] if node_deletion else [True]
    clique_finding_funcs = [partial(greedy_find_clique_number, progress=False, ordering_func=of,
        node_deletion=nd) for nd in node_deletions for of in ordering_funcs]
    clique_finding_funcs.append(nx.algorithms.graph_clique_number)
    
    results = []
    max_clique_numbers = []
    for clique_finding_func in clique_finding_funcs:
        results.append(0)
        max_clique_numbers.append([])
        for graph in tqdm.tqdm(graphs):
            G = graph.copy()
            s = time.time()
            max_clique_numbers[-1].append(clique_finding_func(G))
            results[-1] += time.time() - s
            
    return results, max_clique_numbers


def clique_finding_speed_comparison2(n=60, p=0.5):
    """Times various clique finding algorithms and compares their outputs"""
    graphs = [get_random_graph(n, p) for _ in range(10)]
    ordering_funcs = [None, incr_deg_ordering, decr_deg_ordering]
    clique_finding_funcs = [partial(f, progress=False, ordering_func=of) for f in [greedy_find_clique_number, greedy_find_clique_number2] for of in ordering_funcs]
    clique_finding_funcs.append(nx.algorithms.graph_clique_number)
    
    results = []
    max_clique_numbers = []
    for clique_finding_func in clique_finding_funcs:
        results.append(0)
        max_clique_numbers.append([])
        for graph in tqdm.tqdm(graphs):
            G = graph.copy()
            s = time.time()
            max_clique_numbers[-1].append(clique_finding_func(G))
            results[-1] += time.time() - s
            
    return results, max_clique_numbers


if __name__=='__main__':
    #q4_plot_chromatic_num_lower_upper_bounds(40, 0.5, clique_finder=partial(greedy_find_clique_number, ordering_func=decr_deg_ordering))
    # results, max_clique_numbers = clique_finding_speed_comparison(60, 0.5, node_deletion=False)
    # results, max_clique_numbers = clique_finding_speed_comparison2(40, 0.5)
    idk = q5_plot_chromatic_num_bounds_by_prob(60, [0.4, 0.6], 0.02, k=7)
    


#ns = [30*i for i in range(1, 15)]
#graphs = [get_random_graph(n, 0.5) for n in ns]
#node_clique_numbers = []
#for graph in tqdm(graphs):
#    node_clique_numbers.append(sum(nx.algorithms.node_clique_number(graph, [1,2,3,4,5]).values())/5)
#    
#plt.plot(ns, node_clique_numbers)