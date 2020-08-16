# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:02:52 2020
"""

import networkx as nx
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
import tqdm
import time
import random

from graph_utils import get_random_graph, incr_deg_ordering, \
    decr_deg_ordering, max_sub_deg_ordering, expected_num_ks, variance_num_ks,\
    rough_expected_node_num_cliques
from graph_algorithms import greedy_color_vertices, greedy_find_clique_number, \
     greedy_complete_subgraph_find, clique_colouring_algorithm,\
     greedy_find_clique_number2


def assign_node_colors(graph, colors):
    """Assigns colors from {node:color} dict colors to nx.Graph graph"""
    for node, color in colors.items():
        graph.add_node(node, color=color)
        

def draw_colored_graph(graph):
    """Displays the graph, coloring nodes - nodes should have 'color' data"""
    node_colors = [x[1] for x in graph.nodes(data='color')]
    nx.draw(graph, with_labels=True, node_color=node_colors, font_weight='bold', \
            font_color='white')
    
def draw_random_colored_graph(n, p):
    """Generates a random graph, then greedy colors its nodes with default
    order 1,2,...,n. Then displays the graph"""
    G = get_random_graph(n, p)
    assign_node_colors(G, greedy_color_vertices(G, G.nodes))
    plt.figure()
    draw_colored_graph(G)


def q1_plot_num_colors(graphs, ordering_funcs, ordering_func_names):
    """Draws a plot of number of colours used by the greedy node colouring
    algorithm for each graph in graphs
    
    ordering_funcs:     list of funcs which take a nx.Graph and give an ordering
        of nodes for the greedy algorithm
    ordering_func_names:    list of strings - label on legend for each ordering_func 
    """
    num_colors_used = {}
    plt.figure()
    for ordering_func, ordering_func_name in zip(ordering_funcs, ordering_func_names):
        num_colors_used[ordering_func_name] = []
        for graph in graphs:
            node_ordering = ordering_func(graph)
            colors = greedy_color_vertices(graph, node_ordering)
            num_colors_used[ordering_func_name].append(max(colors.values()))

    for k, v in num_colors_used.items():
        plt.plot(v, label=k)
    
    plt.legend()
    return num_colors_used


def get_chromatic_number_bounds(graph, clique_finder=greedy_find_clique_number):
    """Gives lower and upper bounds on chromatic number of graph"""
    
    # lower bound by size of a randomly extracted complete subgraph
    lb_complete = len(greedy_complete_subgraph_find(graph))
    # lower bound by size of largest complete subgraph (called a clique)
    lb_clique = clique_finder(graph.copy())
    # upper bound by number of colours used by clique colouring algorithm
    ub_clique = max(clique_colouring_algorithm(graph.copy())[0].keys())
    # upper bound - greedy colouring algorithm
    ub_greedy_rand = max(greedy_color_vertices(graph, graph.nodes).values())
    # upper bound - greedy colouring algorithm with max_sub_deg_ordering
    ub_greedy_msd = \
        max(greedy_color_vertices(graph, max_sub_deg_ordering(graph)).values())

    
    # From empirical observations often observe
    # lb_greedy <= lb_clique <= chromatic num <= ub_clique <= ub_greedy 
    return lb_complete, lb_clique, ub_clique, ub_greedy_rand, ub_greedy_msd

     
def q4_print_chromatic_num_lower_upper_bounds(n, p, k=None, \
                                    clique_finder=greedy_find_clique_number):
    """prints latex table style rows of lower/upper bound data on chromatic
    number of random G(n, p) graphs"""

    graphs = [get_random_graph(n, p, k) for _ in range(10)] 
    bounds = map(get_chromatic_number_bounds, tqdm.tqdm(graphs))
    complete_subgraphs = [greedy_complete_subgraph_find(graph) for graph in graphs]
    
    for complete_subgraph, lb_complete, lb_clique, ub_greedy, ub_clique in \
        zip(complete_subgraphs, *zip(*bounds)):
        print("{} & {} & {} & {} & {}\\\\".format(lb_complete, lb_clique,
            ub_clique, ub_greedy, " ".join(map(lambda x: str(x), complete_subgraph))))
 

def tempf(x):
    time.sleep(0.5)
    return x +  random.random()

def multiprocessing_chrom_bounds_func(graphs_list):
    return list(map(get_chromatic_number_bounds, graphs_list))

def q5_plot_chromatic_num_bounds_by_prob(n, prange, pstep, k=None,\
    clique_finder=greedy_find_clique_number, multi=False):
    """Plots a graph of number of colours against edge probability,
    for each of the various lower/upper bounds of chromatic number
    multi: True/False/int multiprocessing - yes/no/ num processes (default 4 if true)
    """
    probs = np.arange(prange[0], prange[1], pstep)
    graphs = [[get_random_graph(n, p, k) for _ in range(10)] for p in probs]
    mean_bounds = []
    pool = Pool(multi if type(multi) is int else 4)
    # graph_generator = pool.imap(multiprocessing_chrom_bounds_func, graphs) if multi else map(f, graphs)
    f = lambda graphs_list: list(map(get_chromatic_number_bounds, graphs_list))
    graph_generator = pool.imap(f, graphs) if multi else map(f, graphs)
    
    for bounds in tqdm.tqdm(graph_generator, total=len(graphs)):
        mean_bounds.append(np.mean(bounds, axis=0))
        
    pool.close()
    pool.join()
    
        
    mean_bounds = np.array(mean_bounds)
    plt.figure()
    for i, label in zip(range(mean_bounds.shape[1]), \
        ['lb_comp', 'lb_clique', 'ub_clique', 'ub_greedy_rand', 'ub_greedy_msd']):
        plt.plot(probs, mean_bounds[:, i], label=label)
    plt.legend()
    
    return probs, mean_bounds


def clique_finding_speed_comparison(n=60, p=0.5, \
    clique_finding_funcs=[greedy_find_clique_number], node_deletions=[True]):
    """Returns running times and output clique size of various clique finding 
    algorithms on the same set of random G(n, p) graphs"""
    
    graphs = [get_random_graph(n, p) for _ in range(10)]
    ordering_funcs = [None, incr_deg_ordering, decr_deg_ordering]
    
    # We'll compare the running time of these different clique finders on our random graphs
    clique_finding_funcs = [partial(cff, progress=False, ordering_func=of, node_deletion=nd) \
        for cff in clique_finding_funcs for of in ordering_funcs for nd in node_deletions]
    clique_finding_funcs.append(nx.algorithms.graph_clique_number)
    
    running_times = []
    max_clique_numbers = []
    for clique_finding_func in clique_finding_funcs:
        running_times.append(0)
        max_clique_numbers.append([])
        for graph in tqdm.tqdm(graphs):
            G = graph.copy()
            s = time.time()
            max_clique_numbers[-1].append(clique_finding_func(G))
            running_times[-1] += time.time() - s
        
    
    return running_times, max_clique_numbers

    