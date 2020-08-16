# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 14:42:15 2020

@author: benja

This is some quick experimental crappy code.  
"""

import math
import networkx as nx
import itertools
import tqdm
import random

def get_all_2edgecoloured_complete_graphs(n):
    """iterator over all 2 edge colourings of graph with n nodes"""
    nodes = set(range(1, n+1))
    edges = set(itertools.combinations(nodes, 2))
    for k in range(len(edges)):
        for red_edges in itertools.combinations(edges, k):
            G = nx.Graph()
            G.add_nodes_from(nodes)
            
            red_edges = set(red_edges)
            blue_edges = edges - red_edges
            G.add_edges_from(red_edges, colour=1)
            G.add_edges_from(blue_edges, colour=0)
    
            yield G
            
def get_edge_attributes(graph):
    return [graph.get_edge_data(u, v) for u, v in graph.edges]

def draw_coloured_graph(graph):
    ea =  get_edge_attributes(graph)
    nx.draw(graph, with_labels=True, edge_color=[i['colour'] for i in ea], font_weight='bold')
    

def is_edge_monochromatic(graph, colour=None):
    """True/False whether graph is edge monochromatic. If colour is specified will
    only return True if monochromatic for given colour
    Return:
    boolean
    """
    def get_edge_colour(graph, u, v):
        return graph.get_edge_data(u, v)['colour']
    edge_iterator = iter(graph.edges)
    seen_colour = get_edge_colour(graph, *next(edge_iterator))
    if seen_colour != colour:
        return False
    for u, v in edge_iterator:
        if get_edge_colour(graph, u, v) != colour:
            return False
    return True

def has_monochrome_subgraph(graph, n, colour=None):
    """Tries all subgraphs of n nodes in graph to see if if they're edge monochromatic"""
    for subgraph in itertools.combinations(graph.nodes, n):
        subgraph = graph.subgraph(subgraph)
        if is_edge_monochromatic(subgraph, colour=colour):
            return True
    return False

def try_ramsey(n, s):
    for graph in tqdm.tqdm(get_all_2edgecoloured_complete_graphs(n)):
        has_monochrome = has_monochrome_subgraph(graph, s)
        if not has_monochrome:
            return False
    return True

#def construct_iteratively(n, filterfunc, extensionfunc, initset):
#    def k_to_kp1(graphset, k):
#        for graph in graphset:
#            graph.add_node(k+1)
#            edges = [[i, k+1] for i in range(1, k+1)]
#            for k in range(k+1):
#                for red_edges in itertools.combinations(edges, k):
#                    graph.add_edges_from([[i, k+1] for i in range(1, k+1)])
#
#def try_ramsey2(n, s):
    
def temp(p):
    stuff = list(range(1, p))
    squares = [x**2 % p for x in stuff]
    qrs = [i in squares for i in stuff]
    return qrs

def temp2(p):
    qrs = temp(p)
    length = len(qrs)
    l = math.floor(length/2) - 1
    return [qrs[i] == qrs[length-1-i] for i in range(l)]

def make_random_graph(n, p):
    nodes = set(range(1, n+1))
    edges = set(itertools.combinations(nodes, 2))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for edge in edges:
        if random.random() < p:
            G.add_edges_from([edge])
    
    return G
#
#n = 300
#p = 0.2
#results = []
#for _ in tqdm.tqdm(range(100)):
#    diameter = nx.diameter(make_random_graph(n, p))
#    results.append(diameter)
#        

    
    
