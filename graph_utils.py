# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:50:13 2020
"""

import random
import itertools
import networkx as nx
import operator as op
from functools import reduce

def get_random_graph(n, p, k=None):
    """Random graph of n nodes and prob p for each edge
    k: int/None. Adds constraint that note i not adjacent to j if i-j = 0 mod k
    """
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

####
# Different graph orderings
####

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
    """returns ordering of nodes {v1,..., vn} where
    starting from vn, vn := min deg v in G
    vn-1 := min deg v in G - {vn}, 
    vj := min deg v in G - {vj+1, ..., vn}"""
    
    node_degrees = [[node, G.degree(node)] for node in G.nodes]
    ordering = []
    

    for _ in range(len(node_degrees)):
        node_degrees.sort(key=lambda x: x[1])   # sorted from min to max degree
        node = node_degrees.pop(0)[0]           # take min degree node as vj
        ordering.append(node)
        for node_degree in node_degrees:        # update degrees per removing vj
            if G.has_edge(node, node_degree[0]):
                node_degree[1] -= 1
        
    #reverse the ordering to get [v1, v2, ..., vn]
    return list(reversed(ordering))

# min_sub_deg_ordering and max_sub_deg_ordering sound similar but not same, e.g.
# [1,2,3,4,5,6,7] nodes and [[1,4], [2,4], [3,4], [5,6], [6,7], [7,5]] edges

def max_sub_deg_ordering(G):
    """returns ordering of nodes {v1,..., vn} where
    starting from v1, take v1 := max deg v in G
    v2 := max deg v in G - {v1}, 
    vj := min deg v in G - {v1, ..., vj-1}"""
    
    node_degrees = [[node, G.degree(node)] for node in G.nodes]
    ordering = []
    
    for _ in range(len(node_degrees)):
        # sorted from max to min degree
        node_degrees.sort(key=lambda x: x[1], reverse=True) 
        # take max degree node as vj
        node = node_degrees.pop(0)[0]                       
        ordering.append(node)                               
        for node_degree in node_degrees:       # update degrees per removing vj
            if G.has_edge(node, node_degree[0]):
                node_degree[1] -= 1
        
    return ordering

            
def random_ordering(G):
    nodes = list(G.nodes)
    random.shuffle(nodes)
    return nodes


def get_complete_graph(n):
    """Returns a K_n"""
    G = nx.Graph()
    nodes = list(range(1, n+1))
    G.add_nodes_from(nodes)
    G.add_edges_from(list(itertools.combinations(nodes, 2)))
    return G

def factorial(n):
    """factorial(4) = 4*3*2*1 = 4! = 24"""
    return reduce(op.mul, range(1, n+1))

def ncr(n, r):
    """n choose r e.g. ncr(n, 2) = n*(n-1)/2"""
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def expected_num_ks(n, p, s):
    """expected number of K_s complete subgraphs in randomly generated G(n, p)
    """
    return ncr(n, s) * p ** ncr(s, 2)

def variance_num_ks(n, p, s):
    """variance of number of K_s complete subgraphs in randomly generated G(n, p)
    """
    output = 0
    for l in range(2, s+1):
        output += ncr(n-s, s-l) * ncr(s, l) * (p**(-ncr(l, 2)) - 1)
        
    output *= ncr(n, s) * p ** ncr(s, 2)
    return output

def rough_expected_node_num_cliques(n, p, j):
    """This function doesn't seem quite right as I thought it would agree with
    expected_num_ks, i.e. f(n, p, j) = g(pn, p, j-1) for f, g = rough,expect
    This func aims to estimate the expected number of K_j a randomly chosen node
    of G(n, p) would be in"""
    j = j-1
    return n**j * p**(j*(j+1)/2) / factorial(j)



