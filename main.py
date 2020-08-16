# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:31:49 2020

@author: benja
"""
import matplotlib.pyplot as plt
from graph_outputs import q5_plot_chromatic_num_bounds_by_prob
from graph_utils import variance_num_ks, expected_num_ks, rough_expected_node_num_cliques

if __name__=='__main__':
    v = variance_num_ks(1000, 0.5, 13)
    e = expected_num_ks(1000, 0.5, 13)
    other_e = rough_expected_node_num_cliques(2000, 0.5, 14)
    print(v/e**2)
    print(e)
    print(other_e)
    idk = q5_plot_chromatic_num_bounds_by_prob(60, [0.4, 0.6], 0.02, multi=5)
    plt.show()
    
    #    pool = multiprocessing.Pool(4)
    #    results = pool.map(tempf, range(15))
    #    x, y = np.mean(results), min(results)
    #    pool.close()
    #    pool.join()
    
    
    #q4_print_chromatic_num_lower_upper_bounds(40, 0.5, clique_finder=partial(greedy_find_clique_number, ordering_func=decr_deg_ordering))
    # results, max_clique_numbers = clique_finding_speed_comparison(60, 0.5)
    # idk = q5_plot_chromatic_num_bounds_by_prob(60, [0.4, 0.6], 0.02, k=7)