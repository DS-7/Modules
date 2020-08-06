import scipy.stats
import networkx as nx
import numpy as np
import pandas as pd
import math
import powerlaw

feature_names = ['degs std %', 'skew', 'skew p-val']

def degs_features(n, degs):
    degs_std = np.std(list(degs))
    skew = scipy.stats.skew(list(degs))
    skew_pval = scipy.stats.skewtest(list(degs)).pvalue
    #sigma = nx.algorithms.smallworld.sigma(g), computationally expensive
    #omega = nx.algorithms.smallworld.omega(g), computationally expensive
    return pd.Series([degs_std / n, skew, skew_pval], index = feature_names)

def graph_features(g):
    n = len(g.nodes())
    degs = dict(g.degree()).values()
    return degs_features(n, degs)

def d_graph_in_features(g):
    n = len(g.nodes())
    in_degs = dict(g.in_degree()).values()
    return degs_features(n, in_degs)

def d_graph_out_features(g):
    n  = len(g.nodes())
    out_degs = dict(g.out_degree()).values()
    return degs_features(n, out_degs)

def simulate_labeled_graphs(sz, samples = 120):
    df = pd.DataFrame()
    power_law_pvalue = []
    for m in np.random.uniform(1, 9, samples):
        m = int(m)
        pam = nx.barabasi_albert_graph(sz, m)
        s = graph_features(pam)
        df = df.append(s.append(pd.Series(['PA'], index = ['Target'])), ignore_index = True)
    for p in np.random.uniform(.00001, .1, samples):
        wsg_l = nx.watts_strogatz_graph(sz, 6, p)
        s = graph_features(wsg_l)
        df = df.append(s.append(pd.Series(['SW_L'], index = ['Target'])), ignore_index = True)
    for p in np.random.uniform(.1, 1, samples):
        wsg_h = nx.watts_strogatz_graph(sz, 6, p)
        s = graph_features(wsg_h)
        df = df.append(s.append(pd.Series(['SW_H'], index = ['Target'])), ignore_index = True)

    return df

def network_summary(g):
    if(g.is_directed()):
        print("Considering underlying undeirected graph")
        g = g.to_undirected()
    node_count = len(g.nodes())
    print("Nodes:", node_count)
    print("Edges:", g.size())
    print("Mean degree:", np.mean(list(dict(g.degree()).values())))
    print("Components#:", len(list(nx.connected_components(g))))
    largest_component = sorted(nx.connected_components(g), key = len, reverse = True)[0]
    largest_component_subgraph = g.subgraph(largest_component)
    print("Largest component node percentage:", len(largest_component) / node_count)
    fit = powerlaw.Fit(list(dict(nx.degree(largest_component_subgraph)).values()), discrete = True)
    print("Powerlaw Alpha of largest component:", fit.power_law.alpha)
    print("Clustering coefficient of largest component:", nx.transitivity(largest_component_subgraph))
    print("Average local clustering of largest component:", nx.average_clustering(largest_component_subgraph))
    print("Diameter of largest component:", nx.diameter(largest_component_subgraph, usebounds=True))

if __name__ == "__main__":
    # execute only if run as a script
    g = nx.barabasi_albert_graph(1000, 5)
    network_summary(g)