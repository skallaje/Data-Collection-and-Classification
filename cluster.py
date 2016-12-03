"""
cluster.py
"""
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import pickle
from twitter import *
from collections import Counter, defaultdict, deque
from itertools import chain, combinations

def create_graph(users):    
    G=nx.Graph()
    G.add_node('HillaryClinton')
    for u in users:
        G.add_edge('HillaryClinton',str(u))
        G.add_node(str(u))
        for friend in users[u]['friends']:
            G.add_node(str(friend))
            G.add_edge(str(u),str(friend))
    return G
    pass
    
def create_clusters(G):
    components=[]
    for component in nx.connected_component_subgraphs(G):
        components.append(component)
    return components
    
def girvan_newman_clustering(G, min, max):
    if ((G.order() >= min) and (G.order() <= max)):
        return [G.nodes()]
    betweenness = nx.edge_betweenness_centrality(G)
    edges = sorted(betweenness.items(), key=lambda x: -x[1])
    components = create_clusters(G)
    count = 0
    while len(components) == 1:
        G.remove_edge(*edges[count][0])
        count=count + 1
        components = create_clusters(G)
    print('Removed ' + str(count) + ' edge(s) among nodes')
    clusters=[]   
    for component in components:
        if component.order() > min:
            clusters.extend(girvan_newman_clustering(component, min, max))
    return clusters
    
def draw_network(graph,users,filename):
    n=graph.nodes()
    lbs={}    
    for node in n:
        if node=='HillaryClinton':
            lbs['HillaryClinton']='HillaryClinton'
        elif int(node) in users:
            lbs[node]=users[int(node)]['names']
         
    pos=nx.spring_layout(graph)
    nx.draw(graph,pos,node_color='#FF0000',edge_color='#00FFFF',node_size=18,width=0.5,edge_cmap=plt.cm.Blues,with_labels=False)
    nx.draw_networkx_labels(graph,pos,lbs,font_size=8,font_color='black')
    plt.savefig(filename, papertype=None, format=None,transparent=True, bbox_inches=None, pad_inches=0.1,dpi=500, facecolor='w', edgecolor='w',orientation='portrait')
    plt.clf()
    pass

def log_cluster_summary(clusters, users, G):
    f = open('clustering_summary.txt', 'w')
    f.write("SUMMARY OF CLUSTERING\n")
    f.write('The graph has %d nodes and %d edges' %(G.order(), G.number_of_edges())+"\n")
    i=0
    for res in clusters:
        i=i+1
        f.write('Size of Cluster'+str(i)+' is '+str(len(res))+"\n")
        g=G.subgraph(res)
        f.write('Subgraph'+str(i)+' has %d nodes and %d edges' % (g.order(), g.number_of_edges())+"\n")
        draw_network(g,users,"cluster"+str(i)+".png")
    
def main():
    users=defaultdict()
    with open('users.pkl', 'rb') as f:
        users = pickle.load(f)
    G=create_graph(users)
    draw_network(G,users,"Graph.png")
    print('Removing edges with high betweenness')
    min=90
    max=200
    clusters=girvan_newman_clustering(G, min, max)
    log_cluster_summary(clusters, users, G)
    
if __name__ == '__main__':
    main()