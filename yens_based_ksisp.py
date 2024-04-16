#!/usr/bin/env python3

from itertools import islice
from networkit import graphtools 
import networkx as nx
import statistics

import argparse

import time
import networkit as nk

        

def hist2nk(name):
    with open(name, "r", encoding='UTF-8') as fhandle:
        print("READING GRAPH:",name)
        firstline = True
        for line in fhandle:
            # print(line)
            if firstline == True:
                fields = line.split(" ")
                firstline = False
                # print(fields)
                n = int(fields[0])
                m = int(fields[1])
                weighted = int(fields[2])==1
                directed = int(fields[3])==1
                graph = nk.graph.Graph(n,weighted,directed)
            else:
                fields = line.split(" ")
                graph.addEdge(int(fields[1]),int(fields[2]),int(fields[3]))
                    
        if not graph.numberOfEdges()==m:
            print(graph.numberOfEdges(),m)
            raise Exception('misreading of graph')
        wgraph = nk.graph.Graph(graph.numberOfNodes(),graph.isWeighted(),graph.isDirected())
        assert graph.numberOfNodes()==wgraph.numberOfNodes()
        if weighted==True:
            for vertice in range(graph.numberOfNodes()):
                for vicino in graph.iterNeighbors(vertice):
                    wgraph.addEdge(vertice,vicino,graph.weight(vertice,vicino))
        else:
            for vertice in range(graph.numberOfNodes()):
                for vicino in graph.iterNeighbors(vertice):
                    wgraph.addEdge(vertice,vicino)
        return wgraph

 









def k_shortest_paths(grafo_, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(grafo_, source, target, weight=weight), k))


def read_nde(filename:str):
    with open(filename,'r', encoding='UTF-8') as f:
        grafo_ = nk.Graph()
        vertices = False
        for line in f.readlines():
            edge = line.split(' ')
    
            if not vertices:
                print("vertices:",int(edge[0]))
                vertices = True
                continue
            if int(edge[0])>int(edge[1]):
                grafo_.addEdge(int(edge[0]), int(edge[1]), addMissing=True)
            else:
                grafo_.addEdge(int(edge[1]), int(edge[0]), addMissing=True)
        return grafo_


        




from yens_based_labeling_handler import labeling_handler

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()

    parser.add_argument('--g',metavar="GRAPH", required=True,  help='Path to the graph file (.hist format)')
    parser.add_argument('--k', metavar="K_VALUE", required=True, help='Number of top shortest paths to seek for', default=2)
    # parser.add_argument('--b', metavar="BRIDGES_OPT", required=False, help='Bridges optimization is on [0: false, 1: true]', default=0)

    args = parser.parse_args()


    
    if ".hist" in str(args.g):
        G_prime_prime = hist2nk(str(args.g))
    else:
        G_prime_prime = read_nde(str(args.g))
    
 

    G_prime=graphtools.toUndirected(G_prime_prime)
    G = graphtools.toUnweighted(G_prime)    
    G.removeMultiEdges()
    G.removeSelfLoops()
    G.indexEdges()
    
    # if not __debug__:
    #     nk.overview(G)

    # """ LARGEST CONNECTED COMPONENT ONLY"""
    cc = nk.components.ConnectedComponents(G)
    cc.run()
    G = cc.extractLargestConnectedComponent(G, True)

    print("vertices:",G.numberOfNodes(),"arcs:",G.numberOfEdges())
 


     
    # if not __debug__:
    #     print("AFTER SCC")
    #     nk.overview(G)
    K = int(args.k)
    print("Value of K:",K)
    

    
    handler = labeling_handler(G,K)
    if __debug__ and handler.graph.numberOfNodes()<100:
        handler.draw_graph()

    handler.assign_ordering("bet")

    # handler.build(bridges_optimization)
    handler.build()

    cpu_labeling=[]
    cpu_yen=[]

    from datetime import datetime
    import csv
    
    from progress.bar import IncrementalBar

    now = datetime.now() # current date and time

    date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    # date_time = now.strftime("%d_%m_%Y_%H_%M_%S_%f")

    statsfile = str(args.g)+"_"+str(K)+"_"+date_time+'.csv'
    N_QUERIES = 25000 #if G.numberOfNodes()<=100 else 100000


    bar = IncrementalBar('Performing queries:', max = N_QUERIES)
    pairs = []
    queries_done = 0
    aux_graph = nk.nxadapter.nk2nx(G)
    # import sys
    while queries_done < N_QUERIES:

        
        
        try:
            first = graphtools.randomNode(G)    
            second = graphtools.randomNode(G)    
            
            cpu6 = time.perf_counter_ns() / 1000
            yen_paths = k_shortest_paths(aux_graph, first, second, K)
            
            
        except nx.NoPathException:    
            continue
        
        cpu7 = time.perf_counter_ns() / 1000

        cpu_yen.append(round(cpu7-cpu6,2))

        pairs.append((first,second))
        queries_done+=1

        cpu8 = time.perf_counter_ns() / 1000
        handler.top_k_paths(first,second,True)


        cpu9 = time.perf_counter_ns() / 1000

        cpu_labeling.append(round(cpu9-cpu8,2))
        
        Yres = sorted(yen_paths, key = lambda i: (len(i), i))
        Lres = handler.getResultPaths()
        
        if len(Lres)!=len(Yres):
            print("query:",first,second)
            print("paths yen:",Yres)
            print("indices yen:",[[handler.ordering[x] for x in p] for p in Yres])

            print("paths labeling:",list(Lres))
            print("indices labeling:",[[handler.ordering[x] for x in p] for p in Lres])
            
            handler.top_k_paths(first,second,True)
            if handler.graph.numberOfNodes()<100:
                handler.draw_graph()
                nodes = handler.draw_subgraph_rooted_at(first,second,Yres,Lres)
                for j in range(min(K,len(Yres))):
                    assert all(i in nodes for i in Yres[j])
            handler.loopy_top_k_paths(first,second,True)

            print("looppaths labeling:",list(i for i in handler.getResultPaths()))

            raise Exception('cardinality exception')
        
        
        for i in range(len(Yres)):
            
            if len(Yres[i])-1!=len(Lres[i])-1:
                print("\nWrong path is #:",i,"\nfor query:",first,second)
                print("path yen:",Yres[i],"\npath labeling is:",Lres[i])
                print("length yen:",len(Yres[i])-1,"\nlength labeling:",len(Lres[i])-1)
                print("\n\npaths (indices):\t")
                for c in range(len(Yres)):
                    print(c)
                    print(Yres[c],"(",[handler.ordering[x] for x in Yres[c]],")")
                    print(Lres[c],"(",[handler.ordering[x] for x in Lres[c]],")")

                
                

                handler.top_k_paths(first,second,True)
                if handler.graph.numberOfNodes()<100:
                    nodes = handler.draw_subgraph_rooted_at(first,second,Yres,Lres,i)
                    for j in range(K):
                        assert all(i in nodes for i in Yres[j])
                print(handler.labels[first])
                print(handler.labels[second])
                raise Exception('correctness exception')
            
            if Yres[i]!=Lres[i]:
                if len(Yres[i])==len(set(Yres[i])) and len(Lres[i])==len(set(Lres[i])):
                    continue
                print(Yres)
                print(Lres)
                print("\nWrong Path#:",i,"\nvertices:",first,second,"path yen:",Yres[i],"path labeling:",Lres[i])
                raise Exception('sorting exception')

        bar.next()
        
    bar.finish()
    if len(cpu_labeling)!=len(cpu_yen):
        raise Exception('timing exception')
    print(pairs[:5],pairs[-5:])

    print(cpu_labeling[:5],cpu_labeling[-5:])
    print(cpu_yen[:5],cpu_yen[-5:])


    print("Avg query time:",round(statistics.mean(cpu_labeling),2),"us")
    print("Avg yen time:",round(statistics.mean(cpu_yen),2),"us")
    print("Med query time:",round(statistics.median(cpu_labeling),2),"us")
    print("Med yen time:",round(statistics.median(cpu_yen),2),"us")



    with open(statsfile, 'a', newline='', encoding='UTF-8') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(["graph_name",\
                         "date",\
                         "vertices",\
                         "arcs",\
                         "k",\
                         "diameter",\
                         "centralityTime",\
                         "construction_time",\
                         "avg_query_time", \
                         "avg_yen_time",\
                         "med_query_time",\
                         "med_yen_time",\
                         "label_entries", \
                         "space", \
                         "avg_entries",\
                         "med_entries",\
                         "max_entries",\
                         "avg_same_hub_entries",\
                         "med_same_hub_entries",\
                         "max_same_hub_entries",\
                         "prunings"])
        
        finish_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    
        writer.writerow([str(args.g),\
                         str(finish_time),\
                         handler.graph.numberOfNodes(),\
                         handler.graph.numberOfEdges(),\
                         str(K),\
                         handler.diametro,\
                         handler.getCentralityTime(),\
                         handler.getConstructionTime(),\
                         round(statistics.mean(cpu_labeling),2),\
                         round(statistics.mean(cpu_yen),2),\
                         round(statistics.median(cpu_labeling),2),\
                         round(statistics.median(cpu_yen),2),\
                         handler.getLabelingSize(),\
                         handler.getLabelingSpace(),\
                         handler.getAvgLabelingSize(),\
                         handler.getMedianLabelingSize(),\
                         handler.getMaxLabelingSize(),\

                         handler.getAvgSameHubLabels(),\
                         handler.getMedianSameHubLabels(),\
                         handler.getMaxSameHubLabels(),\
                         handler.getPrunings()])
            

