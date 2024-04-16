#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:46:35 2020

@author: anonym
"""


import networkit as nk 
import argparse
import sys

def hist2nk(name,to_directed=False):
    fhandle = open(name, "r")
    print("Reading:",name)
    firstline = True
    natively_directed = False;
    for line in fhandle:

        if firstline == True:
            fields = line.split(" ");
            firstline = False
            # print(fields)
            n = int(fields[0])
            m = int(fields[1])
            weighted = int(fields[2])==1
            natively_directed = int(fields[3])==1
            graph = nk.graph.Graph(n,weighted,natively_directed or to_directed==True)
        else:
            fields = line.split(" ");
            graph.addEdge(int(fields[1]),int(fields[2]),int(fields[3]))
            if to_directed==True and natively_directed==False:
                graph.addEdge(int(fields[2]),int(fields[1]),int(fields[3]))

                
    print("To_be_oriented:",to_directed==True)       
    #assert graph.numberOfEdges()==m or (to_directed==True and natively_directed==False and graph.numberOfEdges()==2*m)
    wgraph = nk.graph.Graph(graph.numberOfNodes(),graph.isWeighted(),graph.isDirected())
    assert graph.numberOfNodes()==wgraph.numberOfNodes()
    if weighted==True:
        for i in range(graph.numberOfNodes()):
            for v in graph.iterNeighbors(i):
                wgraph.addEdge(i,v,graph.weight(i,v))
    else:
        for i in range(graph.numberOfNodes()):
            for v in graph.iterNeighbors(i):
                wgraph.addEdge(i,v);
    return wgraph;

def writeedge(f,u,v,w,eid):
    assert u!=v;
    f.write(str(0)+" "+str(u)+" "+str(v)+" "+str(int(w))+"\n")
    
def mapping2hist(name,graph,mapp):
    print("saving:",name)
    f = open(name, "w")
    if graph.isWeighted():
        we = 1;
    else:
        we = 0;
    if graph.isDirected():
        di = 1;
    else:
        di = 0;
            
    graph.removeMultiEdges()   
    graph.removeSelfLoops()    

    f.write(str(graph.numberOfNodes())+" "+str(graph.numberOfEdges())+" "+str(we)+" "+str(di)+"\n")
    graph.forEdges(lambda u,v,w,eid : writeedge(f,mapp[u],mapp[v],w,eid));
        
    f.close()
    print("saved:",name)

    
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='subgraph extractor')
    parser.add_argument("--g",metavar="GRAPH FILE", required=True, default="", help="input hist graph file")   
    parser.add_argument("--o",metavar="OPERATION [0: shrink to size 1: to directed 2: to undirected 3: to weighted (random) 4: to unweighted 5: (strongly) connected component])", required=True, default="-1", help="operation to perform on graph")    
    parser.add_argument("--s",metavar="DESIRED SIZE (FRACTION IN (0,1] in case of OP=0)", required=False, default="0",  help="fraction of original size")    
    parser.add_argument("--r",metavar="ROOT NODE (IN CASE OF OP=6)", required=False, default="0",  help="BFS root node")    
    parser.add_argument("--d",metavar="MAX DISTANCE FROM ROOT NODE (IN CASE OF OP=6)", required=False, default="0",  help="Threshold distance from root r (d included)")    


    args = parser.parse_args()
    
    filename = str(args.g)
    operation = int(str(args.o)) 
    fraction = float(str(args.s))
    root = int(str(args.r))
    max_distance = int(str(args.d))
    
            
    if filename=="" or "hist" not in filename:
        raise Exception('wrong input file')
    if operation<0 or operation>6:
        raise Exception('wrong operation')

    if operation==0 and (fraction<=0 or fraction>1):
        raise Exception('wrong desired size (must be in (0,1])')


         
    if operation == 0:
        G = hist2nk(filename)

        threshold_n = round(G.numberOfNodes()*fraction)
        assert threshold_n>=3
        assert threshold_n<=G.numberOfNodes()
        nodes=set()
        source = nk.graphtools.randomNode(G)
        while len(nodes)<threshold_n:
            bfs = nk.distance.BFS(G, source, storePaths=False, storeNodesSortedByDistance = True)
            bfs.run()
            dist_vector = bfs.getNodesSortedByDistance();
            for i in range(len(dist_vector)):
                assert dist_vector[i]<sys.maxsize
                assert dist_vector[i]<float("inf")
                nodes.add(i)
                if len(nodes)==threshold_n:
                    break
            source = nk.graphtools.randomNode(G)

        subgraph = nk.graphtools.subgraphFromNodes(G,nodes)

            
        print("original:",G.numberOfNodes(),G.numberOfEdges()," density: ", nk.graphtools.density(G))
        stng = nk.overview(G)
        print(stng)  
        print("scaled:",subgraph.numberOfNodes(),subgraph.numberOfEdges()," density: ",nk.graphtools.density(subgraph))  
        stng = nk.overview(G)
        print(stng)  
        subgraph.checkConsistency()
        subgraph.indexEdges()
    
        remappedgraph=nk.Graph(subgraph.numberOfNodes())
        assert len(nodes)==subgraph.numberOfNodes()
        mapping=[-1 for i in range(G.numberOfNodes())]
        count=0
        for i in nodes:
            mapping[i]=count
            count+=1
        print(mapping[0:10])
        mapping2hist("scaled_"+str(fraction)+"_"+filename,subgraph,mapping)         
        
    if operation == 1:
        G = hist2nk(filename,True)
        stng = nk.overview(G)
        print(stng)  
        nodes=set([i for i in G.iterNodes()])
        mapping=[-1 for i in range(G.numberOfNodes())]
        count=0
        for i in nodes:
            mapping[i]=count
            count+=1
        mapping2hist("directed_"+filename,G,mapping)
    if operation == 2:
        G = hist2nk(filename)
        G_prime = nk.graphtools.toUndirected(G)
        G_prime.removeMultiEdges()
        G_prime.removeSelfLoops()
        G_prime.indexEdges()
        nodes=set([i for i in G.iterNodes()])
        mapping=[-1 for i in range(G.numberOfNodes())]
        count=0
        for i in nodes:
            mapping[i]=count
            count+=1
        mapping2hist("undirected_"+filename,G_prime,mapping)
    if operation == 3:
        raise Exception('not yet implemented')

    if operation == 4:
        G = hist2nk(filename)
        G_prime = nk.graphtools.toUnweighted(G)
        G_prime.removeMultiEdges()
        G_prime.removeSelfLoops()
        G_prime.indexEdges()
        nodes=set([i for i in G.iterNodes()])
        mapping=[-1 for i in range(G.numberOfNodes())]
        count=0
        for i in nodes:
            mapping[i]=count
            count+=1
        mapping2hist("unweighted"+filename,G_prime,mapping)    
    if operation == 5:
        G = hist2nk(filename)
        cc = nk.components.ConnectedComponents(G)
        cc.run()
        G_prime = cc.extractLargestConnectedComponent(G, True)
        G_prime.removeMultiEdges()
        G_prime.removeSelfLoops()
        G_prime.indexEdges()
        nodes=set([i for i in G.iterNodes()])
        mapping=[-1 for i in range(G.numberOfNodes())]
        count=0
        for i in nodes:
            mapping[i]=count
            count+=1
        mapping2hist("unweighted"+filename,G_prime,mapping)    
    
    if operation == 6:
        G = hist2nk(filename)
        
        diam = nk.distance.Diameter(G,algo=1)
        diam.run()
        diametro = diam.getDiameter()
        print("DIAMETER:",diametro)
        visited = [False] * G.numberOfNodes()
        queue = []
        queue.append([root,0])
        visited[root] = True
        reached_nodes = []
        while len(queue) > 0:
            v,d = queue.pop(0)
            if d > max_distance:
                break
            reached_nodes.append(v)
            for neighbor in G.iterNeighbors(v):
                if visited[neighbor]:
                    continue
                queue.append([neighbor, d+1])
                visited[neighbor] = True
        G_prime = nk.graphtools.subgraphFromNodes(G, reached_nodes)
        diam = nk.distance.Diameter(G_prime,algo=1)
        diam.run()
        diametro = diam.getDiameter()
        print("DIAMETER:",diametro)
        mapping=[-1 for i in range(G.numberOfNodes())]
        count=0
        nodes=set([i for i in G_prime.iterNodes()])
        for i in nodes:
            mapping[i]=count
            count+=1
        mapping2hist("bfs_induced_"+filename, G_prime,mapping)
