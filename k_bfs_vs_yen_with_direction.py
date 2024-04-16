# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 20:50:49 2023

@author: anonym
"""


#!/usr/bin/env python3
from itertools import islice
from networkit import graphtools 
import networkx as nx
import argparse
from heapq import heappush,heappop
import networkit as nk
from networkx.utils import pairwise
import time
from progress.bar import IncrementalBar, PixelBar
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import math
import statistics
from collections import deque
import bisect,sys

class PathException(Exception):
    """Base class for exceptions in Searching Paths."""
class NoPathException(PathException):
    """Exception for algorithms that should return a path when running
    on graphs where such a path does not exist."""

    
from enum import IntEnum

# class syntax

class PacketType(IntEnum):
    NORMAL = 0
    TARGETED = 1
    
    
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

class yens_algorithm():

    def __init__(self, gr, kvalue):
        
        self.G = gr
        self.K = kvalue
        self.skip_node = [False for _ in self.G.iterNodes()]
        self.skip_edge = [False for _ in self.G.iterEdges()]
        self.queue_skip_node = deque()
        self.queue_skip_edge = deque()
        
    def is_simple(self,path):
            
        assert len(path) > 0
        assert type(path)==tuple
        assert all(v in self.G.iterNeighbors(u) for u, v in pairwise(path))
    
        s_path = set()
        for el in path:
            if el in s_path:
                assert len(set(path)) < len(path)
                return False
            s_path.add(el)
        assert len(set(path)) == len(path)
        return True
    
    def yen_bidir_BFS(self, source, target, bound_on_length):
    
    
        pred, succ, w = self.yen_bidir_pred_succ(source, target, bound_on_length)
        
        path = deque()
        # d = -1
    
        while w is not None:
            path.append(w)
            w = succ[w]
            # d+=1
    
    
        w = pred[path[0]]
        while w is not None:
            path.appendleft(w)
            w = pred[w]
            # d+=1
    
        # assert d==len(path)-1
        assert len(path)-1<bound_on_length
        return tuple(path)
                
    def yen_bidir_pred_succ(self, source, target, bound_on_length):
        

    
        if self.skip_node[source] or self.skip_node[target]:
            raise NoPathException
        if target == source:
            return ({target: None}, {source: None}, source)
    
    
        
        # predecesssor and successors in search
        pred = {source: None}
        succ = {target: None}
    
        # initialize fringes, start with forward
        forward_fringe = deque([source])
        reverse_fringe = deque([target])
        
        lunghezza=0
        
        while forward_fringe and reverse_fringe:
            lunghezza+=1
            if lunghezza>=bound_on_length:
                raise NoPathException

            if len(forward_fringe) <= len(reverse_fringe):
                this_level = forward_fringe
                forward_fringe = deque()
                for v in this_level:
                    for w in self.G.iterNeighbors(v):
                        assert self.G.edgeId(v,w) == self.G.edgeId(w,v)
                        if self.skip_edge[self.G.edgeId(v,w)]:
                            continue
                        if w==source or self.skip_node[w]:
                            assert w not in succ
                            continue
                        if w not in pred:
                            forward_fringe.append(w)
                            pred[w] = v
                        if w in succ:
                            del forward_fringe
                            del reverse_fringe
                            return pred, succ, w
            else:
                this_level = reverse_fringe
                reverse_fringe = deque()
                for v in this_level:
                    for w in self.G.iterNeighbors(v):
                        assert self.G.edgeId(v,w) == self.G.edgeId(w,v)
                        if self.skip_edge[self.G.edgeId(v,w)]:
                            continue
                        if w==target or self.skip_node[w]:
                            assert w not in pred
                            continue
                        if w not in succ:
                            succ[w] = v
                            reverse_fringe.append(w)
                        if w in pred:
                            del forward_fringe
                            del reverse_fringe
                            return pred, succ, w
        del forward_fringe
        del reverse_fringe
        
        raise NoPathException(f"No path between {source} and {target}.")
        
        
    def run(self, source, target):
        
        results = []
        
        yen_PQ = []
        paths = set()
        
        assert all(not self.skip_node[v] for v in self.G.iterNodes())
        assert all(not self.skip_edge[self.G.edgeId(u, v)] for u,v in self.G.iterEdges())          
            
        
        try:
            p = self.yen_bidir_BFS(source,target,sys.maxsize)
            assert p[0]==source
            assert p[-1]==target            
            assert p not in paths
            heappush(yen_PQ, (len(p)-1,p))
            paths.add(p)
            
            
        except NoPathException:
            del yen_PQ[:]
            del yen_PQ
            paths.clear()
            return results
        
        
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
            
    
        while len(yen_PQ)>0:
    
            _, P_det = heappop(yen_PQ)
            assert P_det[0]==source
            assert P_det[-1]==target
            
            assert type(P_det)==tuple
            results.append(P_det)
            assert sorted(results,key=len)==results
            
            if len(results)==self.K:
                del yen_PQ[:]
                del yen_PQ
                paths.clear()
                return results
          
            assert all(not self.skip_node[v] for v in self.G.iterNodes())
            assert all(not self.skip_edge[self.G.edgeId(u, v)] for u,v in self.G.iterEdges())          
                  

            
            for index in range(1, len(P_det)):
                
                l_root = P_det[:index]
                assert type(l_root)==tuple
                
                for path in results:
                    assert type(path)==tuple
                    if path[:index] == l_root:
                        eid = self.G.edgeId(path[index - 1], path[index])
                        if not self.skip_edge[eid]:
                            self.skip_edge[eid]=True
                            self.queue_skip_edge.append(eid)
                                          
                        
                try:
                    
                    pgt = self.yen_bidir_BFS(l_root[-1],target,sys.maxsize)                    
                    
                    assert len(l_root[:-1] + pgt)-1==len(pgt)-1+len(l_root[:-1])               

                    
                    new_path = l_root[:-1] + pgt
                    assert new_path[0]==source and new_path[-1]==target
                    assert self.is_simple(new_path)   
    
                    if new_path in paths:
                        if not self.skip_node[l_root[-1]]:
                            self.skip_node[l_root[-1]]=True
                            self.queue_skip_node.append(l_root[-1])
                        del new_path
                        continue
                        
                    heappush(yen_PQ,(len(new_path)-1,new_path))
                    paths.add(new_path)
    
                except NoPathException:
                    pass
                
                except Exception as err:
                    print(f"Unexpected {err=}, {type(err)=}")
                    raise
                    
                if not self.skip_node[l_root[-1]]:
                    self.skip_node[l_root[-1]]=True
                    self.queue_skip_node.append(l_root[-1])
            
            while len(self.queue_skip_node)>0:
                x = self.queue_skip_node[0]
                assert self.skip_node[x]
                self.skip_node[x]=False
                self.queue_skip_node.popleft()
                
            while len(self.queue_skip_edge)>0:
                e_id = self.queue_skip_edge[0]
                assert self.skip_edge[e_id]
                self.skip_edge[e_id]=False
                self.queue_skip_edge.popleft()
            
    
                
                
        return results
        
    

    

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

def test_equality(flag, array_A, array_B):
    assert flag == "YY" or flag =="BY"
    if flag == "BY":
        array_bfs = array_A
        array_yen = array_B
       
       
        assert all(type(i)==tuple for i in array_bfs)
        assert all(type(i)==tuple for i in array_yen)
        
        if len([i for i in array_bfs])!=len(set([i for i in array_bfs])):
            raise Exception('uniqueness exception')
        if len(array_yen)!=len(array_bfs):
            print("Y",len(array_yen))
            print("B",len(array_bfs))   
            for i in range(min(len(array_yen),len(array_bfs))):
                if array_yen[i]==array_bfs[i] or len(array_yen[i])==len(array_bfs[i]):
                    print(i,"correct",array_yen[i],array_bfs[i])
                else:
                    print(i,"mismatch",array_yen[i],array_bfs[i])
            for i in [x for x in array_yen if x not in array_bfs]:
                    print("missing",i)
            
            # if __debug__:
            #
            #
            #     pos = nx.kamada_kawai_layout(aux_graph)
            #
            #     colori = ['#029386' for _ in range(aux_graph.number_of_nodes())]  #'#029386'
            #     fig, ax = plt.subplots(figsize=(10,10))
            #     nx.draw(aux_graph,pos, node_color=colori,node_size=800,font_size=14)
            #     nx.draw_networkx_labels(aux_graph,pos,font_family="sans-serif",font_size=14)
            #     ax.set_axis_off()
            #     fig.tight_layout()
            #     plt.show()
            
            raise Exception('card exception')
        indice = 0
        for indice in range(len(array_yen)):
            if len(array_yen[indice])!=len(array_bfs[indice]):
                for xy in array_yen:
                    print("YEN",array_yen.index(xy),xy)
                for xy in array_bfs:
                    print("BFS",array_bfs.index(xy),xy)

                print("index",indice)
                print(array_yen[indice],"Y")  
                print(array_bfs[indice],"B")  
                
                
                # if __debug__:
                #
                #
                #     pos = nx.kamada_kawai_layout(aux_graph)
                #
                #     colori = ['#029386' for _ in range(aux_graph.number_of_nodes())]  #'#029386'
                #     fig, ax = plt.subplots(figsize=(10,10))
                #     nx.draw(aux_graph,pos, node_color=colori,node_size=800,font_size=14)
                #     nx.draw_networkx_labels(aux_graph,pos,font_family="sans-serif",font_size=14)
                #     ax.set_axis_off()
                #     fig.tight_layout()
                #     plt.show()
                raise Exception('correctness exception')
    else:
        array_yen_custom = array_A
        array_yen = array_B
        assert all(type(i)==tuple for i in array_yen_custom)
        assert all(type(i)==tuple for i in array_yen)
        
        if len([i for i in array_yen_custom])!=len(set([i for i in array_yen_custom])):
            raise Exception('uniqueness exception')
        if len(array_yen)!=len(array_yen_custom):
            print("Y",len(array_yen))
            print("Ycustom",len(array_yen_custom))   
            for i in range(min(len(array_yen),len(array_yen_custom))):
                if array_yen[i]==array_yen_custom[i] or len(array_yen[i])==len(array_yen_custom[i]):
                    print(i,"correct",array_yen[i],array_yen_custom[i])
                else:
                    print(i,"mismatch",array_yen[i],array_yen_custom[i])
            for i in [x for x in array_yen if x not in array_yen_custom]:
                    print("missing",i)
            
            # if __debug__:
            #
            #
            #     pos = nx.kamada_kawai_layout(aux_graph)
            #
            #     colori = ['#029386' for _ in range(aux_graph.number_of_nodes())]  #'#029386'
            #     fig, ax = plt.subplots(figsize=(10,10))
            #     nx.draw(aux_graph,pos, node_color=colori,node_size=800,font_size=14)
            #     nx.draw_networkx_labels(aux_graph,pos,font_family="sans-serif",font_size=14)
            #     ax.set_axis_off()
            #     fig.tight_layout()
            #     plt.show()
            
            raise Exception('card exception')
        indice = 0
        for indice in range(len(array_yen)):
            if len(array_yen[indice])!=len(array_yen_custom[indice]):
                print("Y",array_yen)
                print("Ycustom",array_yen_custom)    
                print("Y",array_yen[indice])
                print("Ycustom",array_yen_custom[indice])  
                
                # if __debug__:
                #
                #
                #     pos = nx.kamada_kawai_layout(aux_graph)
                #
                #     colori = ['#029386' for _ in range(aux_graph.number_of_nodes())]  #'#029386'
                #     fig, ax = plt.subplots(figsize=(10,10))
                #     nx.draw(aux_graph,pos, node_color=colori,node_size=800,font_size=14)
                #     nx.draw_networkx_labels(aux_graph,pos,font_family="sans-serif",font_size=14)
                #     ax.set_axis_off()
                #     fig.tight_layout()
                #     plt.show()
                raise Exception('correctness exception')
class single_source_top_k():

    def __init__(self, grph, num_k, r):
        
        self.graph = grph
        self.kappa = num_k
        self.pruned_branches = 0
        
        self.top_k = [deque() for _ in G.iterNodes()]


        # self.predecessors_set = [set() for _ in self.graph.iterNodes()]
        self.pigreco_set = [set() for _ in self.graph.iterNodes()]

        self.ignore_nodes = [False for _ in self.graph.iterNodes()]
        self.queue_ignore_nodes = deque()
        
        self.locally_ignore_nodes = [False for _ in self.graph.iterNodes()]
        self.locally_ignore_edges = [False for _ in self.graph.iterEdges()]
        self.locally_queue_ignore_nodes = deque()
        self.locally_queue_ignore_edges = deque()
        
        self.queue_dist_profile = [deque() for _ in G.iterNodes()]
        self.bound = [sys.maxsize for _ in self.graph.iterNodes()]

        self.visited = [False for _ in self.graph.iterNodes()]
        self.queue_visited = deque()


        self.non_sat = [True for v in self.graph.iterNodes()]
        self.num_non_sat = self.graph.numberOfNodes()-1

        self.root = r
        self.non_sat[r] = False
        self.pruned_branches = 0
        self.extra_visits = 0
        
        self.distance_profile = deque()
        self.detours = []
        
   
        
    
    def binary_search(self, arr, low, high, x):
 
        index = bisect.bisect_left(arr, x[0],key=lambda z:z[0])
        
        while index<len(arr) and arr[index][0]==x[0]:
            if arr[index]==x:
                assert x in arr
                return True
            index+=1
        
        assert x not in arr

        return False
    
    def binary_search_alt(self, arr, low, high, x):
 
        index = bisect.bisect_left(arr,len(x),key=len)
        
        while index<len(arr) and len(arr[index])==len(x):
            if arr[index]==x:
                assert x in arr
                return True
            index+=1
        
        assert x not in arr

        return False
    
    def deallocation(self):

        self.PQ.clear()
        del self.PQ
        del self.pigreco_set[:]
        del self.pigreco_set
        del self.ignore_nodes[:]
        del self.ignore_nodes
        del self.locally_ignore_nodes[:]
        del self.locally_ignore_nodes
        del self.locally_ignore_edges[:]
        del self.locally_ignore_edges

        self.locally_queue_ignore_nodes.clear()
        self.locally_queue_ignore_edges.clear()
        del self.queue_dist_profile[:]
        del self.queue_dist_profile
        del self.bound[:]
        del self.bound
        
        del self.visited[:]
        del self.visited

        self.queue_visited.clear()

        del self.non_sat[:]
        del self.non_sat


        
        self.distance_profile.clear()
        
    def generalized_bfs(self):
        

        

        assert type(self.top_k)==list and all(len(self.top_k[v])==0 for v in self.graph.iterNodes())
        assert type(self.non_sat)==list
        
        self.PQ = deque()
        assert not self.non_sat[self.root]

        
        for ngx in self.graph.iterNeighbors(self.root):
            tpl =  (1,ngx,tuple([self.root,ngx]),{self.root,ngx},PacketType.NORMAL) 
            assert tpl not in self.PQ
            self.PQ.append(tpl)
            assert list(self.PQ)==sorted(self.PQ,key=lambda x:x[0])
            assert self.non_sat[ngx]
            self.queue_dist_profile[ngx].append(1)
            assert sorted(self.queue_dist_profile[ngx])==list(self.queue_dist_profile[ngx])    

        while len(self.PQ)>0:
            
            wgt, vtx, ptx, setptx_or_route, tipologia = self.PQ.popleft()

            assert type(ptx)==tuple
            assert all(v in self.graph.iterNeighbors(u) for u, v in pairwise(ptx))
            assert wgt==len(ptx)-1
            assert vtx!=self.root
            assert ptx[0]==self.root and ptx[-1]==vtx
            assert self.num_non_sat>0
            
            assert tipologia in [PacketType.NORMAL,PacketType.TARGETED]

            if tipologia == PacketType.NORMAL:
            
                assert type(setptx_or_route)==set
                assert len(ptx)==len(setptx_or_route)
    
                if len(self.top_k[vtx]) < self.kappa:

                    assert self.non_sat[vtx]
                    assert wgt == self.queue_dist_profile[vtx][0]
                    self.queue_dist_profile[vtx].popleft()
                    
                    if self.binary_search_alt(self.top_k[vtx],0,len(self.top_k[vtx])-1,ptx):
                        del ptx
                        del setptx_or_route
                        continue
                    
                    self.standard(wgt,vtx,ptx,setptx_or_route)    

                    del ptx
                    del setptx_or_route
                    if self.num_non_sat==0:
                        return
                    continue
                else:
                    assert not self.non_sat[vtx]
                    assert len(self.queue_dist_profile[vtx])==0
                    assert self.num_non_sat>0
                    if self.binary_search_alt(self.top_k[vtx],0,len(self.top_k[vtx])-1,ptx):
                        del ptx
                        del setptx_or_route
                        continue
                    
                    
                    self.beyond(wgt,vtx,ptx,setptx_or_route)
                    assert self.num_non_sat>0
                    del ptx
                    del setptx_or_route
                    continue
            else:

                assert type(setptx_or_route)==tuple
                
                target = setptx_or_route[-1]
    
                if target == vtx:
                    
                    assert len(setptx_or_route)==1            
                    
                    if len(self.top_k[vtx]) < self.kappa:
                        if self.binary_search_alt(self.top_k[vtx],0,len(self.top_k[vtx])-1,ptx):
                            continue
                        
                        assert not self.binary_search_alt(self.top_k[vtx],0,len(self.top_k[vtx])-1,ptx)
                        
                        self.standard(wgt,vtx,ptx,set(ptx))    
                        del ptx
                        del setptx_or_route         
                        if self.num_non_sat==0:

                            return
                        continue
                    else:
                        assert not self.non_sat[vtx]
                        assert ptx in self.top_k[vtx] or wgt>=len(self.top_k[vtx][-1])-1
                        del ptx
                        del setptx_or_route
                        continue
                else:
                    if not self.non_sat[target]:# or self.non_sat[vtx]:
                        del ptx
                        del setptx_or_route
                        continue
                    if self.binary_search_alt(self.top_k[vtx],0,len(self.top_k[vtx])-1,ptx):
                        del ptx
                        del setptx_or_route
                        continue
                    assert len(setptx_or_route)>1
                    assert self.is_simple(ptx+(setptx_or_route[1],))

                    assert ptx[-1]==vtx
                    
                    assert len(ptx)>=2
                        
                    tpl = (wgt+1,setptx_or_route[1],ptx+(setptx_or_route[1],),setptx_or_route[1:],PacketType.TARGETED)
                    
                    assert list(self.PQ)==sorted(self.PQ,key=lambda x:x[0])
                                             
                    self.PQ.append(tpl)                      
                    assert self.binary_search(self.PQ,0,len(self.PQ)-1,tpl)

                    del ptx
                    del setptx_or_route
                    continue
                    
                
               
                

    
                
                
    def standard(self,WEIG,VERT,PATH,PATHSET):
        
        assert len(self.top_k[VERT])<self.kappa
        assert PATH not in self.top_k[VERT]    
        self.top_k[VERT].append(PATH)
        
        assert list(self.top_k[VERT])==sorted(self.top_k[VERT],key=len)    
        assert len(self.top_k[VERT])<=self.kappa
        
        if len(self.top_k[VERT])==self.kappa:  
            assert self.non_sat[VERT]
            self.non_sat[VERT] = False
            self.num_non_sat -= 1
            assert(len([val for val in self.non_sat if val==True])==self.num_non_sat)
            self.queue_dist_profile[VERT].clear()
            if self.num_non_sat==0:
                return

        assert PATH[-2] in self.graph.iterNeighbors(VERT)
        assert(type(self.pigreco_set[VERT]==set))
        for element in PATH:
            if element!=self.root and element!=VERT and self.non_sat[element]:
                self.pigreco_set[VERT].add(element)

    
        
        assert self.num_non_sat>0
        for ngx in self.graph.iterNeighbors(VERT):
            if ngx in PATHSET:
                continue
            assert  ngx != self.root
            tpl = (WEIG+1,ngx,PATH+(ngx,),PATHSET.union({ngx}),PacketType.NORMAL) 
            assert tpl not in self.PQ
            self.PQ.append(tpl)
            assert list(self.PQ)==sorted(self.PQ,key=lambda x:x[0])
            if self.non_sat[ngx]:
                self.queue_dist_profile[ngx].append(len(PATH)) 
            assert sorted(self.queue_dist_profile[ngx])==list(self.queue_dist_profile[ngx])    

    def beyond(self,WEIG,VERT,PATH,PATHSET):
        
        self.extra_visits += 1
        assert self.num_non_sat>0
        assert not self.non_sat[VERT] and len(self.top_k[VERT])>=self.kappa and PATH not in self.top_k[VERT] and type(PATH) == tuple
        assert self.root in PATHSET
        assert VERT in PATHSET
        assert self.visited[VERT]==False
        assert not self.visited[self.root]

        prev_size = len(self.PQ)
        num_tested = 0
        merge_set = set()
        
        for vr in self.pigreco_set[VERT]:
            
            """ si è saturato nel frattempo, mi prendo merge dei due set pigreco """
            if not self.non_sat[vr]: 
                merge_set.update(self.pigreco_set[vr])
                assert vr not in self.pigreco_set[vr]
          
        assert self.root not in merge_set
        self.pigreco_set[VERT].update(merge_set)
        self.pigreco_set[VERT] = set(filter(lambda x: self.non_sat[x], self.pigreco_set[VERT]))
        assert VERT not in self.pigreco_set[VERT]
        assert self.root not in self.pigreco_set[VERT]
        assert all(self.non_sat[x] for x in self.pigreco_set[VERT])

    
        if len(self.pigreco_set[VERT])>0:

            num_tested = 0
            assert self.pigreco_set[VERT] is not None
                
            self.init_avoidance(PATH)    
            
            for vr in self.pigreco_set[VERT]:    
                assert self.non_sat[vr]
    
                assert list(self.queue_dist_profile[vr])==sorted(self.queue_dist_profile[vr])
                
                max_to_generate = self.kappa-len(self.top_k[vr])
                assert max_to_generate>0      
    
                n_generated = 0  
                num_tested+=1
    
     
                
                for DET in self.find_detours(VERT,vr,max_to_generate,len(PATH)-1):
                    if DET is None:
                        break
                    
                    n_generated+=1
                    
    
                    assert DET[1] in self.graph.iterNeighbors(VERT)
                    assert DET[1] not in PATHSET
                    assert self.is_simple(PATH+DET[1:])
                    assert len(PATH+DET[1:])-1==len(PATH)-1+len(DET)-1
                    
                    tpl = (WEIG+1,DET[1],PATH+(DET[1],),DET[1:],PacketType.TARGETED)
                    self.PQ.append(tpl)  
                    del DET
                    if n_generated==max_to_generate:
                        self.distance_profile.clear()
                        self.detours.clear()
                        break
    
                if num_tested==self.num_non_sat:
                    break

            
            assert self.num_non_sat>0
            self.clean_avoidance()
 
       
        if len(self.PQ)-prev_size==0:
            self.pruned_branches += 1
        

                                       
    def init_avoidance(self,avoid):
        
        lungh = 0
        assert avoid[0]==self.root        
        while lungh<len(avoid)-1:
            u = avoid[lungh]           
            assert not self.ignore_nodes[u]
            self.ignore_nodes[u]=True
            self.queue_ignore_nodes.append(u)            
            lungh+=1
            
    def clean_avoidance(self):
        
        while len(self.queue_ignore_nodes)>0:
            x = self.queue_ignore_nodes[0]
            assert self.ignore_nodes[x]
            self.ignore_nodes[x]=False
            self.queue_ignore_nodes.popleft()
            
    def find_detours(self,source,target,at_most_to_be_found,dist_path):
          
        assert len(self.top_k[target])>=1
        assert len(self.top_k[target])<self.kappa
        assert(sorted(self.top_k[target],key=len)==list(self.top_k[target]))
        assert len(self.top_k[source])==self.kappa
        

        yen_PQ = []
        paths = set()

            
        assert not self.ignore_nodes[source]
        
        if self.ignore_nodes[target]:
            assert len(self.distance_profile)==0
            yield None
        
        self.bound[target]=sys.maxsize
        assert len(self.distance_profile)==0
        assert len(self.detours)==0
        for i in self.queue_dist_profile[target]:
            self.distance_profile.append(i)
            if len(self.distance_profile)>=self.kappa:
                break

        
        
        assert sorted(self.distance_profile)==list(self.distance_profile)

        if len(self.distance_profile)>=at_most_to_be_found:
            self.bound[target] = min(self.bound[target],self.distance_profile[at_most_to_be_found-1])
        

            
        try:
            alt_p = self.bidir_BFS(source,target,self.bound[target]-dist_path)
            assert alt_p[0]==source
            assert alt_p[-1]==target           
            assert len(alt_p)-1+dist_path<self.bound[target]
            assert alt_p not in paths
            heappush(yen_PQ, (len(alt_p)-1,alt_p))
            paths.add(alt_p)
            
            
        except NoPathException:
            self.distance_profile.clear()
            self.detours.clear()     
            yield None
        
        
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
            

        while len(yen_PQ)>0:

            _,P_det = heappop(yen_PQ)
            
            
            if len(self.distance_profile)>=at_most_to_be_found and len(P_det)-1+dist_path>=self.distance_profile[at_most_to_be_found-1]:
                del yen_PQ[:]
                del yen_PQ
                del P_det
                self.distance_profile.clear()
                self.detours.clear()
                yield None

            
            assert len(P_det)-1+dist_path<self.bound[target]            
            bisect.insort(self.distance_profile, len(P_det)-1+dist_path)   
            assert sorted(self.distance_profile)==list(self.distance_profile)

            assert P_det[0]==source
            assert P_det[-1]==target
            yield P_det
            
            self.detours.append(P_det)
            assert len(self.detours)<at_most_to_be_found
            assert sorted(self.detours,key=len)==self.detours
            
            if len(self.distance_profile)>=at_most_to_be_found:
                self.bound[target] = min(self.bound[target],self.distance_profile[at_most_to_be_found-1])
                

            assert len(P_det)-1+dist_path<=self.bound[target]
            
            if len(P_det)-1+dist_path==self.bound[target]:
                del yen_PQ[:]
                del yen_PQ
                del P_det
                self.distance_profile.clear()
                self.detours.clear()
                yield None
                

            assert all(not self.locally_ignore_nodes[v] for v in self.graph.iterNodes())
            assert all(not self.locally_ignore_edges[self.graph.edgeId(u, v)] for u,v in self.graph.iterEdges())            

            for index in range(1, len(P_det)):
                
                l_root = P_det[:index]
                assert type(l_root)==tuple
                
                for path in self.detours:
                    assert type(path)==tuple
                    if path[:index] == l_root:
                        eid = self.graph.edgeId(path[index - 1], path[index])
                        if not self.locally_ignore_edges[eid]:
                            self.locally_ignore_edges[eid]=True
                            self.locally_queue_ignore_edges.append(eid)
                                          
                        
                try:
                    
                    pgt = self.bidir_BFS(l_root[-1],target,self.bound[target]-dist_path)
                    
                    
                    assert len(l_root[:-1] + pgt)-1==len(pgt)-1+len(l_root[:-1])               
                    assert len(pgt)-1+len(l_root[:-1])+dist_path<self.bound[target]
                    
                    new_path = l_root[:-1] + pgt
                   
                    assert len(new_path)-1<self.bound[target]-dist_path
                    assert new_path[0]==source and new_path[-1]==target
                    assert self.is_simple(new_path)   

                    if new_path in paths:
                        if not self.locally_ignore_nodes[l_root[-1]]:
                            self.locally_ignore_nodes[l_root[-1]]=True
                            self.locally_queue_ignore_nodes.append(l_root[-1])
                        del new_path
                        continue
                        
                    heappush(yen_PQ, (len(new_path)-1,new_path))
                    paths.add(new_path)
    
                except NoPathException:
                    pass
                
                except Exception as err:
                    print(f"Unexpected {err=}, {type(err)=}")
                    raise
                    
                if not self.locally_ignore_nodes[l_root[-1]]:
                    self.locally_ignore_nodes[l_root[-1]]=True
                    self.locally_queue_ignore_nodes.append(l_root[-1])
            
            while len(self.locally_queue_ignore_nodes)>0:
                x = self.locally_queue_ignore_nodes[0]
                assert self.locally_ignore_nodes[x]
                self.locally_ignore_nodes[x]=False
                self.locally_queue_ignore_nodes.popleft()
                
            while len(self.locally_queue_ignore_edges)>0:
                e_id = self.locally_queue_ignore_edges[0]
                assert self.locally_ignore_edges[e_id]
                self.locally_ignore_edges[e_id]=False
                self.locally_queue_ignore_edges.popleft()
                
        self.distance_profile.clear()
        self.detours.clear()
        paths.clear()
                
                
        
   
    def is_simple(self,path):
            
        assert len(path) > 0
        assert type(path)==tuple
        assert all(v in self.graph.iterNeighbors(u) for u, v in pairwise(path))

        s_path = set()
        for el in path:
            if el in s_path:
                assert len(set(path)) < len(path)
                return False
            s_path.add(el)
        assert len(set(path)) == len(path)
        return True
       
    
    
    def bidir_BFS(self, source, target, bound_on_length):
    
    
        pred, succ, w = self.bidir_pred_succ(source, target, bound_on_length)
        
        path = deque()
        # d = -1
    
        while w is not None:
            path.append(w)
            w = succ[w]
            # d+=1
    
    
        w = pred[path[0]]
        while w is not None:
            path.appendleft(w)
            w = pred[w]
            # d+=1
    
        # assert d==len(path)-1
        del pred
        del succ
        assert len(path)-1<bound_on_length
        return tuple(path)
                
    def bidir_pred_succ(self, source, target, bound_on_length):
        

    
        if self.ignore_nodes[source] or self.ignore_nodes[target] or self.locally_ignore_nodes[source] or self.locally_ignore_nodes[target]:
            raise NoPathException
        if target == source:
            return ({target: None}, {source: None}, source)
    
    
        
        # predecesssor and successors in search
        pred = {source: None}
        succ = {target: None}
    
        # initialize fringes, start with forward
        forward_fringe = deque([source])
        reverse_fringe = deque([target])
        
        lunghezza=0
        
        while forward_fringe and reverse_fringe:
            lunghezza+=1
            if lunghezza>=bound_on_length:
                raise NoPathException

            if len(forward_fringe) <= len(reverse_fringe):
                this_level = forward_fringe
                forward_fringe = deque()
                for v in this_level:
                    for w in self.graph.iterNeighbors(v):
                        assert self.graph.edgeId(v,w) == self.graph.edgeId(w,v)
                        if self.locally_ignore_edges[self.graph.edgeId(v,w)]:
                            continue
                        if w==source or self.ignore_nodes[w] or self.locally_ignore_nodes[w]:
                            assert w not in succ
                            continue
                        if w not in pred:
                            forward_fringe.append(w)
                            pred[w] = v
                        if w in succ:
                            del forward_fringe
                            del reverse_fringe
                            return pred, succ, w
            else:
                this_level = reverse_fringe
                reverse_fringe = deque()
                for v in this_level:
                    for w in self.graph.iterNeighbors(v):
                        assert self.graph.edgeId(v,w) == self.graph.edgeId(w,v)
                        if self.locally_ignore_edges[self.graph.edgeId(v,w)]:
                            continue
                        if w==target or self.ignore_nodes[w] or self.locally_ignore_nodes[w]:
                            assert w not in pred
                            continue
                        if w not in succ:
                            succ[w] = v
                            reverse_fringe.append(w)
                        if w in pred:
                            del forward_fringe
                            del reverse_fringe
                            return pred, succ, w
        del forward_fringe
        del reverse_fringe
        
        raise NoPathException(f"No path between {source} and {target}.")
        
            


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()

    parser.add_argument('--g',metavar="GRAPH", required=True,  help='Path to the graph file (.hist or .nde format)')
    parser.add_argument('--k', metavar="K_VALUE", required=True, type=int, help='Number of top shortest paths to seek for', default=2)
    parser.add_argument('--r', metavar="NUM_ROOTS", required=False, type=int, help='Number of roots to consider (Default sqrt n)', default=-1)

    args = parser.parse_args()


    
    if ".hist" in str(args.g):
        G_prime_prime = hist2nk(str(args.g))
    elif ".nde" in str(args.g):
        G_prime_prime = read_nde(str(args.g))
    else:
        raise Exception('unknown graph format')

    
    num_roots = int(args.r)
    

    G_prime=graphtools.toUndirected(G_prime_prime)
    G = graphtools.toUnweighted(G_prime)    
    G.removeMultiEdges()
    G.removeSelfLoops()
    G.indexEdges()
    

    cc = nk.components.ConnectedComponents(G)
    cc.run()
    G = cc.extractLargestConnectedComponent(G, True)
    G.indexEdges()
    print("vertices:",G.numberOfNodes(),"arcs:",G.numberOfEdges())
 



    if not __debug__:
        print("AFTER SCC")
        nk.overview(G)
        
    K = int(args.k)
    if K<1:
        K=2    
    print("Value of K:",K)
    
    diam = nk.distance.Diameter(G,algo=1)
    diam.run()
    diametro = diam.getDiameter()[0]  
    print("DIAMETER:",diametro)
    del diam

    now = datetime.now() # current date and time
    date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    import os

    statsfile = str(os.path.basename(args.g))+"_"+str(K)+"_kbfs_orient_"+date_time+'.csv'    


    cpu_yen  = []
    cpu_kbfs = []
    speedups = []
    pruned_branches = []
    extra_visits = []
    fraction_of_pruned = []
    randomly_selected_sqrt_roots = set()
    
    rootsfile = str(os.path.basename(args.g))+"_"+str(K)+"_roots.csv"


    if os.path.isfile(rootsfile):
        # Open file 
        with open(rootsfile) as csvfile: 
              
            # Skips the heading 
            # Using next() method 
            # heading = next(csvfile) 
              
            # Create reader object by passing the file  
            # object to reader method 
            reader_obj = csv.DictReader(csvfile,delimiter=";")
              
            # Iterate over each row in the csv file  
            # using reader object 
            for row in reader_obj: 
                
                
                cnt = 0
                splittaggio = row['roots'][1:-1].split(',')
                l = set()
                while cnt<len(splittaggio):
                    randomly_selected_sqrt_roots.add(int(row['roots'][1:-1].split(',')[cnt]))
                    cnt+=1
                break
                
            

                # randomly_selected_sqrt_roots=set(list(row[2]))
                # print(randomly_selected_sqrt_roots)
    else:
        if num_roots == -1:
            
            num_roots = round(math.sqrt(G.numberOfNodes()))
            if G.numberOfNodes()>=20000:
                num_roots = 30
            if G.numberOfNodes()>=1000000:
                num_roots = 5
        if num_roots < 1:
            num_roots = 1
            
        if num_roots > G.numberOfNodes():
            num_roots = round(math.sqrt(G.numberOfNodes()))
            if G.numberOfNodes()>=30000:
                num_roots = 30                  
            if G.numberOfNodes()>=1000000:
                num_roots = 5
        
        while len(randomly_selected_sqrt_roots)<num_roots:
            randomly_selected_sqrt_roots.add(graphtools.randomNode(G))
            
        with open(rootsfile, 'w', newline='', encoding='UTF-8') as csvfile:
            writer = csv.writer(csvfile,delimiter=";")
            writer.writerow(["graph_name",\
                             "k",\
                             "roots"])
            writer.writerow([str(os.path.basename(args.g)),\
                             str(K),\
                             list(randomly_selected_sqrt_roots)])
    print("N_ROOTS:",len(randomly_selected_sqrt_roots))
    # num_roots = 1
    # randomly_selected_sqrt_roots = {61} #problematic root oregon 4,2
    # randomly_selected_sqrt_roots = {100,4197,4440,4849,6333,7069,8859,9257,10475,10813} 
    # randomly_selected_sqrt_roots = {47,9009} 


    bar = PixelBar('Yen vs BFS:', max = len(randomly_selected_sqrt_roots))
    aux_graph = nk.nxadapter.nk2nx(G)

    if __debug__ and G.numberOfNodes()<100:
    
        pos = nx.kamada_kawai_layout(aux_graph)    
        colori = ['#029386' for _ in range(aux_graph.number_of_nodes())]  #'#029386'
        fig, ax = plt.subplots(figsize=(10,10))
        nx.draw(aux_graph,pos, node_color=colori,node_size=800,font_size=14)
        nx.draw_networkx_labels(aux_graph,pos,font_family="sans-serif",font_size=14)
        ax.set_axis_off()
        fig.tight_layout()
        plt.show()
        
    randomly_selected_sqrt_roots = sorted(list(randomly_selected_sqrt_roots))    
    
    
    for first_root in randomly_selected_sqrt_roots:
        
        
       
        print("Performing genBFS for root:",first_root,"...",end="",flush=True)
        SSTK = single_source_top_k(G,K,first_root) 
        local_cpu_bfs = time.perf_counter()        
        SSTK.generalized_bfs()
        local_cpu_bfs = time.perf_counter()-local_cpu_bfs
        SSTK.deallocation()
        
        cpu_kbfs.append(local_cpu_bfs)
        pruned_branches.append(SSTK.pruned_branches)
        extra_visits.append(SSTK.extra_visits)
        if SSTK.extra_visits>0:
            fraction_of_pruned.append(round(SSTK.pruned_branches/SSTK.extra_visits,2))
        else:
            fraction_of_pruned.append(1.0) 
        
        print(" done in:", round(local_cpu_bfs,2), "seconds", flush=True)
        nested_bar = IncrementalBar('Yen Iterations:', max = G.numberOfNodes())


        local_cpu_yen = 0
        
        YENTK = yens_algorithm(G,K)
        
        for second in range(G.numberOfNodes()):
            nested_bar.next()

            if first_root==second:
                continue
            
            cpu_yen_single_pair = time.perf_counter()
            
            top_k_yen = YENTK.run(first_root, second)
            
            local_cpu_yen += (time.perf_counter()-cpu_yen_single_pair)
            
            assert all(type(i)==tuple for i in top_k_yen)
            
            test_equality("BY",SSTK.top_k[second],top_k_yen)
            if __debug__:
                test_equality("YY",top_k_yen, [tuple(i) for i in islice(nx.shortest_simple_paths(aux_graph, first_root, second, weight=None), K)])


            for i in top_k_yen:
                del i
            del top_k_yen
            
        cpu_yen.append(local_cpu_yen)
        
        speedups.append(local_cpu_yen/local_cpu_bfs)
        
        for i in G.iterNodes():
            for j in SSTK.top_k[i]:
                del j
                
        nested_bar.finish()
 
        bar.next()

        print("\nTotal KBFS CPUTime:", round(local_cpu_bfs,2),"Total Yen CPUTime:", round(local_cpu_yen,2),"Speedup:",round(local_cpu_yen/local_cpu_bfs,2), "Extra-visits:", SSTK.extra_visits,"of which pruned:", SSTK.pruned_branches, flush=True)

        del SSTK
        del YENTK



            

        

    bar.finish()

    assert len(cpu_yen)==len(cpu_kbfs)
    print("Total CPUTime Yen", round(sum(cpu_yen),2), "s")
    print("Total CPUTime KBFS", round(sum(cpu_kbfs),2), "s")
    print("Avg CPUTime Yen", round(statistics.mean(cpu_yen),2), "s")
    print("Avg CPUTime KBFS", round(statistics.mean(cpu_kbfs),2), "s")
    print("Avg Speedup", round(statistics.mean(speedups),2))
    print("Med Speedup", round(statistics.median(speedups),2))
    print("Avg Fraction Pruned", round(statistics.mean(fraction_of_pruned),2))



    assert len(cpu_kbfs)==len(cpu_yen)
    assert len(cpu_kbfs)==len(extra_visits)
    assert len(cpu_kbfs)==len(pruned_branches)
    assert len(cpu_kbfs)==len(speedups)


    assert len(cpu_kbfs)==len(randomly_selected_sqrt_roots)

    with open(statsfile, 'a', newline='', encoding='UTF-8') as csvfile:
        
        writer = csv.writer(csvfile)
        
        finish_time = now.strftime("%d_%m_%Y_%H_%M_%S")


        writer.writerow(["graph_name",\
                         "date",\
                         "vertices",\
                         "arcs",\
                         "k",\
                         "diameter",\
                         "root",\
                         "yen_time",\
                         "kbfs_time",\
                         "speedup",\
                         "extra_visits",\
                         "pruned_visits",\
                         "fraction"])
        
        for s in range(len(cpu_kbfs)):
            writer.writerow([str(args.g),\
                             str(finish_time),\
                             G.numberOfNodes(),\
                             G.numberOfEdges(),\
                             str(K),\
                             diametro,\
                             randomly_selected_sqrt_roots[s],\
                             round(cpu_yen[s],2),\
                             round(cpu_kbfs[s],2),\
                             round(speedups[s],2),\
                             round(extra_visits[s],0),\
                             round(pruned_branches[s],0),\
                             fraction_of_pruned[s]])
            
        writer.writerow(["graph_name",\
                         "date",\
                         "vertices",\
                         "arcs",\
                         "k",\
                         "diameter",\
                         "n_roots",\
                         "tot_yen_time",\
                         "tot_kbfs_time",\
                         "avg_yen_time",\
                         "avg_kbfs_time",\
                         "med_yen_time",\
                         "med_kbfs_time",\
                         "avg_fraction",\
                         "avg_speedup",\
                         "med_speedup"])

        
        writer.writerow([str(args.g),\
                         str(finish_time),\
                         G.numberOfNodes(),\
                         G.numberOfEdges(),\
                         str(K),\
                         diametro,\
                         len(randomly_selected_sqrt_roots),\
                         round(sum(cpu_yen),2),\
                         round(sum(cpu_kbfs),2),\
                         round(statistics.mean(cpu_yen),2),\
                         round(statistics.mean(cpu_kbfs),2),\
                         round(statistics.median(cpu_yen),2),\
                         round(statistics.median(cpu_kbfs),2),\
                         round(statistics.mean(fraction_of_pruned),2),\
                         round(statistics.mean(speedups),2),\
                         round(statistics.median(speedups),2)])

            


