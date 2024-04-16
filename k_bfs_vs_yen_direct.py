#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:28:41 2023

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
    
    def yen_pred(self,source,target,bound_on_length):
        if self.skip_node[source] or self.skip_node[target]:
          raise NoPathException
        if target == source:
            return {source:None}
        pred={source:None}
       
        fringe=deque([source])
       
     
        lunghezza=0
        while fringe:
            lunghezza+=1
            
            if lunghezza>=bound_on_length:
                raise NoPathException
            this_level=fringe
            fringe=deque()
            while this_level:
                vtx=this_level.popleft()
                for i in self.G.iterNeighbors(vtx):
                    if self.skip_edge[self.G.edgeId(vtx,i)]:
                        continue
                    if i in pred or self.skip_node[i]:
                        
                        continue
                    if i==target:
                       del fringe
                       pred[i]=vtx
                       return pred
                     
                    pred[i]=vtx               
                    fringe.append(i)
       
        del fringe
        
        raise NoPathException(f"No path between {source} and {target}.")
    def yen_BFS(self,source,target,bound_on_length):
        pred=self.yen_pred(source, target, bound_on_length)
        path=deque()
        w=target
        while w is not None:
            path.appendleft(w)
            w=pred[w]
        del pred
        assert len(path)-1<bound_on_length
        return tuple(path)
    
        
        
    def run(self, source, target):
        
        results = []
        
        yen_PQ = []
        paths = set()
        
        assert all(not self.skip_node[v] for v in self.G.iterNodes())
        assert all(not self.skip_edge[self.G.edgeId(u, v)] for u,v in self.G.iterEdges())          
            
        
        try:
            p = self.yen_BFS(source,target,sys.maxsize)
            assert p[0]==source
            assert p[-1]==target
            # assert w == len(p)-1
            
            assert p not in paths
            heappush(yen_PQ, (len(p)-1,p))
            paths.add(p)
            
            
        except NoPathException:
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
                del yen_PQ
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
                    
                    pgt = self.yen_BFS(l_root[-1],target,sys.maxsize)
                    
                    
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
        # print("bfs ",array_bfs,"\n")
        # print("yen ",array_yen,"\n")
       
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
        self.detour_done=[False for _ in G.iterNodes()]
        self.path_to_add=[[] for _ in G.iterNodes()]
        self.predecessors_set = [set() for _ in self.graph.iterNodes()]
        self.pigreco_set = [None for _ in self.graph.iterNodes()]
        self.last_det_path=[0 for _ in G.iterNodes()]
        self.ignore_nodes = [False for _ in self.graph.iterNodes()]
        self.queue_ignore_nodes = deque()
        
        
        self.locally_ignore_nodes = [False for _ in self.graph.iterNodes()]
        self.locally_ignore_edges = [False for _ in self.graph.iterEdges()]
        self.locally_queue_ignore_nodes = deque()
        self.locally_queue_ignore_edges = deque()
        
        self.queue_dist_profile = [deque() for _ in G.iterNodes()]

        #self.bound = [sys.maxsize for _ in self.graph.iterNodes()]

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

        
    
        
    def generalized_bfs(self):
        

        

        assert type(self.top_k)==list and all(len(self.top_k[v])==0 for v in self.graph.iterNodes())
        

        assert type(self.non_sat)==list
        
        
        self.PQ = deque()
        assert not self.non_sat[self.root]

        
        
        for ngx in self.graph.iterNeighbors(self.root):
            assert (1,ngx,tuple([self.root,ngx]),{self.root,ngx}) not in self.PQ
            self.PQ.append((1,ngx,tuple([self.root,ngx]),{self.root,ngx},"reg",None,None))
            assert list(self.PQ)==sorted(self.PQ,key=lambda x:x[0])
            assert self.non_sat[ngx]
            self.queue_dist_profile[ngx].append((self.root,))
            assert sorted(self.queue_dist_profile[ngx])==list(self.queue_dist_profile[ngx])    

        while len(self.PQ)>0:
            
            wgt, vtx, ptx, setptx, flag, source, paths = self.PQ.popleft()
           
                    
           
            assert wgt==len(ptx)-1

            assert ptx[0]==self.root and ptx[-1]==vtx
            assert vtx!=self.root
            assert type(ptx)==tuple
            assert type(setptx)==set

            assert len(ptx)==len(setptx)
            assert all(v in self.graph.iterNeighbors(u) for u, v in pairwise(ptx))

            assert self.num_non_sat>0
            assert ptx not in self.top_k[vtx]
          
        

            if len(self.top_k[vtx]) < self.kappa:
                assert self.non_sat[vtx]
                
                # assert wgt == len(self.queue_dist_profile[vtx][0])
                # self.queue_dist_profile[vtx].popleft()
                
                self.standard(wgt,vtx,ptx,setptx,flag,source,paths)    
                del ptx
                del setptx
                
                if self.num_non_sat==0:
                    del self.PQ
                    return
            else:
                if(flag=="spec" and vtx in self.predecessors_set[source] and True):
                    neighbors = set()             
                 
                    
                    for i in paths:
                            if(len(i)==0) or self.non_sat[i[-1]]==False:
                               continue
                            if i[0] not in neighbors:
                                neighbors.add(i[0])
                                self.path_to_add[i[0]].append(i[1:])
                            else:
                                self.path_to_add[i[0]].append(i[1:])
                            
                    for ngx in neighbors:
                       
                        assert ngx not in setptx
                        assert ngx not in ptx

                        assert ngx in self.graph.iterNeighbors(vtx)
                        assert ptx[-1]==vtx
                        assert (wgt+1,ngx,ptx+(ngx,),setptx.union({ngx})) not in self.PQ
                        self.PQ.append((wgt+1,ngx,ptx+(ngx,),setptx.union({ngx}),"spec",source,self.path_to_add[ngx]))  
                        self.path_to_add[ngx]=[]
                        assert list(self.PQ)==sorted(self.PQ,key=lambda x:x[0])
                        if self.non_sat[ngx]:
                            
                            if(self.detour_done[ngx] ):
                                self.binary_search_alt(self.queue_dist_profile[ngx], 0, len(self.queue_dist_profile[ngx])-1, ptx)
                                if(self.last_det_path[ngx]<len(ptx)):
                                    self.detour_done[ngx]=False
                                    self.last_det_path[ngx]=0
                            else:
                                if(len(self.queue_dist_profile[ngx])<self.kappa):
                                    self.queue_dist_profile[ngx].append(ptx)
                                assert all (len(self.queue_dist_profile[ngx][i])==len(sorted(self.queue_dist_profile[ngx],key=len)[i])for i in range(len(self.queue_dist_profile[ngx])))
                       # assert sorted(self.queue_dist_profile[ngx],key=lambda x:x[0])==list(self.queue_dist_profile[ngx])  
                    del ptx
                    del setptx  
                else:            
                    assert not self.non_sat[vtx]
                    assert len(self.queue_dist_profile[vtx])==0
                    if(self.pigreco_set[vtx]==None or len(self.pigreco_set[vtx])>0):
                        self.beyond(wgt,vtx,ptx,setptx,flag,paths)
                    del ptx
                    del setptx
                
    def binary_search_alt(self, arr, low, high, x):
       
       
        index = bisect.bisect_left(arr,len(x),key=len)
        
        while index<len(arr) and len(arr[index])==len(x):
            if arr[index]==x:
                assert x in arr
              
                return True
            index+=1
        
        assert x not in arr
       
        arr.insert(index, x)
        if(len(arr)>self.kappa):
            arr.pop()
      
        assert len(arr)<= self.kappa    
        assert all (len(arr[i])==len(sorted(arr,key=len)[i])for i in range(len(arr)))
        return False
    
    
    def count(self,vr):
      
        wg=len(self.top_k[vr][-1])-1
        pos=len(self.top_k[vr])
        count=0
        while pos<len(self.queue_dist_profile[vr]):
            if wg==len(self.queue_dist_profile[vr][pos]):
                count+=1
                pos+=1
            else:
                break 
        
        return count    
    def standard(self,WEIG,VERT,PATH,PATHSET,flag,source,paths):
        
        assert len(self.top_k[VERT])<self.kappa
        
    
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
            
            
            # if __debug__ and G.numberOfNodes()<100:                    
            #     pos = nx.kamada_kawai_layout(aux_graph)
            #     colori = ['#029386' for _ in range(aux_graph.number_of_nodes())]  
            #     for v in saturi: 
            #         colori[v] = 'red'
            #     fig, ax = plt.subplots(figsize=(10,10))
            #     nx.draw(aux_graph,pos, node_color=colori,node_size=800,font_size=14)
            #     nx.draw_networkx_labels(aux_graph,pos,font_family="sans-serif",font_size=14)
            #     ax.set_axis_off()
            #     fig.tight_layout()
            #     plt.show()
                            
        #assert PATH[-2] in self.graph.iterNeighbors(VERT)
        assert VERT in self.graph.iterNeighbors(PATH[-2])
        assert(type(self.predecessors_set[VERT]==set))
        
        
        self.predecessors_set[VERT].add(PATH[-2])
        
        assert self.num_non_sat>0
    
        neighbors = set()             
     
        if(flag=="spec"):
            
            for i in paths:
                if(len(i)==0) or self.non_sat[i[-1]]==False:
                   continue
                if i[0] not in neighbors:
                    neighbors.add(i[0])
                    self.path_to_add[i[0]].append(i[1:])
                else:
                    self.path_to_add[i[0]].append(i[1:])
        assert self.num_non_sat>0
        for ngx in self.graph.iterNeighbors(VERT):
            if ngx in PATHSET:
                continue
            assert  ngx != self.root
            assert (WEIG+1,ngx,PATH+(ngx,),PATHSET.union({ngx})) not in self.PQ
            if ngx in neighbors:
                self.PQ.append((WEIG+1,ngx,PATH+(ngx,),PATHSET.union({ngx}),flag,source,self.path_to_add[ngx]))
                self.path_to_add[ngx]=[]
            else:
                self.PQ.append((WEIG+1,ngx,PATH+(ngx,),PATHSET.union({ngx}),"reg",None,None))
            assert list(self.PQ)==sorted(self.PQ,key=lambda x:x[0])
            if self.non_sat[ngx]:
                
                if(self.detour_done[ngx] and not(flag=="reg" and len(PATH)==self.last_det_path[ngx])):
                    self.binary_search_alt(self.queue_dist_profile[ngx], 0, len(self.queue_dist_profile[ngx])-1, PATH)
                
                    if(self.last_det_path[ngx]<len(PATH)):
                        self.last_det_path[ngx]=0
                        self.detour_done[ngx]=False
                  
                else:
                    if(len(self.queue_dist_profile[ngx])<self.kappa):
                        self.queue_dist_profile[ngx].append(PATH)
                    assert all (len(self.queue_dist_profile[ngx][i])==len(sorted(self.queue_dist_profile[ngx],key=len)[i])for i in range(len(self.queue_dist_profile[ngx])))
               # assert sorted(self.queue_dist_profile[ngx],key=lambda x:x[0])==list(self.queue_dist_profile[ngx]) 
          

    def beyond(self,WEIG,VERT,PATH,PATHSET,flag,paths):
       
           
        self.extra_visits += 1
        assert self.num_non_sat>0
        assert not self.non_sat[VERT] and len(self.top_k[VERT])>=self.kappa and PATH not in self.top_k[VERT] and type(PATH) == tuple
        assert self.root in PATHSET
        assert VERT in PATHSET
        assert self.visited[VERT]==False
        assert not self.visited[self.root]
        skip=set()
        neighbors = set()             
     
        if(flag=="spec"):
           
            for i in paths:
                
                if(len(i)==0) or self.non_sat[i[-1]]==False:
                   continue
                if i[0] not in neighbors:
                    neighbors.add(i[0])
                
                skip.add(i[-1])
                self.path_to_add[i[0]].append(i[1:])
        num_tested = 0
        
        if self.pigreco_set[VERT] is not None:
            
            self.init_avoidance(PATH)
            it_index = 0
            
            while it_index < len(self.pigreco_set[VERT]):
                
                vr = self.pigreco_set[VERT][it_index]
              
                if not self.non_sat[vr]:
                    del self.pigreco_set[VERT][it_index]
                    continue
                if vr in skip:
                    it_index+=1
                    continue
                it_index+=1
            
                
                assert self.non_sat[vr]       
                assert sorted(self.queue_dist_profile[vr],key=lambda x:x[0])==list(self.queue_dist_profile[vr]) 
            
                count_value=self.count(vr)
                max_to_generate = self.kappa-(len(self.top_k[vr])+count_value)
              
                if max_to_generate<=0:
                  
                   num_tested+=1
                   continue

                n_generated = 0    
                num_tested+=1

                
                for DET in self.find_detours(VERT,vr,max_to_generate,len(PATH)-1,PATH,count_value):
                    if DET is None:
                        break
                
                    n_generated+=1                 
                        
                    assert DET[1] in self.graph.iterNeighbors(VERT)
                    assert DET[1] not in PATHSET
                    assert not self.ignore_nodes[DET[1]] or DET[1] in neighbors
                    assert self.is_simple(PATH+DET[1:])
                    assert all(i[2]!=PATH+DET[1:] for i in self.PQ)
                    assert len(PATH+DET[1:])-1==len(PATH)-1+len(DET)-1
                    
                  
                    assert len(neighbors)<=self.graph.degree(VERT)-1 
                    
                    if DET[1] not in neighbors:
                        neighbors.add(DET[1])
                        self.path_to_add[DET[1]].append(DET[2:])
                   
                    else:    
                        assert DET[2:] not in self.path_to_add[DET[1]]
                        self.path_to_add[DET[1]].append(DET[2:])
                        assert DET[1] in neighbors
                        assert len(neighbors)<=self.graph.degree(VERT)-1

                        if n_generated==max_to_generate:
                            self.distance_profile.clear()
                            self.detours.clear()
                            break
                        continue

                   
                    
                    if n_generated==max_to_generate:
                        self.distance_profile.clear()
                        self.detours.clear()
                        break
                    
                if num_tested==self.num_non_sat:
                    break
                
              
                
            while it_index < len(self.pigreco_set[VERT]):                
                vr = self.pigreco_set[VERT][it_index]
                if not self.non_sat[vr]:
                    del self.pigreco_set[VERT][it_index]
                    continue
                it_index+=1    
                
            assert self.num_non_sat>0
      
            self.clean_avoidance()
            
        else:
            
            self.pigreco_set[VERT] = deque()
        
            localPQ = deque()          

            
            self.visited[VERT] = True
            self.queue_visited.append(VERT)
            self.visited[self.root] = True
            self.queue_visited.append(self.root)
            
            assert VERT!=self.root
            
            for prd in self.predecessors_set[VERT]:
                if self.visited[prd]:
                    continue   
                
                assert prd!=self.root            
                assert self.visited[prd]==False
                self.visited[prd] = True
                self.queue_visited.append(prd)
                localPQ.append(prd)
    
            if not self.visited[self.root]:
                self.visited[self.root] = True
                self.queue_visited.append(self.root)
                
            
    
            self.init_avoidance(PATH)
            
            while len(localPQ)>0:
                
                vr = localPQ.popleft()
                detour=True
                assert vr != self.root
                assert self.visited[vr] == True
                assert num_tested<self.num_non_sat
    
               
                if self.non_sat[vr]:
                    assert vr not in self.pigreco_set[VERT]
                    self.pigreco_set[VERT].append(vr)
                    assert sorted(self.queue_dist_profile[vr],key=lambda x:x[0])==list(self.queue_dist_profile[vr]) 
                   
                    count_value=self.count(vr)
                    max_to_generate = self.kappa-(len(self.top_k[vr])+count_value)
                  
                    if max_to_generate<=0 or vr in skip:
                       
                        detour=False
                        
                    
                    n_generated = 0        
                    num_tested+=1
                    if(detour):
                        for DET in self.find_detours(VERT,vr,max_to_generate,len(PATH)-1,PATH,count_value):
                            
                            if DET is None:
                                break
                          
                            n_generated+=1
                           
                            assert DET[1] in self.graph.iterNeighbors(VERT)
                            assert DET[1] not in PATHSET
                            assert self.is_simple(PATH+DET[1:])
                            assert all(i[2]!=PATH+DET[1:] for i in self.PQ)
                            assert len(PATH+DET[1:])-1==len(PATH)-1+len(DET)-1
                            
                            if DET[1] not in neighbors:
                                neighbors.add(DET[1])
                                self.path_to_add[DET[1]].append(DET[2:])
                         
                            else:    
                                assert DET [2:]not in self.path_to_add[DET[1]]
                                self.path_to_add[DET[1]].append(DET[2:])
                                assert DET[1] in neighbors
                           
                              
    
                                if n_generated==max_to_generate:
                                    self.distance_profile.clear()
                                    self.detours.clear()
                                    break
                                continue
        
                           
                             
                            """ importante commentare questo break altrimenti rischio di non trovare tutti i pigreco """
                            # if len(neighbors)==self.graph.degree(VERT)-1:
                            #     assert PATH[-1] not in neighbors
                            #     break
                            if n_generated==max_to_generate:
                                self.distance_profile.clear()
                                self.detours.clear()
                                break
                        
                    if num_tested==self.num_non_sat:
                        del localPQ
                        break
                    """ importante commentare questo break altrimenti rischio di non trovare tutti i pigreco """
                    # if len(neighbors)==self.graph.degree(VERT)-1:
                    #     assert PATH[-1] not in neighbors
                    #     break
    
    
                    
                assert num_tested<self.num_non_sat
                assert self.num_non_sat>0
                for prd in self.predecessors_set[vr]:
       
                    if self.visited[prd]:
                        continue    
                    
                    assert prd != self.root
    
                    self.visited[prd] = True
                    self.queue_visited.append(prd)
                    
                    localPQ.append(prd)
    
           
            while len(self.queue_visited)>0:
                x = self.queue_visited[0]
                assert self.visited[x]
                self.visited[x]=False
                self.queue_visited.popleft()        
                
                
            assert self.num_non_sat>0
            self.clean_avoidance()

        if len(neighbors)==0:
         
            self.pruned_branches += 1
            return
        
        for ngx in neighbors:
            
            assert ngx not in PATHSET
            assert ngx not in PATH

            assert ngx in self.graph.iterNeighbors(VERT)
            assert PATH[-1]==VERT
            assert (WEIG+1,ngx,PATH+(ngx,),PATHSET.union({ngx})) not in self.PQ
            self.PQ.append((WEIG+1,ngx,PATH+(ngx,),PATHSET.union({ngx}),"spec",VERT,self.path_to_add[ngx]))   
            self.path_to_add[ngx]=[]
            assert list(self.PQ)==sorted(self.PQ,key=lambda x:x[0])
            if self.non_sat[ngx]:
               
                if(self.detour_done[ngx]):
                    self.binary_search_alt(self.queue_dist_profile[ngx], 0, len(self.queue_dist_profile[ngx])-1, PATH)
                    if(self.last_det_path[ngx]<len(PATH)):
                        self.detour_done[ngx]=False
                        self.last_det_path[ngx]=0
                else:
                    if(len(self.queue_dist_profile[ngx])<self.kappa):
                        self.queue_dist_profile[ngx].append(PATH)
                  
                    assert all (len(self.queue_dist_profile[ngx][i])==len(sorted(self.queue_dist_profile[ngx],key=len)[i])for i in range(len(self.queue_dist_profile[ngx])))
           # assert sorted(self.queue_dist_profile[ngx],key=lambda x:x[0])==list(self.queue_dist_profile[ngx])     
          
                                       
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
            
    def find_detours(self,source,target,at_most_to_be_found,dist_path,coming_path,count_value):
          
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
        
       
        
        assert len(self.distance_profile)==0
        assert len(self.detours)==0

        

        
        
        assert sorted(self.distance_profile)==list(self.distance_profile)

        # if len(self.queue_dist_profile[target])-count_value>=at_most_to_be_found:
         
        #     self.bound[target] = min(self.bound[target],len(self.queue_dist_profile[target][(count_value+at_most_to_be_found)-1]))
        if(len(self.queue_dist_profile[target])==self.kappa):
            bound=len(self.queue_dist_profile[target][self.kappa-1])
        else:
            bound=sys.maxsize
            
        if(bound==len(self.top_k[target][-1])-1):
           
             yield None 
     
           
        
               
        try:
            
            # alt_p = self.bidir_BFS(source,target,bound-dist_path)
            
            alt_p = self.BFS(source,target,bound-dist_path)
            assert alt_p[0]==source
            assert alt_p[-1]==target
            # assert alt_w == len(alt_p)-1            
            assert len(alt_p)-1+dist_path<bound
            assert alt_p not in paths
            heappush(yen_PQ,(len(alt_p)-1,alt_p))
            paths.add(alt_p)
            
            
        except NoPathException:
           
            self.distance_profile.clear()
            self.detours.clear()
            yield None
            # return 
        
        
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
            

        while len(yen_PQ)>0:


            
            
            _ , P_det = heappop(yen_PQ)
            
            
            if len(self.queue_dist_profile[target])-self.count(target)>=at_most_to_be_found and len(P_det)-1+dist_path>=bound:
                del yen_PQ
                self.distance_profile.clear()
                self.detours.clear()
                paths.clear()
                yield None

          
            assert len(P_det)-1+dist_path<bound
           
            if self.binary_search_alt(self.queue_dist_profile[target], 0, len(self.queue_dist_profile[target])-1, coming_path[:-1]+P_det[:-1]):
            
               continue
            
            assert sorted(self.queue_dist_profile[target],key=lambda x:x[0])==list(self.queue_dist_profile[target]) 
           
            assert sorted(self.distance_profile)==list(self.distance_profile)

            assert P_det[0]==source
            assert P_det[-1]==target
            self.detour_done[target]=True
            if(len(coming_path[:-1]+P_det[:-1])>self.last_det_path[target]):
                self.last_det_path[target]=len(coming_path[:-1]+P_det[:-1])    
            yield P_det
           
            self.detours.append(P_det)
            assert len(self.detours)<at_most_to_be_found
            assert sorted(self.detours,key=len)==self.detours

            # if len(detours)>=at_most_to_be_found:
            #     del yen_PQ
            #     break
            
            # if len(self.queue_dist_profile[target])>=at_most_to_be_found:
            #     self.bound[target] = min(self.bound[target],len(self.queue_dist_profile[target][at_most_to_be_found-1]))
            # if len(self.queue_dist_profile[target])-self.count(target)>=at_most_to_be_found:
            
            #     self.bound[target] = min(self.bound[target],len(self.queue_dist_profile[target][(count_value+at_most_to_be_found)-1]))  cambio
            if(len(self.queue_dist_profile[target])==self.kappa):
                bound=len(self.queue_dist_profile[target][self.kappa-1])
            else:
                bound=sys.maxsize
            if(bound==len(self.top_k[target][-1])-1):
                 
                 yield None    

            assert len(P_det)-1+dist_path<=bound
            
            if len(P_det)-1+dist_path==bound:
                del yen_PQ
                paths.clear()
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
                    
                 
                    #pgt = self.bidir_BFS(l_root[-1],target,bound-(dist_path+len(l_root[:-1])))
                    pgt = self.BFS(l_root[-1],target,bound-(dist_path+len(l_root[:-1])))
                    
                    assert len(l_root[:-1] + pgt)-1==len(pgt)-1+len(l_root[:-1])               
                    assert len(pgt)-1+len(l_root[:-1])+dist_path<bound
                    
                    new_path = l_root[:-1] + pgt
                   
                    assert len(new_path)-1<bound-dist_path
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
       
    
    def standard_BFS(self,source,target,bound_on_length):
        
        if self.ignore_nodes[source] or self.ignore_nodes[target] or self.locally_ignore_nodes[source] or self.locally_ignore_nodes[target]:
          
            raise NoPathException
        if target == source:
            return {source:None}
        pred={source:None}
        # visited=set()
        # visited.add(source)
        # fringe=deque()
        # fringe.append((source,(source,)))
        fringe=deque([source])

        lunghezza=0
        while fringe:
            lunghezza+=1
           
            if lunghezza>=bound_on_length:
              
                raise NoPathException
            this_level=fringe
            fringe=deque()
            while this_level:
                vtx=this_level.popleft()
                for i in self.graph.iterNeighbors(vtx):
                    if self.locally_ignore_edges[self.graph.edgeId(vtx,i)]:
                        continue
                    if i in pred or self.ignore_nodes[i] or self.locally_ignore_nodes[i]:
                        
                        continue
                    if i==target:
                       del fringe
                       pred[i]=vtx
                       return pred
                       # assert len(ptx+(i,))-1<bound_on_length
                       # assert self.is_simple(ptx+(i,))
                       # return ptx+(i,)
                    #visited.add(i)
                    pred[i]=vtx
                    fringe.append(i)
        
        del fringe
        
        raise NoPathException(f"No path between {source} and {target}.")
    def BFS(self,source,target,bound_on_length):
        pred=self.standard_BFS(source, target, bound_on_length)
        path= deque()
        w=target
        while w is not None:
            path.appendleft(w)
            w=pred[w]
        assert len(path)-1<bound_on_length
        assert self.is_simple(tuple(path))
        return tuple(path)     
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
                     #   assert self.graph.edgeId(v,w) == self.graph.edgeId(w,v)
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
                   # continue
                    for w in self.graph.iterNeighbors(v):
                     #   assert self.graph.edgeId(v,w) == self.graph.edgeId(w,v)
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

    parser.add_argument('--g',metavar="GRAPH", required=True,  help='Path to the graph file (.hist format)')
    parser.add_argument('--k', metavar="K_VALUE", required=True, type=int, help='Number of top shortest paths to seek for', default=2)
    parser.add_argument('--r', metavar="NUM_ROOTS", required=False, type=int, help='Number of roots to consider (Default sqrt n)', default=-1)
    args = parser.parse_args()


    
    if ".hist" in str(args.g):
        
        G = hist2nk(str(args.g))
    else:
        G = read_nde(str(args.g))
    
    num_roots = int(args.r)
    
    
    if not G.isDirected():
        raise Exception('wrong graph input')

    G = graphtools.toUnweighted(G)    

    cc = nk.components.StronglyConnectedComponents(G)
    cc.run()
    part=cc.getPartition()
    index= max(range(len(part)), key=lambda i: len(part.getMembers(i)))
    G=nk.graphtools().subgraphFromNodes(G,[i for i in part.getMembers(index)],compact=True)    

    G.removeMultiEdges()
    G.removeSelfLoops()
    G.indexEdges()
    
    print("vertices:",G.numberOfNodes(),"arcs:",G.numberOfEdges())
    
    
    
    
    if not __debug__:
        print("AFTER SCC")
        nk.overview(G)
        
    K = int(args.k)
    if K<1:
        K=2
        
    print("Value of K:",K)
    
    diam = nk.distance.Diameter(G,algo=nk.distance.DiameterAlgo.ESTIMATED_SAMPLES,nSamples=10)
    diam.run()
    diametro = diam.getDiameter()[0]  
    print("DIAMETER:",diametro)
    del diam
    
    now = datetime.now() # current date and time
    date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    import os
    
    statsfile = str(os.path.basename(args.g))+"_"+str(K)+"_kbfs_global_directed_"+date_time+'.csv'
    
    
    # aux_graph = nk.nxadapter.nk2nx(G)

    # print("K-EDGE-CONNECTED:",nx.is_k_edge_connected(aux_graph, k=K))

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
        
    print("N_ROOTS:",num_roots)
    # num_roots = 1
    # randomly_selected_sqrt_roots = {5541} #problematic root oregon 4,2
    # randomly_selected_sqrt_roots = {2} 
    # randomly_selected_sqrt_roots = {47,9009} 
    #randomly_selected_sqrt_roots = {10} 
    # randomly_selected_sqrt_roots=set()
    # for i in G.iterNodes():
        
    #     randomly_selected_sqrt_roots.add(i)
   
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
        
        cpu_kbfs.append(local_cpu_bfs)
        pruned_branches.append(SSTK.pruned_branches)
        extra_visits.append(SSTK.extra_visits)
        if SSTK.extra_visits>0:
            fraction_of_pruned.append(round(SSTK.pruned_branches/SSTK.extra_visits,2))
        else:
            fraction_of_pruned.append(0.0)

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
                test_equality("YY",top_k_yen,[tuple(i) for i in islice(nx.shortest_simple_paths(aux_graph, first_root, second, weight=None), K)])
                

            for i in top_k_yen:
                del i
            del top_k_yen
            
        cpu_yen.append(local_cpu_yen)
        
        speedups.append(local_cpu_yen/local_cpu_bfs)
        
        for i in G.iterNodes():
            for j in SSTK.top_k[i]:
                del j
       # time.sleep(0.1)        
        nested_bar.finish()
        
        bar.next()

        print("Total KBFS CPUTime:", round(local_cpu_bfs,2),"Total Yen CPUTime:", round(local_cpu_yen,2),"Speedup:",round(local_cpu_yen/local_cpu_bfs,2), "Extra-visits:", SSTK.extra_visits,"of which pruned:", SSTK.pruned_branches, flush=True)

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

        
        # fraction_of_pruned = list(filter(lambda x:x>0, fraction_of_pruned))
        # print("Avg Fraction", round(statistics.mean(fraction_of_pruned),2))
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

            

