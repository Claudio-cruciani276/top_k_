#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:28:41 2023

@author: anonym
"""
    
#!/usr/bin/env python3
from itertools import islice
from networkit import graphtools 
import gc
import networkx as nx
import argparse
from heapq import heappush,heappop
import networkit as nk
from networkx.utils import pairwise
import time
from progress.bar import IncrementalBar, PixelBar

from memory_profiler import profile
import bisect
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import math
import statistics
from collections import deque
import sys

class PathException(Exception):
    """Base class for exceptions in Searching Paths."""
class NoPathException(PathException):
    """Exception for algorithms that should return a path when running
    on graphs where such a path does not exist."""

from enum import Enum

# class syntax

class PathEntryType(Enum):

    IS_IN_PQ = 1
    IS_IN_PATH_PROFILE_ONLY = 2
    
    
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
        del pred
        del succ
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
                
               
                raise Exception('correctness exception')
                
class single_source_top_k():

    def __init__(self, grph, num_k, r):
        
        self.graph = grph
        self.kappa = num_k
        self.pruned_branches = 0
        
        self.top_k = [deque() for _ in G.iterNodes()]
        self.pigreco_set = [set() for _ in self.graph.iterNodes()]
        
        self.ignore_nodes = [False for _ in self.graph.iterNodes()]
        self.queue_ignore_nodes = deque()
        
        self.locally_ignore_nodes = [False for _ in self.graph.iterNodes()]
        self.locally_ignore_edges = [False for _ in self.graph.iterEdges()]
        self.locally_queue_ignore_nodes = deque()
        self.locally_queue_ignore_edges = deque()
        
        self.pt_profile = [deque() for _ in G.iterNodes()]
        self.yenized = [False for _ in G.iterNodes()]    

        self.visited = [False for _ in self.graph.iterNodes()]
        self.queue_visited = deque()


        self.non_sat = [True for v in self.graph.iterNodes()]
        self.num_non_sat = self.graph.numberOfNodes()-1

        self.root = r
        self.non_sat[r] = False
        self.pruned_branches = 0
        self.extra_visits = 0
        self.detours = []
        # self.auxiliary = list()
        self.merge_set = set()
        self.neighbors = set()
        
        
        
    # def search_and_insert_vertex(self, arr, x):
 
    #     index = bisect.bisect_left(arr,x)
        
    #     if index<len(arr) and arr[index]==x:
    #         assert x in arr
    #         return
            
        
    #     assert x not in arr
    #     arr.insert(index, x)
    #     assert x in arr
    #     return     
    
    def search_and_insert_path(self, arr, path):
 
        index = bisect.bisect_left(arr,len(path),key=lambda x: len(x[0]))
        
        while index<len(arr) and len(arr[index][0])==len(path):
            if arr[index][0]==path:
                assert any(path == a[0] for a in arr)
                return (index,True)
            index+=1
        
        assert path not in arr
        assert all(path != a[0] for a in arr)
        # arr.insert(index, path)
        # assert path in arr
        # assert any(path == a[0] for a in arr)

        return (index,False)
    
    def deallocation(self):

        for i in self.PQ:
            del i
        
        del self.PQ
        for i in self.pigreco_set:
            for j in i:
                del j
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
        for i in self.pt_profile:
            for j in i:
                del j
        del self.pt_profile[:]
        del self.pt_profile
        del self.yenized[:]
        del self.yenized
        
        del self.visited[:]
        del self.visited
        
        
        self.queue_visited.clear()
        assert self.num_non_sat == len([i for i in self.non_sat if i==True])
        assert self.num_non_sat == len([v for v in self.graph.iterNodes() if self.non_sat[v]])
        
        assert self.num_non_sat == len([v for v in self.graph.iterNodes() if len(self.top_k[v])<self.kappa and v!=self.root])
        
        del self.non_sat[:]
        del self.non_sat    

        #garbage collection forced
        gc.collect()
    # # instantiating the decorator
    # # code for which memory has to
    # # be monitored    
    # @profile
    def generalized_bfs(self):
        

        

        assert type(self.top_k)==list and all(len(self.top_k[v])==0 for v in self.graph.iterNodes())     
        assert type(self.non_sat)==list        
        self.PQ = deque()
        assert not self.non_sat[self.root]

        for ngx in self.graph.iterNeighbors(self.root):
            cm = tuple([self.root,ngx])
            tpl = (1,ngx,cm,{self.root,ngx})
            assert tpl not in self.PQ
            assert len(cm)-1==1

            self.PQ.append(tpl)
            assert list(self.PQ)==sorted(self.PQ,key=lambda x:x[0])
            assert self.non_sat[ngx]
            assert cm not in self.pt_profile[ngx]
            assert not self.yenized[ngx]
            self.pt_profile[ngx].append((cm,PathEntryType.IS_IN_PQ))
            

            assert any(a[0]==cm for a in self.pt_profile[ngx])
            assert sorted(self.pt_profile[ngx],key=lambda x:len(x[0]))==list(self.pt_profile[ngx])

        while len(self.PQ)>0:
            
            tpl = self.PQ.popleft()
            wgt, vtx, ptx, setptx = tpl
            del tpl
            assert wgt==len(ptx)-1
            assert ptx[0]==self.root and ptx[-1]==vtx
            assert vtx!=self.root
            assert type(ptx)==tuple
            assert type(setptx)==set
            assert len(ptx)==len(setptx)
            assert all(v in self.graph.iterNeighbors(u) for u, v in pairwise(ptx))
            assert self.num_non_sat>0
            assert ptx not in self.top_k[vtx]
          
            assert sorted(self.pt_profile[vtx],key=lambda x:len(x[0]))==list(self.pt_profile[vtx])


            if len(self.top_k[vtx]) < self.kappa:
                assert self.non_sat[vtx]
           
                assert (ptx,PathEntryType.IS_IN_PQ) in self.pt_profile[vtx]
                assert (ptx,PathEntryType.IS_IN_PATH_PROFILE_ONLY) not in self.pt_profile[vtx]

                assert wgt == len(self.pt_profile[vtx][0][0])-1 
                count = 0
                while count<len(self.pt_profile[vtx]) and wgt == len(self.pt_profile[vtx][count][0])-1:
                    
                    if ptx == self.pt_profile[vtx][count][0]:
                        self.pt_profile[vtx][0],self.pt_profile[vtx][count]=self.pt_profile[vtx][count],self.pt_profile[vtx][0]        
                        self.pt_profile[vtx].popleft()
                        break
                    count+=1
                    
                assert (ptx,PathEntryType.IS_IN_PQ) not in self.pt_profile[vtx]
                assert (ptx,PathEntryType.IS_IN_PATH_PROFILE_ONLY) not in self.pt_profile[vtx]

                assert sorted(self.pt_profile[vtx],key=lambda x:len(x[0]))==list(self.pt_profile[vtx])


                self.standard(wgt,vtx,ptx,setptx)    

                del ptx
                del setptx
                if self.num_non_sat==0:
                    return
                continue
            
            else:
                assert not self.non_sat[vtx]
                # assert len(self.pt_profile[vtx])==0
                self.beyond(wgt,vtx,ptx,setptx)
                del ptx
                del setptx
                
                
    def standard(self,WEIG,VERT,PATH,PATHSET):
        
        assert len(self.top_k[VERT])<self.kappa
        
    
        self.top_k[VERT].append(PATH)
        
        assert list(self.top_k[VERT])==sorted(self.top_k[VERT],key=len)    
        assert len(self.top_k[VERT])<=self.kappa
        
        if len(self.top_k[VERT])==self.kappa:  
            assert self.non_sat[VERT]    
            self.non_sat[VERT] = False
            self.num_non_sat -= 1
            assert(len([val for val in self.non_sat if val==True])==self.num_non_sat)
            
            # self.pt_profile[VERT].clear()

            if self.num_non_sat==0:
                return

                              
        assert PATH[-2] in self.graph.iterNeighbors(VERT)
        assert(type(self.pigreco_set[VERT]==set))
        
        for vpath in PATH[1:-1]:
            assert vpath is not self.root and vpath is not VERT
            if self.non_sat[vpath]:
                self.pigreco_set[VERT].add(vpath)
                # self.search_and_insert_vertex(self.pigreco_set[VERT],vpath)

        
        assert self.num_non_sat>0
        for ngx in self.graph.iterNeighbors(VERT):
            if ngx in PATHSET:
                continue
            assert  ngx != self.root
            cm = PATH+(ngx,)
            assert WEIG+1==len(cm)-1
            tpl = (WEIG+1,ngx,cm,PATHSET.union({ngx}))
            assert tpl not in self.PQ
            self.PQ.append(tpl)
            
            assert list(self.PQ)==sorted(self.PQ,key=lambda x:x[0])
            
            
            if self.non_sat[ngx]:

                if not self.yenized[ngx]:
                    assert (cm,PathEntryType.IS_IN_PQ) not in self.pt_profile[ngx]
                    assert (cm,PathEntryType.IS_IN_PATH_PROFILE_ONLY) not in self.pt_profile[ngx]

                    self.pt_profile[ngx].append((cm,PathEntryType.IS_IN_PQ))

                else:
                    (indice,was_there) = self.search_and_insert_path(self.pt_profile[ngx],cm)
                    if was_there:
                        self.pt_profile[ngx][indice] = (cm,PathEntryType.IS_IN_PQ) #now enqueued, change status
                    else:
                        self.pt_profile[ngx].insert(indice,(cm,PathEntryType.IS_IN_PQ))
                        
                assert (cm,PathEntryType.IS_IN_PQ) in self.pt_profile[ngx]
                assert (cm,PathEntryType.IS_IN_PATH_PROFILE_ONLY) not in self.pt_profile[ngx]
                
            assert sorted(self.pt_profile[ngx],key=lambda x:len(x[0]))==list(self.pt_profile[ngx])


    def beyond(self,WEIG,VERT,PATH,PATHSET):
        
        self.extra_visits += 1
        assert self.num_non_sat>0
        assert not self.non_sat[VERT] and len(self.top_k[VERT])>=self.kappa and PATH not in self.top_k[VERT] and type(PATH) == tuple
        assert self.root in PATHSET
        assert VERT in PATHSET
        assert self.visited[VERT]==False
        assert not self.visited[self.root]

        assert self.merge_set is not None
        self.merge_set.clear()
        # for vr in self.pigreco_set[VERT]:
        # assert len(self.auxiliary)==0
        # contatore = 0
        # lunghezza_corrente = len(self.pigreco_set[VERT])
        # while contatore < lunghezza_corrente:
        for scanned_vertex in self.pigreco_set[VERT]:
            """ si è saturato nel frattempo, mi prendo merge dei due set pigreco """
            assert scanned_vertex not in self.pigreco_set[scanned_vertex]

            if not self.non_sat[scanned_vertex]:
                assert scanned_vertex not in self.merge_set
                for element in self.pigreco_set[scanned_vertex]:
                    assert element != scanned_vertex
                    if self.non_sat[element] and element != VERT:
                        self.merge_set.add(element)
                
                # self.pigreco_set[VERT].pop(contatore)
                # assert scanned_vertex not in self.pigreco_set[VERT]
            #     continue
            # else:
            # contatore+=1
        self.pigreco_set[VERT]=set(filter(lambda x : self.non_sat[x],self.pigreco_set[VERT]))
        self.pigreco_set[VERT].update(self.merge_set)
        assert all(self.non_sat[x] for x in self.pigreco_set[VERT])
        # while len(self.auxiliary)>0:
        #     aux_vertex = self.auxiliary.pop()
        #     for element in self.pigreco_set[aux_vertex]:
        #         assert element != aux_vertex
        #         if self.non_sat[element] and element != VERT:
        #             self.pigreco_set[VERT].add(element)

                    # self.search_and_insert_vertex(self.pigreco_set[VERT],element)

        assert VERT not in self.pigreco_set[VERT]
        # assert sorted(self.pigreco_set[VERT])==self.pigreco_set[VERT]

        
        assert self.root not in self.pigreco_set[VERT]


        assert len(self.neighbors) == 0
        
        if len(self.pigreco_set[VERT])>0:

            num_tested = 0
            assert self.pigreco_set[VERT] is not None
                
            self.init_avoidance(PATH)
            
            for target_vertex in self.pigreco_set[VERT]:  
                
                assert self.non_sat[target_vertex]                    
                assert sorted(self.pt_profile[target_vertex],key=lambda x:len(x[0]))==list(self.pt_profile[target_vertex])
                
                max_to_generate = self.kappa-len(self.top_k[target_vertex])
                
                
                assert max_to_generate>0      
    
                n_generated = 0   
                num_tested+=1
    
                
                for DET in self.find_detours(VERT,target_vertex,max_to_generate,len(PATH)-1):
                    
                    if DET is None:
                        break
                    
                    n_generated+=1                 
                        
                    assert DET[1] in self.graph.iterNeighbors(VERT)
                    assert DET[1] not in PATHSET
                    assert not self.ignore_nodes[DET[1]] or DET[1] in self.neighbors
                    assert self.is_simple(PATH+DET[1:])
                    assert all(i[2]!=PATH+DET[1:] for i in self.PQ)
                    assert len(PATH+DET[1:])-1==len(PATH)-1+len(DET)-1              
                    assert len(self.neighbors)<self.graph.degree(VERT)-1
                    
                    
                    path_to_target = PATH+DET[1:]
           
                    assert path_to_target[0]==self.root
                    assert path_to_target[-1]==target_vertex
                    assert len(path_to_target)-1 == len(PATH)-1+len(DET)-1
                    
                    
                    if not self.yenized[target_vertex]:
                        assert all(path_to_target!=a[0] for a in self.pt_profile[target_vertex])
                        assert (path_to_target,PathEntryType.IS_IN_PATH_PROFILE_ONLY) not in self.pt_profile[target_vertex]
                        assert (path_to_target,PathEntryType.IS_IN_PQ) not in self.pt_profile[target_vertex]
                        self.pt_profile[target_vertex].append((path_to_target,PathEntryType.IS_IN_PATH_PROFILE_ONLY))
                        self.yenized[target_vertex]=True

         
                    else:
                        (indice,was_there) = self.search_and_insert_path(self.pt_profile[target_vertex],path_to_target)
                        if not was_there:
                            self.pt_profile[target_vertex].insert(indice,(path_to_target,PathEntryType.IS_IN_PATH_PROFILE_ONLY))
                            self.yenized[target_vertex]=True

                        assert (path_to_target,PathEntryType.IS_IN_PATH_PROFILE_ONLY) in self.pt_profile[target_vertex]
         
                    assert sorted(self.pt_profile[target_vertex],key=lambda x:len(x[0]))==list(self.pt_profile[target_vertex])   


                            
                            
                            
                    if DET[1] in self.neighbors:
                        assert DET[1] in self.neighbors
                        assert len(self.neighbors)<self.graph.degree(VERT)-1
    
                        if n_generated==max_to_generate:
                            self.detours.clear()
                            break
                        
                        del DET
                        continue
    
                    assert DET[1] not in self.neighbors
                    self.neighbors.add(DET[1])                     
                    
                    if len(self.neighbors)==self.graph.degree(VERT)-1:
                        assert PATH[-1] not in self.neighbors
                        self.detours.clear()
                        del DET
                        break
                    
                    if n_generated==max_to_generate:
                        self.detours.clear()
                        del DET
                        break
                    
                    del DET
                

                if len(self.pt_profile[target_vertex])>=max_to_generate and self.ignore_nodes[target_vertex]==False:
                    starting_index = max_to_generate-1
                    soglia = len(self.pt_profile[target_vertex][starting_index][0])-1
                    
                    while starting_index>=0:
                        
                        cmt, tip = self.pt_profile[target_vertex][starting_index]
                        
                        starting_index-=1
                        
                        if tip==PathEntryType.IS_IN_PQ:
                            continue
                        
                        assert tip==PathEntryType.IS_IN_PATH_PROFILE_ONLY
                        if len(cmt)-1!=soglia:
                            break
                        
                        assert len(cmt)-1==soglia
                        assert cmt[-1]==target_vertex
                        
                        end = len(cmt)-1
                        
                        while end>=1:
                            if cmt[end-1]!=VERT and cmt[end-1] in PATHSET:
                                break
                            if cmt[end-1]==VERT:
                                self.neighbors.add(cmt[end])
                                break
                            end-=1

                        if len(self.neighbors)==self.graph.degree(VERT)-1:
                            assert PATH[-1] not in self.neighbors
                            break
                        continue    
                        
                    
                if num_tested==self.num_non_sat:
                    break
                
                if len(self.neighbors)==self.graph.degree(VERT)-1:
                    assert PATH[-1] not in self.neighbors
                    break
                
    
            self.clean_avoidance(PATH)
            
        assert self.num_non_sat>0

        if len(self.neighbors)==0:
            self.pruned_branches += 1

            return
        
        for ngx in self.neighbors:
            assert ngx not in PATHSET
            assert ngx not in PATH
            assert ngx in self.graph.iterNeighbors(VERT)
            assert PATH[-1]==VERT
            cm = PATH+(ngx,)
            
            tpl = (WEIG+1,ngx,cm,PATHSET.union({ngx}))
            assert tpl not in self.PQ
            assert WEIG+1==len(cm)-1

            self.PQ.append(tpl)        
            assert list(self.PQ)==sorted(self.PQ,key=lambda x:x[0])
            if self.non_sat[ngx]:
                
                if not self.yenized[ngx]:
                    assert (cm,PathEntryType.IS_IN_PQ) not in self.pt_profile[ngx]
                    assert (cm,PathEntryType.IS_IN_PATH_PROFILE_ONLY) not in self.pt_profile[ngx]

                    self.pt_profile[ngx].append((cm,PathEntryType.IS_IN_PQ))

                else:
                    (indice,was_there) = self.search_and_insert_path(self.pt_profile[ngx],cm)
                    if was_there:
                        self.pt_profile[ngx][indice] = (cm,PathEntryType.IS_IN_PQ) #now enqueued, change status
                    else:
                        self.pt_profile[ngx].insert(indice,(cm,PathEntryType.IS_IN_PQ))
                        
                assert (cm,PathEntryType.IS_IN_PQ) in self.pt_profile[ngx]
                assert (cm,PathEntryType.IS_IN_PATH_PROFILE_ONLY) not in self.pt_profile[ngx]
                    
            
            assert sorted(self.pt_profile[ngx],key=lambda x:len(x[0]))==list(self.pt_profile[ngx])
        
        self.neighbors.clear()
        del PATH
        del PATHSET                          
        
    def init_avoidance(self,avoid):        
        lungh = 0
        assert avoid[0]==self.root        
        while lungh<len(avoid)-1:
            u = avoid[lungh]           
            assert not self.ignore_nodes[u]
            self.ignore_nodes[u]=True
            # self.queue_ignore_nodes.append(u)            
            lungh+=1
            
    def clean_avoidance(self,avoid):
        
        lungh = 0
        assert avoid[0]==self.root        
        while lungh<len(avoid)-1:
            u = avoid[lungh]           
            assert self.ignore_nodes[u]
            self.ignore_nodes[u]=False
            # self.queue_ignore_nodes.append(u)            
            lungh+=1
            
    def find_detours(self,source,target,at_most_to_be_found,dist_to_vert):
          
        assert len(self.top_k[target])>=1
        assert len(self.top_k[target])<self.kappa
        assert(sorted(self.top_k[target],key=len)==list(self.top_k[target]))
        assert len(self.top_k[source])==self.kappa
        

        yen_PQ = []
        paths = set()

            
        assert not self.ignore_nodes[source]
        
        if self.ignore_nodes[target]:
            yield None
        
        
        assert len(self.detours)==0


        THR = len(self.pt_profile[target][at_most_to_be_found-1][0])-1 - dist_to_vert if len(self.pt_profile[target])>=at_most_to_be_found else sys.maxsize

        try:
            alt_p = self.bidir_BFS(source,target,THR)

            assert alt_p[0]==source
            assert alt_p[-1]==target
            assert len(alt_p)-1<THR
            assert alt_p not in paths
            heappush(yen_PQ,(len(alt_p)-1,alt_p))
            paths.add(alt_p)
            
            
        except NoPathException:
            self.detours.clear()
            yield None
        
        
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise
            

        while len(yen_PQ)>0:


            
            
            _ , P_det = heappop(yen_PQ)
            
            # if len(self.pt_profile[target])>=at_most_to_be_found and len(P_det)-1+dist_path>=len(self.pt_profile[target][at_most_to_be_found-1])-1:
            #     del yen_PQ[:]
            #     del yen_PQ
            #     self.detours.clear()
            #     paths.clear()
            #     yield None

            assert P_det[0]==source
            assert P_det[-1]==target
            
            
            # if len(P_det)-1+dist_path>=threshold:
            #     del yen_PQ[:]
            #     del yen_PQ
            #     self.detours.clear()
            #     paths.clear()
            #     yield None
                
            yield P_det
            assert len(self.detours)<at_most_to_be_found
            assert sorted(self.detours,key=len)==self.detours

            self.detours.append(P_det)
            

            if len(self.detours)>=at_most_to_be_found: # or len(P_det)-1+dist_path==threshold:
                del yen_PQ[:]
                del yen_PQ
                self.detours.clear()
                paths.clear()
                yield None
                

            assert all(not self.locally_ignore_nodes[v] for v in self.graph.iterNodes())
            assert all(not self.locally_ignore_edges[self.graph.edgeId(u, v)] for u,v in self.graph.iterEdges())            

            THR = len(self.pt_profile[target][at_most_to_be_found-1][0])-1 - dist_to_vert if len(self.pt_profile[target])>=at_most_to_be_found else sys.maxsize

                
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
                    
                    pgt = self.bidir_BFS(l_root[-1],target,THR)
                    
                    
                    assert len(l_root[:-1] + pgt)-1==len(pgt)-1+len(l_root[:-1])               
                    assert len(pgt)-1<THR
                    
                    new_path = l_root[:-1] + pgt
                   
                    assert len(new_path)-1<THR+dist_to_vert
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

    statsfile = str(os.path.basename(args.g))+"_"+str(K)+"_kbfs_alt_glob_"+date_time+'.csv'
    

    cpu_yen  = []
    cpu_kbfs = []
    speedups = []
    pruned_branches = []
    extra_visits = []
    num_non_saturi = [] 

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
    # randomly_selected_sqrt_roots = {848,4481} #problematic root oregon 4,2
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
        num_non_saturi.append(SSTK.num_non_sat)

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
        
        for i in SSTK.top_k:
            for j in i:
                del j
        del SSTK.top_k[:]
        nested_bar.finish()
 
        bar.next()

        print("\nTotal KBFS CPUTime:", round(local_cpu_bfs,2),"Total Yen CPUTime:", round(local_cpu_yen,2),"Speedup:",round(local_cpu_yen/local_cpu_bfs,2), "Extra-visits:", SSTK.extra_visits,"of which pruned:", SSTK.pruned_branches, "Non saturi rimasti: ",SSTK.num_non_sat,flush=True)

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
                         "fraction",
                         "num_non_saturi"])
        
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
                             fraction_of_pruned[s],\
                             round(num_non_saturi[s],0)])
            
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
                         "avg_non_saturi",\
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
                         round(statistics.mean(num_non_saturi),2),\
                         round(statistics.mean(speedups),2),\
                         round(statistics.median(speedups),2)])

            


