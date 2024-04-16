"""
Created on Mon Oct  4 14:14:21 2021

@author: anonym
"""


from heapq import heappush,heappop 
import sys
import time
import networkit as nk
from progress.bar import IncrementalBar
import math
import statistics

# from collections import deque
from networkx.utils import pairwise
import networkx as netwx
from sortedcontainers import SortedDict
from bisect import bisect_left, bisect_right
from collections import deque

class PathException(Exception):
    """Base class for exceptions in Searching Paths."""
class NoPathException(PathException):
    """Exception for algorithms that should return a path when running
    on graphs where such a path does not exist."""
    

class PathBuffer:
    def __init__(self):
        self.paths = set()
        self.sortedpaths = []
        # self.counter = count()
    def clear(self):
        self.paths.clear()
        self.sortedpaths.clear()
        # self.counter = count()
        
    def __len__(self):
        return len(self.sortedpaths)

    def push(self, cost, hashable_path, hub):
        
        assert type(hashable_path)==tuple

        
        if hashable_path not in self.paths:
            heappush(self.sortedpaths, (cost, hashable_path, hub))

            self.paths.add(hashable_path)

    def pop(self):
        (cost, hashable_path, hub) = heappop(self.sortedpaths)
        self.paths.remove(hashable_path)
        return cost, hashable_path, hub


    
class labeling_handler():

    def __init__(self, G, num_k):
        
        self.graph = G

        self.num_k = num_k
        self.ordering = [0]*self.graph.numberOfNodes()
        self.reverse_ordering = [0]*self.graph.numberOfNodes()
        
        self.labels =  [SortedDict() for _ in range(self.graph.numberOfNodes())]
        
        self.intersection_tests = [False]*self.graph.numberOfNodes()
        


        
        self.root_index = 0
        self.root = 0



        self.labeling_size = 0
        self.prunings = 0
        self.constrTime = 0
        self.centralityTime = 0
        self.results = []
        self.lengths = []

        #ROBA PER YEN SIMULTANEO
        # self.lowest_index = None
        # self.length_func = len
        # self.shortest_path_func = self._bidirectional_shortest_path
        

        # self.ignore_edges = None        
        # self.reset_nodes = deque()
        # self.avoid_nodes = [False]*self.graph.numberOfNodes()
        # self.listB = [PathBuffer() for _ in range(self.graph.numberOfNodes()-1)]
        # self.listA = [[] for _ in range(self.graph.numberOfNodes()-1)]
        # self.prev_path = [None for _ in range(self.graph.numberOfNodes()-1)]
        # self.touched = deque()
        # self.yen_root = []
        # self.aux_indice = 0
        # self.root_length = 0
        # self.hub_of_sub_path = sys.maxsize
        # self.PQ = []
        # self.settled = set()
        # self.prunati = set()
        self.diametro = 0.0
        self.density = 0.0
        # self.equality = True
        # self.aux_idx = 0
        # self.indice = 1
        self.aux_graph = nk.nxadapter.nk2nx(self.graph)
        # self.K_COMPONENTS = netwx.k_edge_components(, k=self.num_k)

        
    def is_in_results(self,x):
        assert len(x)==3 and type(x)==tuple

        i = bisect_left(self.results, x[0], key=lambda z:z[0])
        #The return value i is such that all e in a[:i] have e < x, and all e in a[i:] have e >= x.

        if i != len(self.results):
            while i<len(self.results) and self.results[i][0]==x[0]:
                if self.results[i] == x:
                    assert x in self.results
                    return True, i
                i+=1
        assert x not in self.results    
        return False, i
    
    def add_to_results_in_position(self,p,x):
        assert len(x)==3 and type(x)==tuple
        assert x not in self.results
        self.results.insert(p,x)
        assert self.results == sorted(self.results,key=lambda z:z[0])    

    def add_to_results(self,x):
        assert len(x)==3 and type(x)==tuple
        #bisect_right riduce il numero di shift
        i = bisect_right(self.results, x[0], key=lambda z:z[0]) #search weights
        #The return value i is such that all e in a[:i] have e <= x, and all e in a[i:] have e > x.        
        assert i == len(self.results) or (all(self.results[k][0] <= x[0] for k in range(0,i)) and all(self.results[k][0] > x[0] for k in range(i,len(self.results))))
        assert all(j != x for j in self.results)
        assert all(j[1] != x[1] for j in self.results)

        self.results.insert(i,x)
        
        assert x in self.results
        assert self.results == sorted(self.results,key=lambda z:z[0])    

    def add_to_external(self,a, x):
        assert len(x)==3 and type(x)==tuple
        i = bisect_right(a, x[0], key=lambda z:z[0])
        #The return value i is such that all e in a[:i] have e <= x, and all e in a[i:] have e > x.

        assert i == len(a) or\
            (all(a[k][0] <= x[0] for k in range(0,i)) and all(a[k][0] > x[0] for k in range(i,len(a))))
        assert all(j != x for j in a)
        a.insert(i,x) 
        assert x in a

        assert a == sorted(a,key=lambda z:z[0])    

    def add_to_lengths(self,x):
        assert type(x)==int
        i = bisect_right(self.lengths, x)
        #The return value i is such that all e in a[:i] have e <= x, and all e in a[i:] have e > x.

        self.lengths.insert(i,x)    
        assert self.lengths == sorted(self.lengths)    
        
    def is_path_in_label_of_host(self,host,hub,path,peso):

        if hub not in self.labels[host]:
            return False, -1
        
        
        return self.is_path_in_label(self.labels[host][hub],path,peso)
    
    def is_path_in_label(self,label,path,peso):

        
        #bisect_left necessario per scorrere path con stesso peso
        i = bisect_left(label, peso, key=lambda x:x[1])
        #The return value i is such that all e in a[:i] have e < x, and all e in a[i:] have e >= x.

        if i != len(label):
            while i<len(label) and label[i][1]==peso:
                if label[i][0] == path:
                    assert (path,peso) in label
                    return True, i
                i+=1
        assert (path,peso) not in label
        assert i>=0
        return False, i   
    
    # def reset_ignored(self):
    #     for node in self.reset_nodes:
    #         assert self.avoid_nodes[node]
    #         self.avoid_nodes[node] = False
    #     assert all(self.avoid_nodes[node]==False for node in self.graph.iterNodes())
    #     self.reset_nodes = deque()

    # def conditional_reset_ignored(self,forbidden_vertices,target=-1):
    #     for node in self.reset_nodes:
    #         assert self.avoid_nodes[node]
    #         self.avoid_nodes[node] = False
        
    #     self.reset_nodes = deque()
    #     for node in forbidden_vertices:
    #         if node == target:
    #             assert target != -1
    #             raise NoPathException
    #         self.avoid_nodes[node] = True
    #         self.reset_nodes.append(node)
            

    #     assert all(self.avoid_nodes[node]==False for node in self.graph.iterNodes() if node not in forbidden_vertices)
    #     assert all(self.avoid_nodes[node] for node in forbidden_vertices)
        
    """ def dense_graphs(self):
        
        assert self.lowest_index == self.root_index
        
        self.lowest_index = -1

        assert len(self.listB) == self.graph.numberOfNodes()-self.root_index-1 and  len(self.listA) == self.graph.numberOfNodes()-self.root_index-1 and len(self.prev_path) == self.graph.numberOfNodes()-self.root_index-1

    
        assert all(len(i)==0 for i in self.listB) and all(len(i)==0 for i in self.listA) and all(i==None for i in self.prev_path) and len(self.touched)==0
        
    
        self.append_entry(host=self.root,hub=self.root_index,path=(self.root,),peso=0)
    
        
        source_index = self.root_index+1
        
        self.PQ = []
        self.settled = set()

        while source_index<self.graph.numberOfNodes():
            
    
            assert self.prev_path[source_index-self.root_index-1] is None

            self.reset_ignored()
            self.ignore_edges = set()                
                
            assert self.lowest_index<=self.ordering[self.root] and self.lowest_index<=self.ordering[self.reverse_ordering[source_index]]
            
            try:
                wgt, pgt, hgt = self.shortest_path_func(self.root, self.reverse_ordering[source_index])
            except NoPathException:
                self.touched.append(source_index-self.root_index-1)
                source_index+=1
                continue
            
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise
            
            assert wgt == len(pgt)-1          
            assert type(pgt)==tuple
            assert len(self.listB[source_index-self.root_index-1])==0
            
    
            self.listA[source_index-self.root_index-1].append(pgt)
            self.prev_path[source_index-self.root_index-1] = pgt
            assert source_index-self.root_index-1 not in self.touched
            self.touched.append(source_index-self.root_index-1)
            heappush(self.PQ, (wgt,0,self.reverse_ordering[source_index],pgt,hgt))
            source_index+=1
        
        
    
        while len(self.PQ)>0:
            
            W_PT, ITER_VERTEX, VERTEX, PT, H_PT = heappop(self.PQ)
            REF_VERTEX = self.ordering[VERTEX]-self.root_index-1
            
            assert REF_VERTEX in self.touched
            assert all(self.ordering[i]>=H_PT for i in PT)
            assert W_PT==len(PT)-1 and PT[-1]==VERTEX and PT[0]==self.root and len(PT)>=2
            


            if self.root_index == 0:
                assert H_PT == self.root_index
                if self.root_index in self.labels[VERTEX] and len(self.labels[VERTEX][self.root_index])>=self.num_k and self.labels[VERTEX][self.root_index][self.num_k-1][1]<=W_PT:
                    self.prunings+=1
                    continue 
                assert not self.k_shorter_simple_paths(self.root,VERTEX,W_PT)
            else: 
                if self.k_shorter_simple_paths(self.root,VERTEX,W_PT):
                    self.prunings+=1
                    continue



            L,w_L,R,w_R = self.break_path_at(path=PT, hub_vertex=H_PT,peso=W_PT)
            assert L[0]==self.reverse_ordering[H_PT]
            assert L[-1]==self.root
            assert R[0]==self.reverse_ordering[H_PT]
            assert R[-1]==VERTEX
            
            del PT
            
            p_1,i_1 = self.is_path_in_label_of_host(host=self.root, hub=H_PT, path=L,peso=w_L)
            p_2,i_2 = self.is_path_in_label_of_host(host=VERTEX,hub=H_PT,path=R,peso=w_R)

            if not p_1:
                if i_1 == -1:
                    self.labels[self.root][H_PT] = [(L,w_L)]
                else:
                    self.labels[self.root][H_PT].insert(i_1,(L,w_L))
                self.labeling_size+=1
            if not p_2:
                if i_2 == -1:
                    # self.append_entry(host=VERTEX,hub=H_PT,path=R,peso=w_R)
                    self.labels[VERTEX][H_PT] = [(R,w_R)]
                else:
                    self.labels[VERTEX][H_PT].insert(i_2,(R,w_R))

                self.labeling_size+=1

            if ITER_VERTEX<self.num_k:
                (vv, pp, ww, hh) = self.hub_get_next(VERTEX,REF_VERTEX)
                if (vv,pp,ww,hh)!=(None,None,None,None):
                    assert vv==VERTEX
                    assert ww>=W_PT
                    heappush(self.PQ, (ww, ITER_VERTEX+1, vv, pp, hh))

                else:
                    self.settled.add(VERTEX)
            else:
                self.settled.add(VERTEX)
            
                
                    
        assert len(self.touched)<=len(self.listB)
        assert len(self.touched)==len(set(self.touched))
    """
    def loopy_top_k_paths(self, u: int, v: int, break_at_k:bool=True):
        
        self.results.clear()
    
    
        if u == v:
            self.add_to_results((0,(u,),self.ordering[u]))
            return
        if len(self.labels[u])==0 or len(self.labels[v])==0:
            return 
        maxhub = min(self.labels[u].keys()[-1],self.labels[v].keys()[-1])
        it_u = iter(self.labels[u].keys())
        it_v = iter(self.labels[v].keys())
        el_u = next(it_u,-1)
        el_v = next(it_v,-1)
        while True:
            if el_u==-1 or el_v==-1 or el_u>maxhub or el_v>maxhub:
                break
            if el_u<el_v:
                el_u = next(it_u,-1)
                continue
            if el_u>el_v:
                el_v = next(it_v,-1)
                continue
           
            assert el_u==el_v
            assert el_u!=-1
            hub = el_u
            el_u = next(it_u,-1)
            el_v = next(it_v,-1)
            assert hub in self.labels[u] and hub in self.labels[v]
            P_u = self.labels[u][hub]
    
            assert type(P_u)==list #sortedcontainers.sortedlist.SortedKeyList  
    
            assert len(P_u)>0
            P_v = self.labels[v][hub]
            assert type(P_v)==list #sortedcontainers.sortedlist.SortedKeyList  
    
            assert len(P_v)>0
    
            
            
            
            for p_u,d_u in P_u:
                
                
                assert d_u == len(p_u)-1
            
    
                if break_at_k and len(self.results)>=self.num_k and d_u >= self.results[self.num_k-1][0]:
                    assert len(self.results)==self.num_k
                    break

                for p_v,d_v in P_v:
                    
    
    
                    assert d_v == len(p_v)-1
    
                    assert p_u[0]==p_v[0]
                    assert p_u[0]==self.reverse_ordering[hub]
                    assert p_v[0]==self.reverse_ordering[hub]
    
                    if break_at_k and len(self.results)>=self.num_k and d_u+d_v >= self.results[self.num_k-1][0]:
                        assert len(self.results)==self.num_k
                        break
    
                        
                    assert not break_at_k or len(self.results)<self.num_k or (d_u+d_v<self.results[self.num_k-1][0])
                    
                    w_pt, pt = self.combination_without_intersection_test(d_u, d_v, p_u, p_v)
    
                    
    
                    assert pt is not None
                    assert type(pt)==tuple
                    
                    self.add_to_results((w_pt, pt, hub))
    
    
                    if break_at_k and len(self.results)>self.num_k:
                        el = self.results.pop()
                        assert len(self.results)==self.num_k
                        assert el[0]>=self.results[self.num_k-1][0]
          
        
    def break_path_at(self,path:list,hub_vertex:int,peso:int):
        
        
        assert type(path)==tuple and type(hub_vertex)==int
        assert peso==len(path)-1

        
        idx = peso
        w_r = 0
        right = tuple()
        while True:
            if path[idx] == self.reverse_ordering[hub_vertex]:
                right = (path[idx],)+right
                break
            right = (path[idx],)+right
            idx-=1
            w_r+=1
        
        assert right == path[idx:peso+1]        
        assert right[0] == self.reverse_ordering[hub_vertex]
        assert right[-1] == path[-1]
        assert w_r == len(right)-1
        
        
        
        left = tuple()
        w_l = -1
        while True:
            if idx<0:
                break
            left = left + (path[idx],)
            idx-=1
            w_l+=1 
        # left = path[0:idx+1][::-1]
        
        assert left[0] == self.reverse_ordering[hub_vertex]
        assert left[-1] == path[0]
        # print(w_l,left)
        assert w_l == len(left)-1
        assert w_l == peso - w_r
        return left,w_l,right,w_r
    
    def bidir_BFS(self, source, target, lowest_index, ignore_nodes=None, ignore_edges=None):
    
    
        pred, succ, w = self.bidir_pred_succ(source, target, lowest_index, ignore_nodes, ignore_edges)
        path = deque()
        h, d = sys.maxsize, -1
    
        while w is not None:
            path.append(w)
            if self.ordering[w]<h:
                h = self.ordering[w]
            w = succ[w]
            d+=1

    
        w = pred[path[0]]
        while w is not None:
            path.appendleft(w)
            if self.ordering[w]<h:
                h = self.ordering[w]
            w = pred[w]
            d+=1

        assert d==len(path)-1
        
        assert min(self.ordering[i] for i in path)==h
        assert h>=lowest_index
        return d, tuple(path), h
    

            
    def bidir_pred_succ(self, source, target, lowest_index, ignore_nodes=None, ignore_edges=None):
    
    
        if ignore_nodes is not None and (source in ignore_nodes or target in ignore_nodes):
            raise NoPathException
        if target == source:
            return ({target: None}, {source: None}, source)
    
    
        
        # predecesssor and successors in search
        pred = {source: None}
        succ = {target: None}
    
        # initialize fringes, start with forward
        forward_fringe = deque([source])
        reverse_fringe = deque([target])
    
        while forward_fringe and reverse_fringe:
            if len(forward_fringe) <= len(reverse_fringe):
                this_level = forward_fringe
                forward_fringe = deque()
                for v in this_level:
                    for w in self.graph.iterNeighbors(v):
                        if ignore_edges is not None and ((v, w) in ignore_edges or (w, v) in ignore_edges):
                            continue
                        if self.ordering[w]<=lowest_index or w==source or (ignore_nodes is not None and w in ignore_nodes):
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
                        if ignore_edges is not None and ((v, w) in ignore_edges or (w, v) in ignore_edges):
                            continue
                        if self.ordering[w]<=lowest_index or w==target or (ignore_nodes is not None and w in ignore_nodes):
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
        
    def oriented_bfs(self):
               
        
        self.append_entry(host=self.root,hub=self.root_index,path=(self.root,),peso=0)
        prioQ = []
        paths = set()
        listA = {}
        pruned = set()
        target_index = self.root_index+1
        
        while target_index<self.graph.numberOfNodes():
            
            target = self.reverse_ordering[target_index]
            try:
                wgt, pgt, hgt = self.bidir_BFS(source=self.root,target=target, lowest_index=self.root_index, ignore_nodes=None, ignore_edges=None)
            except NoPathException:
                target_index+=1
                continue
            
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise
            
            
            assert wgt == len(pgt)-1          
            assert type(pgt)==tuple
            assert pgt[0]==self.root
            assert pgt[-1]==target
            assert (wgt,pgt,hgt) not in prioQ
            assert target not in listA
            assert pgt not in paths
            assert pgt[-1] not in pruned
            heappush(prioQ, (wgt,pgt,hgt))
            paths.add(pgt)

            target_index+=1
            continue
        
        
        while len(prioQ)>0:
            
            wgt, ptx, htx  = heappop(prioQ)
            paths.remove(ptx)

            vtx = ptx[-1]
            if len(pruned)+self.root_index+1==self.graph.numberOfNodes():
                break
            if vtx in pruned:
                self.prunings+=1
                continue
            
            assert self.is_simple(ptx)

            assert self.ordering[vtx]>self.root_index
            
            if vtx not in listA:
               listA[vtx]=[ptx]
            else:
               assert ptx not in listA[vtx]
               listA[vtx].append(ptx)
            
            
            if self.k_shorter_simple_paths(self.root, vtx, wgt):
                self.prunings+=1
                pruned.add(vtx)
                continue
            assert not self.path_is_encoded(self.root, vtx, ptx, wgt)

            # if htx == self.root_index:
            #     p_1,i_1 = self.is_path_in_label_of_host(host=vtx, hub=htx, path=ptx,peso=wgt)
            #     if not p_1:
            #         assert not self.path_is_encoded(self.root, vtx, ptx, wgt)
            self.append_entry(host=vtx,hub=htx,path=ptx,peso=wgt)
                    # self.labeling_size+=1
                
            #     
            # else:
            #     assert htx < self.root_index
            #     L,w_L,R,w_R = self.break_path_at(path=ptx, hub_vertex=htx,peso=wgt)
            #     assert L[0]==self.reverse_ordering[htx]
            #     assert L[-1]==self.root
            #     assert R[0]==self.reverse_ordering[htx]
            #     assert R[-1]==vtx
            #     p_1,i_1 = self.is_path_in_label_of_host(host=self.root, hub=htx, path=L,peso=w_L)
            #     p_2,i_2 = self.is_path_in_label_of_host(host=vtx,hub=htx,path=R,peso=w_R)
                
                
            #     if not p_1:
            #         if i_1 == -1:
            #             self.labels[self.root][htx] = [(L,w_L)]
            #         else:
            #             self.labels[self.root][htx].insert(i_1,(L,w_L))
            #         self.labeling_size+=1
            #     if not p_2:
            #         if i_2 == -1:
            #             self.labels[vtx][htx] = [(R,w_R)]
            #         else:
            #             self.labels[vtx][htx].insert(i_2,(R,w_R))
    
            #         self.labeling_size+=1

            assert self.path_is_encoded(self.root, vtx, ptx, wgt)
            assert vtx in listA
            assert len(listA[vtx])>0

            ignore_nodes = set()
            ignore_edges = set()
            assert ptx==listA[vtx][-1]
            for index in range(1, len(ptx)):
                root = ptx[:index]
                assert type(root)==tuple
                for path in listA[vtx]:
                    assert type(path)==tuple
                    if path[:index] == root:
                        ignore_edges.add((path[index - 1], path[index]))
                try:
                    
                    wgt, pgt, hgt = self.bidir_BFS(root[-1],vtx,lowest_index=self.root_index, ignore_nodes=ignore_nodes, ignore_edges=ignore_edges)
                    
                    new_path = root[:-1] + pgt
                    assert self.is_simple(new_path)

                    if new_path in paths:
                        ignore_nodes.add(root[-1])
                        continue
                    assert new_path[0]==self.root and new_path[-1]==vtx
                    
                    new_weight = len(new_path)-1
                    sp_h = sys.maxsize if len(root[:-1])<=0 else min([self.ordering[i] for i in root[:-1]])
                    new_hub = min(hgt,sp_h)
                    
                    assert new_hub == min([self.ordering[i] for i in new_path])
                    # assert (new_weight,,new_hub) not in prioQ
                    assert new_path not in paths
                    heappush(prioQ, (new_weight,new_path,new_hub))
                    paths.add(new_path)


                except NoPathException:
                    pass
                except Exception as err:
                    print(f"Unexpected {err=}, {type(err)=}")
                    raise
                    
                ignore_nodes.add(root[-1])
            del ignore_nodes    
            del ignore_edges
        prioQ.clear()
        del prioQ
        del listA
        del paths
        del pruned
    
    def final_repair(self):
               
        

        prioQ = []
        paths = set()
        listA = {}
        pruned = set()
        target_index = self.root_index+1
        assert self.graph.degree(self.root)>1
        
        while target_index<self.graph.numberOfNodes():
            
            target = self.reverse_ordering[target_index]
            try:
                wgt, pgt, hgt = self.bidir_BFS(source=self.root,target=target, lowest_index=-1, ignore_nodes=None, ignore_edges=None)
            except NoPathException:
                target_index+=1
                continue
            
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise
            
            
            assert wgt == len(pgt)-1          
            assert type(pgt)==tuple
            assert pgt[0]==self.root
            assert pgt[-1]==target
            assert (wgt,pgt,hgt) not in prioQ
            assert target not in listA
            assert pgt not in paths
            assert pgt[-1] not in pruned
            heappush(prioQ, (wgt,pgt,hgt))
            paths.add(pgt)
    
            target_index+=1
            continue
        
        
        while len(prioQ)>0:
            
            wgt, ptx, htx  = heappop(prioQ)
            paths.remove(ptx)
    
            vtx = ptx[-1]
            
            if len(pruned)+self.root_index+1==self.graph.numberOfNodes():
                break
            
            if vtx in pruned:
                self.prunings+=1
                continue
            
            assert self.is_simple(ptx)
    
            assert self.ordering[vtx]>self.root_index
            
            if vtx not in listA:
               listA[vtx]=[ptx]
            else:
               listA[vtx].append(ptx)

            
            if self.k_shorter_simple_paths(self.root, vtx, wgt):
                self.prunings+=1
                pruned.add(vtx)
                continue

            
            assert htx<self.root_index or self.path_is_encoded(self.root, vtx, ptx, wgt)
                
            if htx<self.root_index:
            
                assert htx <= self.root_index
                L,w_L,R,w_R = self.break_path_at(path=ptx, hub_vertex=htx,peso=wgt)
                assert L[0]==self.reverse_ordering[htx]
                assert L[-1]==self.root
                assert R[0]==self.reverse_ordering[htx]
                assert R[-1]==vtx
                p_1,i_1 = self.is_path_in_label_of_host(host=self.root, hub=htx, path=L,peso=w_L)
                p_2,i_2 = self.is_path_in_label_of_host(host=vtx,hub=htx,path=R,peso=w_R)
                
                
                if not p_1:
                    if i_1 == -1:
                        self.labels[self.root][htx] = [(L,w_L)]
                    else:
                        self.labels[self.root][htx].insert(i_1,(L,w_L))
                    self.labeling_size+=1
                if not p_2:
                    if i_2 == -1:
                        self.labels[vtx][htx] = [(R,w_R)]
                    else:
                        self.labels[vtx][htx].insert(i_2,(R,w_R))    
                    self.labeling_size+=1
                

            assert self.path_is_encoded(self.root, vtx, ptx, wgt)
            assert vtx in listA
            assert len(listA[vtx])>0
    
            ignore_nodes = set()
            ignore_edges = set()
            assert ptx==listA[vtx][-1]
            for index in range(1, len(ptx)):
                root = ptx[:index]
                assert type(root)==tuple
                for path in listA[vtx]:
                    assert type(path)==tuple
                    if path[:index] == root:
                        ignore_edges.add((path[index - 1], path[index]))
                try:
                    if self.graph.degree(root[-1])>2 or (self.graph.degree(root[-1])==2 and root[-1] not in self.art):
                        
                        wgt, pgt, hgt = self.bidir_BFS(root[-1], vtx, lowest_index=-1, ignore_nodes=ignore_nodes, ignore_edges=ignore_edges)
                        
                        new_path = root[:-1] + pgt
                        assert self.is_simple(new_path)
        
                        if new_path in paths:
                            ignore_nodes.add(root[-1])
                            continue
                        assert new_path[0]==self.root and new_path[-1]==vtx
                        
                        new_weight = len(new_path)-1
                        
                        sp_h = sys.maxsize if len(root[:-1])<=0 else min([self.ordering[i] for i in root[:-1]])
                        
                        assert sp_h != hgt
                        
                        new_hub = min(hgt,sp_h)
                        
                        assert new_hub == min([self.ordering[i] for i in new_path])
                        assert new_hub <= self.root_index
                        assert new_path not in paths
                        
                        heappush(prioQ, (new_weight,new_path,new_hub))
                        paths.add(new_path)
                    # else:
                    #     print("su")
    
                except NoPathException:
                    pass
                except Exception as err:
                    print(f"Unexpected {err=}, {type(err)=}")
                    raise
                    
                ignore_nodes.add(root[-1])
            del ignore_nodes    
            del ignore_edges
        del prioQ
        del listA
        del paths
        del pruned

            
            
            
        

        
     
        
        
       

                    

                
        
        


    """    
    def hub_get_next(self,vertice,ref_vertice):
        
                
        assert self.prev_path[ref_vertice] is not None


        self.reset_ignored()
        self.ignore_edges = set()          

        self.indice = 1
        lunghezza = len(self.prev_path[ref_vertice])
        
        while self.indice<lunghezza:

            self.yen_root = []
            self.aux_indice = 0
            self.hub_of_sub_path = sys.maxsize 

            while self.aux_indice<self.indice:
                self.yen_root.append(self.prev_path[ref_vertice][self.aux_indice])
                self.hub_of_sub_path = min(self.ordering[self.prev_path[ref_vertice][self.aux_indice]],self.hub_of_sub_path)
                self.aux_indice+=1

            assert all(self.yen_root[i]==self.prev_path[ref_vertice][i] for i in range(self.indice ))# == (self.prev_path)[:indice]
            assert self.length_func(self.yen_root)==self.aux_indice
            
            self.root_length = self.aux_indice                       
            
            for path in self.listA[ref_vertice]:
                self.equality = True
                self.aux_idx = 0
                while self.aux_idx<min(self.indice ,len(path)):
                    if self.yen_root[self.aux_idx]!=path[self.aux_idx]:
                        self.equality = False
                        break
                    self.aux_idx+=1
                    
                if self.equality:
                    assert all(self.yen_root[i]==path[i] for i in range(self.indice ))
                    self.ignore_edges.add((path[self.indice  - 1], path[self.indice ]))
                    continue

                assert any(self.yen_root[i]!=path[i] for i in range(self.indice))
                    
            self.indice +=1
            
            if self.yen_root[-1] in self.settled:
                self.avoid_nodes[self.yen_root[-1]]=True
                self.reset_nodes.append(self.yen_root[-1])
                continue
            
            try:
                length, spur, hub_of_spur = self.shortest_path_func(self.yen_root[-1],vertice)
                self.aux_indice = len(self.yen_root)-2
                
                spur = tuple(self.yen_root[0:len(self.yen_root)-1])+spur
                
                self.listB[ref_vertice].push(self.root_length + length-1, spur, min(self.hub_of_sub_path,hub_of_spur))
                
            except NoPathException:
                pass
            except Exception as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise
            assert not self.avoid_nodes[self.yen_root[-1]]
            
            self.avoid_nodes[self.yen_root[-1]]=True
            self.reset_nodes.append(self.yen_root[-1])


        if self.listB[ref_vertice]:
            
            wgt, pgt, hgt = self.listB[ref_vertice].pop()
            
            assert wgt==len(pgt)-1
            assert ref_vertice in self.touched
            

            self.listA[ref_vertice].append(pgt)
            self.prev_path[ref_vertice] = pgt
            return (vertice, pgt, wgt, hgt)  
        
        return (None,None,None,None)   
    
    
    
                
                
    
    def drop(self):
        try:
            # self.global_lengths.pop() 
            self.listB.pop()
            self.listA.pop()
            self.prev_path.pop()
            # self.pruned.pop()
            
            # assert len(self.global_lengths) == self.graph.numberOfNodes()-self.root_index-2
            assert len(self.listB) == self.graph.numberOfNodes()-self.root_index-2
            assert len(self.listA) == self.graph.numberOfNodes()-self.root_index-2
            assert len(self.prev_path) == self.graph.numberOfNodes()-self.root_index-2
            # assert len(self.pruned) == self.graph.numberOfNodes()-self.root_index-2
        except IndexError:
            pass

    def clean_yen_data_structures(self):
        resetter = 0
        lunghezza = len(self.touched)

        while resetter<lunghezza:
            ref = self.touched[resetter]

            self.listB[ref] = PathBuffer()
            self.listA[ref] = []
            self.prev_path[ref] = None
            # self.pruned[ref] = False
            resetter+=1
            

            
    def reset(self):
        for ref in self.touched:

            self.listB[ref] = PathBuffer()
            self.listA[ref] = []
            self.prev_path[ref] = None
            # self.pruned[ref] = False
            # self.global_lengths[ref] = []

            
        self.touched = []
    
        self.drop()
    """
    

    
    """
    def _bidirectional_shortest_path(self, source, target):
        
        
        pred, succ, w = self._bidirectional_pred_succ(source, target)
        path = deque()
        h, d = sys.maxsize, -1
        # indice = 0
        
        # from w to target
        # indice_h = 0
        
        while w is not None:
            path.append(w)
            if self.ordering[w]<h:
                h = self.ordering[w]
                # indice_h = indice
            # indice += 1
            w = succ[w]
            d+=1
            
        # assert self.ordering[path[indice_h]]==h
        
        
        # len_first = d

        # indice = -1

        w = pred[path[0]]
        while w is not None:
            path.appendleft(w)
            if self.ordering[w]<h:
                h = self.ordering[w]
                # indice_h = indice
            # indice -= 1
            w = pred[w]
            d+=1
            
        # indice_h = d - len_first + indice_h


        assert d==len(path)-1
        assert min(self.ordering[i] for i in path)==h
        # assert self.ordering[path[indice_h]]==h

        return d, tuple(path), h
    
    def _bidirectional_pred_succ(self, source, target):


        if self.avoid_nodes[source] or self.avoid_nodes[target]:
            raise NoPathException
        if target == source:
            return ({target: None}, {source: None}, source)


        Gpred = self.graph.iterNeighbors
        Gsucc = self.graph.iterNeighbors



        # predecesssor and successors in search
        pred = {source: None}
        succ = {target: None}

        # initialize fringes, start with forward
        forward_fringe = deque([source])
        reverse_fringe = deque([target])

        while forward_fringe and reverse_fringe:
            if len(forward_fringe) <= len(reverse_fringe):
                this_level = forward_fringe
                forward_fringe = deque()
                for v in this_level:
                    for w in Gsucc(v):
                        if (v, w) in self.ignore_edges or (w, v) in self.ignore_edges:
                            continue
                        if self.ordering[w]<=self.lowest_index or self.avoid_nodes[w]:
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
                    for w in Gpred(v):
                        if (v, w) in self.ignore_edges or (w, v) in self.ignore_edges:
                            continue                            
                        if self.ordering[w]<=self.lowest_index or self.avoid_nodes[w]:
                            assert w not in succ
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
    """
    def draw_subgraph_rooted_at(self,source:int,target:int,Yres:list,Lres:list,indice:int=None):
        # import networkx as netwx

        assert type(Yres)==list
        assert type(Lres)==list
        nodes = set()
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        bfs = nk.distance.BFS(self.graph, source, storePaths=False, storeNodesSortedByDistance = True)
        bfs.run()
        
        dist_vector = bfs.getNodesSortedByDistance()
        
        for i in range(len(dist_vector)):
            assert dist_vector[i]<sys.maxsize
            assert dist_vector[i]<float("inf")
            nodes.add(i)     
            
        assert source in nodes
        

        subgraph = nk.graphtools.subgraphFromNodes(self.graph, nodes)

        nxG = nk.nxadapter.nk2nx(subgraph)
        pos = netwx.kamada_kawai_layout(nxG)
        
        
        
        
        if indice is not None:
            for path in [Yres[indice], Lres[indice]]:
                fig, ax = plt.subplots(figsize=(10,10))
                colori = ['orange' for _ in nodes]
                c=0
                for i in nodes:
                    if i in path:
                        colori[c]='#029386'
                    if i in [source,target]:
                        colori[c]='#c20078'    
                    c+=1
                    
                netwx.draw(nxG,pos, node_color=colori,node_size=800,font_size=14)
                netwx.draw_networkx_labels(nxG,pos,font_family="sans-serif",font_size=14)
                ax.set_axis_off()
                fig.tight_layout()
                plt.show()
                
        else:
            
            # print([mcolors.to_rgb(color) for color in mcolors.XKCD_COLORS])
            
            for cammino,colore in zip(Yres,mcolors.XKCD_COLORS):
                fig, ax = plt.subplots(figsize=(10,10))
                colori = ['orange' for _ in nodes]
                c=0
                for i in nodes:
                    if i in cammino:
                        colori[c]=colore
                    if i in [source,target]:
                        colori[c]='gray' 
                    c+=1
                netwx.draw(nxG,pos, node_color=colori,node_size=800,font_size=14)
                netwx.draw_networkx_labels(nxG,pos,font_family="sans-serif",font_size=14)
                ax.set_axis_off()
                fig.tight_layout()
                plt.show()
            for cammino,colore in zip(Lres,mcolors.XKCD_COLORS):
                fig, ax = plt.subplots(figsize=(10,10))
                colori = ['orange' for _ in nodes]
                c=0
                for i in nodes:
                    if i in cammino:
                        colori[c]=colore
                    if i in [source,target]:
                        colori[c]='gray' 
                    c+=1
                netwx.draw(nxG,pos, node_color=colori,node_size=800,font_size=14)
                netwx.draw_networkx_labels(nxG,pos,font_family="sans-serif",font_size=14)
                ax.set_axis_off()
                fig.tight_layout()
                plt.show()
        return nodes
                
            
    def draw_graph(self):
       

        import matplotlib.pyplot as plt
        # import networkx as netwx
        nxG = nk.nxadapter.nk2nx(self.graph)

        pos = netwx.kamada_kawai_layout(nxG)
        
        colori = ['#029386' for _ in range(nxG.number_of_nodes())]  #'#029386'
        fig, ax = plt.subplots(figsize=(10,10))
        netwx.draw(nxG,pos, node_color=colori,node_size=800,font_size=14)
        netwx.draw_networkx_labels(nxG,pos,font_family="sans-serif",font_size=14)
        ax.set_axis_off()
        fig.tight_layout()
        plt.show()
         
        
    
    
    def combination_or_intersect_without_allocation(self,d_u,d_v,p_u,p_v,h):
        
        assert p_u[0]==self.reverse_ordering[h]
        assert p_v[0]==self.reverse_ordering[h]       
        assert p_u[0]==p_v[0]

            
        assert self.is_simple(p_u)
        assert self.is_simple(p_v)
        
        assert type(p_u) == tuple
        assert type(p_v) == tuple
        assert all(self.intersection_tests[i]==False for i in self.graph.iterNodes())

        if d_u+d_v>self.graph.numberOfNodes()-1:
            if __debug__:
                resulting_path = tuple(reversed(p_u[1:]))+p_v
                assert resulting_path[0]==p_u[-1] and resulting_path[-1]==p_v[-1]                     

                assert len(resulting_path)-1>self.graph.numberOfNodes()-1    
                assert len(resulting_path)-1==d_u+d_v
                assert not self.is_simple(resulting_path)

            return None, True #Intersection


        idx = d_u
        while idx>=1:
            assert not self.intersection_tests[p_u[idx]]
            self.intersection_tests[p_u[idx]]=True
            idx-=1
            
        assert not self.intersection_tests[self.reverse_ordering[h]]

        idx = 1
        while idx<d_v+1:
            v = p_v[idx]
            if self.intersection_tests[v]: #Intersezione
                for x in p_u: #reset for next try
                    self.intersection_tests[x]=False

                assert not self.is_simple(tuple(reversed(p_u[1:]))+p_v)

                return None, True #Intersection
            #v cannot be in p_v, since it is simple by design here
            idx+=1


        if __debug__:
            resulting_path = tuple(reversed(p_u[1:]))+p_v
            assert resulting_path[0]==p_u[-1]
            assert resulting_path[-1]==p_v[-1]
            assert self.is_simple(resulting_path)
            assert len(resulting_path)-1==d_u+d_v
            
           
        for x in p_u: #reset for next try
            self.intersection_tests[x]=False
        assert all(self.intersection_tests[x]==False for x in p_v) #reset for next try

        return d_u+d_v, False

    def combination_or_intersect(self,d_u,d_v,p_u,p_v,h):
        assert type(p_u) == tuple
        assert type(p_v) == tuple
        assert p_u[0]==self.reverse_ordering[h]
        assert p_v[0]==self.reverse_ordering[h]       
 

        assert self.is_simple(p_u)
        assert self.is_simple(p_v)
        

        assert all(self.intersection_tests[i]==False for i in self.graph.iterNodes())

        if d_u+d_v>self.graph.numberOfNodes()-1:
            if __debug__:
                resulting_path = tuple(reversed(p_u[1:]))+p_v
                assert resulting_path[0]==p_u[-1] and resulting_path[-1]==p_v[-1]                     

                assert len(resulting_path)-1>self.graph.numberOfNodes()-1    
                assert len(resulting_path)-1==d_u+d_v
                assert not self.is_simple(resulting_path)
                del resulting_path
            return None, None, True #Intersection


        idx = d_u
        resulting_path = tuple()
        while idx>=1:
            assert not self.intersection_tests[p_u[idx]]
            self.intersection_tests[p_u[idx]]=True
            resulting_path = resulting_path + (p_u[idx],)
            idx-=1
        
        assert p_u[idx]==self.reverse_ordering[h]
        
        resulting_path = resulting_path + (p_u[0],)
        
        assert not self.intersection_tests[self.reverse_ordering[h]]
        idx = 1
        while idx<d_v+1:
            v = p_v[idx]


            if self.intersection_tests[v]: #Intersezione
                for x in p_u: #reset for next try
                    self.intersection_tests[x]=False

                assert not self.is_simple(tuple(reversed(p_u[1:]))+p_v)
                return None, None, True #Intersection
            #v cannot be in p_v, since it is simple by design here
            resulting_path = resulting_path + (v,)

            idx+=1

        assert resulting_path == tuple(reversed(p_u[1:]))+p_v


        assert resulting_path[0]==p_u[-1]
        assert resulting_path[-1]==p_v[-1]

        for x in resulting_path: #reset for next try
            self.intersection_tests[x]=False
            
        assert all(self.intersection_tests[x]==False for x in p_v) #reset for next try

        weight = d_u+d_v

        assert self.is_simple(resulting_path)

        assert weight == len(resulting_path)-1

        return weight, resulting_path, False

    
    def sp_query(self, u: int, v: int):
        

        shortest_pair = (None, None, None)

        assert u!=v
        

        
        if len(self.labels[u])==0 or len(self.labels[v])==0:
            return 
        maxhub = min(self.labels[u].keys()[-1],self.labels[v].keys()[-1])
        it_u = iter(self.labels[u].keys())
        it_v = iter(self.labels[v].keys())
        el_u = next(it_u,-1)
        el_v = next(it_v,-1)
        while True:
            if el_u==-1 or el_v==-1 or el_u>maxhub or el_v>maxhub:
                break
            if el_u<el_v:
                el_u = next(it_u,-1)
                continue
            if el_u>el_v:
                el_v = next(it_v,-1)
                continue

            assert el_u==el_v and el_u!=-1
            hub = el_u
            el_u = next(it_u,-1)
            el_v = next(it_v,-1)
        
            assert hub in self.labels[u] and hub in self.labels[v]

            P_u = self.labels[u][hub]
            assert len(P_u)>0
            
            P_v = self.labels[v][hub]
            assert len(P_v)>0

            
            for p_u,d_u in P_u:
                
                assert d_u==len(p_u)-1
                
                if shortest_pair[0] is not None and d_u>=shortest_pair[0]:
                    break
                
                
                for p_v,d_v in P_v:
                    
                    assert d_v==len(p_v)-1    
                    
                    assert p_u[0]==p_v[0]
                    assert p_u[0]==self.reverse_ordering[hub]
                    assert p_v[0]==self.reverse_ordering[hub]

                    if shortest_pair[0] is not None and d_u+d_v>=shortest_pair[0]:
                        break
                    
                    w_pt, pt, intersection_test_outcome = self.combination_or_intersect(d_u,d_v,p_u, p_v,hub)
                    
                    if intersection_test_outcome:
                        assert len(set(p_u).intersection(set(p_v)))>1
                        assert pt is None
                        assert not self.is_simple(self.combination_without_intersection_test(d_u,d_v,p_u,p_v)[1])      
                        break
                        
                    assert w_pt==d_u+d_v
                    assert set(p_u).intersection(set(p_v))=={self.reverse_ordering[hub]}
                    assert pt is not None
                    assert not intersection_test_outcome
                    assert self.is_simple(pt)
                    
                    if shortest_pair[0] is None or w_pt<shortest_pair[0]:
                        shortest_pair = (w_pt, pt, hub)
                        break
                break
            return shortest_pair
                
              
                
           
    def top_k_paths(self, u: int, v: int, break_at_k:bool=True):
        
        self.results.clear()


        if u == v:
            self.add_to_results((0,(u,),self.ordering[u]))
            return
                

        
        if len(self.labels[u])==0 or len(self.labels[v])==0:
            return 
        maxhub = min(self.labels[u].keys()[-1],self.labels[v].keys()[-1])
        it_u = iter(self.labels[u].keys())
        it_v = iter(self.labels[v].keys())
        el_u = next(it_u,-1)
        el_v = next(it_v,-1)
        while True:
            if el_u==-1 or el_v==-1 or el_u>maxhub or el_v>maxhub:
                break
            if el_u<el_v:
                el_u = next(it_u,-1)
                continue
            if el_u>el_v:
                el_v = next(it_v,-1)
                continue

            assert el_u==el_v
            assert el_u!=-1
            hub = el_u
            el_u = next(it_u,-1)
            el_v = next(it_v,-1)
            assert hub in self.labels[u] and hub in self.labels[v]
            P_u = self.labels[u][hub]
            assert type(P_u)==list #sortedcontainers.sortedlist.SortedKeyList  

            assert len(P_u)>0
            P_v = self.labels[v][hub]
            assert type(P_v)==list #sortedcontainers.sortedlist.SortedKeyList  
            assert len(P_v)>0

            
            
            
            for p_u,d_u in P_u:
                
                
                assert d_u == len(p_u)-1
            

                if break_at_k and len(self.results)>=self.num_k and d_u >= self.results[self.num_k-1][0]:
                    assert len(self.results)==self.num_k
                    break
                if v!=self.reverse_ordering[hub] and v in p_u:
                    continue
                for p_v,d_v in P_v:
                    


                    assert d_v == len(p_v)-1

                    assert p_u[0]==p_v[0]
                    assert p_u[0]==self.reverse_ordering[hub]
                    assert p_v[0]==self.reverse_ordering[hub]

                    if break_at_k and len(self.results)>=self.num_k and d_u+d_v >= self.results[self.num_k-1][0]:
                        assert len(self.results)==self.num_k
                        break

                        
                    assert not break_at_k or len(self.results)<self.num_k or (d_u+d_v<self.results[self.num_k-1][0])

                    w_pt, pt, intersection_test_outcome = self.combination_or_intersect(d_u, d_v, p_u, p_v, hub)
                    
                    if intersection_test_outcome:
                        assert len(set(p_u).intersection(set(p_v)))>1
                        assert pt is None
                        assert not self.is_simple(self.combination_without_intersection_test(d_u,d_v,p_u,p_v)[1])      
                        continue
                        
                    
                    assert set(p_u).intersection(set(p_v))=={self.reverse_ordering[hub]}
                    assert pt is not None
                    assert not intersection_test_outcome
                    assert self.is_simple(pt)
                    assert type(pt)==tuple
                    
                    self.add_to_results((w_pt, pt, hub))


                    if break_at_k and len(self.results)>self.num_k:
                        el = self.results.pop()
                        assert len(self.results)==self.num_k
                        assert el[0]>=self.results[self.num_k-1][0]
   
    def top_k_paths_through(self, u: int, v: int, hub:int, break_at_k:bool=True):
        
        self.results.clear()
        
        assert u!=v

        if hub not in self.labels[u] or hub not in self.labels[v]:
            return
        

        P_u = self.labels[u][hub]
        assert type(P_u)==list #sortedcontainers.sortedlist.SortedKeyList  

        assert len(P_u)>0
        P_v = self.labels[v][hub]
        assert type(P_v)==list #sortedcontainers.sortedlist.SortedKeyList  

        assert len(P_v)>0

        
        
        
        for p_u,d_u in P_u:
            
            
            assert d_u == len(p_u)-1
        

            if break_at_k and len(self.results)>=self.num_k and d_u >= self.results[self.num_k-1][0]:
                assert len(self.results)==self.num_k
                break
            if v!=self.reverse_ordering[hub] and v in p_u:
                continue
            for p_v,d_v in P_v:
                


                assert d_v == len(p_v)-1

                assert p_u[0]==p_v[0]
                assert p_u[0]==self.reverse_ordering[hub]
                assert p_v[0]==self.reverse_ordering[hub]

                if break_at_k and len(self.results)>=self.num_k and d_u+d_v >= self.results[self.num_k-1][0]:
                    assert len(self.results)==self.num_k
                    break

                    
                assert not break_at_k or len(self.results)<self.num_k or (d_u+d_v<self.results[self.num_k-1][0])

                w_pt, pt, intersection_test_outcome = self.combination_or_intersect(d_u, d_v, p_u, p_v, hub)
                
                if intersection_test_outcome:
                    assert len(set(p_u).intersection(set(p_v)))>1
                    assert pt is None
                    assert not self.is_simple(self.combination_without_intersection_test(d_u,d_v,p_u,p_v)[1])      
                    continue
                    
                
                assert set(p_u).intersection(set(p_v))=={self.reverse_ordering[hub]}
                assert pt is not None
                assert not intersection_test_outcome
                assert self.is_simple(pt)
                assert type(pt)==tuple
                
                self.add_to_results((w_pt, pt,hub))


                if break_at_k and len(self.results)>self.num_k:
                    el = self.results.pop()
                    assert len(self.results)==self.num_k
                    assert el[0]>=self.results[self.num_k-1][0]                    
                
    def filtered_query(self, u: int, v: int, w:int):
          
        self.results.clear()


        if u == v:
            self.add_to_results((0,(u,),self.ordering[u]))
            return

        if len(self.labels[u])==0 or len(self.labels[v])==0:
            return 
        maxhub = min(self.labels[u].keys()[-1],self.labels[v].keys()[-1])
        it_u = iter(self.labels[u].keys())
        it_v = iter(self.labels[v].keys())
        el_u = next(it_u,-1)
        el_v = next(it_v,-1)
        while True:
            if el_u==-1 or el_v==-1 or el_u>maxhub or el_v>maxhub:
                break
            if el_u<el_v:
                el_u = next(it_u,-1)
                continue
            if el_u>el_v:
                el_v = next(it_v,-1)
                continue
            
            assert el_u==el_v and  el_u!=-1
            hub = el_u
            el_u = next(it_u,-1)
            el_v = next(it_v,-1)
            assert hub in self.labels[u] and hub in self.labels[v]
            P_u = self.labels[u][hub]
            assert type(P_u)==list and len(P_u)>0

            
            P_v = self.labels[v][hub]
            assert type(P_v)==list and len(P_v)>0
            

            for p_u,d_u in P_u:
                
                assert d_u == len(p_u)-1
            
                if d_u>=w:
                    break
                if v!=self.reverse_ordering[hub] and v in p_u:
                    continue
                for p_v,d_v in P_v:
                    
                    assert d_v == len(p_v)-1

                    assert p_u[0]==p_v[0]
                    assert p_u[0]==self.reverse_ordering[hub]
                    assert p_v[0]==self.reverse_ordering[hub]

                    if d_u+d_v>w:
                        break
                        

                    w_pt, pt, intersection_test_outcome = self.combination_or_intersect(d_u, d_v, p_u, p_v, hub)
                    
                    if intersection_test_outcome:
                        assert len(set(p_u).intersection(set(p_v)))>1
                        assert pt is None
                        assert not self.is_simple(self.combination_without_intersection_test(d_u,d_v,p_u,p_v)[1])      
                        continue
                        
                    
                    assert set(p_u).intersection(set(p_v))=={self.reverse_ordering[hub]}
                    assert pt is not None
                    assert not intersection_test_outcome
                    assert self.is_simple(pt)
                    assert type(pt)==tuple
                    assert w_pt<=w
                    
                    self.add_to_results((w_pt, pt,hub))

                                        
    def stand_alone_top_k_paths(self, u: int, v: int, break_at_k:bool=True):
        

        container = []
        
        if u == v:
            self.add_to_external(container, (0, [u],self.ordering[u]))
            return container
        

        
        if len(self.labels[u])==0 or len(self.labels[v])==0:
            return container
        maxhub = min(self.labels[u].keys()[-1],self.labels[v].keys()[-1])
        it_u = iter(self.labels[u].keys())
        it_v = iter(self.labels[v].keys())
        el_u = next(it_u,-1)
        el_v = next(it_v,-1)
        while True:
            if el_u==-1 or el_v==-1 or el_u>maxhub or el_v>maxhub:
                break
            if el_u<el_v:
                el_u = next(it_u,-1)
                continue
            if el_u>el_v:
                el_v = next(it_v,-1)
                continue
            assert el_u==el_v
            assert el_u!=-1
            hub = el_u
            el_u = next(it_u,-1)
            el_v = next(it_v,-1)
            assert hub in self.labels[u] and hub in self.labels[v]
            P_u = self.labels[u][hub]
            assert len(P_u)>0
            P_v = self.labels[v][hub]
            assert len(P_v)>0

            for p_u,d_u in P_u:
                
                
                
                assert d_u == len(p_u)-1

                
                if break_at_k and len(container)>=self.num_k and d_u >= container[self.num_k-1][0]:
                    assert len(container)==self.num_k
                    break
                if v!=self.reverse_ordering[hub] and v in p_u:
                    continue
                for p_v,d_v in P_v:
                    
                    assert d_v == len(p_v)-1
                    
                    
                    assert p_u[0]==p_v[0]
                    assert p_u[0]==self.reverse_ordering[hub]
                    assert p_v[0]==self.reverse_ordering[hub]

                    if break_at_k and len(container)>=self.num_k and d_u+d_v >= container[self.num_k-1][0]:
                        assert len(container)==self.num_k
                        break
                    

                    assert not break_at_k or len(container)<self.num_k or (d_u+d_v<container[self.num_k-1][0])

                    w_pt, pt, intersection_test_outcome = self.combination_or_intersect(d_u, d_v, p_u, p_v, hub)
                    
                    if intersection_test_outcome:
                        assert len(set(p_u).intersection(set(p_v)))>1
                        assert pt is None
                        assert not self.is_simple(self.combination_without_intersection_test(d_u,d_v,p_u,p_v)[1])         
                        continue
                        
                    
                    assert set(p_u).intersection(set(p_v))=={self.reverse_ordering[hub]}
                    assert pt is not None
                    assert not intersection_test_outcome
                    assert self.is_simple(pt)
                    
                    
                    self.add_to_external(container,(w_pt,pt,hub))
                    
                    if break_at_k and len(container)>self.num_k:
                        el = container.pop()
                        assert len(container)==self.num_k
                        assert el[0]>=container[self.num_k-1][0]
                             

        
        return container


    def build(self):

        print("Construction")
        cpu4 = time.perf_counter()    
      
        
        self.art = set(netwx.articulation_points(self.aux_graph))
        print("Number Articulation Points",len(self.art))
        # print(self.art)
        bar = IncrementalBar('Iterations:', max = self.graph.numberOfNodes())
  
        for node in self.reverse_ordering:
                        
            self.root_index = self.ordering[node]
            self.root = node
            
            

            self.oriented_bfs()              
            
            if __debug__:
                print("visit root:",node,"label entries:",self.labeling_size,"prunings:",self.prunings)
            bar.next()
            
        bar.finish()
        bar = IncrementalBar('Repairing Iterations:', max = self.graph.numberOfNodes()-1)
        # bar.next()

        for node in self.reverse_ordering[1:]:
                        
            self.root_index = self.ordering[node]
            self.root = node
            
            if __debug__:
                if self.graph.degree(self.root)==1:
                    assert self.root not in self.art

            if self.graph.degree(self.root)>2 or (self.graph.degree(self.root)==2 and self.root not in self.art):
                self.final_repair()              
            
            if __debug__:
                print("repair root:",node,"label entries:",self.labeling_size,"prunings:",self.prunings)
            bar.next()
            
        bar.finish()
       



        assert all(list(self.labels[node].keys()) == sorted(self.labels[node].keys())  for node in self.reverse_ordering)
        assert all(list(self.labels[node][h]) == sorted(self.labels[node][h],key=lambda x:x[1])  for node in self.reverse_ordering for h in self.labels[node].keys())

        cpu5 = time.perf_counter()

        self.constrTime = round((cpu5-cpu4),5)
        print("Elapsed construction time:",self.constrTime,"s")
        print("Labeling size:",self.getLabelingSize(),"Total prunings:",self.prunings)#,"Total saved yens (if bridges):", self.saved_yens_executions)
        print("Labeling space:",self.getLabelingSpace())

                                 
        print("Avg entries per vertex:",self.getAvgLabelingSize())
        print("Med entries per vertex:",self.getMedianLabelingSize())
        print("Max entries per vertex:",self.getMaxLabelingSize())

        print("Avg same hub entries per vertex:",self.getAvgSameHubLabels())
        print("Med same hub entries per vertex:",self.getMedianSameHubLabels())
        print("Max same hub entries per vertex:",self.getMaxSameHubLabels())
        if self.graph.numberOfNodes()<=50:
            self.print_labeling()
            
    def print_labeling(self):
        for v in self.graph.iterNodes():
            print("Vertex:",v,[(self.reverse_ordering[k],[list(i) for i in self.labels[v][k]]) for k in self.labels[v].keys()])
      



    

    

    
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
        
    
    
    def combination_without_intersection_test(self,d_u,d_v,p_u,p_v):
        
        assert type(p_u) == tuple
        assert type(p_v) == tuple
        
        assert p_u[0]==p_v[0]
        
        
        resulting_path = tuple(reversed(p_u[1:]))+p_v
        
        assert resulting_path[0]==p_u[-1] and resulting_path[-1]==p_v[-1]
        
        weight = d_u+d_v
        
        assert weight == len(resulting_path)-1
        
        
        return weight, resulting_path

            
    def top_k_lengths(self, u: int, v: int, break_at_k:bool=True):
            

        assert u!=v

        
        
        self.lengths = [] #SortedList()

        
        if len(self.labels[u])==0 or len(self.labels[v])==0:
            return 
        maxhub = min(self.labels[u].keys()[-1],self.labels[v].keys()[-1])
        it_u = iter(self.labels[u].keys())
        it_v = iter(self.labels[v].keys())
        el_u = next(it_u,-1)
        el_v = next(it_v,-1)
        while True:
            if el_u==-1 or el_v==-1 or el_u>maxhub or el_v>maxhub:
                break
            if el_u<el_v:
                el_u = next(it_u,-1)
                continue
            if el_u>el_v:
                el_v = next(it_v,-1)
                continue
        
            assert el_u==el_v
            assert el_u!=-1
            hub = el_u
            el_u = next(it_u,-1)
            el_v = next(it_v,-1)
        
            P_u = self.labels[u][hub]
            assert type(P_u)==list #sortedcontainers.sortedlist.SortedKeyList or (self.root==u and type(P_u)==list)
            assert len(P_u)>0
            
            P_v = self.labels[v][hub]
            assert type(P_v)==list #sortedcontainers.sortedlist.SortedKeyList  or (self.root==u and type(P_v)==list)
            assert len(P_v)>0
            

            

            for p_u,d_u in P_u:
                
                assert d_u == len(p_u)-1
                if break_at_k and len(self.lengths)>=self.num_k and d_u>=self.lengths[self.num_k-1]:
                    break
                if v!=self.reverse_ordering[hub] and v in p_u:
                    continue
                for p_v,d_v in P_v:
                    
                    assert d_v == len(p_v)-1
                    assert p_u[0]==p_v[0]
                    
                    if break_at_k and len(self.lengths)>=self.num_k and d_u+d_v>=self.lengths[self.num_k-1]:
                        break
                        
                    w_pt, intersection = self.combination_or_intersect_without_allocation(d_u, d_v, p_u, p_v, hub)
                    
                    if intersection:
                        assert len(set(p_u).intersection(set(p_v)))>1
                        assert not self.is_simple(self.combination_without_intersection_test(d_u, d_v, p_u, p_v)[1])
                        continue
                        
                    
                    assert set(p_u).intersection(set(p_v))=={self.reverse_ordering[hub]}
                    assert not intersection
                    assert self.is_simple(self.combination_without_intersection_test(d_u, d_v, p_u, p_v)[1])
                    assert d_u+d_v==w_pt
                    
                    
                    self.add_to_lengths(d_u+d_v)
                    if break_at_k and len(self.lengths)>self.num_k:
                        el = self.lengths.pop()
                        assert len(self.lengths)==self.num_k
                        assert el>=self.lengths[self.num_k-1]
                
                        
    def top_k_lengths_through(self, u: int, v: int, hub:int, break_at_k:bool=True):
             

         assert u!=v
         self.lengths = [] #SortedList()
         if hub not in self.labels[u] or hub not in self.labels[v]:
             return

         
         
         

         
         
    
         P_u = self.labels[u][hub]
         assert type(P_u)==list #sortedcontainers.sortedlist.SortedKeyList or (self.root==u and type(P_u)==list)
         assert len(P_u)>0
        
         P_v = self.labels[v][hub]
         assert type(P_v)==list #sortedcontainers.sortedlist.SortedKeyList  or (self.root==u and type(P_v)==list)
         assert len(P_v)>0
        
    
        
    
         for p_u,d_u in P_u:
            
            assert d_u == len(p_u)-1
            if break_at_k and len(self.lengths)>=self.num_k and d_u>=self.lengths[self.num_k-1]:
                break
            if v!=self.reverse_ordering[hub] and v in p_u:
                continue
            for p_v,d_v in P_v:
                
                assert d_v == len(p_v)-1
                assert p_u[0]==p_v[0]
                
                if break_at_k and len(self.lengths)>=self.num_k and d_u+d_v>=self.lengths[self.num_k-1]:
                    break
                    
                w_pt, intersection = self.combination_or_intersect_without_allocation(d_u, d_v, p_u, p_v, hub)
                
                if intersection:
                    assert len(set(p_u).intersection(set(p_v)))>1
                    assert not self.is_simple(self.combination_without_intersection_test(d_u, d_v, p_u, p_v)[1])
                    continue
                    
                
                assert set(p_u).intersection(set(p_v))=={self.reverse_ordering[hub]}
                assert not intersection
                assert self.is_simple(self.combination_without_intersection_test(d_u, d_v, p_u, p_v)[1])
                assert d_u+d_v==w_pt
                
                
                self.add_to_lengths(d_u+d_v)
                if break_at_k and len(self.lengths)>self.num_k:
                    el = self.lengths.pop()
                    assert len(self.lengths)==self.num_k
                    assert el>=self.lengths[self.num_k-1]
                        
    def path_through_hub(self, u: int, v: int, path:tuple, peso:int, hub:int):
        
        assert type(path)==tuple 
        assert len(path)-1==peso
        assert u!=v


        if hub not in self.labels[u] or hub not in self.labels[v]:
            return False
        
        P_u = self.labels[u][hub]
        
        assert type(P_u)==list #sortedcontainers.sortedlist.SortedKeyList or (self.root==u and type(P_u)==list) 
        assert len(P_u)>0
        
        P_v = self.labels[v][hub]
        
        assert type(P_v)==list #sortedcontainers.sortedlist.SortedKeyList  or (self.root==u and type(P_v)==list)
        assert len(P_v)>0

        for p_u,d_u in P_u:
        
            assert d_u == len(p_u)-1
               
            if d_u>peso:                            
                break 
            if v!=self.reverse_ordering[hub] and v in p_u:
                continue
            for p_v,d_v in P_v:
                
                assert d_v == len(p_v)-1
                
                assert p_u[0]==p_v[0]
                assert p_u[0]==self.reverse_ordering[hub]
                assert p_v[0]==self.reverse_ordering[hub]
                
                
                if d_u+d_v>peso:    
                    break 
                
                if d_u+d_v!=peso:
                    assert d_u+d_v<peso
                    continue
                
                assert d_u+d_v == peso
                
            
                
                pt = tuple(reversed(p_u[1:]))+p_v
                assert type(pt)==tuple and d_u+d_v==len(pt)-1
            
                equality = True
                aux_idx = 0
            
                while aux_idx<=peso:
                    if pt[aux_idx]!=path[aux_idx]:
                        equality = False
                        break
                    aux_idx+=1
                if equality:
                    assert pt == path and self.is_simple(pt)
                    return True
                assert pt!=path
        return False
        
        
    def path_is_encoded(self, u: int, v: int, path:tuple, peso:int):
        

        
        assert type(path)==tuple 
        assert len(path)-1==peso
        assert u!=v
        assert path[0]==u
        assert path[-1]==v



        if len(self.labels[u].keys())==0 or len(self.labels[v].keys())==0:
            return False



        # for hub in common_hubs: 
        if len(self.labels[u])==0 or len(self.labels[v])==0:
            return False
        maxhub = min(self.labels[u].keys()[-1],self.labels[v].keys()[-1])
        it_u = iter(self.labels[u].keys())
        it_v = iter(self.labels[v].keys())
        el_u = next(it_u,-1)
        el_v = next(it_v,-1)
        while True:
            if el_u==-1 or el_v==-1 or el_u>maxhub or el_v>maxhub:
                break
            if el_u<el_v:
                el_u = next(it_u,-1)
                continue
            if el_u>el_v:
                el_v = next(it_v,-1)
                continue
            
            assert el_u==el_v
            assert el_u!=-1
            hub = el_u
            el_u = next(it_u,-1)
            el_v = next(it_v,-1)
            assert hub in self.labels[u] and hub in self.labels[v]
            
            P_u = self.labels[u][hub]
            assert type(P_u)==list #sortedcontainers.sortedlist.SortedKeyList or (self.root==u and type(P_u)==list) 
            assert len(P_u)>0
            
            P_v = self.labels[v][hub]

            assert type(P_v)==list #sortedcontainers.sortedlist.SortedKeyList  or (self.root==u and type(P_v)==list)
            assert len(P_v)>0


            for p_u,d_u in P_u:
            
                assert d_u == len(p_u)-1
           
                if d_u>peso:                            
                    break 
                if v!=self.reverse_ordering[hub] and v in p_u:
                    continue
                for p_v,d_v in P_v:
                    
                    assert d_v == len(p_v)-1
                    
                    
                    
                    assert p_u[0]==p_v[0]
                    assert p_u[0]==self.reverse_ordering[hub]
                    assert p_v[0]==self.reverse_ordering[hub]
                    
                    
                    if d_u+d_v>peso:    
                        break 
                    if d_u+d_v!=peso:
                        assert d_u+d_v<peso
                        continue
                    
                    assert d_u+d_v == peso
                    
                    #TODO: TESTARE SENZA ALLOCARE
                    
                    # pt = tuple(reversed(p_u[1:]))+p_v
                    # assert type(pt)==tuple
                    # assert d_u+d_v==len(pt)-1
        
                    # equality = True
                    # aux_idx = 0
    
                    # while aux_idx<=peso:
                    #     if pt[aux_idx]!=path[aux_idx]:
                    #         equality = False
                    #         break
                    #     aux_idx+=1
                    # if equality:
                    #     assert pt == path
                    #     assert self.is_simple(pt)
                    #     return True
                    

                    assert(u==p_u[-1])
                    assert(v==p_v[-1])
    
                    assert path[0]==p_u[-1]
                    assert path[-1]==p_v[-1]

    
                    equal = True
                    cnt = 0;
                    while True:
                        if p_u[len(p_u)-1-cnt]!=path[cnt]:    
                            equal=False
                            break
                        elif path[cnt] == self.reverse_ordering[hub]:                            
                            break                        
                        cnt+=1;
                   
                    
                    
                    if equal==False:
                        assert tuple(reversed(p_u[1:]))+p_v !=path
                        continue
                    
                    assert path[cnt] == p_v[0]
                    assert path[cnt] == p_v[cnt-len(p_u)+1]

                    
                    while True:
                        assert cnt-len(p_u)+1<=len(p_v)-1
                        
                        if p_v[cnt-len(p_u)+1]!=path[cnt]: 
                            equal=False
                            break
                        elif path[cnt]==p_v[len(p_v)-1]:
                            assert cnt==len(path)-1
                            break
                        cnt+=1;
                        assert cnt<=len(path)-1
                    
                    if equal:
                        assert tuple(reversed(p_u[1:]))+p_v == path
                        assert d_u+d_v==len(tuple(reversed(p_u[1:]))+p_v)-1
                        assert self.is_simple(tuple(reversed(p_u[1:]))+p_v)
                        return True
                    
                    
                    assert tuple(reversed(p_u[1:]))+p_v !=path
            
        return False
        
    def k_shorter_simple_paths(self, u: int, v: int, peso:int):
            

        assert u!=v

        
        
        self.lengths = [] #SortedList()

        
        if len(self.labels[u])==0 or len(self.labels[v])==0:
            return 
        maxhub = min(self.labels[u].keys()[-1],self.labels[v].keys()[-1])
        it_u = iter(self.labels[u].keys())
        it_v = iter(self.labels[v].keys())
        el_u = next(it_u,-1)
        el_v = next(it_v,-1)
        
        while True:
            if el_u==-1 or el_v==-1 or el_u>maxhub or el_v>maxhub:
                break
            if el_u<el_v:
                el_u = next(it_u,-1)
                continue
            if el_u>el_v:
                el_v = next(it_v,-1)
                continue
            
            assert el_u==el_v
            assert el_u!=-1
            hub = el_u
            el_u = next(it_u,-1)
            el_v = next(it_v,-1)
        
            P_u = self.labels[u][hub]
            assert type(P_u)==list #sortedcontainers.sortedlist.SortedKeyList or (self.root==u and type(P_u)==list)
            assert len(P_u)>0
            
            P_v = self.labels[v][hub]
            assert type(P_v)==list #sortedcontainers.sortedlist.SortedKeyList  or (self.root==u and type(P_v)==list)
            assert len(P_v)>0
            

            

            for p_u,d_u in P_u:
                
                assert d_u == len(p_u)-1
                
                if len(self.lengths)>=self.num_k and peso>=self.lengths[self.num_k-1]:    
                    return True
                
                if d_u>peso:
                    break
                if v!=self.reverse_ordering[hub] and v in p_u:
                    continue
                for p_v,d_v in P_v:
                    
                    assert d_v == len(p_v)-1
                    
                    if d_u+d_v>peso:
                        break 
                    
                    assert p_u[0]==p_v[0]
                    
                    w_pt, intersection = self.combination_or_intersect_without_allocation(d_u, d_v, p_u, p_v, hub)
                    
                    if intersection:
                        assert len(set(p_u).intersection(set(p_v)))>1
                        assert not self.is_simple(self.combination_without_intersection_test(d_u, d_v, p_u, p_v)[1])
                        continue
                        
                    
                    assert set(p_u).intersection(set(p_v))=={self.reverse_ordering[hub]}
                    assert not intersection
                    assert self.is_simple(self.combination_without_intersection_test(d_u, d_v, p_u, p_v)[1])
                    assert d_u+d_v<=peso

                    assert d_u+d_v==w_pt
                    
                    
                    self.add_to_lengths(d_u+d_v)
                    if len(self.lengths)>=self.num_k and peso>=self.lengths[self.num_k-1]:
                        return True
                
                   
        assert len(self.lengths)<self.num_k or peso<self.lengths[self.num_k-1]
        return False  
        
   
    def k_shorter_simple_paths_through(self, u: int, v: int, peso:int, hub:int):
            

        assert u!=v

        
        
        if hub not in self.labels[u] or hub not in self.labels[v]:
            return False
        self.lengths = [] 

 
        P_u = self.labels[u][hub]
        assert type(P_u)==list #sortedcontainers.sortedlist.SortedKeyList or (self.root==u and type(P_u)==list)
        assert len(P_u)>0
        
        P_v = self.labels[v][hub]
        assert type(P_v)==list #sortedcontainers.sortedlist.SortedKeyList  or (self.root==u and type(P_v)==list)
        assert len(P_v)>0
        

        

        for p_u,d_u in P_u:
            
            assert d_u == len(p_u)-1
            if len(self.lengths)>=self.num_k and peso>=self.lengths[self.num_k-1]:
                return True
            
            if d_u>peso:
                break
            if v!=self.reverse_ordering[hub] and v in p_u:
                continue
            for p_v,d_v in P_v:
                
                assert d_v == len(p_v)-1
                
                if d_u+d_v>peso:
                    break 
                
                assert p_u[0]==p_v[0]
                
                w_pt, intersection = self.combination_or_intersect_without_allocation(d_u, d_v, p_u, p_v, hub)
                
                if intersection:
                    assert len(set(p_u).intersection(set(p_v)))>1
                    assert not self.is_simple(self.combination_without_intersection_test(d_u, d_v, p_u, p_v)[1])
                    continue
                    
                
                assert set(p_u).intersection(set(p_v))=={self.reverse_ordering[hub]}
                assert not intersection
                assert self.is_simple(self.combination_without_intersection_test(d_u, d_v, p_u, p_v)[1])
                assert d_u+d_v<=peso

                assert d_u+d_v==w_pt
                
                
                self.add_to_lengths(d_u+d_v)
                if len(self.lengths)>=self.num_k and peso>=self.lengths[self.num_k-1]:
                    return True
                
                   
        assert len(self.lengths)<self.num_k or peso<self.lengths[self.num_k-1]
        return False      
    
    
    
            
    
    
    def append_entry(self,host:int,hub:int,path:tuple,peso:int):
        
        
        assert type(path)==tuple 
        assert peso==len(path)-1
        assert len(self.labels[host])>=0
        
        if hub not in self.labels[host]:
            assert self.root == host or not self.path_is_encoded(u=self.reverse_ordering[hub],v=host,path=path,peso=peso)
            self.labels[host][hub] = [(path,peso)]
            self.labeling_size+=1
            assert list(self.labels[host][hub]) == sorted(self.labels[host][hub],key=lambda x:x[1])
            assert type(self.labels[host][hub])==list
            assert type(self.labels[host])==SortedDict
            assert list(self.labels[host].keys()) == sorted(self.labels[host].keys())
            return
 
        
        assert type(self.labels[host][hub])==list
        
        
        
        
        l_h = self.labels[host][hub]
        
        assert (path,peso) not in l_h
        assert not self.path_is_encoded(u=self.reverse_ordering[hub],v=host,path=path,peso=peso)
        assert not self.path_through_hub(u=self.reverse_ordering[hub],v=host,path=path,peso=peso,hub=hub)             
        assert not self.path_is_encoded(path[0],path[-1],path,peso)   
        
    
        l_h.append((path,peso))
        self.labeling_size+=1
        assert list(self.labels[host][hub]) == sorted(self.labels[host][hub],key=lambda x:x[1])
        assert type(self.labels[host][hub])==list
        assert type(self.labels[host])==SortedDict
        assert list(self.labels[host].keys()) == sorted(self.labels[host].keys())





    def getResultPaths(self):

        return [list(p[1]) for p in self.results]

    

    
    
    
 
    def assign_ordering(self,policy="deg"):
        
        print("Ordering Computation")
        
        start_centrality = time.perf_counter()
        
        

        if policy=="deg":
            centrality = nk.centrality.DegreeCentrality(self.graph, normalized=True, outDeg=True, ignoreSelfLoops=True)

        else:
            assert policy=="bet"
            SAMPLES = round(math.pow(self.graph.numberOfNodes(),2/3))
            print("Samples:",SAMPLES)
            centrality = nk.centrality.EstimateBetweenness(self.graph, nSamples=SAMPLES, normalized=True, parallel=True)

        
        centrality.run()
        
        end_centrality = time.perf_counter()
            
        
        print("Centrality computation time:",round((end_centrality-start_centrality),5),"s")
        
        total_centrality = end_centrality-start_centrality
       
        while round((end_centrality-start_centrality),5)<30 and policy=="bet" and SAMPLES<self.graph.numberOfNodes():
            
            start_centrality = time.perf_counter()

            SAMPLES*=2
            print("Resamples:",SAMPLES)
            centrality = nk.centrality.EstimateBetweenness(self.graph, nSamples=SAMPLES, normalized=True, parallel=True)

            centrality.run()
            end_centrality = time.perf_counter()

            print("Centrality re-computation time:",round((end_centrality-start_centrality),5),"s")
            total_centrality=total_centrality+end_centrality-start_centrality
        
        start_init =  time.perf_counter()
        
        for index,(vertex,_) in enumerate(centrality.ranking()):
            self.ordering[vertex]=index
            self.reverse_ordering[index]=vertex
        
        end_init = time.perf_counter() 
        init_time = end_init-start_init
        print("Init computation time:",round(init_time,5),"s")
        
        # while [25,5,61,64,67,39] != self.reverse_ordering[0:6]:
        #     print(self.reverse_ordering[0:20])
        #     centrality = nk.centrality.EstimateBetweenness(self.graph, nSamples=SAMPLES, normalized=True, parallel=True)
        #     centrality.run()
        #     for index,(vertex,score) in enumerate(centrality.ranking()):
        #         self.ordering[vertex]=index
        #         self.reverse_ordering[index]=vertex
        # while [5,25,61] != self.reverse_ordering[0:3]:
        #     print(self.reverse_ordering[0:20])
        #     centrality = nk.centrality.EstimateBetweenness(self.graph, nSamples=SAMPLES, normalized=True, parallel=True)
        #     centrality.run()
        #     for index,(vertex,score) in enumerate(centrality.ranking()):
        #         self.ordering[vertex]=index
        #         self.reverse_ordering[index]=vertex


        self.centralityTime = round(total_centrality+init_time,5)

        print("Total centrality computation time:",self.centralityTime,"s")
            

   
        start_diameter = time.perf_counter()
        diam = nk.distance.Diameter(self.graph,algo=1)
        diam.run()
        self.diametro = diam.getDiameter()[0]
        print("Diameter:",self.diametro)
        del diam
        self.density = nk.graphtools.density(self.graph)

        end_diameter = time.perf_counter()
        print("Diameter computation time:",round((end_diameter-start_diameter),5),"s")




            
    def getConstructionTime(self):
        return round(self.constrTime,2)
    
    def getCentralityTime(self):
        return round(self.centralityTime,2)
    def getLabelingSize(self):
        assert self.labeling_size==sum([len(self.labels[i][k]) for i in self.graph.iterNodes() for k in self.labels[i].keys()]) #num encoded paths
        return self.labeling_size
    
    def getLabelingSpace(self):
        space = 0
        for v in self.graph.iterNodes():
            for hub in self.labels[v]:
                space += sys.getsizeof(int())
                for path,weight in self.labels[v][hub]:
                    assert len(path)-1==weight
                    space += (weight+1) * sys.getsizeof(int())
                    
        return space
    
    def getAvgLabelingSize(self):
        n_entries = []
        for i in self.graph.iterNodes():
            local = 0
            for k in self.labels[i].keys():
                local+=len(self.labels[i][k])
            n_entries.append(local)
        assert round(statistics.mean(n_entries),2)==round(self.labeling_size/self.graph.numberOfNodes(),2)
        return round(self.labeling_size/self.graph.numberOfNodes(),2)
    
    def getMedianLabelingSize(self):
        n_entries = []
        for i in self.graph.iterNodes():
            local = 0
            for k in self.labels[i].keys():
                local+=len(self.labels[i][k])
            n_entries.append(local)
        
        return statistics.median(n_entries)
    
    def getMaxLabelingSize(self):
        n_entries = []
        for i in self.graph.iterNodes():
            local = 0
            for k in self.labels[i].keys():
                local+=len(self.labels[i][k])
            n_entries.append(local)
        assert round(max(n_entries))==max(n_entries)
        return max(n_entries)
    
    def getPrunings(self):
        return self.prunings
    
    
    def getAvgSameHubLabels(self):
        values = []
        
        for v in self.graph.iterNodes():
            
            for hub in self.labels[v]:
                values.append(len(self.labels[v][hub]))

        return round(statistics.mean(values),2)
    
    def getMedianSameHubLabels(self):
        values = []
        
        for v in self.graph.iterNodes():
            
            for hub in self.labels[v]:
                values.append(len(self.labels[v][hub]))

        return statistics.median(values)

    

    def getMaxSameHubLabels(self):
        values = []
        
        for v in self.graph.iterNodes():
            
            for hub in self.labels[v]:
                values.append(len(self.labels[v][hub]))

        return max(values)

    
