//
// Created by anonym on 27/05/23.
//
#ifndef KSISPINDEX_H
#define KSISPINDEX_H
#include <set>
#include <sys/time.h>
#include "networkit/graph/Graph.hpp"
#include "progressBar.h"
#include <map>
#include <unordered_set>
#include <boost/functional/hash.hpp>
#include <algorithm>
#include <iostream>
#include <queue>
#include <cassert>
#include <climits>
#include <networkit/centrality/EstimateBetweenness.hpp>
#include <networkit/centrality/KPathCentrality.hpp>
#include <networkit/components/ConnectedComponents.hpp>
#include "networkit/centrality/DegreeCentrality.hpp"
#include "mytimer.h"

using vertex = int32_t;
using dist = uint32_t;


using edge_id = uint64_t;

class path{
    public:
        std::vector<vertex> seq;
        dist w;
        vertex h;
        path(){
            this->h=0;
            this->w=0;
            this->seq.clear();
        };
        void init(std::deque<vertex> sorgente){
            this->w = sorgente.size()-1;
            this->seq.resize(sorgente.size());
            for(size_t t = 0 ; t <sorgente.size(); t++){
                this->seq[t]=sorgente[t];
            }
        }
        
        void init(vertex uniq,vertex hub){
            this->w = 0;
            this->seq.resize(1,uniq);//this->seq = new vertex[1];
            // this->seq[0]=uniq; 
            this->h = hub;           
        }
        void combined_init(path& ptx,int thr_ptx, path& det, int thr_det){
            

            this->w = std::max(0,thr_ptx)+thr_det-1;

            this->seq.resize(this->w+1);//= new vertex[this->w+1];
            int cnt = 0;
            for(size_t l = 0; l < thr_ptx; l++){
                this->seq[cnt]=ptx.seq[l];
                assert(l==cnt);
                cnt+=1;
            }
            for(size_t l = 0; l < thr_det; l++){
                this->seq[cnt]=det.seq[l];
                cnt+=1;
            }
            #ifndef NDEBUG
            std::vector<vertex> tmp;
            for(size_t l = 0; l < thr_ptx; l++)
                tmp.push_back(ptx.seq[l]);
            for(size_t l = 0; l < thr_det; l++)
                tmp.push_back(det.seq[l]);
            assert(tmp.size()==this->w+1);
            assert(tmp==this->seq);
            #endif
        
        }
        void combined_init(path& ptx,int thr_ptx, path& det, int thr_det, vertex* ord){

            this->w = std::max(0,thr_ptx)+thr_det - 1;
            this->h = std::numeric_limits<vertex>::max()/2;
            this->seq.resize(this->w+1);//= new vertex[this->w+1];
            int cnt = 0;
            for(size_t l = 0; l < thr_ptx; l++){
                this->seq[cnt]=ptx.seq[l];
                assert(l==cnt);
                this->h = std::min(this->h,ord[this->seq[l]]);
                cnt+=1;
            }
            for(size_t l = 0; l < thr_det; l++){
                this->seq[cnt]=det.seq[l];
                this->h = std::min(this->h,ord[this->seq[cnt]]);
                cnt+=1;
            }

            #ifndef NDEBUG
            std::vector<vertex> tmp;
            for(size_t l = 0; l < thr_ptx; l++)
                tmp.push_back(ptx.seq[l]);
            for(size_t l = 0; l < thr_det; l++)
                tmp.push_back(det.seq[l]);
            assert(tmp.size()==this->w+1);
            assert(tmp==this->seq);
            for(size_t l = 0; l < this->seq.size(); l++)
                assert(this->h<=ord[seq[l]]);
            #endif




        }
        inline bool operator==(path const & rhs) const{
		    if(this->w != rhs.w){
                return false;
            }
            for(size_t l = 0; l != this->w; l++){
                if(this->seq[l]!=rhs.seq[l]){
                    return false;
                }            
            }
            return true;
	    }
        inline bool operator!=(path const & rhs) const{
		    if(this->w != rhs.w){
                return true;
            }
            for(size_t l = 0; l != this->w; l++){
                if(this->seq[l]!=rhs.seq[l]){
                    return true;
                }            
            }
            return false;
	    }
        inline bool operator<(path const & rhs) const{
            return this->w < rhs.w;
        }
        ~path(){
            this->seq.clear();
        }
        
        
};
struct heap_min_comparator{
  bool operator()(const path& a,const path& b) const{
    return a.w>b.w;
  }
};
struct heap_max_comparator{
  bool operator()(const path& a,const path& b) const{
    return a.w<b.w;
  }
};
// class for hash function
class HashPaths {
public:
    // id is returned as hash function
    size_t operator()(const path& t) const
    {
        return boost::hash_range(t.seq.rbegin(),t.seq.rend());//t.seq,t.seq+t.w);
    }
};

using pathlist = std::vector<path>;

// struct heap_data
// {
// 	path P;
// 	heap_data(path n)
// 	{
// 		P=n;
// 	}

// 	inline bool operator<(heap_data const & rhs) const
// 	{
// 		return this->P.w > rhs.P.w;
// 	}
// };
// struct heap_data_hub
// {
// 	path P;
//     vertex h;
// 	heap_data_hub(path n,vertex v)
// 	{
// 		P=n;
//         h=v;
// 	}
//     inline  bool operator==(heap_data_hub const & rhs) const
// 	{
// 		return this->P.w == rhs.P.w && this->h==rhs.h && this->P.seq==rhs.P.seq;
// 	}
// 	inline  bool operator<(heap_data_hub const & rhs) const
// 	{
// 		return this->P.w > rhs.P.w || (this->P.w == rhs.P.w && this->h>rhs.h);
// 	}
// };
class kSiSPIndex {

public:
    
    NetworKit::Graph* graph;

    size_t K;
    static const vertex null_vertex; 
    static const dist null_distance; 

    dist max_dist;
    vertex* ordering;
    vertex* reverse_ordering;

    
    uint32_t index_size;
    uint32_t prunings;

    double constr_time;
    pathlist results;
    ~kSiSPIndex();

    double init(NetworKit::Graph*, int, int);
    void build();
    void query(vertex, vertex);
    dist length_query(vertex, vertex);
    void fill_queue_by_query(vertex, vertex);
    void print_path(const path&);


private:
    // friend class set_comp;
    // class set_comp {
    //     public:
    //         inline bool operator()(const path& a,
    //                     const path& b) const
    //         {
    //             return a.size()<b.size() || a!=b;
    //         }
    // };  
    // friend class res_comp;
    // class res_comp {
    //     public:
    //         inline bool operator()(const path& a,
    //                     const path& b) const
    //         {
    //             return a.w<b.w;
    //         }
    // };  
    // friend class hub_res_comp;
    // class hub_res_comp {
    //     public:
    //         inline bool operator()(const heap_data_hub& a, const heap_data_hub& b) const
    //         {
    //             return a.P.size()<b.P.size() || (a.P.size()==b.P.size() && a.h<b.h);
    //         }
    // };  

    //boost::heap::fibonacci_heap<heap_data>* basicQ;
    //boost::heap::fibonacci_heap<heap_data_hub>* extendedQ;

    std::vector<path> Q;

    // std::vector<heap_data_hub> extendedQ;

    bool* pruned;
    std::queue<vertex> queue_pruned_reset;

    vertex num_pruned;
    vertex* bidir_bfs_pred;
    std::queue<vertex> queue_pred_reset;
    
    std::deque<vertex> forward_fringe, reverse_fringe;

    
    vertex* bidir_bfs_succ;
    std::queue<vertex> queue_succ_reset;

    vertex join_vertex;
    pathlist* listA;


    std::map<vertex, pathlist>* index;

    std::pair<double,vertex>* ordering_rank;

    vertex root_index;
    vertex root;

    std::vector<dist> lengths;
    std::vector<dist> Q_lengths;

    bool* ignore_nodes;
    bool* ignore_edges;
    bool* intersection_tests;
    std::queue<vertex> reset_ignored_nodes;
    std::queue<edge_id> reset_ignored_edges;
    
    


    bool is_simple(const path&);


    pathlist::iterator lower;
    std::map<vertex, pathlist>::iterator hub_iterator;
    std::map<vertex, pathlist>::iterator it_s,it_t;
    
    vertex max_hub;
    
	std::unordered_set<path,HashPaths>* trace_paths;
    // std::map<vertex, int> freq;    
    // std::map<vertex, int>::iterator it_freq;    


    size_t s_s,s_t;

    pathlist::iterator up_it;

    bool is_traced(vertex, path&);
    void trace(vertex, path&);
    void untrace(vertex, path&);

    // void add_result(path);   
    bool is_path_in(const pathlist&,const path&);
    bool is_path_list_sorted(const pathlist&);

    void append_entry(vertex, vertex, path&);    
    void add_entry(vertex, vertex, path&);

    bool bidirectional_bfs(vertex, vertex, vertex, path&, dist);
    bool bidirectional_pred_succ(vertex, vertex, vertex,  dist);

    bool auto_bidirectional_bfs(vertex, vertex, vertex, path&, dist, std::unordered_set<vertex>&, std::unordered_set<edge_id>&);
    bool auto_bidirectional_pred_succ(vertex, vertex, vertex, dist, std::unordered_set<vertex>&, std::unordered_set<edge_id>&,vertex&,
                                        std::unordered_map<vertex,vertex>&,std::unordered_map<vertex,vertex>&);


    void break_path_at(path&, vertex, path&, path&);
    void combine_bfs();
    void pair_repair(vertex,vertex);

    bool k_shorter_simple_paths(vertex, vertex, dist);

    short int is_combination_non_simple_no_allocation(const path&, const path&);

    bool combination_or_intersection(const path&,const path&, vertex, path&);
    

    bool is_path_encoded(vertex, vertex, const path&);
};


#endif //CPP_KSISPINDEX_H
