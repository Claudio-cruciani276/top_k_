//
// Created by anonym on 27/05/23
//

#include "kSiSPIndex.h"
#include <boost/range/join.hpp>

using namespace std;
namespace {

// Return the number of threads that would be executed in parallel regions
int get_max_threads() {
#ifdef _OPENMP
	return omp_get_max_threads();
#else
	return 1;
#endif
}

int get_current_procs(){
#ifdef _OPENMP
	return omp_get_num_procs();
#else
	return 1;
#endif
}

// Set the number of threads that would be executed in parallel regions
void set_num_threads(int num_threads) {
#ifdef _OPENMP
	omp_set_num_threads(num_threads);
#else
	if (num_threads != 1) {
		throw std::runtime_error("compile with -fopenmp");
	}
#endif
}

// Return my thread ID
int get_thread_id() {
#ifdef _OPENMP
	return omp_get_thread_num();
#else
	return 0;
#endif
}

template<typename T>
struct parallel_vector {
	parallel_vector(size_t size_limit)
	: v(get_max_threads(), std::vector<T>(size_limit)),
	  n(get_max_threads(), 0) {}

	void push_back(const T &x) {
		int id = get_thread_id();
		v[id][n[id]++] = x;
	}

	void clear() {
		for(int i = 0;i<get_max_threads();i++)
			n[i] = 0;
	}

	std::vector<std::vector<T> > v;
	std::vector<size_t> n;
};
}  // namespace



const vertex kSiSPIndex::null_vertex = round(std::numeric_limits<vertex>::max()/2);
const dist kSiSPIndex::null_distance = round(std::numeric_limits<dist>::max()/2);



kSiSPIndex::~kSiSPIndex(){
        
    
    delete[] this->ignore_nodes;
    delete[] this->ignore_edges;



    delete[] this->bidir_bfs_pred;
    delete[] this->bidir_bfs_succ;

    delete[] this->index;
    delete[] this->intersection_tests;
    this->results.clear();
    this->lengths.clear();
    delete[] this->pruned;
    delete[] this->listA;
    

    delete[] this->ordering;
    delete[] this->reverse_ordering;
    


    this->Q.clear();
    // this->Q.clear();

}



// inline void kSiSPIndex::add_length(dist d) {
//     lengths.insert(std::upper_bound(lengths.begin(),lengths.end(), d), d);
//     if(lengths.size()>this->K){
//         lengths.pop_back();
//     }
// }

// inline void kSiSPIndex::add_result(path pt) {
//     assert(!is_path_in(results, pt));    
//     if(results.empty()){
//         results.push_back(pt);
//         assert(is_path_list_sorted(results));
//         return;
//     }


//     up_it = std::upper_bound(results.begin(), results.end(), pt);

//     assert(!is_path_in(results, pt));    

//     assert(up_it==results.end() || *up_it!=pt);
//     results.insert(up_it,pt);
//     assert(is_path_list_sorted(results));
//     assert(is_path_in(results, pt));    

// }

// inline void kSiSPIndex::add_H_result(path pt, vertex ht) {
//     assert(!is_H_path_in(H_results,pt,ht));    
//     if(H_results.empty()){
//         H_results.push_back({pt,ht});
//         assert(is_H_path_list_sorted(H_results));
//         return;
//     }
//     std::pair<path,vertex> p = {pt,ht};


//     up_H_it = std::upper_bound(H_results.begin(), H_results.end(), p, hub_res_comp());

//     assert(!is_H_path_in(H_results, p.first, p.second));    

//     assert(up_H_it==H_results.end() || *up_H_it!=p);
//     H_results.insert(up_H_it,p);
//     assert(is_H_path_list_sorted(H_results));
//     assert(is_H_path_in(H_results, p.first, p.second));    
// }

inline bool kSiSPIndex::is_path_in(const pathlist& plist, const path &pt){
    //inefficiente ma tanto viene invocata solo per debug
    return std::find(plist.begin(),plist.end(),pt)!=plist.end();
    
}
// inline bool kSiSPIndex::is_H_path_in(const hubpathlist& plist, const path &pt, const vertex& ht){
//     //inefficiente ma tanto viene invocata solo per debug
//     std::pair<path,vertex> p = {pt,ht};
//     return std::find(plist.begin(),plist.end(),p)!=plist.end();
    
// }
inline void kSiSPIndex::append_entry(vertex host, vertex hub, path& pt) {

    assert(pt.seq[0] == reverse_ordering[hub]);
    assert(pt.w>=0);

    hub_iterator = index[host].find(hub); //log

    if(hub_iterator != index[host].end()){    
        assert(!is_path_in(hub_iterator->second,pt));        
        hub_iterator->second.push_back(pt); 
        index_size++;
        assert(is_path_in(hub_iterator->second,pt));        
        return;
    }   
    else{
        pathlist& handle  = index[host][hub];
        handle = pathlist();
        handle.push_back(pt);
        index_size++;
        return;
    }
}
inline void kSiSPIndex::add_entry(vertex host, vertex hub, path& pt) {

    
    assert(pt.w>0);
    assert(pt.seq[0] == reverse_ordering[hub]);

    hub_iterator = index[host].find(hub); //log

    if(hub_iterator != index[host].end()){
        
        lower = std::lower_bound(hub_iterator->second.begin(), hub_iterator->second.end(), pt);

        if(lower==hub_iterator->second.end()){
            assert(pt.w>=hub_iterator->second.back().w);

            assert(!is_path_in(hub_iterator->second,pt));        
            hub_iterator->second.push_back(pt); 
            index_size++;
            assert(is_path_in(hub_iterator->second,pt));           
            assert(is_path_list_sorted(hub_iterator->second));

            return;

        }
        else{ 
            while(pt.w==(*lower).w && lower!=hub_iterator->second.end()){
                if(pt==*lower){
                    assert(is_path_in(hub_iterator->second,pt));    
                    return; //found path
                }
                std::advance(lower, 1);
            }
            assert(!is_path_in(hub_iterator->second,pt));    
            hub_iterator->second.insert(lower,pt);
            assert(is_path_list_sorted(hub_iterator->second));
            index_size++;
            assert(is_path_in(hub_iterator->second,pt));    
            return;
        }

    }
    else{
        pathlist& handle = index[host][hub];
        handle = pathlist();
        handle.push_back(pt);
        index_size++;
        return;
    }
}


inline bool kSiSPIndex::bidirectional_bfs(vertex source, vertex target, vertex lowest_index, path & path_to, dist largest) {
    
    

    

    

    
    bool path_found = bidirectional_pred_succ(source, target, lowest_index, largest);


    if(!path_found){
        assert(join_vertex==null_vertex);
        while(!this->queue_pred_reset.empty()){
            this->bidir_bfs_pred[this->queue_pred_reset.front()]=-1;
            this->queue_pred_reset.pop();
        }
        while(!this->queue_succ_reset.empty()){
            this->bidir_bfs_succ[this->queue_succ_reset.front()]=-1;
            this->queue_succ_reset.pop();
        }
    
        return false;
    }
    
    assert(this->join_vertex!=null_vertex);

    std::vector<vertex> right;

    
    path_to.h = null_vertex;

    while(this->join_vertex != null_vertex){
        right.push_back(this->join_vertex);
        path_to.h = min(ordering[this->join_vertex], path_to.h);
        this->join_vertex = this->bidir_bfs_succ[this->join_vertex];
    }

    this->join_vertex = this->bidir_bfs_pred[right[0]];

    while(this->join_vertex != null_vertex){
        path_to.seq.push_back(this->join_vertex);
        path_to.h = min(ordering[this->join_vertex], path_to.h);
        this->join_vertex = bidir_bfs_pred[this->join_vertex];
    }

    std::reverse(std::begin(path_to.seq), std::end(path_to.seq));

    // path_to.seq.reserve(left.size()+right.size());

    path_to.seq.insert(path_to.seq.end(), right.begin(), right.end());

    // path_to.seq = boost::copy_range<std::vector<vertex>>(boost::join(left, right));
    #ifndef NDEBUG
        vertex min_h = null_vertex;
        for(size_t t=0;t<path_to.seq.size();t++){
            min_h = min(min_h,ordering[path_to.seq[t]]);
        }
        assert(min_h==path_to.h);
    #endif

    assert(path_to.seq.size()-1<largest);
    
    path_to.w = path_to.seq.size()-1;

    // left.clear();
    right.clear();
    while(!this->queue_pred_reset.empty()){
        this->bidir_bfs_pred[this->queue_pred_reset.front()]=-1;
        this->queue_pred_reset.pop();
    }
    while(!this->queue_succ_reset.empty()){
        this->bidir_bfs_succ[this->queue_succ_reset.front()]=-1;
        this->queue_succ_reset.pop();
    }
    

    // path_to.init(temp, hub_to);
    
        
    return true;




}
inline bool kSiSPIndex::auto_bidirectional_bfs(vertex source, vertex target, vertex lowest_index, path & path_to, dist largest, std::unordered_set<vertex>& vertici, std::unordered_set<edge_id>& archi) {
    
    

    

    

    vertex touch_vertex = null_vertex;
    std::unordered_map<vertex,vertex> pred;
    std::unordered_map<vertex,vertex> succ;
    bool path_found = auto_bidirectional_pred_succ(source, target, lowest_index, largest, vertici,archi,touch_vertex,pred,succ);


    if(!path_found){
        assert(touch_vertex==null_vertex);    
        return false;
    }
    
    assert(touch_vertex!=null_vertex);

    std::vector<vertex> right;

    
    path_to.h = null_vertex;

    while(touch_vertex != null_vertex){
        right.push_back(touch_vertex);
        path_to.h = min(ordering[touch_vertex], path_to.h);
        assert(succ.find(touch_vertex)!=succ.end());
        touch_vertex = succ[touch_vertex];
    }
    assert(pred.find(right[0])!=pred.end());

    touch_vertex = pred[right[0]];

    while(touch_vertex != null_vertex){
        path_to.seq.push_back(touch_vertex);
        path_to.h = min(ordering[touch_vertex], path_to.h);
        assert(pred.find(touch_vertex)!=pred.end());
        touch_vertex = pred[touch_vertex];
    }

    std::reverse(std::begin(path_to.seq), std::end(path_to.seq));

    path_to.seq.insert(path_to.seq.end(), right.begin(), right.end());

    #ifndef NDEBUG
        vertex min_h = null_vertex;
        for(size_t t=0;t<path_to.seq.size();t++){
            min_h = min(min_h,ordering[path_to.seq[t]]);
        }
        assert(min_h==path_to.h);
    #endif

    assert(path_to.seq.size()-1<largest);
    
    path_to.w = path_to.seq.size()-1;

    right.clear();
    pred.clear();
    succ.clear();
    

    
        
    return true;




}
inline bool kSiSPIndex::bidirectional_pred_succ(vertex source, vertex  target, vertex lowest_index, dist LG) {

    
    if(this->ignore_nodes[source] || this->ignore_nodes[target]){
        return false;
    }


    assert(std::count(this->bidir_bfs_pred, this->bidir_bfs_pred+this->graph->numberOfNodes(), -1)==graph->numberOfNodes());
    assert(std::count(this->bidir_bfs_succ, this->bidir_bfs_succ+this->graph->numberOfNodes(), -1)==graph->numberOfNodes());


    this->bidir_bfs_pred[source] = null_vertex;
    this->queue_pred_reset.push(source);
    this->bidir_bfs_succ[target] = null_vertex;
    this->queue_succ_reset.push(target);

    if(source == target){

        this->join_vertex = source;
        return true;
    }


    this->forward_fringe.clear();
    this->reverse_fringe.clear();
    this->forward_fringe.push_back(source);
    this->reverse_fringe.push_back(target);
    vertex v,stop_vertex;
    dist steps = 0;
    while(!this->forward_fringe.empty() && !this->reverse_fringe.empty()){
        
        steps+=1;
        if(steps>=LG){
            return false;
        }
        if(this->forward_fringe.size() <= this->reverse_fringe.size()){
            
            stop_vertex = this->forward_fringe.back();

            
            while(true){
                v = this->forward_fringe.front();
                this->forward_fringe.pop_front();
                
                
                for(vertex w : graph->neighborRange(v)){


                        assert(this->graph->edgeId(v,w)==this->graph->edgeId(w,v));

                        if(this->ignore_edges[graph->edgeId(v,w)]==true){
                            continue;
                        }
                        if(this->ordering[w] <= lowest_index || w == source || this->ignore_nodes[w]==true){
                            assert(this->bidir_bfs_succ[w]==-1);
                            continue;                            
                        }
                        
                        if(this->bidir_bfs_pred[w]==-1){
                            this->forward_fringe.push_back(w);
                            this->bidir_bfs_pred[w] = v;
                            this->queue_pred_reset.push(w);
                        }
                        if(this->bidir_bfs_succ[w]!=-1){
                            this->join_vertex = w;
                            return true;
                        }

                }
                if(v==stop_vertex){
                    break;
                }
            }
        }
                   


                
            


        
        else{
            stop_vertex = this->reverse_fringe.back();
     
            while(true){
                v = this->reverse_fringe.front();
                this->reverse_fringe.pop_front();
                
                for(vertex w : graph->neighborRange(v)){

                    assert(this->graph->edgeId(v,w)==this->graph->edgeId(w,v));

                    
                    if(this->ignore_edges[graph->edgeId(v,w)]==true){
                        continue;

                    }
                    if(this->ordering[w] <= lowest_index || w == target || this->ignore_nodes[w]==true){
                        assert(this->bidir_bfs_pred[w]==-1);
                        continue;
                    }
                    if(this->bidir_bfs_succ[w]==-1){
                        this->reverse_fringe.push_back(w);
                        this->bidir_bfs_succ[w] = v;
                        this->queue_succ_reset.push(w);
                    }
                    if(this->bidir_bfs_pred[w]!=-1){
                        this->join_vertex = w;
                        return true;
                    }
                }
                if(v==stop_vertex){
                    break;
                }
            }

        }
    }
    return false;

}

inline bool kSiSPIndex::auto_bidirectional_pred_succ(vertex source, vertex  target, vertex lowest_index, dist LG, std::unordered_set<vertex>& vertici, std::unordered_set<edge_id>& archi,
                     vertex& contact_vertex,std::unordered_map<vertex,vertex>&pred, std::unordered_map<vertex,vertex>&succ){

    
    if(vertici.find(source)!=vertici.end() || vertici.find(target)!=vertici.end()){
        return false;
    }




    pred[source] = null_vertex;
    succ[target] = null_vertex;



    if(source == target){
        contact_vertex = source;
        return true;
    }

    std::deque<vertex> fwd;
    std::deque<vertex> bck;
    fwd.push_back(source);
    bck.push_back(target);
    vertex v,stop_vertex;
    dist steps = 0;
    while(!fwd.empty() && !bck.empty()){
        
        steps+=1;
        if(steps>=LG){
            return false;
        }
        if(fwd.size() <= bck.size()){
            
            stop_vertex = fwd.back();

            
            while(true){
                v = fwd.front();
                fwd.pop_front();
                
                
                for(vertex w : graph->neighborRange(v)){


                        assert(this->graph->edgeId(v,w)==this->graph->edgeId(w,v));

                        if(archi.find(graph->edgeId(v,w))!=archi.end()){
                            continue;
                        }
                        if(this->ordering[w] <= lowest_index || w == source || vertici.find(w)!=vertici.end()){
                            assert(succ.find(w)==succ.end());
                            continue;                            
                        }
                        
                        if(pred.find(w)==pred.end()){
                            fwd.push_back(w);
                            pred[w]=v;
                        }
                        if(succ.find(w)!=succ.end()){
                            contact_vertex = w;
                            return true;
                        }

                }
                if(v==stop_vertex){
                    break;
                }
            }
        }
                   


                
            


        
        else{
            stop_vertex = bck.back();
     
            while(true){
                v = bck.front();
                bck.pop_front();
                
                for(vertex w : graph->neighborRange(v)){

                    assert(this->graph->edgeId(v,w)==this->graph->edgeId(w,v));

                    
                    if(archi.find(graph->edgeId(v,w))!=archi.end()){
                        continue;

                    }
                    if(this->ordering[w] <= lowest_index || w == target || vertici.find(w)!=vertici.end()){
                        assert(pred.find(w)==pred.end());
                        continue;
                    }
                    if(succ.find(w)==succ.end()){
                        bck.push_back(w);
                        succ[w] = v;
                    }
                    if(pred.find(w)!=pred.end()){
                        contact_vertex = w;
                        return true;
                    }
                }
                if(v==stop_vertex){
                    break;
                }
            }

        }
    }
    return false;

}

inline void kSiSPIndex::break_path_at(path& pt, vertex hub, path& left, path& right){

    int32_t idx = pt.w;
    

    // std::deque<vertex> tmp_r,tmp_l;
    while(true){
        if(pt.seq[idx] == reverse_ordering[hub]){
            right.seq.push_back(pt.seq[idx]); 
            break;
        }
        right.seq.push_back(pt.seq[idx]); 
        idx--;
    }
    std::reverse(std::begin(right.seq), std::end(right.seq));
    assert(right.seq[0]==reverse_ordering[hub]);
    assert(right.seq[right.seq.size()-1]==pt.seq[pt.w]);
    right.w = right.seq.size()-1;


    assert(idx>=0 && pt.w>0);
    while(true){
        if(idx<0){
            break;
        }
        left.seq.push_back(pt.seq[idx]); 
        idx--;
        
    }
    assert(left.seq[0]==reverse_ordering[hub]);
    assert(left.seq[left.seq.size()-1]==pt.seq[0]);
    left.w = left.seq.size()-1;


    
        

}

void kSiSPIndex::build() {

    std::cout << "Construction\n";
    
    mytimer time_counter;
	time_counter.restart();
    

    ProgressStream combine(graph->numberOfNodes());
    combine.label() << "Combining Iterations:";
    
    // int card = 0;


    for(size_t u = 0; u < graph->numberOfNodes(); u++){
        root_index = u;
        root = reverse_ordering[u];
        combine_bfs();

        ++combine;
    }

    ProgressStream pairs_rep(graph->numberOfNodes()-1);
    pairs_rep.label() << "Pair Repairing:";

    for(size_t u = 1; u < graph->numberOfNodes(); u++){
        if(graph->degree(reverse_ordering[u]) <= 1){
            ++pairs_rep;
            continue;
        }

        
        for(size_t v = u+1; v < graph->numberOfNodes(); v++){
            pair_repair(reverse_ordering[u],reverse_ordering[v]);
        }

        

        ++pairs_rep;
    }


    constr_time = time_counter.elapsed();
    //SORTING TEST
    #ifndef NDEBUG
    for(size_t u = 0; u < graph->numberOfNodes(); u++){
   
        for(auto & entry : this->index[u]){
            assert(is_path_list_sorted(entry.second));
        }
    }
    #endif
}

inline void kSiSPIndex::combine_bfs() {
    #ifndef NDEBUG
        this->graph->parallelForNodes([&] (vertex i){
            assert(graph->hasNode(i));
            assert(this->pruned[i]==false);

        });
        assert(this->num_pruned==0);
        assert(this->queue_pruned_reset.empty());
    #endif 

    assert(this->Q.empty());
    
    path rootpath;
    rootpath.init(root,root_index);
    append_entry(root, root_index, rootpath);

    vertex next_root;
    for(vertex next_root_index = root_index+1; next_root_index < graph->numberOfNodes(); next_root_index++){

        //top-k towards previous_root
        path path_to_root;
        next_root = reverse_ordering[next_root_index];
        
        #ifndef NDEBUG
            assert(root!=next_root);
            assert(std::count(this->ignore_nodes, this->ignore_nodes+this->graph->numberOfNodes(), true)==0);
            assert(std::count(this->ignore_edges, this->ignore_edges+this->graph->numberOfEdges(), true)==0);
            assert(this->reset_ignored_nodes.empty());
            assert(this->reset_ignored_edges.empty());
            assert(this->pruned[next_root]==false);
            
        #endif
  
        bool path_exists = bidirectional_bfs(root, next_root, root_index, path_to_root, null_distance);

        if(!path_exists){
            assert(std::count(this->ignore_nodes, this->ignore_nodes+this->graph->numberOfNodes(), true)==0);
            assert(std::count(this->ignore_edges, this->ignore_edges+this->graph->numberOfEdges(), true)==0);
            assert(this->pruned[next_root]==false);
            this->pruned[next_root] = true;
            this->queue_pruned_reset.push(next_root);
            this->num_pruned += 1;
            assert(this->num_pruned+1+this->root_index <= this->graph->numberOfNodes());
            continue;
        }

        this->listA[next_root].clear();
        this->trace_paths[next_root].clear();

        assert(path_to_root.seq[0] == root);
        assert(path_to_root.seq[path_to_root.w] == next_root);

        this->Q.push_back(path_to_root);

        assert(!is_traced(next_root,path_to_root));
        this->trace(next_root,path_to_root);


    }
    make_heap(this->Q.begin(), this->Q.end(),heap_min_comparator());


    path ptx;
    vertex vtx;
    dist bound = null_distance;
    while(!this->Q.empty()){
        
        ptx = this->Q.front(); 
        pop_heap(this->Q.begin(), this->Q.end(),heap_min_comparator());
        this->Q.pop_back();
        vtx = ptx.seq[ptx.w];

        assert(is_simple(ptx));
        assert(ptx.w>0);
        assert(is_traced(vtx,ptx));

        if(this->num_pruned+1+this->root_index==this->graph->numberOfNodes()){
            assert(this->pruned[vtx]);
            this->Q.clear();
            break;
        }
        if(this->pruned[vtx]==true){
            continue;
        }
        
        



        bound = length_query(root, vtx);
        if(ptx.w>=bound){
            assert(k_shorter_simple_paths(root,vtx,ptx.w));
            assert(this->pruned[vtx]==false);
            this->prunings++;
            this->pruned[vtx] = true;
            this->queue_pruned_reset.push(vtx);            
            this->num_pruned++;
            continue;
        }
    
        assert(!k_shorter_simple_paths(root,vtx,ptx.w));
        #ifndef NDEBUG
            vertex minimum = null_vertex; 
            for(size_t t=0;t<ptx.w+1;t++){
                minimum = std::min(minimum,ordering[ptx.seq[t]]);
            }
            assert(minimum==root_index);

        #endif
        assert((int)this->listA[vtx].size()<this->K);
        assert(!is_path_in(this->listA[vtx],ptx));
        this->listA[vtx].push_back(ptx);
        assert(is_path_in(this->listA[vtx],ptx));
        assert(is_path_list_sorted(this->listA[vtx]));
        assert(!is_path_encoded(root,vtx,ptx));

        append_entry(vtx, root_index, ptx);

        assert(is_path_encoded(root,vtx,ptx));
    





        assert(std::count(this->ignore_nodes, this->ignore_nodes+this->graph->numberOfNodes(), true)==0);
        assert(std::count(this->ignore_edges, this->ignore_edges+this->graph->numberOfEdges(), true)==0);
        
        vertex local_back_root;
        for(int i = 1; i != ptx.w+1; i++){

            local_back_root = ptx.seq[i-1];


            assert(is_path_list_sorted(this->listA[vtx]));

            for(int t = 0;t!=this->listA[vtx].size();t++){
                
                // path local_path(this->listA[vtx][t].begin(),this->listA[vtx][t].begin() +i);
                bool uguali = true;
                for(int j=0;j<i && j<this->listA[vtx][t].w+1;j++){
                    if(this->listA[vtx][t].seq[j]!=ptx.seq[j]){
                        uguali = false;
                        break;
                    }
                }

                if(uguali){
                    // assert(local_root==local_path);
                    if(this->ignore_edges[graph->edgeId(this->listA[vtx][t].seq[i-1],this->listA[vtx][t].seq[i])]==false){
                        this->ignore_edges[graph->edgeId(this->listA[vtx][t].seq[i-1],this->listA[vtx][t].seq[i])]=true;
                        this->reset_ignored_edges.push(graph->edgeId(this->listA[vtx][t].seq[i-1],this->listA[vtx][t].seq[i]));
                    }
                }   


                
            }
            path detour_to_target;

            bool detour_exists = bidirectional_bfs(local_back_root, vtx, root_index, detour_to_target, bound-std::max(0,i-1));

            if(!detour_exists){                
                this->ignore_nodes[local_back_root]=true;
                this->reset_ignored_nodes.push(local_back_root);
                continue;
            }
            assert(detour_to_target.h<null_vertex);
            assert(is_simple(detour_to_target));
            assert(detour_to_target.w+std::max(0,i-1)<bound);


            path new_path;
            new_path.combined_init(ptx,i-1,detour_to_target,detour_to_target.w+1);
            assert(new_path.w<bound);

            #ifndef NDEBUG
                for(size_t t=0;t<new_path.w+1;t++){
                    assert(ordering[new_path.seq[t]]>=root_index);
                }
            #endif
            assert(new_path.seq[0] == root && new_path.seq[new_path.w] == vtx);
            assert(is_simple(new_path));

            if(is_traced(vtx,new_path)){
                assert(this->ignore_nodes[local_back_root]==false);
                this->ignore_nodes[local_back_root]=true;
                this->reset_ignored_nodes.push(local_back_root);

                continue;
            }

            this->Q.push_back(new_path);
            std::push_heap(this->Q.begin(), this->Q.end(),heap_min_comparator());

            assert(!is_traced(vtx,new_path));
            this->trace(vtx,new_path);
            this->ignore_nodes[local_back_root]=true;
            this->reset_ignored_nodes.push(local_back_root);


        }

        while(!this->reset_ignored_nodes.empty()){
            this->ignore_nodes[this->reset_ignored_nodes.front()]=false;
            this->reset_ignored_nodes.pop();
        }
        while(!this->reset_ignored_edges.empty()){
            this->ignore_edges[this->reset_ignored_edges.front()]=false;
            this->reset_ignored_edges.pop();
        }   
 


    }

    assert(this->Q.empty());
    assert(std::count(this->ignore_nodes, this->ignore_nodes+this->graph->numberOfNodes(), true)==0);
    assert(std::count(this->ignore_edges, this->ignore_edges+this->graph->numberOfEdges(), true)==0);


    while(!this->queue_pruned_reset.empty()){
        assert(this->pruned[this->queue_pruned_reset.front()]);
        this->pruned[this->queue_pruned_reset.front()] = false;
        this->queue_pruned_reset.pop();
    }
    this->num_pruned = 0;



 
}

inline bool kSiSPIndex::is_traced(vertex v, path & P){
    return this->trace_paths[v].find(P)!=this->trace_paths[v].end();

}
inline void kSiSPIndex::trace(vertex v, path & P){
    assert(this->trace_paths[v].find(P)==this->trace_paths[v].end());
    this->trace_paths[v].insert(P);
}
inline void kSiSPIndex::untrace(vertex v, path & P){
    assert(this->trace_paths[v].find(P)!=this->trace_paths[v].end());
    this->trace_paths[v].erase(P);
}





inline void kSiSPIndex::pair_repair(vertex u, vertex v) {

    #ifndef NDEBUG
        this->graph->parallelForNodes([&] (vertex i){
            assert(graph->hasNode(i));
            assert(this->pruned[i]==false);

        });
        assert(this->num_pruned==0);
        assert(this->queue_pruned_reset.empty());
    #endif 


    assert(this->Q.empty());
    

    dist bound = null_distance;

    assert(u!=v);
    assert(std::count(this->ignore_nodes, this->ignore_nodes+this->graph->numberOfNodes(), true)==0);
    assert(std::count(this->ignore_edges, this->ignore_edges+this->graph->numberOfEdges(), true)==0);
    assert(this->reset_ignored_nodes.empty());
    assert(this->reset_ignored_edges.empty());



    


    assert(u!=v);
    

    fill_queue_by_query(u,v);

    assert(is_heap(this->Q.begin(), this->Q.end(),heap_max_comparator()));

    if(this->Q.size()==0){
        //non connessi
        this->prunings++;
        INFO("disconnected graph");
        #ifndef NDEBUG
        path tmppt;
        assert(!bidirectional_bfs(u, v, -1, tmppt, null_distance));
        #endif
        return;
    }
    
    this->listA[v].clear();
    this->trace_paths[v].clear();

    this->Q_lengths.resize(this->Q.size(),0);

    vertex l_index = 0;

    for(size_t f = 0;f<this->Q.size();f++){       
        this->trace(v,(this->Q[f]));
        this->Q_lengths[f]=this->Q[f].w;
        if(this->Q[f].h!=0){
            l_index = -1;
        }
    }

    
    assert(this->Q.size()<=this->K);
    assert(this->Q.size()==this->Q_lengths.size());

    std::make_heap(this->Q_lengths.begin(), this->Q_lengths.end());

    assert(is_heap(this->Q.begin(), this->Q.end(),heap_max_comparator()));
    assert(is_heap(this->Q_lengths.begin(), this->Q_lengths.end()));

    if(this->Q.size()>=this->K){
        bound = this->Q[0].w;
        assert(Q_lengths.size()>=this->K);
        assert(bound<null_distance);
        assert(bound==this->Q_lengths[0]);

    }
    else{
        l_index = -1;
    }

    make_heap(this->Q.begin(), this->Q.end(),heap_min_comparator());
    assert(is_heap(this->Q.begin(), this->Q.end(),heap_min_comparator()));




    while(!this->Q.empty()){
        
        path ptx = this->Q.front(); 
        pop_heap(this->Q.begin(), this->Q.end(),heap_min_comparator());
        this->Q.pop_back();
        assert(ptx.seq[0]==u);
        assert(is_simple(ptx));
        assert(ptx.w>0);
        assert(is_traced(v,ptx));
        assert(ptx.h<=ordering[u] && ptx.h<=ordering[v]);      

        if((ptx.w>=bound) && (Q_lengths.size()>=this->K)){

            if(ptx.h<ordering[u]){
                this->listA[v].push_back(ptx);
            }
            while(!this->Q.empty() && ptx.w == bound){
                path ptx = this->Q.front(); 
                pop_heap(this->Q.begin(), this->Q.end(),heap_min_comparator());
                this->Q.pop_back();
                assert(ptx.seq[0]==u);
                assert(is_simple(ptx));
                assert(ptx.w>0);
                assert(is_traced(v,ptx));
                if(ptx.h<ordering[u]){
                    this->listA[v].push_back(ptx);
                }


            }
            
            this->Q.clear();
            this->prunings++;
            
            for(auto camm:listA[v]){
                if(camm.h>=ordering[u])
                    continue;
                if(k_shorter_simple_paths(u,v,camm.w)){
                    return; 
                }   
                path L,R;                 
                break_path_at(camm, camm.h, L, R);
                add_entry(u, camm.h, L);
                add_entry(v, camm.h, R);
            }
            return;
      

        }
        

        assert(!k_shorter_simple_paths(u,v,ptx.w));

        assert(ptx.h<=ordering[u]);
        assert(is_path_encoded(u,v,ptx) || ptx.h<ordering[u]);


        assert(this->listA[v].size()<this->K);
        assert(!is_path_in(this->listA[v],ptx));
        this->listA[v].push_back(ptx);
        assert(is_path_in(this->listA[v],ptx));
        assert(is_path_list_sorted(this->listA[v]));
        assert(std::count(this->ignore_nodes, this->ignore_nodes+this->graph->numberOfNodes(), true)==0);
        assert(std::count(this->ignore_edges, this->ignore_edges+this->graph->numberOfEdges(), true)==0);
     
        vertex local_back_root;
        for(int i = 1; i != ptx.w+1; i++){

            local_back_root = ptx.seq[i-1];
            assert(is_path_list_sorted(listA[v]));

            for(int t = 0;t!=listA[v].size();t++){
                
                bool uguali = true;
                for(int j=0;j<i && j!=this->listA[v][t].w+1;j++){
                    if(this->listA[v][t].seq[j]!=ptx.seq[j]){
                        uguali = false;
                        break;
                    }
                }

                if(uguali){
                    if(this->ignore_edges[graph->edgeId(this->listA[v][t].seq[i-1],this->listA[v][t].seq[i])]==false){
                        this->ignore_edges[graph->edgeId(this->listA[v][t].seq[i-1],this->listA[v][t].seq[i])]=true;
                        this->reset_ignored_edges.push(graph->edgeId(this->listA[v][t].seq[i-1],this->listA[v][t].seq[i]));
                    }
                }   
      

                
            }

            path detour_to_target;
            bool detour_exists = bidirectional_bfs(local_back_root, v, l_index, detour_to_target, bound-std::max(0,i-1));

            if(!detour_exists){
                this->ignore_nodes[local_back_root]=true;
                this->reset_ignored_nodes.push(local_back_root);
                continue;
            }
            assert(detour_to_target.h<null_vertex);
            assert(is_simple(detour_to_target));
            assert(detour_to_target.w+std::max(0,i-1)<bound);

            path new_path;
            new_path.combined_init(ptx,i-1,detour_to_target,detour_to_target.w+1,ordering);

            assert(new_path.seq[0] == u && new_path.seq[new_path.w] == v);
            assert(is_simple(new_path));

            if(is_traced(v,new_path)){
                assert(this->ignore_nodes[local_back_root]==false);
                this->ignore_nodes[local_back_root]=true;
                this->reset_ignored_nodes.push(local_back_root);
                continue;
            }
            assert(new_path.w<bound);

            assert(this->Q.size()<=this->K);

            this->Q.push_back(new_path);
            std::push_heap(this->Q.begin(), this->Q.end(),heap_min_comparator());

            assert(this->Q_lengths.size()<=this->K);
            
            #ifndef NDEBUG
            dist old_b = bound;
            #endif
            
            this->Q_lengths.push_back(new_path.w);
            std::push_heap(this->Q_lengths.begin(), this->Q_lengths.end());
            
            if(this->Q_lengths.size()>this->K){
                assert(this->Q_lengths.size()==1+this->K);
                std::pop_heap(this->Q_lengths.begin(), this->Q_lengths.end());
                this->Q_lengths.pop_back();   
            }
            if(this->Q_lengths.size()>=this->K){
                bound = this->Q_lengths[0];
            }
            assert(this->Q_lengths.size()<=this->K);

            #ifndef NDEBUG
                assert(old_b >= bound);
                assert(this->Q_lengths.size()<=this->K);
                for(size_t t=0;t<new_path.w+1;t++){
                    assert(ordering[new_path.seq[t]]>=new_path.h);
                }
            #endif

            assert(!is_traced(v,new_path));
            this->trace(v,new_path);
            this->ignore_nodes[local_back_root]=true;
            this->reset_ignored_nodes.push(local_back_root);

        }

        while(!this->reset_ignored_nodes.empty()){
            this->ignore_nodes[this->reset_ignored_nodes.front()]=false;
            this->reset_ignored_nodes.pop();
        }
        while(!this->reset_ignored_edges.empty()){
            this->ignore_edges[this->reset_ignored_edges.front()]=false;
            this->reset_ignored_edges.pop();
        }


    }

    assert(this->Q.empty());
    assert(std::count(this->ignore_nodes, this->ignore_nodes+this->graph->numberOfNodes(), true)==0);
    assert(std::count(this->ignore_edges, this->ignore_edges+this->graph->numberOfEdges(), true)==0);


    assert(this->queue_pruned_reset.empty());
    for(auto camm:listA[v]){
        if(camm.h>=ordering[u])
            continue;
        assert(!k_shorter_simple_paths(u,v,camm.w));
        
        path L,R;                 
        break_path_at(camm, camm.h, L, R);
        add_entry(u, camm.h, L);
        add_entry(v, camm.h, R);
    }
    
}



double kSiSPIndex::init(NetworKit::Graph *G, int K, int ordinamento) {

    this->graph = G;
    this->K = K;


    this->max_dist = graph->numberOfNodes()-1;    
    this->index_size = 0;
    this->prunings = 0;

    this->index = new  std::map<vertex, pathlist>[graph->numberOfNodes()];
    this->listA = new pathlist[graph->numberOfNodes()];
    this->trace_paths = new std::unordered_set<path,HashPaths>[graph->numberOfNodes()];

    this->pruned = new bool[graph->numberOfNodes()];
    this->intersection_tests = new bool[graph->numberOfNodes()];
    this->ignore_nodes = new bool[graph->numberOfNodes()];
    this->ignore_edges = new bool[graph->numberOfEdges()];
    
    this->ordering_rank = new std::pair<double,vertex>[graph->numberOfNodes()];
    this->ordering = new vertex[graph->numberOfNodes()];
    this->reverse_ordering = new vertex[graph->numberOfNodes()];
    this->bidir_bfs_pred = new vertex[graph->numberOfNodes()];
    this->bidir_bfs_succ = new vertex[graph->numberOfNodes()];

    this->graph->parallelForNodes([&] (vertex i){
        assert(graph->hasNode(i));
        this->index[i].clear();
        this->listA[i].clear();
        this->pruned[i]=false;
        this->intersection_tests[i]=false;
        this->ignore_nodes[i]=false;
        this->ordering[i]=null_vertex;
        this->reverse_ordering[i]=null_vertex;
        this->ordering_rank[i]={null_vertex,null_vertex};
        this->bidir_bfs_pred[i]=-1;
        this->bidir_bfs_succ[i]=-1;

    });
    this->graph->parallelForEdges([&] (NetworKit::node i, NetworKit::node j){
        assert(this->graph->edgeId(i,j)==this->graph->edgeId(j,i));
        this->ignore_edges[this->graph->edgeId(i,j)]=false;
    });

    this->num_pruned = 0;
    this->Q.clear();// = new boost::heap::fibonacci_heap<heap_data_hub>();   
    double centr_time = 0.0;

    if(ordinamento==0){

        INFO("BY DEGREE");       
        mytimer local_constr_timer;
        local_constr_timer.restart();
        const NetworKit::Graph& hand = *graph;
        NetworKit::DegreeCentrality* rank = new NetworKit::DegreeCentrality(hand);
        rank->run();
        this->graph->forNodes([&] (vertex i){
            assert(graph->hasNode(i));
            this->ordering_rank[i]=std::make_pair(rank->score(i),i);

        });
        delete rank;
        centr_time = local_constr_timer.elapsed();


    }
    else if(ordinamento==1){
        INFO("BY APX BETW");
        mytimer local_constr_timer;
        double max_time = 30.0;
        double cumulative_time = 0.0;
        double fract = 0.66;
        double n_samples =  round(std::pow((double)graph->numberOfNodes(),fract));

        const NetworKit::Graph& hand = *graph;
        while(cumulative_time<max_time && n_samples<(double)graph->numberOfNodes()){
            local_constr_timer.restart();

            std::cout<<"fract: "<<fract<<" "<<n_samples<<" SAMPLES\n";
            NetworKit::EstimateBetweenness* rank = new NetworKit::EstimateBetweenness(hand,n_samples,false,true);

            
            rank->run();
            
            this->graph->forNodes([&] (vertex i){
                assert(graph->hasNode(i));
                assert(i<graph->numberOfNodes());
                this->ordering_rank[i]=std::make_pair(rank->score(i),i);

            });
            delete rank;
            cumulative_time+=local_constr_timer.elapsed();
            n_samples*=2;
        }
        centr_time = cumulative_time;


    }
    else{
        assert(ordinamento==2);
        INFO("BY kPATH");
        mytimer local_constr_timer;
        local_constr_timer.restart();
        const NetworKit::Graph& hand = *graph;

        NetworKit::KPathCentrality* rank = new NetworKit::KPathCentrality(hand,0.0,round(std::pow((double)graph->numberOfNodes(),0.3)));

            
        rank->run();
            
        this->graph->forNodes([&] (vertex i){
            assert(graph->hasNode(i));
            assert(i<graph->numberOfNodes());
            this->ordering_rank[i]=std::make_pair(rank->score(i),i);

        });
        delete rank;

        centr_time = local_constr_timer.elapsed();
        


    }
        

    std::sort(this->ordering_rank, this->ordering_rank+graph->numberOfNodes(), [](const std::pair<double,vertex>  &a, const std::pair<double,vertex>  &b) {
        if(a.first == b.first){
            return a.second > b.second;
        }
        else{
            return a.first > b.first;
        }
    });
    

    for(size_t count = 0; count < graph->numberOfNodes();count++){
        // std::cout<<count<<" "<<graph->numberOfNodes()<<" "<<this->ordering_rank[count].first<<" "<<this->ordering_rank[count].second<<"\n";
        this->reverse_ordering[count]=this->ordering_rank[count].second;
        this->ordering[this->ordering_rank[count].second]=count;
        

    }

    for(size_t count = 0; count < 10 ;count++)
        std::cout<<"In position "<<count<<" we have vertex "<<this->reverse_ordering[count]<<" rank "<<this->ordering_rank[count].first<<std::endl;

    delete[] this->ordering_rank;

    return centr_time;
    // for(size_t count = 0; count < 10 ;count++)
    //     std::cout<<"Position "<<count<<" of vertex "<<this->ordering[count]<<std::endl;

    // ordering.resize(V);
    // reverse_ordering.resize(V);
    // auto rank = deg->ranking();
    // for(uint32_t s = 0; s < V; s++){
    //     ordering[rank[s].first] = s;
    //     reverse_ordering[s] = rank[s].first;
    //     assert(rank[s].second == graph->degree(rank[s].first));
    // }
    
}

// std::pair<bool, uint32_t> kSiSPIndex::IsPathInLabel(uint32_t hub, uint32_t host, const path_data &path) {

//     std::upper_bound(lengths.begin(),lengths.end(), distance)


//     size_t i = 0;
//     for(; i < index[host][hub].size(); i++){
//         if(index[host][hub][i].size() < vertices.size())
//             continue;
//         if(index[host][hub][i].size() > vertices.size())
//             break;
//         auto c = std::vector<uint32_t> (path.begin(), vertices.end());
//         reverse(c.begin(),c.end());
//         if(index[host][hub][i] == path || index[host][hub][i] == c)
//             return make_pair(true, i);
//     }
//     return make_pair(false, i);
// }


inline bool kSiSPIndex::is_simple(const path& path) {
    
    std::unordered_set<vertex> temp;
    for(size_t t=0;t<path.w+1;t++){
        assert(t==0 || graph->hasEdge(path.seq[t-1],path.seq[t]));
        if(temp.find(path.seq[t])!=temp.end()){
            return false;
        }
        assert(temp.find(path.seq[t])==temp.end());        
        temp.insert(path.seq[t]);
    }
    assert(path.w+1 == temp.size());
    return true;
}

inline bool kSiSPIndex::k_shorter_simple_paths(vertex source, vertex target, dist threshold) {
    
    lengths.clear();
    
    assert(threshold>0);
    

    if(index[source].empty() || index[target].empty()){
        return false;
    }
    
    this->max_hub = min(index[source].rbegin()->first, index[target].rbegin()->first);





    it_s = index[source].begin();
    it_t = index[target].begin();
    short int res_simpl;
    while(true){

        if(it_s == index[source].end() || it_t == index[target].end() || it_s->first > this->max_hub || it_t->first > this->max_hub){
            break;
        }

        if(it_s->first < it_t->first){
            it_s++;
            continue;
        }

        if(it_s->first > it_t->first){
            it_t++;
            continue;
        }
        
        assert(it_s->first == it_t->first);
        for(this->s_s=0;this->s_s!=it_s->second.size();this->s_s++){
            
            #ifndef NDEBUG
            if(this->lengths.size() >= this->K ) {
                for(auto&valore:this->lengths){
                    assert(this->lengths[0]>=valore);
                }
            }
            #endif
            if(this->lengths.size() >= this->K && threshold >= this->lengths[0]){
                return true;
            }

            if(it_s->second[this->s_s].w > threshold){
                break;
            }
            
            // if(find(it_s->second[this->s_s].seq.begin()+1, it_s->second[this->s_s].seq.end(), target) != it_s->second[this->s_s].seq.end()){
            //     continue; 
            // }
            for(this->s_t=0;this->s_t!=it_t->second.size();this->s_t++){

                assert(this->lengths.size() < this->K || threshold < this->lengths[0]);
                #ifndef NDEBUG
                if(this->lengths.size() >= this->K ) {
                    for(auto&valore:this->lengths){
                        assert(this->lengths[0]>=valore);
                    }
                }
                #endif
                if(it_s->second[this->s_s].w+it_t->second[this->s_t].w>threshold){
                    break;
                }
                
                assert(it_s->first==it_t->first);

                res_simpl = is_combination_non_simple_no_allocation(it_s->second[this->s_s], it_t->second[this->s_t]);
                if(res_simpl==1){
                    continue;
                }
                if(res_simpl==2){
                    break;
                }
                assert(res_simpl==0);

                if(this->lengths.empty()){
                    this->lengths.push_back(it_s->second[this->s_s].w+it_t->second[this->s_t].w);
                    continue;
                    
                }

                else if(this->lengths.size()<this->K-1){
                    this->lengths.push_back(it_s->second[this->s_s].w+it_t->second[this->s_t].w);
                    assert(std::find(this->lengths.begin(),this->lengths.end(),it_s->second[this->s_s].w+it_t->second[this->s_t].w)!=this->lengths.end());
                    assert(this->lengths.size() < this->K);
                    continue;

                }
                else{   
                    assert(this->lengths.size() < this->K || threshold < this->lengths[0]);
                    assert(this->lengths.size()>=this->K-1);
                    this->lengths.push_back(it_s->second[this->s_s].w+it_t->second[this->s_t].w);
                    if(this->lengths.size()==this->K){
                        std::make_heap(this->lengths.begin(), this->lengths.end());
                    }
                    else{
                        assert(this->lengths.size()==this->K+1);
                        std::push_heap(this->lengths.begin(), this->lengths.end());
                        std::pop_heap(this->lengths.begin(), this->lengths.end());
                        this->lengths.pop_back();       
                    }
                    //sort(this->lengths.begin(),this->lengths.end());
                    

                    assert(this->lengths.size()==this->K);

                    #ifndef NDEBUG
                    for(auto&valore:this->lengths){
                        assert(this->lengths[0]>=valore);
                    }
                    #endif
                    if(this->lengths.size() >= this->K && threshold >= this->lengths[0]){ //max element is in position 0
                        return true;
                    }   
                    
                    
                    
                }


                

                
            }
        }
        it_s++;
        it_t++;
    }
    


    return false;
}


bool kSiSPIndex::is_path_encoded(vertex s, vertex t, const path& pt) {

    if(index[s].empty() || index[t].empty())
        return false;
    
    this->max_hub = min(index[s].rbegin()->first, index[t].rbegin()->first);
    assert(this->max_hub!=null_vertex);
    
    
    
    it_s = index[s].begin();
    it_t = index[t].begin();

    assert(is_simple(pt));

    while(true){
        if(it_s == index[s].end() || it_t == index[t].end() || it_s->first > this->max_hub || it_t->first > this->max_hub){
            break;
        }

        if(it_s->first < it_t->first){
            it_s++;
            continue;
        }
        if(it_s->first > it_t->first){
            it_t++;
            continue;
        }
        
        assert(it_s->first == it_t->first);

        for(this->s_s=0;this->s_s!=it_s->second.size();this->s_s++){
            
            path& s_path = it_s->second[this->s_s];

            if(s_path.w > pt.w){
                break;
            }
            // if(find(it_s->second[this->s_s].seq.begin()+1, it_s->second[this->s_s].seq.end(), t) != it_s->second[this->s_s].seq.end()){
            //     continue; 
            // }
            for(this->s_t=0;this->s_t!=it_t->second.size();this->s_t++){
                
                path & t_path = it_t->second[this->s_t];

                if(s_path.w+t_path.w>pt.w){
                    break;
                }
                if(s_path.w+t_path.w<pt.w){
                    continue;
                }
                
                assert(s_path.w+t_path.w==pt.w);
                assert(it_s->first==it_t->first);

                assert(reverse_ordering[it_t->first]==s_path.seq[0]);
                assert(reverse_ordering[it_t->first]==t_path.seq[0]);
                assert(s==s_path.seq[s_path.w]);
                assert(t==t_path.seq[t_path.w]);

                assert(pt.seq[0]==s_path.seq[s_path.w]);
                assert(pt.seq[pt.w]==t_path.seq[t_path.w]);

                bool equal = true;
                dist cnt = 0;
                for(;;cnt++){
                    if(s_path.seq[s_path.w-cnt]!=pt.seq[cnt]){    
                        equal=false;
                        break;
                    }   
                    else{
                        if(pt.seq[cnt]==reverse_ordering[it_t->first]){
                            break;
                        }
                    }
                }
                
                
                if(!equal){
                    continue;
                }
                assert(pt.seq[cnt]==t_path.seq[0]);
                assert(pt.seq[cnt]==t_path.seq[cnt-s_path.w]);

                for(;;cnt++){
                    assert(cnt-s_path.w<=t_path.w);
                    if(t_path.seq[cnt-s_path.w]!=pt.seq[cnt]){    
                        equal=false;
                        break;
                    } 
                    else{
                        if(pt.seq[cnt]==t_path.seq[t_path.w]){
                            assert(cnt==pt.w);
                            break;
                        }  
                    }
                }

                
                if(equal){
                    return true;
                }
            }
        }
        it_s++;
        it_t++;
    }
    return false;
}

inline short int kSiSPIndex::is_combination_non_simple_no_allocation(const path &path_u, const path &path_v) {
    
    if(path_u.w + path_v.w > max_dist){
        return 1;
    }

    assert(path_u.seq[0]==path_v.seq[0]);

    

    int32_t idx = path_u.w;
    
    while(idx >= 1){
        if(path_u.seq[idx]==path_v.seq[path_v.seq.size()-1]){
            for(int32_t t=path_u.w;t>idx;t--){
                intersection_tests[path_u.seq[t]] = false;
            }
            #ifndef NDEBUG
            for(dist t=0;t<path_u.w+1;t++)
                assert(!intersection_tests[path_u.seq[t]]);
            #endif
            return 2;
        }
        intersection_tests[path_u.seq[idx]] = true;
        idx--;

    }
    
    idx = 0;


    while(idx < path_v.w + 1){
        
        if(intersection_tests[path_v.seq[idx]]){
            for(dist t=0;t<path_u.w+1;t++){
                intersection_tests[path_u.seq[t]] = false;
            }
            #ifndef NDEBUG
            for(dist t=0;t<path_u.w+1;t++)
                assert(!intersection_tests[path_u.seq[t]]);
            #endif
            
            return 1;
        }
        idx++;
    }

    for(dist t=0;t<path_u.w+1;t++){
        intersection_tests[path_u.seq[t]] = false;
    }
    #ifndef NDEBUG
        for(dist t=0;t<path_u.w+1;t++)
            assert(!intersection_tests[path_u.seq[t]]);
        for(dist t=0;t<path_v.w+1;t++)
            assert(!intersection_tests[path_v.seq[t]]);
    #endif
    
    return 0;
}

// inline void kSiSPIndex::plain_combination(const path &path_u, const path &path_v, path& resulting_path) {
//     resulting_path.clear();
//     for(int t=path_u.w;t>=0;t--)
//         resulting_path.push_back(path_u[t]);
//     for(int t=1;t<=path_v.w;t++)
//         resulting_path.push_back(path_v[t]);

//     assert(resulting_path[0]==path_u[path_u.w]);
//     assert(resulting_path[weight(resulting_path)]==path_v[path_v.w]);
//     return;
// }

inline bool kSiSPIndex::combination_or_intersection(const path& p_u,const path&p_v, vertex hub, path& combination){

    assert(reverse_ordering[hub]==p_u.seq[0]);
    assert(reverse_ordering[hub]==p_v.seq[0]);
    assert(is_simple(p_u));
    assert(is_simple(p_v));

    
    if(p_u.w + p_v.w > max_dist){
        return true;
    }

    combination.seq.resize(p_u.w + p_v.w + 1,0);


    int32_t idx = p_u.w;
    dist len = 0;
    while(idx >= 1){
        assert(!intersection_tests[p_u.seq[idx]]);
        intersection_tests[p_u.seq[idx]] = true;
        combination.seq[len]= p_u.seq[idx];
        idx--;
        len++;
    }
    assert((hub == ordering[p_u.seq[0]] && len==0) || combination.seq[0]==p_u.seq[p_u.w]);
    idx = 0;
    while(idx < p_v.w+1){
        if(intersection_tests[p_v.seq[idx]]){

            for(dist t=0;t<p_u.w+1;t++){
                intersection_tests[p_u.seq[t]] = false;
            }
            #ifndef NDEBUG
                for(dist t=0;t<p_u.w+1;t++)
                    assert(!intersection_tests[p_u.seq[t]]);
            #endif
            combination.seq.clear();

            return true;

        }
        combination.seq[len]=p_v.seq[idx];
        len++;

        idx++;
    }
    assert(combination.seq[0]==p_u.seq[p_u.w]);
    assert(combination.seq[combination.seq.size()-1]==p_v.seq[p_v.w]);
    combination.w = combination.seq.size()-1;
    assert(combination.w==p_u.w + p_v.w);
    combination.h = hub;
    #ifndef NDEBUG
    for(auto&el:combination.seq){
        assert(ordering[el]>=hub);
    }
    #endif
    assert(is_simple(combination));

    for(dist t=0;t<p_u.w+1;t++){
        intersection_tests[p_u.seq[t]] = false;
    }

    #ifndef NDEBUG
    for(dist t=0;t<p_u.w+1;t++)
        assert(!intersection_tests[p_u.seq[t]]);
    for(dist t=0;t<p_v.w+1;t++)
        assert(!intersection_tests[p_v.seq[t]]);
    #endif

    return false;

};


inline bool kSiSPIndex::is_path_list_sorted(const pathlist& plist){
	for (size_t i = 0; i < plist.size()-1; i++){
        if(plist[i+1].w<plist[i].w){
            return false;
        }
    }

	return true;
};

// inline bool kSiSPIndex::is_H_path_list_sorted(const hubpathlist& plist){
// 	for (size_t i = 0; i < plist.size()-1; i++){
//         if(weight(plist[i+1].first)<weight(plist[i].first)){
//             return false;
//         }
//     }

// 	return true;
// };

// inline dist kSiSPIndex::weight(const path& pt) {
//     return pt.size()-1;
// }
inline void kSiSPIndex::print_path(const path& pt) {
    vertex hb = null_vertex;
    for(size_t t=0;t<pt.w+1;t++){
        hb=min(hb,ordering[pt.seq[t]]);
    }

    std::cout<<"H:"<<hb<<"\n";
    for(size_t t=0;t<pt.w+1;t++){
        if(t!=pt.w){
            std::cout << pt.seq[t]<< " ";
        }
        else{
            std::cout << pt.seq[t]<<"\n";
        }
    }

}






void kSiSPIndex::fill_queue_by_query(vertex source, vertex target){

    
    assert(this->Q.empty());

    assert(source!=target);

    if(index[source].empty() || index[target].empty()){
        return;
    }
    int n_occ = 0;

    this->max_hub = min(index[source].rbegin()->first, index[target].rbegin()->first);
    

    // freq.clear();
    it_s = index[source].begin();
    it_t = index[target].begin();
    
    while(true){
        if(it_s == index[source].end() || it_t == index[target].end() || it_s->first > max_hub || it_t->first > max_hub){
            break;
        }
        if(it_s->first < it_t->first){
            it_s++;
            continue;
        }
        if(it_s->first > it_t->first){
            it_t++;
            continue;
        }

        assert(it_s->first == it_t->first);

        for(this->s_s=0;this->s_s!=this->it_s->second.size();this->s_s++){
            
            

            #ifndef NDEBUG
            if(this->Q.size() >= this->K ) {
                for(auto&valore:this->Q){
                    assert(this->Q[0].w>=valore.w);
                }
            }
            #endif
            
            if(this->Q.size() >= this->K && it_s->second[this->s_s].w >= this->Q[0].w){
                #ifndef NDEBUG
                    for(auto& el: this->Q){
                        assert(it_s->second[this->s_s].w>=el.w);
                    }
                #endif
                break;
            }

            
            // if(find(it_s->second[this->s_s].seq.begin()+1, it_s->second[this->s_s].seq.end(), target) != it_s->second[this->s_s].seq.end()){
            //     continue; 
            // }
            

            
            for(this->s_t=0;this->s_t!=this->it_t->second.size();this->s_t++){


                #ifndef NDEBUG
                if(this->Q.size() >= this->K) {
                    for(auto&valore:this->Q){
                        assert(this->Q[0].w>=valore.w);
                    }
                }
                #endif
                if(this->Q.size() >= this->K && it_s->second[this->s_s].w + it_t->second[s_t].w >= this->Q[0].w){     
                    break;
                }
                
                path to_be_built;
                assert(it_s->first==it_t->first);
                
                
                if(combination_or_intersection(it_s->second[this->s_s], it_t->second[s_t], it_s->first, to_be_built)){
                    continue;
                }
                assert(to_be_built.w>0);
                assert(is_simple(to_be_built));

                assert(to_be_built.w==it_s->second[this->s_s].w + it_t->second[s_t].w);

                assert(std::find(this->Q.begin(),this->Q.end(),to_be_built)==this->Q.end());
                
                // it_freq = freq.find(it_s->first);
                // if(it_freq==freq.end()){
                //     freq[it_s->first]=1;
                // }
                // else{
                //     assert(it_freq->first == it_s->first);
                //     it_freq->second++;
                // }

                

                if(this->Q.empty()){
                    this->Q.push_back(to_be_built);
                    continue;
                    
                }
                


                else if(this->Q.size()<this->K-1){
                    assert(std::find(this->Q.begin(),this->Q.end(),to_be_built)==this->Q.end());
                    this->Q.push_back(to_be_built);                    
                    assert(std::find(this->Q.begin(),this->Q.end(),to_be_built)!=this->Q.end());
                    assert(this->Q.size() < this->K);
                    continue;

                }
                else if(this->Q.size()==this->K-1){
                    assert(std::find(this->Q.begin(),this->Q.end(),to_be_built)==this->Q.end());
                    
                    assert(this->Q.size()==this->K-1);
                    this->Q.push_back(to_be_built);
                    std::make_heap(this->Q.begin(), this->Q.end(),heap_max_comparator());
                    assert(this->Q.size()==this->K);
                    assert(is_heap(this->Q.begin(), this->Q.end(),heap_max_comparator()));
                    assert(std::find(this->Q.begin(),this->Q.end(),to_be_built)!=this->Q.end());

                    assert(this->Q.size()==this->K);

                    #ifndef NDEBUG
                    for(auto&valore:this->Q){
                        assert(this->Q[0].w>=valore.w);
                    }
                    #endif
                    continue;
                    
                    
                }
                else{ 
                    assert(to_be_built.w < this->Q[0].w);
                    assert(std::find(this->Q.begin(),this->Q.end(),to_be_built)==this->Q.end());

                    assert(is_heap(this->Q.begin(), this->Q.end(),heap_max_comparator()));
                    

                    assert(this->Q.size()==this->K);
                    this->Q.push_back(to_be_built);
                    std::push_heap(this->Q.begin(), this->Q.end(),heap_max_comparator());

                    assert(is_heap(this->Q.begin(), this->Q.end(),heap_max_comparator()));
                    assert(this->Q.size()==this->K+1);

                    std::pop_heap(this->Q.begin(), this->Q.end(),heap_max_comparator());
                    this->Q.pop_back();    

                    assert(is_heap(this->Q.begin(), this->Q.end(),heap_max_comparator()));
                    assert(this->Q.size()==this->K);
                    continue;

                }
                



                
            }
        }
        it_s++;
        it_t++;
    }
    assert(Q.size()<this->K || is_heap(this->Q.begin(), this->Q.end(),heap_max_comparator()));

    if(Q.size()<this->K){
        std::make_heap(this->Q.begin(), this->Q.end(),heap_max_comparator());
    }
    assert(is_heap(this->Q.begin(), this->Q.end(),heap_max_comparator()));
    // it_freq = freq.find(indice+1);
    // while(it_freq!=freq.end() && it_freq->second==this->K){
    //     indice++;
    //     it_freq = freq.find(indice);
        
    // }


}






   
void kSiSPIndex::query(vertex source, vertex target){

    results.clear();

    if(index[source].empty() || index[target].empty())
        return;
    
    if(source == target){
        assert(results.empty());
        path p_d;
        p_d.init(source,ordering[source]);
        results.push_back(p_d);
        assert(this->is_path_list_sorted(results));
        return;
            
    }
    max_hub = min(index[source].rbegin()->first, index[target].rbegin()->first);
    

    
    it_s = index[source].begin();
    it_t = index[target].begin();
    
    while(true){
        if(it_s == index[source].end() || it_t == index[target].end() || it_s->first > max_hub || it_t->first > max_hub){
            break;
        }
        if(it_s->first < it_t->first){
            it_s++;
            continue;
        }
        if(it_s->first > it_t->first){
            it_t++;
            continue;
        }

        assert(it_s->first == it_t->first);
        for(this->s_s=0;this->s_s!=this->it_s->second.size();this->s_s++){
            
            
            if(this->results.size() >= this->K && it_s->second[this->s_s].w >= this->results[0].w){
                break;
            }
            
            // if(find(it_s->second[this->s_s].seq.begin()+1, it_s->second[this->s_s].seq.end(), target) != it_s->second[this->s_s].seq.end()){
            //     continue; 
            // }

            
            for(this->s_t=0;this->s_t!=this->it_t->second.size();this->s_t++){


                if(this->results.size() >= this->K && it_s->second[this->s_s].w + it_t->second[s_t].w >= this->results[0].w){
                    break;
                }

                path to_be_built;
                assert(it_s->first==it_t->first);

                
                if(combination_or_intersection(it_s->second[this->s_s], it_t->second[s_t], it_s->first, to_be_built)){
                    continue;
                }
                assert(to_be_built.w>0);
                assert(is_simple(to_be_built));

                assert(to_be_built.w==it_s->second[this->s_s].w + it_t->second[s_t].w);
                assert(!is_path_in(results,to_be_built));

                

                if(this->results.empty()){
                    this->results.push_back(to_be_built);
                    continue;
                    
                }
                


                else if(this->results.size()<this->K-1){
                    assert(!is_path_in(results,to_be_built));
                    this->results.push_back(to_be_built);                    
                    assert(is_path_in(results,to_be_built));
                    assert(this->results.size() < this->K);
                    continue;

                }
                else if(this->results.size()==this->K-1){
                    assert(!is_path_in(results,to_be_built));
                    assert(this->results.size()==this->K-1);

                    this->results.push_back(to_be_built);                    
                    std::make_heap(this->results.begin(), this->results.end(),heap_max_comparator());
                    assert(this->results.size() == this->K);
                    assert(is_path_in(results,to_be_built));
                    




                    assert(is_heap(this->results.begin(), this->results.end(),heap_max_comparator()));

                    assert(this->results.size()==this->K);

                    #ifndef NDEBUG
                    for(auto&valore:this->results){
                        assert(this->results[0].w>=valore.w);
                    }
                    #endif
                    continue;

                }
                else{
                    assert(this->results.size()==this->K);
                    assert(to_be_built.w < this->results[0].w);
                    assert(!is_path_in(results,to_be_built));
                    assert(is_heap(this->results.begin(), this->results.end(),heap_max_comparator()));

                    
                    this->results.push_back(to_be_built);
                    std::push_heap(this->results.begin(), this->results.end(),heap_max_comparator());

                    assert(this->results.size()==this->K+1);
                    assert(is_heap(this->results.begin(), this->results.end(),heap_max_comparator()));

                    std::pop_heap(this->results.begin(), this->results.end(),heap_max_comparator());
                    this->results.pop_back();    

                    assert(this->results.size()==this->K);
                    assert(is_heap(this->results.begin(), this->results.end(),heap_max_comparator()));
                    
                    


                    #ifndef NDEBUG
                    for(auto&valore:this->results){
                        assert(this->results[0].w>=valore.w);
                    }
                    #endif

                    
                    
                    
                }
                
            }
        }
        it_s++;
        it_t++;
    }
    assert(results.size()<this->K || is_heap(this->results.begin(), this->results.end(),heap_max_comparator()));

    if(results.size()<this->K){
        std::make_heap(this->results.begin(), this->results.end(),heap_max_comparator());
    }
    assert(is_heap(this->results.begin(), this->results.end(),heap_max_comparator()));
    std::sort_heap(this->results.begin(), this->results.end(),heap_max_comparator());
    assert(is_path_list_sorted(results));
}

dist kSiSPIndex::length_query(vertex source, vertex target){

    lengths.clear();
    

    if(index[source].empty() || index[target].empty()){
        return null_distance;
    }
    assert(source!=target);

    
    max_hub = min(index[source].rbegin()->first, index[target].rbegin()->first);
    


    it_s = index[source].begin();
    it_t = index[target].begin();
    short int res_simpl;
    while(true){

        if(it_s == index[source].end() || it_t == index[target].end() || it_s->first > this->max_hub || it_t->first > this->max_hub){
            break;
        }

        if(it_s->first < it_t->first){
            it_s++;
            continue;
        }

        if(it_s->first > it_t->first){
            it_t++;
            continue;
        }
        
        assert(it_s->first == it_t->first);
        for(this->s_s=0;this->s_s!=it_s->second.size();this->s_s++){
            
            #ifndef NDEBUG
            if(this->lengths.size() >= this->K ) {
                for(auto&valore:this->lengths){
                    assert(this->lengths[0]>=valore);
                }
            }
            #endif

            if(this->lengths.size() >= this->K && it_s->second[this->s_s].w >= this->lengths[0]){
                break;
            }
            // if(find(it_s->second[this->s_s].seq.begin()+1, it_s->second[this->s_s].seq.end(), target) != it_s->second[this->s_s].seq.end()){
            //     continue; 
            // }

            for(this->s_t=0;this->s_t!=it_t->second.size();this->s_t++){


                
                assert(it_s->first==it_t->first);

                if(this->lengths.size() >= this->K && it_s->second[this->s_s].w + it_t->second[this->s_t].w >= this->lengths[0]){
                    break;
                }
                res_simpl = is_combination_non_simple_no_allocation(it_s->second[this->s_s], it_t->second[this->s_t]);
                if(res_simpl==1){
                    continue;
                }
                if(res_simpl==2){
                    break;
                }
                assert(res_simpl==0);
                

                if(this->lengths.empty()){
                    this->lengths.push_back(it_s->second[this->s_s].w+it_t->second[this->s_t].w);
                    continue;
                    
                }
                else if(this->lengths.size()<this->K-1){    
                    this->lengths.push_back(it_s->second[this->s_s].w+it_t->second[this->s_t].w);
                    assert(this->lengths.size() < this->K);
                    continue;

                }
                else if(this->lengths.size()==this->K-1){    
                    this->lengths.push_back(it_s->second[this->s_s].w+it_t->second[this->s_t].w);
                    std::make_heap(this->lengths.begin(), this->lengths.end());
                    assert(this->lengths.size() == this->K);
                    assert(std::is_heap(this->lengths.begin(), this->lengths.end()));

                    continue;

                }
                
                else{   
                    assert(this->lengths.size()==this->K);
                    assert(it_s->second[this->s_s].w+it_t->second[this->s_t].w < this->lengths[0]);
                    assert(std::is_heap(this->lengths.begin(), this->lengths.end()));
                    this->lengths.push_back(it_s->second[this->s_s].w+it_t->second[this->s_t].w);


                    assert(this->lengths.size()==this->K+1);
                    std::push_heap(this->lengths.begin(), this->lengths.end());
                    std::pop_heap(this->lengths.begin(), this->lengths.end());
                    this->lengths.pop_back();            
                    assert(this->lengths.size()==this->K);

                    assert(std::is_heap(this->lengths.begin(), this->lengths.end()));


                    #ifndef NDEBUG
                    for(auto&valore:this->lengths){
                        assert(this->lengths[0]>=valore);
                    }
                    #endif
                    continue;
                }



                
            }
        }
        it_s++;
        it_t++;
    }
    assert(lengths.size()<this->K || is_heap(this->lengths.begin(), this->lengths.end()));


    if(this->lengths.size() < this->K){
        return null_distance;
    }
    else{
        assert(std::is_heap(this->lengths.begin(), this->lengths.end()));

        assert(this->lengths[0]<null_distance);
        #ifndef NDEBUG
        for(auto&valore:this->lengths){
            assert(this->lengths[0]>=valore);
        }
        #endif
        return this->lengths[0];
    }


    
}