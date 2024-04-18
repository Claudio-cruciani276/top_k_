/*
 * KBFS_GLOBAl.HPP
 *
 *  Created on: 
 *      Author: Claudio Cruciani
 */

#ifndef KBFS_GLOBAL_HPP_
#define KBFS_GLOBAL_HPP_

#include "networkit/graph/Graph.hpp"
#include "progressBar.h"
#include <boost/functional/hash.hpp>

#include <iostream>
#include <set>
#include <vector>

#include <list>

#include <deque>
#include <limits>
#include <cstdint>


using vertex = int32_t;
using dist = uint32_t;
using edge_id = uint64_t;

class path {
public:
    std::vector<vertex> seq;
    dist w;

    static const vertex null_vertex; //= round(std::numeric_limits<vertex>::max() / 2);
    static const dist null_distance; //= round(std::numeric_limits<dist>::max() / 2);

    using iterator = std::vector<vertex>::iterator;
    //using const_iterator = std::vector<vertex>::const_iterator;

    iterator begin() { return seq.begin(); }
    iterator end() { return seq.end(); }
    //const_iterator begin() const { return seq.begin(); }
    //const_iterator end() const { return seq.end(); }

    //default
    path() : w(0) { } // 

    // single vertex
    path(vertex v) : w(0) { 
        seq.push_back(v);
    }
    
    //from lsit
    path(const std::list<vertex>& vertices) : w(vertices.size() - 1) {
        seq.assign(vertices.begin(), vertices.end());   
    }   

    //from vectorrr
    path(const std::vector<vertex>& vertices) : w(vertices.size() - 1) {
    seq = vertices; 
    }

    
    // from set of vertices
    path(const std::set<vertex>& vertices) : w(vertices.size()-1) { 
        seq.assign(vertices.begin(), vertices.end());
    }
    

    // Costruttore da due path da combinare
    path(const path& path1, const path& path2) : w(path1.seq.size() + path2.seq.size()-1) { 
        seq.reserve(path1.seq.size() + path2.seq.size()); // Ottimizza la memoria
        seq.insert(seq.end(), path1.seq.begin(), path1.seq.end());
        seq.insert(seq.end(), path2.seq.begin(), path2.seq.end());
    }
    // escludiamo l'ultimo nodo del primo percorso solo se è uguale al primo del secondo percorso
    path(const path& path1, const path& path2, bool exclude_last_if_same) {
        if (exclude_last_if_same && !path1.seq.empty() && !path2.seq.empty() && path1.seq.back() == path2.seq.front()) {
            // Aggiungi tutti gli elementi di path1 tranne l'ultimo
            seq.insert(seq.end(), path1.seq.begin(), path1.seq.end() - 1);
        } else {
            // Altrimenti, aggiungi tutti gli elementi di path1
            seq.insert(seq.end(), path1.seq.begin(), path1.seq.end());
        }
        // Aggiungi tutti gli elementi di path2
        seq.insert(seq.end(), path2.seq.begin(), path2.seq.end());
        // Calcola la lunghezza del percorso (numero di archi)
        w = (seq.size() > 1) ? seq.size() - 1 : 0;
    }

    // Costruttore a partire da un path esistente e prende i primi l nodi
    path(const path& other, size_t l) {
        if (l > other.seq.size()) {
            l = other.seq.size();  // Assicura che l non superi la lunghezza di other.seq
        }
        seq.assign(other.seq.begin(), other.seq.begin() + l);
        w = (seq.size() > 1) ? seq.size() - 1 : 0;  
    }

    template<typename Iter>
    path(Iter begin, Iter end) : seq(begin, end), w(std::distance(begin, end) - 1) {}

    void addVertex(vertex v) {
        seq.push_back(v);
        w = (seq.size() > 1) ? seq.size() - 1 : 0;
    }

    // Restituisce la lunghezza della sequenza di nodi(NON IL PESO)
    size_t getSize() const {
        return seq.size();
    }

    bool isEmpty() const {
        return seq.empty();
    }

    // Metodo per ottenere il primo nodo del percorso
    vertex getFirstNode() const {
        if (isEmpty()) {
            //throw std::range_error("Attempted to access the first node of an empty path.");
            return null_vertex;
        }
        return seq.front();
    }

    vertex getLastNode() const {
        if (!seq.empty()) {
            return seq.back();
        }
        //throw std::runtime_error("Path is empty");
        return null_vertex;
    }

    void printPath() const {
        std::cout << "Path (w=" << w << "): ";
        for (const auto& vertex : seq) {
            std::cout << vertex << " ";
        }
        std::cout << "\n";
    }



    bool operator==(const path& other) const {
        if (w != other.w) return false; 
        return seq == other.seq; 
    }

    bool operator!=(const path& other) const {
        return !(*this == other);
    }

    bool operator<(const path& other) const {
        if (seq.size() != other.seq.size()) {
            return seq.size() < other.seq.size();
        }
        return seq < other.seq;  // Compara le sequenze lessicograficamente
    }  

    // AAA non è necessario definire un distruttore esplicito,
    // grazie alla gestione automatica della memoria di std::vector e 
    //alla non presenza di altre risorse dinamiche nella classe.
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
//AD ESEMPIO:
//#include <queue>
//std::priority_queue<path, std::vector<path>, heap_min_comparator> pq;
// in questo modo pq è una priority queue che ordina gli oggetti
// path in modo che l'oggetto con il valore w più basso sia sempre in cima. L'uso di heap_min_comparator 
//inverte l'ordinamento predefinito da max-heap a min-heap.



//  viene usato per definire una funzione hash
// personalizzata per gli oggetti della classe path
class HashPaths {
public:
    // id is returned as hash function
    size_t operator()(const path& t) const
    {
        return boost::hash_range(t.seq.rbegin(),t.seq.rend());//t.seq,t.seq+t.w);
    }
};

// Definizione delle eccezioni specifiche per la ricerca di percorsi
class PathException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error; 
};

class NoPathException : public PathException {
public:
    using PathException::PathException; 
};


//using entry = std::tuple<dist, vertex, path, std::set<vertex>, std::string, vertex, std::vector<path>>;


struct YenEntry {
    dist d; 
    path p; 

    // Costruttore 
    YenEntry(dist d, path p) : d(d), p(p) {}

    // Definizione delll'operatore < per il confronto, necessario per l'heap
    bool operator<(const YenEntry& other) const {
        return d > other.d; // Per un min-heap basato sulla distanza
    }
};



using entry = std::tuple<dist, vertex, path, std::set<vertex>, std::string, vertex, std::vector<path>>;

// Comparatore per l'ordinamento crescente basato su 'dist', SIMILE AL CODICE PROF  di heap_min_comparator
struct CompareEntry {
    bool operator()(const entry& a, const entry& b) const {
        // Confronta basandosi sul primo elemento delle tuple, cioè 'dist'
        return std::get<0>(a) > std::get<0>(b);
    }
};




using pathlist = std::vector<path>;


class KBFS_global{
public:

    //static const vertex null_vertex; 
    //static const dist null_distance; 
    
    NetworKit::Graph* graph;
    size_t K;
    vertex root;


    // Costruttore
    KBFS_global(NetworKit::Graph* G, int num_k, vertex r);
    //KBFS_global(int num_k, vertex r);
    //double init(NetworKit::Graph*, int, int);
    void printInfo() const;
    void printStructures() const;
    void printEntryPQ(const entry&);
    void printPQ() const;
    void printDistanceProfile() const;
    void printTopK() const;
    void printPredecessorsSet() const ;
    bool binary_TEST();
    //vv
    void standard(dist WEIG, vertex VERT, const path& PATH, const std::set<vertex>& PATHSET, const std::string& flag, vertex source, const pathlist& paths);
    void generalized_bfs();
    void beyond(dist WEIG, vertex VERT, const path& PATH, const std::set<vertex>& PATHSET, const std::string& flag, const pathlist& paths);

    std::vector<path> find_detours(vertex source, vertex target, size_t at_most_to_be_found, dist dist_path, const path& coming_path, size_t count_value) ;
    path bidir_BFS(vertex source, vertex target, dist bound_on_length);

    void init_avoidance(const path& avoid);
    void clean_avoidance();
    int  count(vertex vr);
    bool is_simple(const path& p);

    void PathsForward(std::set<vertex>& neighbors, const std::vector<path>&  paths);
    bool binary_search_alt(std::deque<std::pair<dist,path>>& arr, const path& x);
    void bidir_pred_succ(vertex source, vertex target, dist bound_on_length, std::unordered_map<vertex, vertex>& pred, std::unordered_map<vertex, vertex>& succ, vertex& intersection);

    // Distruttore
    ~KBFS_global();
    
    // Metodi della classe 

private:
     
    //std::priority_queue<entry, std::vector<entry>, CompareEntry> pq;

    //ora è un vectorpoi verrà trasfomata in heap cosi: std::make_heap(pq.begin(), pq.end(), CompareEntry());
    std::vector<entry> PQ;

    std::int64_t pruned_branches;
    std::vector<std::deque<path>> top_k;
    std::vector<bool> detour_done;
    std::vector<std::vector<path>> path_to_add; 
    std::vector<std::set<vertex>> predecessors_set; 
    //std::vector<vertex> pigreco_set; 
    std::vector<std::set<vertex>> pigreco_set;
    std::vector<dist> last_det_path;

    std::vector<bool> ignore_nodes;
    std::deque<vertex> queue_ignore_nodes; 

    std::vector<bool> locally_ignore_nodes;
    std::vector<bool> locally_ignore_edges; // edges ancora da capire il formato
    std::deque<vertex> locally_queue_ignore_nodes;
    std::deque<dist> locally_queue_ignore_edges; // Sostituisci 'dist' con il tipo effettivo sarebbe edge!!!!!!!

    std::vector<std::deque<dist>> queue_dist_profile; 

    std::vector<dist> bound; 

    std::vector<bool> visited;
    std::deque<vertex> queue_visited;

    std::vector<bool> non_sat;
    int num_non_sat;

    std::int64_t extra_visits; // Usa std::int64_t se necessario
    //std::deque<int> distance_profile; // Sostituisci 'int' 
    std::vector<std::deque<std::pair<dist,path>>> distance_profile; // Sostituisci 'int' 
    std::vector<path> detours; 


    //BFS e yen
    std::unordered_map<vertex, vertex> pred, succ;
    vertex intersection;
};




#endif // KBFS_GLOBAL_HPP_