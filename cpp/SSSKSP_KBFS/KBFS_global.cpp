
#include "KBFS_global.hpp"
#include <algorithm> 
#include <cassert>  
#include <boost/functional/hash.hpp> // Assumendo che boost::hash_range sia usato per HashPaths


//gli elmenti della libreria std sono allocati dinamicamente in modo automatico, new non necessario
KBFS_global::KBFS_global(NetworKit::Graph *G, int num_k, vertex r)
    : graph(G), 
    K(num_k),
    root(r),
    pruned_branches(0),
    top_k(G->numberOfNodes()),
    detour_done(G->numberOfNodes(), false),
    predecessors_set(G->numberOfNodes()),
    pigreco_set(G->numberOfNodes()), 
    last_det_path(G->numberOfNodes(),0), // Usa std::numeric_limits per 'null_distance'
    ignore_nodes(G->numberOfNodes(), false),
    locally_ignore_nodes(G->numberOfNodes(), false),
    locally_ignore_edges(G->numberOfEdges(), false), // Assumendo un modo per ottenere il numero di archi
    //queue_dist_profile(G->numberOfNodes()),
    distance_profile(G->numberOfNodes()),
    bound(G->numberOfNodes(), std::numeric_limits<dist>::max()),
    visited(G->numberOfNodes(), false),
    non_sat(G->numberOfNodes(), true),
    num_non_sat(G->numberOfNodes() - 1),
    path_to_add(G->numberOfNodes()),
    extra_visits(0),
    detours (),
    queue_visited()
{
    non_sat[root] = false; 
    // Altre inizializzazioni possono andare qui se necessario
}


void KBFS_global::printInfo() const {
        std::cout << "KBFS Object Information:" << std::endl;
        std::cout << "Number of shortest paths (K): " << K << std::endl;
        std::cout << "Root vertex: " << root << std::endl;
        if (graph) {
            std::cout << "Graph vertices: " << graph->numberOfNodes() << std::endl;
            std::cout << "Graph edges: " << graph->numberOfEdges() << std::endl;
        } else {
            std::cout << "Graph is not initialized." << std::endl;
        }
    }



// Implementazione del distruttore
KBFS_global::~KBFS_global() {
    // Dato che graph è un puntatore a NetworKit::Graph passato nel costruttore,
    // la gestione della memoria di graph è responsabilità di chi fornisce il puntatore.
    // Non deallocare graph qui(non usare delete graph; qui).

    // Non ci sono allocazioni dinamiche esplicite (con new) per gli altri membri,
    // quindi non ci sono operazioni specifiche di deallocazione richieste nel distruttore.
}



//const vertex KBFS_global::null_vertex = round(std::numeric_limits<vertex>::max()/2);
//const dist KBFS_global::null_distance = round(std::numeric_limits<dist>::max()/2);

void KBFS_global::generalized_bfs() {
    //serve per un assert successivo
    size_t num_neighbors_of_root = 0;
        
    for (const auto& deque : top_k) {
        assert(deque.empty()); // Assicurati che ogni deque in top_k sia vuoto.
    }
    // Invece, verifica che non_sat[root] sia false.
    assert(!non_sat[root]);
    
    PQ.clear();
    
    //std::cout << "Metto in coda i vicini della radice "<<root<<std::endl;
    // Itera sui vicini del nodo root e aggiungi entry corrispondenti in PQ
    graph->forNeighborsOf(root, [&](NetworKit::node u) {
        ++num_neighbors_of_root;
        assert(non_sat[u]);
      
        //std::cout << "Vicino: "<<u<<" --> ";

        path p;
        p.addVertex(root);
        p.addVertex(u);
        std::set<vertex> setptx; 
        setptx.insert(root);
        setptx.insert(u);

        // Crea l'entry per PQ
        entry e = std::make_tuple(1, u, p,setptx, "regular", KBFS_global::null_vertex, std::vector<path>());
        PQ.push_back(e);

        //printEntryPQ(e);

        // Aggiungi la pair (1, path) nella deque corrispondente a u in distance_profile
        distance_profile[u].push_back(std::make_pair(1, p));

        // Verifica che distance_profile[u] sia ordinato
        assert(std::is_sorted(distance_profile[u].begin(), distance_profile[u].end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
         }));   
    });

    //std::cout << std::endl;
    //printDistanceProfile();
    //printPQ();
    // Verifica che il numero di entry inserite in PQ sia uguale al numero di vicini di root
    assert(PQ.size() == num_neighbors_of_root);

    // Converti PQ in un min heap(vedi codice prof)
    std::make_heap(PQ.begin(), PQ.end(), CompareEntry());

    while (!PQ.empty()) { 
        // Prima estrarre l'elemento minimo dall'heap, assicurarsi che sia in cima
        std::pop_heap(PQ.begin(), PQ.end(), CompareEntry());
        // Ora, l'elemento minimo è all'ultima posizione di PQ
        entry e = PQ.back();
        //std::cout << std::endl;
        //std::cout << "Estrazione da PQ -> visita sulla entry";
        //printEntryPQ(e);
        

        PQ.pop_back(); // Rimuovi l'elemento dall'heap
        //structured bindings
        auto [wgt, vtx, ptx, setptx, flag, source, paths] = e;


        assert(wgt == ptx.seq.size() - 1);
        assert(ptx.seq.front() == root && ptx.seq.back() == vtx);
        assert(vtx != root);
        assert(ptx.seq.size() == setptx.size());
        assert(num_non_sat > 0); 
        bool ptxFound = std::find(top_k[vtx].begin(), top_k[vtx].end(), ptx) != top_k[vtx].end();
        assert(!ptxFound);  //ptx non sia trovato in top_k[vtx]
        if (top_k[vtx].size() < K) {
            assert(non_sat[vtx]);
            //assert(wgt == distance_profile[vtx].front().first);
            
            distance_profile[vtx].pop_front();
            
            //si esegue una visita regolare
            //std::cout << "VISITA STANDARD: ";
            standard(wgt, vtx, ptx, setptx, flag, source, paths);
            // Non è necessario "eliminare" ptx o setptx in C++ come in Python. 
            // Se necessario, pulisci o reimposta le variabili qui.
            //printDistanceProfile();
            //std::cout << " FINE VISITA STANDARD"<<std::endl;
            if (num_non_sat == 0) {
                PQ.clear(); 
                return;
            }
        }
        else {
            if (flag == "spec" && predecessors_set[source].find(vtx) != predecessors_set[source].end()) {
                std::set<vertex> neighbors;
                PathsForward(neighbors,paths);
                /*
                for (const auto& path : paths) {
                    if (path.empty() || !non_sat[path.back()] || path.size() > bound[path.back()]) continue;
                    
                    vertex firstNode = path.front();
                    std::vector<vertex> subPath(path.begin() + 1, path.end());
                    neighbors.insert(firstNode);
                    path_to_add[firstNode].push_back(subPath);
                }*/
                
                // Processa ogni vicino unico
                for (vertex ngx : neighbors) {
                    //if (PATHSET.find(ngx) != PATHSET.end() || ngx == root) continue; // Salta se ngx è già in PATHSET o è la root

                    //assert(graph->hasEdge(vtx, ngx)); // Assicurati che ngx sia effettivamente un vicino di vtx

                    path newPath = ptx;
                    newPath.addVertex(ngx); // Aggiungi ngx al nuovo percorso

                    std::set<vertex> newSetptx = setptx;
                    newSetptx.insert(ngx); // Aggiungi ngx al set di vertici visitati

                    // Crea e aggiungi la nuova entry in PQ
                    entry newEntry = std::make_tuple(wgt + 1, ngx, newPath, newSetptx, "spec", source, path_to_add[ngx]);
                    PQ.push_back(newEntry);
                    path_to_add[ngx].clear(); // Resetta path_to_add per ngx
            

                    // PQ ordinato dopo l'inserimento di nuove entry
                    assert(std::is_sorted(PQ.begin(), PQ.end(), CompareEntry()));

                    if (non_sat[ngx]) {
                        if (detour_done[ngx]) {
                            
                            bool found = binary_search_alt(distance_profile[ngx], newPath);
                            // < di ptx o di newPath!!!!!!!!!!!!!!!!!!
                            if (last_det_path[ngx] < newPath.w) {
                                detour_done[ngx] = false;
                                last_det_path[ngx] = 0;
                            }
                        } else {
                            
                            distance_profile[ngx].push_back(std::make_pair(newPath.w, newPath));

                        }
                        // Aggiornamenti specifici per 'flag' "spec"
                        // (Potrebbe includere aggiornamenti a detour_done, last_det_path, ecc., a seconda delle necessità)
                    }
                    // Verifica che distance_profile[ngx] sia ordinato per distanza dopo l'inserimento
                     assert(std::is_sorted(distance_profile[ngx].begin(), distance_profile[ngx].end(), 
                        [](const auto& a, const auto& b) { return a.first < b.first; }));
                }
                //non è necessario "eliminare" esplicitamente variabili locali
                //come si farebbe in Python con del. In Python. Basta tenere conto dello scope delle variabili  

            }
            else{
                assert(!non_sat[vtx]); // Verifica che non_sat[vtx] sia false
                //??assert(distance_profile[vtx].empty()); // Verifica che distance_profile[vtx] sia vuoto
                //Versione python ha un of che qui probabilmente non serve
                //std::cout << "VISITA EXTRA: ";
                beyond(wgt, vtx, ptx, setptx, flag, paths); 
            }

        } 
    }   
    printTopK(); 
}



void KBFS_global::standard(dist WEIG, vertex VERT, const path& PATH, const std::set<vertex>& PATHSET, const std::string& flag, vertex source, const pathlist& paths) {
    assert(top_k[VERT].size() < K);
    //AGGIUNGO IL CAMMINO MINIMO ESTRATTO
    top_k[VERT].push_back(PATH);
    
    assert(std::is_sorted(top_k[VERT].begin(), top_k[VERT].end(), [](const path& a, const path& b) {
        return a.seq.size() < b.seq.size();
    }));

    assert(top_k[VERT].size() <= K);

    if (top_k[VERT].size() == K) {
        assert(non_sat[VERT]); // Verifica che VERT sia effettivamente considerato non saturato
        
        non_sat[VERT] = false;
        num_non_sat--; // Decrementa il conteggio dei nodi non saturi

        // Verifica che il conteggio dei non saturi sia corretto
        //size_t count = std::count(non_sat.begin(), non_sat.end(), true);
        assert(std::count(non_sat.begin(), non_sat.end(), true) == num_non_sat);

        // Pulisci LA distance profile per VERT
        distance_profile[VERT].clear();

        // Se non ci sono più nodi non saturi, termina l'esecuzione
        if (num_non_sat == 0) {
            return;
        }
    }
    // è tutto un controllo
    bool isNeighbor = false;
    graph->forNeighborsOf(VERT, [&](NetworKit::node u) {
        if (u == *(PATH.seq.end() - 2)) {
            isNeighbor = true;
        }
    });
    assert(isNeighbor);
    /*
    //sarebbe il nodo prima di VERT
    //non si aggiunge
    vertex penultimoNodo = *(PATH.seq.end() - 2);
    predecessors_set[VERT].insert(penultimoNodo);
    */
    // Inserisci tutti i nodi del percorso escluso l'ultimo in predecessors_set
    //il file python aggiungeva solo il penultimo!strano!
    //è un set, non serve fare il controllo dell'unicità
    for (size_t i = 0; i < PATH.seq.size() - 1; ++i) {
        predecessors_set[VERT].insert(PATH.seq[i]);
    }


    std::set<vertex> neighbors;

    //si potrebbe spostare fuori? prima della chiamata di STANDARD?
    if (flag == "spec") {
        PathsForward(neighbors,paths);   //fare dei TEST
    }

    assert(num_non_sat > 0); 

    graph->forNeighborsOf(VERT, [&](NetworKit::node ngx) {
        if (PATHSET.find(ngx) != PATHSET.end()) { //non devo includere vicini che appartegono a path(cammini semplici)
            return; // il "continue" in Python il return termina l'esecuzione dalla lambda function corrente
        }
        
        assert(ngx != root);

        // Creazione di un nuovo newPATHSET uguale a PATHSET con ngx aggiunto
        std::set<vertex> newPATHSET(PATHSET);
        newPATHSET.insert(ngx);
        
        path newPath = PATH; // Crea una copia del path corrente
        newPath.addVertex(ngx); 
        
        //ngx è in neighbors
        if (neighbors.find(ngx) != neighbors.end()) {
            //Controllo se l'entry esiste già in PQ
            //std::cout<<"Aggiunta in PQ ";
            entry newEntry = std::make_tuple(WEIG + 1, ngx,newPath, newPATHSET, flag, source, path_to_add[ngx]);
            //printEntryPQ(newEntry);
            assert(std::find(PQ.begin(), PQ.end(),std::make_tuple(WEIG + 1, ngx, newPath, newPATHSET, flag, source, path_to_add[ngx])) == PQ.end());
            PQ.push_back(std::make_tuple(WEIG + 1, ngx,newPath, newPATHSET, flag, source, path_to_add[ngx]));
            path_to_add[ngx].clear(); // equivalente a self.path_to_add[ngx]=[] in Python
        } else {
            //Controllo se l'entry esiste già in PQ
            //std::cout<<"Aggiunta in PQ ";
            entry newEntry = std::make_tuple(WEIG + 1, ngx, newPath, newPATHSET, "reg", null_vertex, std::vector<path>());
            //printEntryPQ(newEntry);
            assert(std::find(PQ.begin(), PQ.end(),std::make_tuple(WEIG + 1, ngx, newPath, newPATHSET, "reg", null_vertex, std::vector<path>())) == PQ.end());
            PQ.push_back(std::make_tuple(WEIG + 1, ngx, newPath, newPATHSET, "reg", null_vertex, std::vector<path>()));
        }   
   
       // assert(std::is_sorted(PQ.begin(), PQ.end(), CompareEntry()));


        if (non_sat[ngx]) {
            // la lunghezza di PATH da size_t, serve dist
            //dist lenPath = static_cast<dist>(PATH.seq.size());

            if (detour_done[ngx] && !(flag == "reg" && newPath.w == last_det_path[ngx])) {
                //su python si lavorava con Path perchè è una copia e l'aggiunta di ngx non faceva modifiche sull0tiginale
                //qui usiaimo newPath, creato per non mofdificare l'originale
                bool res = binary_search_alt(distance_profile[ngx], newPath);
                // nel file python controlla len(PATH) invece di newPath.w
                if (last_det_path[ngx] < newPath.w) {
                    last_det_path[ngx] = 0;
                    detour_done[ngx] = false;
                }
            } else {
                distance_profile[ngx].push_back(std::make_pair(newPath.w, newPath));
            }
            // Verifica che distance_profile[ngx] sia ordinato per distanze
            assert(std::is_sorted(distance_profile[ngx].begin(), distance_profile[ngx].end(),
                [](const std::pair<dist, path>& a, const std::pair<dist, path>& b) {
                    return a.first < b.first;
                }));

            // Verifica aggiuntiva che l'elemento sia effettivamente quello con la distanza minore se PQ non è vuoto
            assert(non_sat[ngx]); // Verifica che ngx sia effettivamente considerato non saturato
        }   

    });     

}


void KBFS_global::PathsForward(std::set<vertex>& neighbors,const std::vector<path>& paths) {
    
    for (const auto& p : paths) {
        // Controlla se il percorso è vuoto o se il nodo finale non è saturo
        // oppure se la lunghezza del percorso supera il bound (path.seq.size() > bound[path.seq.back())
        // attenzione sul terzo controllo perchè i p in paths non partono dalla root!!! !!!!!!!!!
        if (p.seq.empty() || !non_sat[p.seq.back()] || p.w > bound[p.seq.back()]) {
            continue;
        }
 
        vertex firstNode = p.seq.front();
        std::vector<int> subVector(p.seq.begin() + 1, p.seq.end());
        path subPath(subVector);
        
        neighbors.insert(firstNode);
        
        // Se path_to_add deve accettare vettori di vertex come sottopercorsi, potrebbe essere necessaria una conversione
        //path newPath(subPath);
        // Imposta newPath.w o qualsiasi altra proprietà necessaria
        path_to_add[firstNode].push_back(subPath);
    }
}


//distance_profile update 
bool KBFS_global::binary_search_alt(std::deque<std::pair<dist,path>>& arr, const path& x) {
    //std::cout<<"binary_search_alt start: "<<std::endl;
    //std::cout<<" prima "<<std::endl;
    //printDistanceProfile();
    // lambda come comparatore che confronta la lunghezza dei path
    auto comp = [](const std::pair<dist, path>& a, const std::pair<dist, path>& b) {
        return a.first< b.first;
        //return a.seq.size() < b.seq.size();
    };

    auto searchValue = std::make_pair(x.w, x);
    // Trova la posizione usando std::lower_bound basato sulla lunghezza di x
    auto it = std::lower_bound(arr.begin(), arr.end(),searchValue, comp);

    // Cerca se x è già presente in arr con la stessa lunghezza
    //while (it != arr.end() && it->seq.size() == x.seq.size()) {
    while (it != arr.end() && it->first == x.w) {
        if (it->second == x) {
            //assert(std::find(arr.begin(), arr.end(), searchValue) != arr.end());
            //std::cout<<" GIA PRESENTE "<<std::endl;
            return true;
        }
        ++it;
    }

     //aggiungi x in posizione it

    // Se x non è presente, inseriscilo nella posizione trovata
    //assert(std::find(arr.begin(), arr.end(), x) == arr.end());
    arr.insert(it, std::make_pair(x.w, x));
    //std::cout<<" dopo "<<std::endl;
    //printDistanceProfile();

    // Verifica che arr sia ancora ordinato per lunghezza dopo l'inserimento
    assert(std::is_sorted(arr.begin(), arr.end(), comp));

    return false;
} 

void KBFS_global::beyond(dist WEIG, vertex VERT, const path& PATH, const std::set<vertex>& PATHSET, const std::string& flag, const pathlist& paths) {
    extra_visits += 1; 
    assert(num_non_sat > 0); 
    // Verifica che il nodo VERT sia considerato saturato, che ci siano almeno K percorsi in top_k[VERT], 
    // che PATH non sia in top_k[VERT]
    assert(!non_sat[VERT] && top_k[VERT].size() >= K && std::find(top_k[VERT].begin(), top_k[VERT].end(), PATH) == top_k[VERT].end());
    // Verifica che la radice e VERT siano in PATHSET
    assert(PATHSET.find(root) != PATHSET.end());
    assert(PATHSET.find(VERT) != PATHSET.end());
    //assert(!visited[VERT]);
    //assert(!visited[root]);
    std::set<vertex> skip; // Inizializza il set di nodi da saltare
    std::set<vertex> neighbors; 

    if (flag == "spec") {
        //PathsForward(neighbors,paths);  ma serve anche skip!!!
        for (const auto& p : paths) {
            // Controlla se il percorso è vuoto o se il nodo finale non è saturo
            // oppure se la lunghezza del percorso supera il bound (path.seq.size() > bound[path.seq.back())
            if (p.seq.empty() || !non_sat[p.seq.back()] || p.w > bound[p.seq.back()]) {
                continue;
            }

            vertex firstNode = p.seq.front();
            std::vector<int> subVector(p.seq.begin() + 1, p.seq.end());
            path subPath(subVector);
            
            neighbors.insert(firstNode);
            
            // Se path_to_add deve accettare vettori di vertex come sottopercorsi, potrebbe essere necessaria una conversione
            //path newPath(subPath);
            // Imposta newPath.w o qualsiasi altra proprietà necessaria
            skip.insert(p.seq.back());
            path_to_add[firstNode].push_back(subPath);
        }
    }
    size_t num_tested = 0;
    
    //se il pigreco set di vert esiste e non è vuoto
    //"""ragionare sul fatto che se è si svuotato nel corso dell'esecuzione
    //questo if non deve essere comunque superato"""
    if (!pigreco_set[VERT].empty()) {
        init_avoidance(PATH);
        auto it = pigreco_set[VERT].begin();
        while (it != pigreco_set[VERT].end()) {
            vertex vr = *it;
            if (!non_sat[vr]) {
                it = pigreco_set[VERT].erase(it); // Rimuove l'elemento corrente e probabilmente spoesta anche l'it al prossimo
                continue;
            } 
            if(skip.find(vr) != skip.end()){
                ++it; 
                continue;
            } 
            ++it; 
             

            // nodo vr deve essere non saturato
            assert(non_sat[vr]);

            // Verifica che distance_profile[vr] sia ordinato per lunghezza dei percorsi
            assert(std::is_sorted(distance_profile[vr].begin(), distance_profile[vr].end(),
                                [](const auto& a, const auto& b) { return a.first < b.first; }));

            int count_value = count(vr);
            int max_to_generate = K - (top_k[vr].size() + count_value);

            if (max_to_generate <= 0) {
                num_tested++;
                continue;
            }

            //int n_generated = 0;
            num_tested++;        
            
           
            //"""Generazione dei detours,il codice python restituisce una DET per voltr
            //e qualndo restituisce un Det==none"""
            std::vector<path> detours = find_detours(VERT, vr, max_to_generate, PATH.seq.size() - 1, PATH, count_value);
            for (const auto& DET : detours) {
                //if (DET.seq.empty()) break;

                //n_generated++;
                 

                //SERIE DI CONTROLLI, ATTENTO AGLI INDICI
                // è tutto un controllo
                //A
                bool isNeighbor = false;
                graph->forNeighborsOf(VERT, [&](NetworKit::node u) {
                    if (u == DET.seq[1]) {
                        isNeighbor = true;
                    }
                });
                assert(isNeighbor); 
                //assert(graph->hasEdge(VERT, DET.seq[0])); // DET[1] è il primo nodo nel detour//sarebbe il check che ho fatto nelle righe pirma
                //B
                assert(PATHSET.find(DET.seq[1]) == PATHSET.end());//per garantire percorsi semplici
                assert(!ignore_nodes[DET.seq[1]] || neighbors.find(DET.seq[1]) != neighbors.end());
                // ancora una serie di verifiche sul nuovo path
                path newPath = PATH; 
                for (size_t i = 1; i < DET.seq.size(); ++i) { // Inizia da 1 per evitare di duplicare il vertice di giunzione
                     newPath.addVertex(DET.seq[i]);
                }
                //1
                assert(this->is_simple(newPath));

                // 2. Verifica che il nuovo percorso non sia già in PQ
                bool foundInPQ = std::any_of(PQ.begin(), PQ.end(), [&newPath](const entry& e) {
                    const path& existingPath = std::get<2>(e);
                    return existingPath == newPath; 
                });
                assert(!foundInPQ);

                // 3. Verifica sulla lunghezza del nuovo percorso
                assert(newPath.seq.size() - 1 == PATH.seq.size() - 1 + DET.seq.size() - 1);

                // 4. Verifica sul numero di vicini
                assert(neighbors.size() <= graph->degree(VERT) - 1);

 

                vertex detFirstVertex = DET.seq[1];
                path detSubPath(DET.seq.begin() + 1, DET.seq.end()); // Crea un sottopercorso escludendo il primo vertice di DET

                // Verifica se detFirstVertex non è tra i vicini e nel caso lo mette
                if (neighbors.find(detFirstVertex) == neighbors.end()) {
                    neighbors.insert(detFirstVertex);
                    path_to_add[detFirstVertex].push_back(detSubPath);
                }
                else{
                    // Se il sottopercorso non è già presente per detFirstVertex, aggiungilo
                    //"""ragionare se questo controllo sull'unicità è esseziale"""
                    auto& pathsForVertex = path_to_add[detFirstVertex];
                    if (std::find(pathsForVertex.begin(), pathsForVertex.end(), detSubPath) == pathsForVertex.end()) {
                        pathsForVertex.push_back(detSubPath);
                    }
                    
                    assert(neighbors.find(detFirstVertex) != neighbors.end());
                    assert(neighbors.size() <= graph->degree(VERT) - 1);

                    /*  SU PYTHON C'È DUE
                    if (n_generated == max_to_generate) {
                        """PERCHE DISTANCE_PROFILE È SVUOTATA? ragionare """
                        distance_profile[VERT].clear(); 
                        detours.clear(); 
                        break; // Esce dal ciclo
                    }*/
                
                }/*
                if (n_generated == max_to_generate) {
                """PERCHE DISTANCE_PROFILE È SVUOTATA? ragionare """
                distance_profile[VERT].clear(); 
                detours.clear(); 
                break; // Esce dal ciclo
                }
                */
            } 
            if (num_tested == num_non_sat) {
                break;
            }
        }
        auto it2 = pigreco_set[VERT].begin();
        while (it2 != pigreco_set[VERT].end()) {
            if (!non_sat[*it2]) {
                it2 = pigreco_set[VERT].erase(it2);
            } else {
                ++it2;
            }
        }

        assert(num_non_sat > 0);
        clean_avoidance();

    } 
    else{   //se pigreco set è vuoto
        // Svuota l'insieme pigreco_set[VERT],clear semplicemente si assicura che è vuoto
        pigreco_set[VERT].clear();

        visited[VERT] = true;
        queue_visited.push_back(VERT);
        visited[root] = true;
        queue_visited.push_back(root);

        assert(VERT != root);

        std::deque<vertex> localPQ;

        //costruisco pgreco
        for (vertex prd : predecessors_set[VERT]) {
        // Salta i predecessori già visitati
            if (visited[prd]) {
                continue;
            }

            assert(prd != root); 
            assert(!visited[prd]); 
            visited[prd] = true; 
            queue_visited.push_back(prd); 

            localPQ.push_back(prd); 
        }

        if (!visited[root]) {
            visited[root] = true;
            queue_visited.push_back(root);
        }

        init_avoidance(PATH);

        while (!localPQ.empty()) {
            vertex vr = localPQ.front(); // Ottiene il primo elemento
            localPQ.pop_front(); // Rimuove il primo elemento dalla coda
            bool detour = true;
            assert(vr != root);
            assert(visited[vr] == true);
            assert(num_tested < num_non_sat);

            if (non_sat[vr]) {
                assert(pigreco_set[VERT].find(vr) == pigreco_set[VERT].end()); // Verifica che vr non sia già in pigreco_set[VERT]
                pigreco_set[VERT].insert(vr); 
                //""" l'assert va rivista"""// Verifica che distance_profile[vr] sia ordinato
                assert(std::is_sorted(distance_profile[vr].begin(), distance_profile[vr].end(), [](const auto& a, const auto& b) { return a.first < b.first; }));
                int count_value = count(vr);
                int max_to_generate = K - (top_k[vr].size() + count_value);
                // se passa l'if non generare detours
                if (max_to_generate <= 0 || skip.find(vr) != skip.end()) {
                    detour = false;
                    // Se non è necessario generare detours, considera il prossimo elemento in localPQ
                    //continue;
                }


                int n_generated = 0;
                num_tested++;
                if (detour) {
                    auto detours = find_detours(VERT, vr, max_to_generate, PATH.w, PATH, count_value);
                    for (const auto& DET : detours) {
                        //if (DET.seq.empty()) break; // Interrompe il ciclo se il detour è vuoto

                        //n_generated++;

                        //assert(graph->hasEdge(VERT, DET.seq[1])); // scrivere l'assert in c++
                        assert(PATHSET.find(DET.seq[1]) == PATHSET.end()); // Verifica che il secondo nodo di DET non sia già in PATHSET
                        /*
                        // Verifica che il nuovo percorso non sia già presente in PQ
                        auto it = std::find_if(PQ.begin(), PQ.end(), [&DET](const entry& e) {
                            return std::get<2>(e).seq == DET.seq; // Confronta le sequenze di vertici
                        });
                        assert(it == PQ.end()); // Assicura che il nuovo percorso non sia in PQ
                        */

                        vertex detFirstVertex = DET.seq[1];
                        //"""controllare: DET+1 o DET+2"""
                        path detSubPath_temp(DET.seq.begin() + 1, DET.seq.end());
                        path detSubPath(PATH,detSubPath_temp); // Escludi il primo nodo di DET dal sottopercorso

                        //""" se non funziona il costtruttore con i due iteratori fare:"""
                        //std::vector<vertex> subVector(DET.seq.begin() + 2, DET.seq.end());
                        //path detSubPath(subVector);
                        assert(is_simple(detSubPath)); 
                        

                        
                        if (neighbors.find(detFirstVertex) == neighbors.end()) {
                            neighbors.insert(detFirstVertex);
                            path_to_add[detFirstVertex].push_back(detSubPath);
                        } else {
                                //Ragionare se occorre questo controllo
                                auto& pathsForVertex = path_to_add[detFirstVertex];
                                if (std::find(pathsForVertex.begin(), pathsForVertex.end(), detSubPath) == pathsForVertex.end()) {
                                    pathsForVertex.push_back(detSubPath);
                                }
                                /*
                                if (n_generated == max_to_generate) {
                                    // Pulisci detours se hai generato il numero massimo di detours
                                    """ragionare perchè pulire anche distance_profile"""
                                    distance_profile[VERT].clear();
                                    detours.clear();
                                    break; // Esce dal ciclo
                                } */
                                                    
                        }/* 
                        if (n_generated == max_to_generate) {
                            // Ripeti la pulizia se hai generato il numero massimo di detours
                            distance_profile[VERT].clear();
                            detours.clear();
                            break; // Esce dal ciclo
                        }*/

                    }

                }
                if (num_tested == num_non_sat) {
                    // Se localPQ fosse una variabile locale che non richiede una cancellazione esplicita, semplicemente termina il ciclo
                    break;
                }
              
            }

            assert(num_tested < num_non_sat);
            assert(num_non_sat > 0);
            //predecessori dei predecessoti....
            for (auto prd : predecessors_set[vr]) {
                if (visited[prd]) {
                    continue; 
                }
                assert(prd != root); 
                visited[prd] = true; 
                queue_visited.push_back(prd); 
                localPQ.push_back(prd);
            }

            
        }

        // Ripristina lo stato di 'visited' per i nodi che erano stati visitati
        while (!queue_visited.empty()) {
            vertex x = queue_visited.front();
            assert(visited[x]); 
            visited[x] = false; 
            queue_visited.pop_front(); 
        }

        assert(num_non_sat > 0); // Verifica che ci siano ancora nodi non saturati
        clean_avoidance();        

     //
    }
    // se passa l'if avviene una potatura!
    if (neighbors.empty()) {
        pruned_branches += 1;
        return;
    }


    for (const auto& ngx : neighbors) {
        assert(PATHSET.find(ngx) == PATHSET.end());
        assert(std::find(PATH.seq.begin(), PATH.seq.end(), ngx) == PATH.seq.end());
        //assert(graph->hasEdge(VERT, ngx)); cercare il rispettico per c++
        assert(PATH.seq.back() == VERT);

        path newPath = PATH;
        newPath.addVertex(ngx);
        std::set<vertex> newPATHSET(PATHSET);
        newPATHSET.insert(ngx);

        entry newEntry(WEIG + 1, ngx, newPath, newPATHSET, "spec", VERT, path_to_add[ngx]);
        assert(std::find(PQ.begin(), PQ.end(), newEntry) == PQ.end());

        PQ.push_back(newEntry);
        path_to_add[ngx].clear();

        assert(std::is_sorted(PQ.begin(), PQ.end(), CompareEntry()));

        if (non_sat[ngx]) {
            if (detour_done[ngx]) {
                //qui nel codice python fa le successive operazioni con PATH e non newPath 
                //ma hon ha troppo senso
                if (binary_search_alt(distance_profile[ngx], newPath)) {
                    if (last_det_path[ngx] < newPath.w) {
                        detour_done[ngx] = false;
                        last_det_path[ngx] = 0;
                    }
                }
            } else {
                    distance_profile[ngx].push_back({newPath.w, newPath});
                    assert(std::is_sorted(distance_profile[ngx].begin(), distance_profile[ngx].end(), 
                                    [](const std::pair<dist, path>& a, const std::pair<dist, path>& b) { return a.first < b.first; }));
            }
        }   
    }


}



void KBFS_global::init_avoidance(const path& avoid) {
    assert(!avoid.seq.empty() && avoid.seq.front() == root);
    //dovrebbe escludere l'ultimo elemento?
    for (size_t i = 0; i < avoid.seq.size() - 1; ++i) {
        vertex u = avoid.seq[i];
        assert(!ignore_nodes[u]);
        ignore_nodes[u] = true;
        queue_ignore_nodes.push_back(u);
    }
}

void KBFS_global::clean_avoidance() {
    while (!queue_ignore_nodes.empty()) {
        // Ottiene il primo elemento della coda
        vertex x = queue_ignore_nodes.front();
        assert(ignore_nodes[x]); 
        ignore_nodes[x] = false; 
        queue_ignore_nodes.pop_front(); 
    }
}




//numero di cammini in distance_profile[vr] che hanno lo stesso peso dell'ultimo cammino minimo trovato per vr
int KBFS_global::count(vertex vr) {
    //peso del cammino più lungo tra quelli dei cammini minimi di vr
    dist wg = top_k[vr].back().w;
    int count = 0;
    for (auto& pair : distance_profile[vr]) {
        if (wg == pair.first) {
            count++;
        } else {
            break;
        }
    }
    return count;
}


bool KBFS_global::is_simple(const path& p) {

    assert(!p.seq.empty());
    //si potrebbe fare il controllo per vedere che due vicini nel path sono effettivamente vixini nel grafo
    //in c++ non è immediato
    std::set<vertex> s_path;
    for (size_t i = 0; i < p.seq.size(); ++i) {
        // Controlla se il vertice è già presente nel set (cioè verifica che non ci siano ripetizioni)
        if (!s_path.insert(p.seq[i]).second) {
            // Se l'inserimento fallisce, significa che il vertice era già presente: il percorso non è semplice
            assert(s_path.size() < p.seq.size());
            return false;
        }
    }
   // Se il set e il percorso hanno la stessa dimensione, non ci sono duplicati: il percorso è semplice
    assert(s_path.size() == p.seq.size());
    return true;
}


std::vector<path> KBFS_global::find_detours(vertex source, vertex target, size_t at_most_to_be_found, dist dist_path, const path& coming_path, size_t count_value) {
    // Verifiche iniziali con asserzioni
    assert(top_k[target].size() >= 1);
    assert(top_k[target].size() < K);
    assert(std::is_sorted(top_k[target].begin(), top_k[target].end(), [](const path& a, const path& b) { return a.seq.size() < b.seq.size(); }));
    assert(top_k[source].size() == K);
    assert(dist_path==coming_path.getSize()-1);

    //int n_generated = 0;
    std::vector<path> detours_found; 

    std::vector<YenEntry> yen_PQ; 
    //trasforma yen_PQ in un heah
    //ho definito il comparatore per strutture del tipo YenEntry
    std::make_heap(yen_PQ.begin(), yen_PQ.end());
    std::set<path> paths; 

    assert(!ignore_nodes[source]);

    if (ignore_nodes[target]) {
        //assert(sin_distance_profile.empty());
        return detours_found; 
    }

    bound[target] = std::numeric_limits<dist>::max(); 
    
    //assert(sin_distance_profile.empty());
    assert(detours_found.empty());
    assert(detours.empty()); 
    //count_value non influisce su bound
    if (distance_profile[target].size() - count_value >= at_most_to_be_found) {
        size_t index = count_value + at_most_to_be_found;
        // cosi garantisco l'indice non superi la dimensione del vettore
        if (index < distance_profile[target].size()) {
            bound[target] = std::min(bound[target], distance_profile[target][index].first);
        }
    }


    /*"""??
    if (bound[target] == top_k[target].back().size()- 1) {
        return detours_found;
        //return std::vector<path>(); // Simula 'yield None' quando la condizione è verificata
    }"""  */


    
    // Inizia cercando il primo cammino minimo 
    try {
        path altP = bidir_BFS(source, target, bound[target] - dist_path);
        //assert(!altP.empty())
        assert(altP.getFirstNode() == source && altP.getLastNode() == target) ;
        //controllare altP.w+dist_path+1, se serve l'1
        assert(altP.w+dist_path+1<bound[target]);
        YenEntry entry(altP.w , altP); 
        yen_PQ.push_back(entry); // Add entry 
        std::push_heap(yen_PQ.begin(), yen_PQ.end()); // Mantieni le proprietà dell'heap
        paths.insert(altP); 
    } catch (const NoPathException& e) {
        //sinDistanceProfile.clear(); // 
        //detours.clear();
        return detours_found;// Restituisce un vettore vuoto indicando l'assenza di ulteriori percorsi
    } catch (const std::exception& e) {
        // Gestisci altre eccezioni potenziali
        std::cerr << "Unexpected error: " << e.what() << std::endl;
        throw; // Rilancia l'eccezione per ulteriore gestione
    }


   while (!yen_PQ.empty()) {
        std::pop_heap(yen_PQ.begin(), yen_PQ.end()); // Prepara l'elemento minore per l'estrazione
        YenEntry minEntry = yen_PQ.back(); // Ottieni l'elemento minore
        yen_PQ.pop_back(); // Rimuovi l'elemento dall'heap
        
        path P_det = minEntry.p; //  percorso dall'entry estratta

        //detours.push_back(P_det)
        
        if (distance_profile[target].size() - count(target) >= at_most_to_be_found &&
            P_det.getSize()- 1 + dist_path >= distance_profile[target][count_value + at_most_to_be_found - 1].first) {
            // Pulisci le strutture e termina
            // cancellare PQ,paths ma c++ gestisce le variabili locali
            //detours.clear(); 
            return detours_found;
        }

        // Verifica che la lunghezza del percorso estratto rispetti il bound
        assert(P_det.seq.size() - 1 + dist_path < bound[target]);
        //Attento al costruttore per due path con vrtice comune
        path newPath(coming_path,P_det,true);
        assert(newPath.w == (P_det.getSize()- 1 + dist_path));
        //"""vedere gli indici,perchè su python esclude l'ultimo?"""
        //newPath.seq.insert(newPath.seq.end(), P_det.seq.begin()+1, P_det.seq.end()); // Combina i percorsi
        

        if (distance_profile[target].size() - count(target) >= at_most_to_be_found) {
            size_t index = count_value + at_most_to_be_found;
            // Assicurati che l'indice non superi la dimensione del vettore
            if (index < distance_profile[target].size()) {
                bound[target] = std::min(bound[target], distance_profile[target][index].first);
            }
        }        

        //se non hanno il vertice in comune devo levare -1 a P_det.w + distPath
        if (P_det.seq.size() -1 + dist_path >= bound[target]) {
            //yen_PQ.clear(); 
            //paths.clear(); 
            //detours_found.clear(); 
            return detours_found; // Restituisci un vettore vuoto per indicare che non ci sono ulteriori percorsi o per simulare 'yield None'            
        }  

        // Effettua la ricerca binaria sul distance_profile
        //
        if (binary_search_alt(distance_profile[target], newPath)) {
            continue; // Se il percoso già è in distance_profile continua con la prossima estrazione
        }
        //global sistance Profile
        //binary_search_alt(distance_profile[target], newPath);

        //distance_profile[target] modificata da YEN
        detour_done[target] = true;

        assert(std::is_sorted(distance_profile[target].begin(), distance_profile[target].end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; }));
        //assert(std::is_sorted(sin_distance_profile.begin(), sin_distance_profile.end()));
        assert(P_det.seq.front() == source);
        assert(P_det.seq.back() == target);
        

        if (newPath.w > last_det_path[target]) {
            last_det_path[target] = newPath.w;
        }

        //SUITABLE PATH
        detours_found.push_back(P_det); //paths da ritornare
        detours.push_back(P_det);//per yen

        if (detours_found.size() == at_most_to_be_found) {
            return detours_found;
        }


        assert(detours_found.size() < at_most_to_be_found);
        assert(std::is_sorted(detours_found.begin(), detours_found.end(), [](const path& a, const path& b) {
            return a.w < b.w;
        }));

        /* """controlla
        if (bound[target] == top_k[target].back().w) {
            return detours_found;
        }   """  */
    
        
        for (NetworKit::node u : graph->nodeRange()) {
                assert(!locally_ignore_nodes[u]);
        } 

    
        for (size_t index = 1; index < P_det.getSize()-1; ++index) {

            //inizialmente stava prima del for
            // Asserzione che verifica se tutti gli archi non sono localmente ignorati
            graph->forEdges([&](NetworKit::node u, NetworKit::node v, NetworKit::index edgeId) {
            assert(!locally_ignore_edges[edgeId]);
            });

            // l_root come un vettore fino all'indice corrente
            //path l_root(P_det.seq.begin(), P_det.seq.begin() + index);
            path l_root(P_det,index);
                
            for (const auto& current_path : detours) {
                path path_to_evaluate(current_path,index);

                // Confrontare l_root con i percorsi esistenti in 'detours'
                if (path_to_evaluate==l_root) {
                    NetworKit::node u = current_path.seq[index - 1];
                    NetworKit::node v = current_path.seq[index];
                    NetworKit::edgeid eid = graph->edgeId(u, v);
                    // Se l'arco non è già marcato come ignorato, marcarlo e aggiungerlo alla coda
                    if (!locally_ignore_edges[eid]) {
                        locally_ignore_edges[eid] = true;
                        locally_queue_ignore_edges.push_back(eid);
                    }
                }            
            }
            
            //Fatto sotto però ad ogni iterazione metto lo spur node tra i nodi da ignoarare considerato che diventerà il penultimo nodo nella prissima iteraizoen
            //invece di fare ogni volta un for
            //for (size_t i = 0; i < l_root.getSize() - 1; ++i) {
            //    vertex node = l_root.seq[i];
                
                // Check and Set Ignore State
            //    if (!locally_ignore_nodes[node]) {
            //        locally_ignore_nodes[node] = true;
            //        locally_queue_ignore_nodes.push_back(node);
            //    }
            //}       

    


            try {
                vertex lRootLastVertex = l_root.getLastNode(); 
                path pgt = bidir_BFS(lRootLastVertex, target, bound[target] - (dist_path + l_root.w));
                

                size_t combinedPathLength = pgt.w - 1 + (l_root.w);
                assert(dist_path + combinedPathLength < bound[target]);

                // Costruttore passando true: lRoot (escluso l'ultimo elemento) +  pgt
                path newPath(l_root,pgt,true);

                assert(newPath.w < bound[target] - dist_path);
                assert(!newPath.isEmpty() && newPath.getFirstNode() == source && newPath.getLastNode()  == target);
                assert(is_simple(newPath));

                //ad ogni iterazione metto lo spur node tra i nodi da ignoarare considerato che diventerà il penultimo nodo nella prissima iteraizoen
                if (!locally_ignore_nodes[lRootLastVertex]) {
                   locally_ignore_nodes[lRootLastVertex] = true;
                   locally_queue_ignore_nodes.push_back(lRootLastVertex);
                }   

                //se il percorso è già in paths
                if (paths.find(newPath) != paths.end()) {
                    //vertex lastNodeBeforeTarget = l_root.back(); 
                    //if (!locally_ignore_nodes[lRootLastVertex]) {
                    //    locally_ignore_nodes[lRootLastVertex] = true;
                    //    locally_queue_ignore_nodes.push_back(lRootLastVertex);
                    //}
                    // Continua con la prossima iterazione del ciclo
                    continue;
                }

                //YenEntry newEntry(static_cast<dist>(newPath.size() - 1), newPath);
                YenEntry newEntry(newPath.w, newPath);
                yen_PQ.push_back(newEntry);
                // Trasforma yen_PQ in un heap (o ristabilisce la proprietà di heap se era già un heap)
                std::push_heap(yen_PQ.begin(), yen_PQ.end());

                // Aggiunta del nuovo percorso a paths
                paths.insert(newPath);


                
            } catch (const NoPathException& e) {
                //  gestione l'eccezione specifica NoPathException
            } catch (const std::exception& e) {
                // Qui gestisci altre eccezioni inaspettate
                std::cerr << "Unexpected error: " << e.what() << std::endl;
                throw; // Puoi rilanciare l'eccezione per ulteriore gestione esterna
            }

            //fatto sopra
            //if (!locally_ignore_nodes[lRootLastVertex]) {
            //    locally_ignore_nodes[lRootLastVertex] = true;
            //    locally_queue_ignore_nodes.push_back(lRootLastVertex);
            //}

            // gli archi vanno ripristinati ad ogni iterazione 
            while (!locally_queue_ignore_edges.empty()) {
                edge_id e_id = locally_queue_ignore_edges.front();
                assert(locally_ignore_edges[e_id]); // Verifica che l'arco e_id sia marcato come ignorato localmente.
                locally_ignore_edges[e_id] = false; // Rimarca l'arco e_id come non ignorato.
                locally_queue_ignore_edges.pop_front(); // Rimuove e_id dalla coda degli archi da ignorare localmente.
            } 
            
        }
        
        //i nodi vanno ripristinati ad ogni estrazione di YENPQ
        while (!locally_queue_ignore_nodes.empty()) {
                vertex x = locally_queue_ignore_nodes.front();
                assert(locally_ignore_nodes[x]); // Verifica che il nodo x sia marcato come ignorato localmente.
                locally_ignore_nodes[x] = false; // Rimarca il nodo x come non ignorato.
                locally_queue_ignore_nodes.pop_front(); // Rimuove x dalla coda dei nodi da ignorare localmente.
        } 



    } 
    
    //distance_profile.clear(); // Pulisce il vettore distance_profile
    //detours.clear(); // Pulisce il vettore detours
    paths.clear();
    yen_PQ.clear();
    



    //return std::vector<path>(paths.begin(), paths.end()); // Converti set in vector per il ritorno
    return detours_found;
}


path KBFS_global::bidir_BFS(vertex source, vertex target, dist bound_on_length) {
    
    pred.clear();
    succ.clear();
    intersection = null_vertex; 

    bidir_pred_succ(source, target, bound_on_length, pred, succ, intersection);
    /* fatto con ECCEZIONE sotto
    if (intersection == null_vertex) {
        return path();  // Ritorna un percorso vuoto se non c'è intersezione
    }*/
    
    if (intersection == null_vertex) {
        throw NoPathException("No path found within the given bound length.");
    }

    vertex current = intersection;

    // Costruzione del percorso dalla sorgente all'intersezione
    path pre_path;  // Uso un vector temporaneo per costruire il percorso inverso
    while (current != null_vertex && current != source) {
        pre_path.addVertex(current);
        current = pred[current];
    }
    pre_path.addVertex(source);  // Aggiungi la sorgente all'inizio del percorso
    std::reverse(pre_path.begin(), pre_path.end());  // Inverte il percorso per avere l'ordine corretto

    // Costruzione del percorso dall'intersezione alla destinazione
    current = intersection;
    path post_path;
    while (current != null_vertex && current != target) {
        post_path.addVertex(current);
        current = succ[current];
    }
    post_path.addVertex(target);  // Aggiungi il target alla fine del percorso
    
    path result_path(pre_path,post_path,true);
    // Concatenazione dei due percorsi
    //result_path.seq.insert(result_path.seq.end(), pre_path.begin(), pre_path.end());
    //result_path.seq.insert(result_path.seq.end(), post_path.begin() + 1, post_path.end());  // Evita duplicazione dell'intersezione
    //result_path.w = result_path.seq.size() - 1;

    assert(result_path.w < bound_on_length);
    return result_path;
}

 

void KBFS_global::bidir_pred_succ(vertex source, vertex target, dist bound_on_length, std::unordered_map<vertex, vertex>& pred, std::unordered_map<vertex, vertex>& succ, vertex& intersection) {
    
    if (ignore_nodes[source] || ignore_nodes[target] || locally_ignore_nodes[source] || locally_ignore_nodes[target]) {
        throw NoPathException("Source or target node is ignored.");
    }

    if (target == source) {
        intersection = source; // L'intersezione è il nodo stesso se sorgente e destinazione coincidono
        pred[source] = null_vertex; // nessun predecessore
        succ[target] = null_vertex; // Nessun successore 
        return;
    }

    pred[source] = null_vertex; // Inizializza il predecessore della sorgente a null
    succ[target] = null_vertex; // Inizializza il successore del target a null

    // Fringes per la ricerca in avanti e indietro
    std::deque<vertex> forward_fringe = {source};
    std::deque<vertex> reverse_fringe = {target};

    dist distance = 0; 

    
    //std::unordered_map<vertex, bool> forward_visited;
    //std::unordered_map<vertex, bool> reverse_visited;
    //forward_visited[source] = true;
    //reverse_visited[target] = true;

    while (!forward_fringe.empty() && !reverse_fringe.empty()) {
        distance++;
        if (distance >= bound_on_length) {
            throw NoPathException("No path exists smaller/equal the bound length.");
        }

        if (forward_fringe.size() <= reverse_fringe.size()) {
            std::deque<vertex> next_level;
            for (vertex v : forward_fringe) {
                graph->forNeighborsOf(v, [&](vertex w) {
                    //QUESTI RETURN INTERROMPONO L'ATTUALE ITERAZIONE DELLA LAMBDA FUNCTION
                    if (locally_ignore_edges[graph->edgeId(v, w)]) return;
                    if (w == source || ignore_nodes[w] || locally_ignore_nodes[w]) return;

                    if (succ.count(w)) {  
                        intersection = w;  // INTERSEZIONE TROVATA!!!
                        return;
                    }
                    //le mappe forniscono il metodo count() che ritorna 0 se la chiave specificata non è presente nella mappa e 1 se la chiave è presente
                    if (!pred.count(w)) {  
                        next_level.push_back(w);
                        pred[w] = v;  
                    }
                });
                // Stop SE L'INTERSEZIONE È TROVATA,QUI È OBBL. PERCHE IL RETURN NON TERMINA LA CHIAMATA MA ESCE DALLA LAMBDA
                if (intersection != null_vertex) break;  
            }
            //MOVE consente di trasferire le risorse di un oggetto (come la memoria allocata) da un oggetto a un altro, invece di copiarle.
            // utile per migliorare le performance quando gli oggetti contengono grandi quantità di dati allocati dinamicament
            //USARLO ANCHE IN ALTRE SITUAZIONI
            forward_fringe = std::move(next_level);
        } else {
            std::deque<vertex> next_level;
            for (vertex v : reverse_fringe) {
                graph->forNeighborsOf(v, [&](vertex w) {
                    if (locally_ignore_edges[graph->edgeId(v, w)]) return;
                    if (w == target || ignore_nodes[w] || locally_ignore_nodes[w]) return;

                    if (pred.count(w)) {  
                        intersection = w;  // ITERSEZIONE TROVATA
                        return;
                    }

                    if (!succ.count(w)) {  
                        next_level.push_back(w);
                        succ[w] = v;  
                    }
                });

                if (intersection != null_vertex) break;  // Stop SE INTERSEZIONE TROVATA
            }
            reverse_fringe = std::move(next_level);
        }

        if (intersection != null_vertex) break;  // Exit while loop 
    }

    if (intersection == null_vertex) {
        throw NoPathException("No intersection found.");
    }
    

}


bool KBFS_global::binary_TEST(){
    std::set<vertex> verticesSet1 = {1,2};
    std::set<vertex> verticesSet2= {1,3};
    std::set<vertex> verticesSet4= {1,2,3};
    std::set<vertex> verticesSet5= {1,2,4,7};
    std::set<vertex> verticesSet6= {2,3,4,10};
    std::set<vertex> verticesSet7= {5,4,7,8};
    std::set<vertex> verticesSet8= {2,3,4,7};
    std::set<vertex> verticesSet9= {11,10,3,2,4};
    std::set<vertex> verticesSet3= {1,8};
    std::set<vertex> verticesSet10= {1,5,7,9};

    path setPath1(verticesSet1);
    path setPath2(verticesSet2);
    path setPath4(verticesSet4);
    path setPath5(verticesSet5);
    path setPath6(verticesSet6);
    path setPath7(verticesSet7);
    path setPath8(verticesSet8);
    path setPath9(verticesSet9);
    path setPath3(verticesSet3);
    path setPath10(verticesSet10);
    
    
    distance_profile[0].push_back(std::make_pair(setPath1.w,setPath1));
    distance_profile[0].push_back(std::make_pair(setPath2.w,setPath2));
    distance_profile[0].push_back(std::make_pair(setPath4.w,setPath4));
    distance_profile[0].push_back(std::make_pair(setPath5.w,setPath5));
    distance_profile[0].push_back(std::make_pair(setPath6.w,setPath6));
    distance_profile[0].push_back(std::make_pair(setPath7.w,setPath7));
    distance_profile[0].push_back(std::make_pair(setPath8.w,setPath8));
    distance_profile[0].push_back(std::make_pair(setPath9.w,setPath9));

    //distance_profile[0].push_back(std::make_pair(setPath3.w,setPath3));
    //distance_profile[0].push_back(std::make_pair(setPath1.w,setPath1));
    //distance_profile[0].push_back(std::make_pair(setPath1.w,setPath1));

    printDistanceProfile();

    binary_search_alt(distance_profile[0],setPath3);
    binary_search_alt(distance_profile[0],setPath10);

    printDistanceProfile();

    return true;



}







void KBFS_global::printTopK() const {
    std::cout << "Contenuto di Top K:" << std::endl;
    for (size_t i = 0; i < top_k.size(); ++i) {
        std::cout << "Vertice " << i << ":" << std::endl;
        const auto& paths = top_k[i];
        for (const auto& path : paths) {
            std::cout << "  Path (w=" << path.w << "): ";
            for (const auto& vertex : path.seq) {
                std::cout << vertex << " ";
            }
            std::cout << std::endl;
        }
        if (paths.empty()) {
            std::cout << "  Nessun percorso." << std::endl;
        }
    }
}

void KBFS_global::printPredecessorsSet() const {
    std::cout << "Contenuto di Predecessors Set:" << std::endl;
    for (size_t i = 0; i < predecessors_set.size(); ++i) {
        std::cout << "Predecessori di vertice " << i << ": ";
        for (const auto& pred : predecessors_set[i]) {
            std::cout << pred << " ";
        }
        std::cout << std::endl;
    }
}



void KBFS_global::printEntryPQ(const entry& e) {
    auto [wgt, vtx, ptx, setptx, flag, source, paths] = e;
    std::cout << "Weight: " << wgt << ", Vertex: " << vtx << ", Path: ";
    for (auto v : ptx.seq) {
        std::cout << v << " ";
    }
    std::cout << ", Set: {";
    for (auto s : setptx) {
        std::cout << s << " ";
    }
    std::cout << "}, Flag: " << flag;
    // Aggiungi qui la stampa per 'source' e 'paths' se necessario
    std::cout << std::endl;
}





void KBFS_global::printPQ() const {
    std::cout << "Contenuto di PQ:" << std::endl;
    for (const auto& e : PQ) {
        auto [dist, vtx, p, setptx, type, nullVertex, paths] = e;
        std::cout << "Dist: " << dist << ", Vertex: " << vtx << ", Type: " << type 
                  << ", NullVertex: " << nullVertex << ", Path: ";
        for (auto& v : p.seq) {
            std::cout << v << " ";
        }
        std::cout << ", Setptx: {";
        for (auto& sv : setptx) {
            std::cout << sv << " ";
        }
        std::cout << "}, Paths: [";
        for (const auto& path : paths) {
            std::cout << "(";
            for (auto v : path.seq) {
                std::cout << v << " ";
            }
            std::cout << ") ";
        }
        std::cout << "]" << std::endl;
    }
}

void KBFS_global::printDistanceProfile() const {
    std::cout << "\nContenuto di Distance Profile:" << std::endl;
    for (size_t i = 0; i < distance_profile.size(); ++i) {
        std::cout << "Vertex " << i << ": ";
        for (const auto& pair : distance_profile[i]) {
            std::cout << "(Dist: " << pair.first << ", Path: ";
            for (auto v : pair.second.seq) {
                std::cout << v << " ";
            }
            std::cout << ") ";
        }
        std::cout << std::endl;
    }
}





void KBFS_global::printStructures() const {
    std::cout << "Printing KBFS_global Structures:" << std::endl;

    // Stampa di top_k
    std::cout << "1. top_k (Paths per node):" << std::endl;
    for (size_t i = 0; i < top_k.size(); ++i) {
        std::cout << "Node " << i << ": ";
        if (top_k[i].empty()) {
            std::cout << "No paths.";
        } else {
            for (const auto& path : top_k[i]) {
                std::cout << "(Path weight: " << path.w << "), ";
            }
        }
        std::cout << std::endl;
    }

    // Stampa di detour_done
    std::cout << "2. detour_done (Nodes):" << std::endl;
    for (size_t i = 0; i < detour_done.size(); ++i) {
        std::cout << "Node " << i << ": " << (detour_done[i] ? "Done" : "Not Done") << std::endl;
    }

    // Stampa di predecessors_set
    std::cout << "3. predecessors_set (Nodes):" << std::endl;
    for (size_t i = 0; i < predecessors_set.size(); ++i) {
        std::cout << "Node " << i << ": ";
        if (predecessors_set[i].empty()) {
            std::cout << "No predecessors.";
        } else {
            for (const auto& pred : predecessors_set[i]) {
                std::cout << pred << ", ";
            }
        }
        std::cout << std::endl;
    }



    // Stampa di last_det_path
    std::cout << "5. last_det_path (Last path weight to each node):" << std::endl;
    for (size_t i = 0; i < last_det_path.size(); ++i) {
        std::cout << "Node " << i << ": Last path weight is " << last_det_path[i] << std::endl;
    }

    // Stampa di ignore_nodes e locally_ignore_nodes
    std::cout << "6. ignore_nodes (Globally ignored nodes):" << std::endl;
    for (size_t i = 0; i < ignore_nodes.size(); ++i) {
        std::cout << "Node " << i << ": " << (ignore_nodes[i] ? "Ignored" : "Not Ignored") << std::endl;
    }

    std::cout << "7. locally_ignore_nodes (Locally ignored nodes):" << std::endl;
    for (size_t i = 0; i < locally_ignore_nodes.size(); ++i) {
        std::cout << "Node " << i << ": " << (locally_ignore_nodes[i] ? "Ignored" : "Not Ignored") << std::endl;
    }

    


}