#include <fstream>
#include "progressBar.h"
#include <networkit/graph/GraphTools.hpp>           
#include <networkit/graph/Graph.hpp>
#include <networkit/io/EdgeListReader.hpp>
//commento per git

class GraphManager
{
public:
    static void read_hist(std::string source, NetworKit::Graph **g)
    {
        std::ifstream ifs(source);
        if (!ifs)
            throw std::runtime_error("Error opening File ");

        int vertices = -1, edges = -1, weighted = -1, directed = -1;

        ifs >> vertices >> edges >> weighted >> directed;

        assert((weighted == 0 || weighted == 1) && (directed == 0 || directed == 1) && (vertices >= 0 && edges >= 0));

        ProgressStream reader(edges);
        std::string t1 = weighted == 0 ? " unweighted" : " weighted";
        std::string t2 = directed == 0 ? " undirected" : " directed";

        reader.label() << "Reading" << t1 << t2 << " graph in " << source << " (HIST FORMAT) containing " << vertices << " vertices and " << edges << " edges ";
        NetworKit::Graph *graph = new NetworKit::Graph(vertices, weighted, directed);
        int time, v1, v2, weight;

        while (true)
        {

            ifs >> time >> v1 >> v2 >> weight;
            if (ifs.eof())
                break;

            ++reader;

            assert(weighted == 1 || weight == 1 || weight == -1);

            if (v1 == v2)
                continue;
            assert(graph->hasNode(v1) && graph->hasNode(v2));
            if (graph->hasEdge(v1, v2))
                // std::cout<<"SKIPPING ROW"<<std::endl;
                ;
            else
            {
                graph->addEdge(v1, v2, weight);
#ifndef NDEBUG
                if (!directed)
                {
                    if (!graph->hasEdge(v1, v2) && !graph->hasEdge(v2, v1))
                        throw std::runtime_error("wrong edge insertion during construction");
                }
                else
                {
                    if (!graph->hasEdge(v1, v2))
                        throw std::runtime_error("wrong edge insertion during construction");
                }
#endif
            }
        }

        ifs.close();
        graph->indexEdges();
        *g = graph;
    }
};



int main() {
    std::string file = "grafo.txt"; // Nome del file contenente il grafo da leggere

    // Definizione di un puntatore a puntatore a un oggetto Graph
    NetworKit::Graph *grafo;

    try {
        // Chiamata alla funzione read_nde per leggere il grafo dal file
        GraphManager::read_hist("critical.hist", &grafo);

        // Ora puoi utilizzare il grafo letto, ad esempio stampando il numero di nodi e archi
        std::cout << "Il grafo contiene " << grafo->numberOfNodes() << " nodi e "
                  << grafo->numberOfEdges() << " archi." << std::endl;

        // Ricordati di deallocare la memoria quando hai finito di usare il grafo
        delete grafo;
    } catch (const std::runtime_error& e) {
        // Gestione dell'eccezione in caso di errore durante la lettura del grafo
        std::cerr << "Errore durante la lettura del grafo: " << e.what() << std::endl;
    }

    return 0;
};