#include <fstream>
#include "progressBar.h"
#include <networkit/graph/GraphTools.hpp>           
#include <networkit/graph/Graph.hpp>
#include <networkit/io/EdgeListReader.hpp>

int main() {
        std::ifstream ifs("critical.hist");
        if (!ifs)
            throw std::runtime_error("Error opening File ");
        else 
            std::cout<<"file Letto"<<std::endl;

        
        int vertices = -1, edges = -1, weighted = -1, directed = -1;

        ifs >> vertices >> edges >> weighted >> directed;

        assert((weighted == 0 || weighted == 1) && (directed == 0 || directed == 1) && (vertices >= 0 && edges >= 0));
        
        ProgressStream reader(edges);
        std::string t1 = weighted == 0 ? " unweighted" : " weighted";
        std::string t2 = directed == 0 ? " undirected" : " directed";

        reader.label() << "Reading" << t1 << t2 << " graph in " << "critical.hist" << " (HIST FORMAT) containing " << vertices << " vertices and " << edges << " edges ";
        std::cout << reader.label().str() << std::endl;
        NetworKit::Graph *graph = new NetworKit::Graph(vertices, weighted, directed,true);
        std::cout << "Numero di nodi: " << graph->numberOfNodes() << std::endl;
        std::cout << "Numero di archi: " << graph->numberOfEdges() << std::endl;
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
// #ifndef NDEBUG
//                 if (!directed)
//                 {
//                     if (!graph->hasEdge(v1, v2) && !graph->hasEdge(v2, v1))
//                         throw std::runtime_error("wrong edge insertion during construction");
//                 }
//                 else
//                 {
//                     if (!graph->hasEdge(v1, v2))
//                         throw std::runtime_error("wrong edge insertion during construction");
//                 }
// #endif
            }
        }
        ifs.close();
        graph->indexEdges();



std::cout << "Lista degli archi del grafo:" << std::endl;
for (NetworKit::node u = 0; u < graph->upperNodeIdBound(); ++u) {
    for (const auto& v : graph->neighborRange(u)) {
        std::cout << u << " -- " << v << std::endl;
    }
}


};