#include <iostream>
#include "GraphManager.hpp"

int main() {
    // Dichiarazione di un puntatore a puntatore a NetworKit::Graph
    NetworKit::Graph* graphPtr = nullptr;
    NetworKit::Graph** graphPtrPtr = &graphPtr;

    try {
        // Chiamata alla funzione read_hist per leggere il grafo dal file
        GraphManager::read_hist("critical.hist", graphPtrPtr);

        // Ottenere il puntatore al grafo letto
        NetworKit::Graph* graph = *graphPtrPtr;

        // Stampare il numero di nodi e archi del grafo
        std::cout << "Numero di nodi: " << graph->numberOfNodes() << std::endl;
        std::cout << "Numero di archi: " << graph->numberOfEdges() << std::endl;
    } catch(const std::runtime_error& e) {
        std::cerr << "Errore: " << e.what() << std::endl;
    }

    // Liberare la memoria allocata per il grafo
    //delete *graphPtrPtr;
    return 0;
}
