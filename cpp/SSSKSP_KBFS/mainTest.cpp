#include "KBFS_global.hpp"
#include "GraphManager.hpp"
#include <networkit/graph/GraphTools.hpp>

#include <networkit/io/EdgeListReader.hpp>
#include <networkit/components/ConnectedComponents.hpp>
#include <networkit/distance/Diameter.hpp>
#include <networkit/io/GraphIO.hpp>
#include <networkit/io/GMLGraphWriter.hpp>
//#include <networkit/graph/Graph.hpp>

int main() {
/*  1
    // Utilizzo del costruttore di default
    path defaultPath;
    std::cout << "Default path:\n";
    defaultPath.addVertex(5);
    defaultPath.printPath();
    defaultPath.addVertex(8);
    defaultPath.printPath();

    // Utilizzo del costruttore con un singolo vertice
    path singleVertexPath(11);
    std::cout << "\nSingle vertex path:\n";
    singleVertexPath.addVertex(10);
    singleVertexPath.printPath();


    // Utilizzo del costruttore da un set di vertici
    std::set<vertex> verticesSet = {11,10};
    path setPath(verticesSet);
    std::cout << "\nPath from a set of vertices:\n";
    //setPath.addVertex();
    //etPath.addVertex();
    setPath.printPath();

    if(singleVertexPath==setPath){
        std::cout << "I path sono uguali." << std::endl;
        }
    else
        std::cout << "I path non sono uguali." << std::endl;
*/

/*
    // Utilizzo del costruttore da due path da combinare
    path combinedPath(singleVertexPath, setPath);
    std::cout << "\nCombined path:\n";
    combinedPath.printPath();

    // Utilizzo del costruttore da due path da combinare
    path combinedPath2(defaultPath, setPath);
    std::cout << "\nCombined path:\n";
    combinedPath2.printPath();
    return 0;
*/ 



    std::vector<int> verticesSet = {1, 2, 3,4,5};
    std::vector<int> verticesSet2 = {4, 5, 6};
    std::vector<int> verticesSet3 = {7, 8, 9,10,12,34,56};
    std::vector<int> verticesSet4 = {7, 8, 9};
    std::vector<int> verticesSet5 = {1,2,3,4,5};

    path newPath(verticesSet);
    path path_to_evaluate(verticesSet2);
    path setPath2(verticesSet3);
    path setPath4(verticesSet4);
    path setPath5(verticesSet5);

    std::set<path> paths;
    paths.insert(newPath);
    paths.insert(path_to_evaluate);
    paths.insert(setPath2);



    // Prova a trovare un path specifico
    std::vector<int> verticesSet100 = {12, 22, 32,33};
    path searchPath(verticesSet100);  // Path che stiamo cercando
    
    newPath.printPath();
    setPath2.printPath();
    std::cout << newPath.w+setPath2.w<< std::endl;
    path nuovo(newPath,setPath2);
    nuovo.printPath();
    std::cout << nuovo.w<< std::endl;
    path p1;
    p1.getFirstNode();
 


  




  
    //gi il grafo utilizzando GraphManager::read_hist
    NetworKit::Graph *graph = new NetworKit::Graph();
    GraphManager::read_hist("further_gab_gadget.hist", &graph);

    // Verifica che il grafo sia stato letto correttamente
    if (!graph) {
        std::cerr << "Failed to read graph." << std::endl;
        return -1;
    }

    *graph = NetworKit::GraphTools::toUnweighted(*graph);
    *graph = NetworKit::GraphTools::toUndirected(*graph);

    const NetworKit::Graph& graph_handle = *graph;
    NetworKit::ConnectedComponents *cc = new NetworKit::ConnectedComponents(graph_handle);
    *graph = cc->extractLargestConnectedComponent(graph_handle, true);
    graph->shrinkToFit();
    graph->indexEdges();

    std::string path = "critical.GML";
    NetworKit::GMLGraphWriter writer;
    // Scrivi il grafo su file
    writer.write(*graph, path);

    std::cout<<"Graph after CC has "<<graph->numberOfNodes()<<" vertices and "<<graph->numberOfEdges()<<" edges\n";
    //ciao
    size_t K = 3; // Numero di percorsi più brevi da trovare
    vertex root = 0;

    //KBFS_global* KBFS_global_MANAGER = new KBFS_global(graph, K, root);
    
    //for (NetworKit::node u : graph->nodeRange()) {
    //    std::cout << "Nodo: " << u << std::endl;
    //} 

    //graph->forEdges([&](NetworKit::node u, NetworKit::node ,NetworKit::index edgeId) {
    //    std::cout << "Edge id: " << edgeId << std::endl;;
    //});

    //NetworKit::edgeid eid = graph->edgeId(5, 6);
    //NetworKit::edgeid eid2 = graph->edgeId(0, 1);
    //std::cout << "Edge id: " << eid << std::endl;
    //std::cout << "Edge id: " << eid2 << std::endl;




/*
    std::cout << "Nodi del Grafo:" << std::endl;
    graph->forNodes([&](NetworKit::node u) {
    std::cout << "Nodo: " << u << std::endl;
    });

    std::cout << "Archi del Grafo:" << std::endl;
    graph->forEdges([&](NetworKit::node u, NetworKit::node v) {
        std::cout << "Arco da " << u << " a " << v << std::endl;
    });
*/
    
    //KBFS_global_MANAGER->generalized_bfs();
    //KBFS_global_MANAGER->binary_TEST();

/*
    // Crea un'istanza di KBFS
    size_t K = 5; // Numero di percorsi più brevi da trovare
    vertex root = 0; // Scegli un vertice radice appropriato
    //std::cout<<"   il numero k è    ";
    KBFS_global kbfs(graph, K, root);
    std::cout<<"   il numero k è    "<<kbfs.K;
    //KBFS_global kbfs();
  
    //kbfs.printStructures();
*/





    // Utilizza kbfs per effettuare operazioni, ad esempio:
    // kbfs.someMethod();

    // Ricordati di deallocare il grafo se KBFS non gestisce la memoria del grafo
    delete graph;





    return 0;

 }   