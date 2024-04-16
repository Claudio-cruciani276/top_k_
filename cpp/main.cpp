//
// Created by anonym on 27/05/23.
//
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <cassert>
#include <map>
#include <algorithm>
#include "kSiSPIndex.h"
#include <string>
#include <bits/stdc++.h>
#include <networkit/io/EdgeListReader.hpp>
#include <networkit/components/ConnectedComponents.hpp>
#include <networkit/distance/Diameter.hpp>

#include "yen_ksp.hpp"
#include <boost/program_options.hpp>
#include "GraphManager.hpp"
#include <networkit/graph/GraphTools.hpp>
#include "mytimer.h"

using namespace std;

typedef
boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
        boost::no_property,
        boost::property<boost::edge_weight_t, int> >
        boost_graph;

boost_graph nk_to_boost(NetworKit::Graph * nk_graph){
    boost_graph b_graph(nk_graph->numberOfNodes());
    boost::property_map<boost_graph, boost::edge_weight_t>::type weights = get(boost::edge_weight, b_graph);
    nk_graph->forEdges([&](NetworKit::node u, NetworKit::node v) {
        weights[boost::add_edge(u, v, b_graph).first] = 1;
        weights[boost::add_edge(v, u, b_graph).first] = 1;
    });
    return b_graph;
}
void write_csv(std::string filename, std::string colname, std::vector<int> vals){
    // Make a CSV file with one column of integer values
    // filename - the name of the file
    // colname - the name of the one and only column
    // vals - an integer vector of values
    
    // Create an output filestream object
    std::ofstream myFile(filename);
    
    // Send the column name to the stream
    myFile << colname << "\n";
    
    // Send data to the stream
    for(int i = 0; i < vals.size(); ++i)
    {
        myFile << vals.at(i) << "\n";
    }
    
    // Close the file
    myFile.close();
}
void write_csv(std::string filename, std::vector<std::pair<std::string, std::vector<int>>> dataset){
    // Make a CSV file with one or more columns of integer values
    // Each column of data is represented by the pair <column name, column data>
    //   as std::pair<std::string, std::vector<int>>
    // The dataset is represented as a vector of these columns
    // Note that all columns should be the same size
    
    // Create an output filestream object
    std::ofstream myFile(filename);
    
    // Send column names to the stream
    for(int j = 0; j < dataset.size(); ++j)
    {
        myFile << dataset.at(j).first;
        if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
    }
    myFile << "\n";
    
    // Send data to the stream
    for(int i = 0; i < dataset.at(0).second.size(); ++i)
    {
        for(int j = 0; j < dataset.size(); ++j)
        {
            myFile << dataset.at(j).second.at(i);
            if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
        }
        myFile << "\n";
    }
    
    // Close the file
    myFile.close();
}
double median(std::vector<double>& arr) { //SORTS
    size_t n = arr.size() / 2;
    if (n % 2 == 0) {
        std::nth_element(arr.begin(),arr.begin() + n/2,arr.end());
        std::nth_element(arr.begin(),arr.begin() + (n - 1) / 2,arr.end());
        return (double) (arr[(n-1)/2]+ arr[n/2])/2.0;
    }

    else{
        std::nth_element(arr.begin(),arr.begin() + n/2,arr.end());
        return (double) arr[n/2];
    }
    // std::nth_element(arr.begin(), arr.begin()+n, arr.end());
    // return arr[n];
}

std::ostream& operator<<(std::ostream& os, path const & rhs){
    for(size_t t=0;t<rhs.w+1;t++)
        if(t!=rhs.w){
            os << rhs.seq[t]<< " ";
        }
        else{
            os << rhs.seq[t];
        }
    return os;
}



double average(std::vector<double> & arr) {

    auto const count = static_cast<double>(arr.size());
    double sum = 0;
    for(double value: arr) sum += value;
    return sum / count;
}

int main(int argc, char **argv) {
    srand (time(NULL));
    
    //declare supported options
	namespace po = boost::program_options;
	po::options_description desc("Allowed options");
	
	desc.add_options()
	("graph_location,g", po::value<std::string>(), "Input Graph File Location")
	("k_paths,k", po::value<int>(), "Number of Top Paths to Compute")
	("num_queries,q", po::value<int>(), "Number of Queries to Be Performed")
	("directed,d",po::value<int>(), "[FALSE(0) TRUE(1)]")
	("ordering,o",po::value<int>(), "Type of Node Ordering [DEGREE(0) APPROX-BETWEENESS(1) k-PATH(2)]")
	;


	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);
    if(vm.empty()){
		std::cout << desc << "\n";
		throw std::runtime_error("empty argument array");
	}
	std::string graph_location;
	
    if(vm.count("graph_location"))
		graph_location = vm["graph_location"].as<std::string>();

    
    int K = -1;

	if (vm.count("k_paths"))
		K = vm["k_paths"].as<int>();

	if (K < 2){
		std::cout << desc << "\n";
		throw std::runtime_error("K must be at least 2");
	}

    int ordering = -1; 

	if (vm.count("ordering")){
		ordering = vm["ordering"].as<int>();
	}
	if(ordering != 0 && ordering != 1 && ordering != 2){
		std::cout << desc << "\n";
		throw std::runtime_error("Wrong ordering selection (0 or 1 or 2)");
	}
    
    int num_queries = -1;
	if (vm.count("num_queries")){
		num_queries = vm["num_queries"].as<int>();
	}
	if(num_queries < 2){
		std::cout << desc << "\n";
		throw std::runtime_error("Wrong num queries");
	}

	int directed = -1;
	if (vm.count("directed")){
		directed = vm["directed"].as<int>();
	}
	if(directed != 0 && directed !=1){
		std::cout << desc << "\n";
		throw std::runtime_error("Wrong directed selection");
	}


    
    if(directed==0){
        std::cout << "Graph is undirected\n";
    }
    else{ 
        throw std::runtime_error("not yet implemented");

    }
    std::cout << "Reading " << graph_location << " building with K = " << K << " num_queries = "<<num_queries<<" ordering "<<ordering<<"\n";

     //timestring
    std::time_t rawtime;
    std::tm* timeinfo;
    char buffer [80];

    std::time(&rawtime);
    timeinfo = std::localtime(&rawtime);
    std::strftime(buffer,80,"%d-%m-%Y-%H-%M-%S",timeinfo);
    std::string tmp_time = buffer;
    // LETTURA
    NetworKit::Graph *graph = new NetworKit::Graph();


	if(graph_location.find(".hist") != std::string::npos){
		GraphManager::read_hist(graph_location,&graph);
	}
	else if(graph_location.find(".nde") != std::string::npos){
		GraphManager::read_nde(graph_location,&graph);
f
	}
	else{
		throw std::runtime_error("Unsupported graph file format");
	}

	*graph = NetworKit::GraphTools::toUnweighted(*graph);
	*graph = NetworKit::GraphTools::toUndirected(*graph);

	const NetworKit::Graph& graph_handle = *graph;
	NetworKit::ConnectedComponents *cc = new NetworKit::ConnectedComponents(graph_handle);
	*graph = cc->extractLargestConnectedComponent(graph_handle, true);
    graph->shrinkToFit();
    graph->indexEdges();
	std::cout<<"Graph after CC has "<<graph->numberOfNodes()<<" vertices and "<<graph->numberOfEdges()<<" edges\n";
    double density = NetworKit::GraphTools::density(*graph);
    NetworKit::Diameter *dm = new NetworKit::Diameter(graph_handle,NetworKit::DiameterAlgo::EXACT,0.0);
    dm->run();

    
    double diameter = dm->getDiameter().first;
    delete dm;
    delete cc;
	std::cout<<"Density: "<<density<<"\n";
	std::cout<<"Diameter: "<<diameter<<"\n";
    
    kSiSPIndex* kindex_manager = new kSiSPIndex();
    
    double centrality_time = kindex_manager->init(graph, K, ordering);
    
    std::string order_string;

    switch(ordering){
        case (0):
            order_string = "DEG";
            break;
        case (1):
            order_string = "BET";
            break;
        case (2):
            order_string = "KPT";
            break;
        default:
            throw new std::runtime_error("problem");

    
    }

    std::cout << "Centrality Time("<<order_string<<"): "<<centrality_time<<"\n";    

    kindex_manager->build();


    std::cout << "Construction time: " << kindex_manager->constr_time << "\n";
    std::cout << "Labeling size: " << kindex_manager->index_size << "\n";

    vector<double> yen_time;
    vector<double> index_time;
    
    ProgressStream query_bar(num_queries);
    query_bar.label() << "Performing "<<num_queries<<" queries";

    boost_graph b_graph = nk_to_boost(graph);
    size_t performed_queries = 0;
    mytimer time_counter;

    while(performed_queries<num_queries){

        vertex u = NetworKit::GraphTools::randomNode(*(graph));
        vertex v = NetworKit::GraphTools::randomNode(*(graph));
        if(u == v) 
            continue;

        performed_queries++;
        
        time_counter.restart();        
        kindex_manager->query(u,v);
        index_time.push_back(time_counter.elapsed());

        time_counter.restart();
        auto yen_paths = boost::yen_ksp(b_graph, u, v, K);
        yen_time.push_back(time_counter.elapsed());

        pathlist converted_yen_paths;
        pathlist converted_index_paths;
        for(auto&el:yen_paths){
            path p;
            p.seq.clear();
            for(auto&arco:el.second){//lista archi
                assert(graph->hasEdge(arco.m_source,arco.m_target));
                p.seq.push_back(arco.m_source);
            }
            p.seq.push_back(el.second.back().m_target);   
            assert(p.seq.size()-1==el.first);
            p.w = p.seq.size()-1;
            converted_yen_paths.push_back(p);
        }
        for(auto & el : kindex_manager->results){

            #ifndef NDEBUG
            for(size_t i = 0; i < el.w; i++)
                assert(graph->hasEdge(el.seq[i],el.seq[i+1]));
            #endif

            
            converted_index_paths.push_back(el);
        }
        if(converted_yen_paths.size()!=converted_index_paths.size()){
            std::cout << "\nPair (" << u << "," << v << ")\n";
            std::cout << "Yen found " << converted_yen_paths.size() << " paths\nIndex provided " << converted_index_paths.size() << " paths\n";
            
            std::cout << "Yen paths\n";
            for(size_t i = 0; i < converted_yen_paths.size(); i++){
                std::cout << converted_yen_paths[i];
                std::cout << " (";
                for(size_t j = 0; j < converted_yen_paths[i].w+1; j++){
                    if(j!=converted_yen_paths[i].w){
                        std::cout << kindex_manager->ordering[converted_yen_paths[i].seq[j]]<<" ";
                    }
                    else{
                        std::cout << kindex_manager->ordering[converted_yen_paths[i].seq[j]]<<")\n";
                    }
                }
            }
            std::cout << "Index paths\n";
            for(size_t i = 0; i < converted_index_paths.size(); i++){
                std::cout << converted_index_paths[i];
                std::cout << " (";
                for(size_t j = 0; j < converted_index_paths[i].w+1;j++){
                    if(j!=converted_index_paths[i].w){

                        std::cout << kindex_manager->ordering[converted_index_paths[i].seq[j]]<<" ";
                    }
                    else{
                        std::cout << kindex_manager->ordering[converted_index_paths[i].seq[j]]<<")\n";

                    }
                }
            }
            // std::cout<<graph->degree(u)<<" "<<graph->degree(v)<<"\n";
            throw std::runtime_error("cardinality exception");
        } 
        else{
            // std::cout << "\nPair (" << u << "," << v << ")\n";
            for(size_t i = 0; i < converted_yen_paths.size(); i++){
                

                if(converted_yen_paths[i].w!=converted_index_paths[i].w){
                    std::cout << "Yen path\n";
                    std::cout << converted_yen_paths[i];
                    std::cout << " (";
                    for(size_t j = 0; j < converted_yen_paths[i].w+1; j++){
                        if(j!=converted_yen_paths[i].w){
                            std::cout << kindex_manager->ordering[converted_yen_paths[i].seq[j]]<<" ";
                        }
                        else{
                            std::cout << kindex_manager->ordering[converted_yen_paths[i].seq[j]]<<")\n";
                        }
                    }
                    std::cout << "Index path\n";
                    std::cout << converted_index_paths[i];
                    std::cout << " (";
                    for(size_t j = 0; j < converted_index_paths[i].w+1;j++){
                        if(j!=converted_index_paths[i].w){
                            std::cout << kindex_manager->ordering[converted_index_paths[i].seq[j]]<<" ";
                        }
                        else{
                            std::cout << kindex_manager->ordering[converted_index_paths[i].seq[j]]<<")\n";

                        }
                    }
                    std::cout<<"\nLength Yen: " << converted_yen_paths[i].w << " Length Index: " << converted_index_paths[i].w<< "\n";
                    std::cout<<graph->degree(u)<<" "<<graph->degree(v)<<"\n";
                    throw std::runtime_error("length exception");

                }

            }
            
        }

        ++query_bar;
    }

    std::cout << "Index average query time: " << average(index_time) << "s --- median query time: " << median(index_time) << "s\n";
    std::cout << "Yen average query time: " << average(yen_time) << "s --- median query time: " << median(yen_time) << "s\n";
    
    
   
    


	std::string shortenedName = graph_location.substr(0, 16);
    stringstream string_object_name;
    string_object_name<<K;
    std::string k_string;
    string_object_name>>k_string;
	std::string timestampstring = shortenedName+"_"+k_string+"_"+order_string+"_"+tmp_time;

	std::string logFile = timestampstring +".csv";
    std::cout<<"Writing: "<<logFile<<"\n";
    std::vector<double> speedup;
    assert(index_time.size()==yen_time.size());

    for(size_t cn = 0;cn<index_time.size();cn++){
        speedup.push_back(yen_time[cn]/index_time[cn]);
    }
    // Create an output filestream object
    std::ofstream myFile(logFile);
    myFile<<"V,E,K,Diameter,Ordering,Density,Centrality,Construction,Size,AvgQTIndex,AvgQTYen,AvgSpeedup,MedQTIndex,MedQTYen,MedSpeedup,Prunings\n";
    myFile<<graph->numberOfNodes()<<",";
    myFile<<graph->numberOfEdges()<<",";
    myFile<<k_string<<",";
    myFile<<diameter<<",";
    myFile<<order_string<<",";
    myFile<<std::scientific<<std::setprecision(2);
    myFile<<density<<",";
    myFile<<centrality_time<<",";
    myFile<<kindex_manager->constr_time<<",";
    myFile<<kindex_manager->index_size<<",";
    myFile<<average(index_time)<<",";
    myFile<<average(yen_time)<<",";
    myFile<<average(speedup)<<",";

    myFile<<median(index_time)<<",";
    myFile<<median(yen_time)<<",";
    myFile<<median(speedup)<<",";
    myFile<<kindex_manager->prunings<<"\n";


    myFile.close();
    


    delete graph;
    delete kindex_manager;
    exit(EXIT_SUCCESS);
}