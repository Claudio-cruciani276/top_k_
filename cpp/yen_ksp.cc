#define BOOST_TEST_MODULE yen_ksp

#include "yen_ksp.hpp"
#include <algorithm>
#include <iostream>
#include <list>

// *******************************************************************
// Generic definitions.
// *******************************************************************

template <typename G>
using Edge = typename G::edge_descriptor;

template <typename G>
using Vertex = typename G::vertex_descriptor;

template <typename G>
using Path = std::list<Edge<G>>;

template <typename G>
using GPath = std::pair<const G&, Path<G>>;

template <typename G, typename T>
using Result = std::pair<T, Path<G>>;

template <typename G, typename T>
using GResult = std::pair<const G &, Result<G, T>>;

template <typename G, typename T>
using Results = std::list<Result<G, T>>;

// Add a directed edge, test it, and set weight.
template<typename G, typename T>
Edge<G>
ade(G &g, Vertex<G> s, Vertex<G> d, T w)
{
  Edge<G> e;
  bool success;

  boost::tie(e, success) = boost::add_edge(s, d, g);
  assert(success);

  boost::get(boost::edge_weight, g, e) = w;

  return e;
}

// Add an undirected edge.
template<typename G, typename T>
std::pair<Edge<G>, Edge<G>>
aue(G &g, Vertex<G> s, Vertex<G> d, T w)
{
  return std::make_pair(ade(g, s, d, w), ade(g, d, s, w));
}

template<typename G>
std::ostream &
operator << (std::ostream &out, const GPath<G> &p)
{
  for(auto const &e: p.second)
    out << e;
  return out;
}  

template<typename G, typename T>
std::ostream &
operator << (std::ostream &out, const GResult<G, T> &gr)
{
  out << gr.second.first << ": "
      << GPath<G>(gr.first, gr.second.second);
  return out;
}  

// *******************************************************************
// Specialized definitions.
// *******************************************************************
typedef
boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                      boost::no_property,
                      boost::property<boost::edge_weight_t, int> >
graph;


typedef Edge<graph> edge;
typedef Vertex<graph> vertex;
typedef Path<graph> path;
typedef Result<graph, int> result;
typedef Results<graph, int> results;

typedef GResult<graph, int> gresult;

// Check whether there is a given result.
bool
check_result(const results &rs, const result &r)
{
  return std::find(rs.begin(), rs.end(), r) != rs.end();
}