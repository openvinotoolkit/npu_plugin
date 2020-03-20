#ifndef TRANSITIVE_REDUCTION_HPP_
#define TRANSITIVE_REDUCTION_HPP_

#include <list>
#include <map>
#include <unordered_set>
#include <vector>

#include "include/mcm/graph/graph.hpp"
#include "include/mcm/algorithms/topological_sort.hpp"
#include "include/mcm/compiler/compilation_profiler.hpp"


namespace mv
{
    template <typename NodeIterator>
    struct OpItComparatorTemplate
    {
        bool operator()(NodeIterator lhs, NodeIterator rhs) const
        {
            return (*lhs) < (*rhs);
        }
    };


    // NOTE: This graph non member function works only on DAGs
    template <typename T_node, typename T_edge, typename EdgeItComp, typename NodeItComp>
    void transitiveReduction_(graph<T_node, T_edge>& g,
                typename graph<T_node, T_edge>::node_list_iterator root,
                const std::set<typename graph<T_node, T_edge>::edge_list_iterator, EdgeItComp>& filteredEdges,
                std::set<typename graph<T_node, T_edge>::node_list_iterator, NodeItComp>& processedNodes)
    {
        // Collecting the set of neighbours, as edges
        // NOTE: Can't use unordered map because node_list_iterator needs to be hashable (requirement too strict)
        std::map<typename graph<T_node, T_edge>::node_list_iterator,
                typename graph<T_node, T_edge>::edge_list_iterator,
                OpItComparatorTemplate<typename graph<T_node, T_edge>::node_list_iterator>> root_adj, toEliminate;

        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
            root_adj[e->sink()] = e;

        // Starting a DFS from each neighbour v
        // If a node u is reachable from v and it's also a neighbour of the root
        // Eliminate the edge between root and u

        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
        {
            // Must skip first edge (itself)
            typename graph<T_node, T_edge>::edge_dfs_iterator edge_dfs(e);
            ++edge_dfs;
            for (; edge_dfs != g.edge_end(); ++edge_dfs)
            {
                auto u = edge_dfs->sink();
                auto it = root_adj.find(u);
                if(it != root_adj.end())
                {
                    auto it2 = filteredEdges.find(it->second);
                    if(it2 != filteredEdges.end())
                        continue;
                    toEliminate[u] = it->second;
                }
            }
        }

        for(auto edgeToEliminatePair : toEliminate)
            g.edge_erase(edgeToEliminatePair.second);

        processedNodes.insert(root);
        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
        {
            auto v = e->sink();
            if (processedNodes.find(v) == processedNodes.end())
                transitiveReduction_(g, v, filteredEdges, processedNodes);
        }
    }

    // NOTE: This graph non member function works only on DAGs
    template <typename T_node, typename T_edge, typename EdgeItComparator, typename NodeItComparator>
    void transitiveReductionOld(graph<T_node, T_edge>& g,
                const std::set<typename graph<T_node, T_edge>::edge_list_iterator, EdgeItComparator>&
                filteredEdges = std::set<typename graph<T_node, T_edge>::edge_list_iterator, EdgeItComparator>())
    {
        MV_PROFILED_FUNCTION(MV_PROFILE_ALGO)
        // Topological sort in this case also checks if the graph is dag. Hence, no explicit check for DAG is needed here.
        auto sortedNodes = topologicalSort(g);

        std::set<typename graph<T_node, T_edge>::node_list_iterator, NodeItComparator> processedNodes;

        for(auto node : sortedNodes)
        {
            if(node->parents_size() != 0)
                return;
            transitiveReduction_<T_node, T_edge, EdgeItComparator>(g, node, filteredEdges, processedNodes);
        }
    }

//
// DAG_Transitive_Reducer: given a DAG G=(V,E) computes a sub DAG G*=(V,E*)
// defined as follows: 
// E* = {(u,v) | (u,v) \in E and  \noexists |path(u,v)| > 1 }
//
// NOTE: NodeItCompType and EdgeItCompType are weak ordering on the nodes and
// edges associated with corresponding iterators.
template<typename DAGType, typename EdgeItCompType, typename NodeItCompType>
class DAG_Transitive_Reducer {

  public:
  //////////////////////////////////////////////////////////////////////////////
    typedef DAGType dag_t;
    typedef NodeItCompType node_comparator_t;
    typedef EdgeItCompType edge_comparator_t;
    typedef typename dag_t::node_list_iterator node_iterator_t;
    typedef typename dag_t::edge_list_iterator edge_iterator_t;
    //TODO(vamsikku): parameterize on the allocator //
    typedef std::map<node_iterator_t, size_t, node_comparator_t> level_map_t;
    typedef std::set<edge_iterator_t, edge_comparator_t> filter_edge_set_t;
  //////////////////////////////////////////////////////////////////////////////

    DAG_Transitive_Reducer(dag_t& dag) : dag_(dag), level_map_(),
      input_edge_count_(0UL), eliminated_edge_count_(0UL){}


    void dump_reduce_info() const {
      printf("[TransitiveReduction] input=%lu eliminated=%lu\n",
          input_edge_count_, eliminated_edge_count_);
    }


    bool reduce(const filter_edge_set_t &filtered_edges=filter_edge_set_t()) {
      MV_PROFILED_FUNCTION(MV_PROFILE_ALGO)

      dag_t &g = dag_;
      input_edge_count_ = g.edge_size();
      eliminated_edge_count_ = 0UL;

      //STEP-1: label each node with its level in the DAG //
      bool is_dag = compute_level_map();
      if (!is_dag) { return false; }

      //STEP-1.1: order the edges based on the level of the source//
      std::map<size_t, std::list<edge_iterator_t> > ordered_edges;
      for (auto e=g.edge_begin(); e!=g.edge_end(); ++e) {
        auto src_itr = level_map_.find(e->source());
        assert(src_itr != level_map_.end());
        ordered_edges[ src_itr->second ].push_back(e);
      }

      for (auto litr=ordered_edges.begin(); litr!=ordered_edges.end();
            ++litr) {
        for (auto eitr=litr->second.begin(); eitr!=litr->second.end();
              ++eitr) {
          edge_iterator_t e = *eitr;
          auto src_itr = level_map_.find(e->source());
          auto sink_itr = level_map_.find(e->sink());
          assert(src_itr != level_map_.end());
          assert(sink_itr != level_map_.end());
          assert(sink_itr->second > src_itr->second);

          // edge (u,w) can be eliminated if there is a path between 'u' and
          // 'w' of at least one node between them.
          size_t level_diff = sink_itr->second - src_itr->second;
          if ((level_diff > 1UL) &&
                has_path_with_atleast_one_node( e->source(), e->sink())) {
            // eliminate the edge //
            if (filtered_edges.find(*eitr) == filtered_edges.end()) {
              g.edge_erase(*eitr);
              ++eliminated_edge_count_;
            }
          }
        }
      }
      return true;
    } // bool reduce //

  private:

    bool has_path_with_atleast_one_node(node_iterator_t src,
          node_iterator_t sink) const {

      const dag_t& g = dag_;
      std::unordered_set<size_t> visited;
      std::list<node_iterator_t> bfs_list;


      //TODO(vamsikku): while doing DFS don't add nodes which exceed the
      //level of the sink into the bfs_list //
      for (auto e=src->leftmost_output(); e!=g.edge_end(); ++e) {
        if (e->sink()->getID() == sink->getID()) { continue; }
        bfs_list.push_back(e->sink());
        visited.insert(e->sink()->getID());
      }

      while (!bfs_list.empty()) {
        node_iterator_t curr_node = bfs_list.front();
        bfs_list.pop_front();
        if (curr_node->getID() == sink->getID()) { return true; }
        for (auto e=curr_node->leftmost_output(); e!=g.edge_end(); ++e) {
          size_t nid = e->sink()->getID();
          if (visited.find(nid) == visited.end()) {
            bfs_list.push_back(e->sink());
            visited.insert(nid);
          }
        }
      }
      return false;
    }

    bool compute_level_map() {
      dag_t& g = dag_;
      std::map<node_iterator_t, size_t, node_comparator_t> in_degree_map;
      std::list<node_iterator_t> zero_in_degree_list[2UL];

      level_map_.clear();

      for (node_iterator_t nitr=g.node_begin(); nitr!=g.node_end(); ++nitr) {
        // compute the in-degree//
        in_degree_map[nitr] = nitr->inputs_size();
        if (!(nitr->inputs_size())) {
          zero_in_degree_list[0].push_back(nitr);
        }
      }


      size_t current_level = 0UL, levelled_nodes = 0UL;
      size_t node_count = in_degree_map.size();

      while (levelled_nodes < node_count) {
        bool curr_parity = ((current_level%2UL) != 0);
        bool next_parity = !curr_parity;

        if (zero_in_degree_list[curr_parity].empty()) { return false; }
        zero_in_degree_list[next_parity].clear();

        for (auto zitr=(zero_in_degree_list[curr_parity]).begin();
              zitr!=zero_in_degree_list[curr_parity].end(); ++zitr) {
          level_map_[*zitr] = current_level;
          levelled_nodes++;

          // now reduce the in-degree of all outgoing nodes //
          for (auto e=(*zitr)->leftmost_output(); e != g.edge_end(); ++e) {
            node_iterator_t sink_node = e->sink();
            auto in_degree_sink_itr = in_degree_map.find(sink_node);
            assert(in_degree_sink_itr != in_degree_map.end());
            assert(in_degree_sink_itr->second > 0UL);
            (in_degree_sink_itr->second)--;

            if (!(in_degree_sink_itr->second)) {
              zero_in_degree_list[next_parity].push_back(sink_node);
            }
          }
        }
        ++current_level;
      }

      return (levelled_nodes == node_count);
    }

  private:
    dag_t &dag_;
    level_map_t level_map_;
    size_t input_edge_count_;
    size_t eliminated_edge_count_;
}; // class DAG_Transitive_Reducer //

#define MV_MCM_USE_NEW_TRANS_REDUCTION

template <typename T_node, typename T_edge, typename EdgeItComparator,
         typename NodeItComparator>
void transitiveReduction(graph<T_node, T_edge>& g,
    const std::set<typename graph<T_node, T_edge>::edge_list_iterator,
      EdgeItComparator>& filteredEdges = std::set<typename graph<T_node,
        T_edge>::edge_list_iterator, EdgeItComparator>()) 
{
  MV_PROFILED_FUNCTION(MV_PROFILE_ALGO)
#ifdef MV_MCM_USE_NEW_TRANS_REDUCTION
  {
    typedef graph<T_node, T_edge> dag_t;
    typedef DAG_Transitive_Reducer<dag_t, EdgeItComparator, NodeItComparator>
       transitive_reducer_t;

    transitive_reducer_t reducer(g);
    reducer.reduce(filteredEdges);
  }
#else
    transitiveReductionOld<T_node, T_edge, EdgeItComparator,
      NodeItComparator>(g, filteredEdges);
#endif

}


} // namespace mv //

#endif
