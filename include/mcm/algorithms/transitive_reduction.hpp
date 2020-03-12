#ifndef TRANSITIVE_REDUCTION_HPP_
#define TRANSITIVE_REDUCTION_HPP_

#include "include/mcm/graph/graph.hpp"
#include "include/mcm/algorithms/topological_sort.hpp"
#include <map>
#include <unordered_set>
#include <vector>
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

    template <typename T_node, typename T_edge, typename NodeItComp>
    bool allChildrenAtSameLevel_(const graph<T_node, T_edge>& g,
        typename graph<T_node, T_edge>::node_list_iterator node,
        const std::map<typename graph<T_node, T_edge>::node_list_iterator,
          size_t, NodeItComp>& level_map) {
      auto e = node->leftmost_output();
      if (e == g.edge_end()) { return true; }

      auto citr = level_map.find(e->sink());
      assert(citr != level_map.end());
      size_t first_child_level = citr->second;
      for (++e; e != g.edge_end(); ++e) {
        citr = level_map.find(e->sink());
        if (citr->second != first_child_level) { return false; }
      }
      return true;
    }


    // NOTE: This graph non member function works only on DAGs
    template <typename T_node, typename T_edge, typename EdgeItComp, typename NodeItComp>
    void transitiveReduction_(graph<T_node, T_edge>& g,
                typename graph<T_node, T_edge>::node_list_iterator root,
                const std::set<typename graph<T_node, T_edge>::edge_list_iterator, EdgeItComp>& filteredEdges,
                std::set<typename graph<T_node, T_edge>::node_list_iterator, NodeItComp>& processedNodes) {


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

        size_t total_edges_traversed=0UL;
        size_t max_edges_per_traversal=0UL;
        size_t eliminated=0UL;
        for(auto e = root->leftmost_output(); e != g.edge_end(); ++e)
        {
            // Must skip first edge (itself)
            typename graph<T_node, T_edge>::edge_dfs_iterator edge_dfs(e);
            ++edge_dfs;
            size_t edges_traversed = 0UL;
            for (; edge_dfs != g.edge_end(); ++edge_dfs)
            {
              ++edges_traversed;
                auto u = edge_dfs->sink();
                auto it = root_adj.find(u);
                if(it != root_adj.end())
                {
                    auto it2 = filteredEdges.find(it->second);
                    if(it2 != filteredEdges.end())
                        continue;
                    toEliminate[u] = it->second;
                    eliminated++;
                }
            }
            if (edges_traversed > max_edges_per_traversal) {
              max_edges_per_traversal = edges_traversed;
            }
            total_edges_traversed += edges_traversed;
        }
       
        assert(mv::isDAG(g));
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


    template<typename T_node, typename T_edge>
    bool hasPathWithAtleastOneNode_(const graph<T_node, T_edge>& g,
        typename graph<T_node, T_edge>::node_list_iterator src,
        typename graph<T_node, T_edge>::node_list_iterator sink) {
      typedef typename graph<T_node, T_edge>::node_list_iterator node_iterator;

      std::unordered_set<size_t> visited;
      std::list<node_iterator> bfs_list;

      for (auto e=src->leftmost_output(); e!=g.edge_end(); ++e) {
        if (e->sink()->getID() == sink->getID()) { continue; }
        bfs_list.push_back(e->sink());
        visited.insert(e->sink()->getID());
      }

      while (!bfs_list.empty()) {
        node_iterator curr_node = bfs_list.front();
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


    template <typename T_node, typename T_edge, typename NodeItComp>
    void computeNodeLevels_(const graph<T_node, T_edge>& g,
        std::map<typename graph<T_node, T_edge>::node_list_iterator, size_t,
                  NodeItComp>& level_map) {
      typedef typename graph<T_node, T_edge>::node_list_iterator
          node_iterator;

      std::map<node_iterator, size_t,
          OpItComparatorTemplate<node_iterator> > in_degree_map;

      std::list<node_iterator> zero_in_degree_list[2UL];

      for (node_iterator nitr=g.node_begin(); nitr!=g.node_end(); ++nitr) {
        // compute the in-degree//
        in_degree_map[nitr] = nitr->inputs_size();
        if (!(nitr->inputs_size())) {
          zero_in_degree_list[0].push_back(nitr);
        }
      }


      assert(!zero_in_degree_list[0].empty());
      size_t current_level = 0UL, levelled_nodes = 0UL;
      size_t node_count = in_degree_map.size();


      printf("zero_in_degree_list = %lu n=%lu\n",
            zero_in_degree_list[0].size(), node_count);
      fflush(stdout);

      while (levelled_nodes < node_count) {
        bool curr_parity = ((current_level%2UL) != 0);
        bool next_parity = !curr_parity;
        zero_in_degree_list[next_parity].clear();

        for (auto zitr=(zero_in_degree_list[curr_parity]).begin();
              zitr!=zero_in_degree_list[curr_parity].end(); ++zitr) {
          level_map[*zitr] = current_level;
          levelled_nodes++;

          // now reduce the in-degree of all outgoing nodes //
          for (auto e=(*zitr)->leftmost_output(); e != g.edge_end(); ++e) {
            node_iterator sink_node = e->sink();
            auto in_degree_sink_itr = in_degree_map.find(sink_node);
            assert(in_degree_sink_itr != in_degree_map.end());
            assert(in_degree_sink_itr->second > 0UL);
            (in_degree_sink_itr->second)--;

            if (!(in_degree_sink_itr->second)) {
              zero_in_degree_list[next_parity].push_back(sink_node);
            }
          }
          printf("levelled_nodes=%lu\n", levelled_nodes);
          fflush(stdout);
        }
        ++current_level;
      }
    }


    template <typename T_node, typename T_edge>
    size_t graphEdgeCount(const graph<T_node, T_edge>& g) {
      size_t diedge_count = 0UL;
      for (auto node = g.node_begin(); node != g.node_end(); ++node) {
        size_t degree = 0UL;
        for (auto e = node->leftmost_output(); e != g.edge_end(); ++e) {
          degree++;
        }
        diedge_count += degree;
      }
      return diedge_count;
    }


    // NOTE: This graph non member function works only on DAGs
    template <typename T_node, typename T_edge, typename EdgeItComparator,
              typename NodeItComparator>
    void transitiveReduction(graph<T_node, T_edge>& g,
        const std::set<typename graph<T_node, T_edge>::edge_list_iterator,
          EdgeItComparator>& filteredEdges =
            std::set<typename graph<T_node, T_edge>::edge_list_iterator,
              EdgeItComparator>())
    {
        typedef typename graph<T_node, T_edge>::edge_list_iterator
            edge_iterator;

        MV_PROFILED_FUNCTION(MV_PROFILE_ALGO)
        size_t input_edges = graphEdgeCount(g);
        printf("[Starting Transitive Reduction] input edges=%lu...\n",
              input_edges);
        std::map<typename graph<T_node, T_edge>::node_list_iterator, size_t, 
            NodeItComparator> nodeLevels;
        std::map<size_t, std::list<edge_iterator> > ordered_edges;


        //STEP-1: label each node with its level in the DAG //
        computeNodeLevels_(g, nodeLevels);
        //STEP-1.1: order the edges based on the level of the source//

        for (auto e=g.edge_begin(); e!=g.edge_end(); ++e) {
          auto src_itr = nodeLevels.find(e->source());
          assert(src_itr != nodeLevels.end());
          ordered_edges[ src_itr->second ].push_back(e);
        }

        size_t eliminated = 0UL;
        for (auto litr=ordered_edges.begin(); litr!=ordered_edges.end();
              ++litr) {
          for (auto eitr=litr->second.begin(); eitr!=litr->second.end();
                ++eitr) {
            edge_iterator e = *eitr;
            auto src_itr = nodeLevels.find(e->source());
            auto sink_itr = nodeLevels.find(e->sink());
            assert(src_itr != nodeLevels.end());
            assert(sink_itr != nodeLevels.end());
            assert(sink_itr->second > src_itr->second);

            // edge (u,w) can be eliminated if there is a path between 'u' and
            // 'w' of at least one node between them.
            size_t level_diff = sink_itr->second - src_itr->second;
            if ((level_diff > 1UL) &&
                  hasPathWithAtleastOneNode_(g, e->source(), e->sink())) {
              // eliminate the edge //
              if (filteredEdges.find(*eitr) == filteredEdges.end()) {
                g.edge_erase(*eitr);
                eliminated++;
              }
            }
          }
        }

        printf("[Eliminated edges= %lu Input edges=%lu]\n",
              eliminated, input_edges);
        printf("[Done Transitive Reduction]...\n");
        fflush(stdout);
    }
}

#endif
