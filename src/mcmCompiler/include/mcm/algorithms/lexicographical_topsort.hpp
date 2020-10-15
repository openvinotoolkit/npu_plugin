#ifndef LEXICOGRAPHICAL_TOPSORT_HPP_
#define LEXICOGRAPHICAL_TOPSORT_HPP_

#include "include/mcm/graph/graph.hpp"
#include "include/mcm/algorithms/is_dag.hpp"
#include <set>
#include <vector>
#include <algorithm>
#include "include/mcm/compiler/compilation_profiler.hpp"

namespace mv
{
    template <typename T_node, typename T_edge, typename T_nodeItComp, typename T_lexComparator>
    void visit(typename graph<T_node, T_edge>::node_list_iterator root,
            std::set<typename graph<T_node, T_edge>::node_list_iterator, T_nodeItComp>& unmarkedNodes,
            std::vector<typename graph<T_node, T_edge>::node_list_iterator>& toReturn,
            graph<T_node, T_edge>& g)
    {
        if(unmarkedNodes.find(root) == unmarkedNodes.end())
            return;

        std::vector<typename graph<T_node, T_edge>::node_list_iterator> sortedNbrs;
        for(auto neighbour = root->leftmost_child(); neighbour != g.node_end(); ++neighbour)
            sortedNbrs.push_back(neighbour);

        std::sort(sortedNbrs.begin(), sortedNbrs.end(), T_lexComparator());

        for (auto nbr: sortedNbrs)
            visit(nbr, unmarkedNodes, toReturn, g);

        unmarkedNodes.erase(root);
        toReturn.push_back(root);
    }

    // NOTE: This graph non member function works only on DAGs
    template <typename T_node, typename T_edge, typename T_nodeItComp, typename T_lexComparator>
    std::vector<typename graph<T_node, T_edge>::node_list_iterator>
    lexTopologicalSort(graph<T_node, T_edge>& g)
    {

        MV_PROFILED_FUNCTION(MV_PROFILE_ALGO)
        std::vector<typename graph<T_node, T_edge>::node_list_iterator> toReturn;

        if(!isDAG(g))
            throw "Trying to execute lexicographical topologicalSort on a graph that is not a DAG";

        std::set<typename graph<T_node, T_edge>::node_list_iterator, T_nodeItComp> unmarkedNodes;
        for(auto node = g.node_begin(); node != g.node_end(); ++node)
            unmarkedNodes.insert(node);

        while(!unmarkedNodes.empty())
        {
            auto toVisit = unmarkedNodes.begin();
            visit<T_node, T_edge, T_nodeItComp, T_lexComparator>(*toVisit, unmarkedNodes, toReturn, g);
        }

        std::reverse(toReturn.begin(), toReturn.end());
        return toReturn;
    }
}

#endif // LEXICOGRAPHICAL_TOPSORT_HPP_
