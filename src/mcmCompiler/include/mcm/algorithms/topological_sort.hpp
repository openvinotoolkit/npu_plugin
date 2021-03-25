#ifndef TOPOLOGICAL_SORT_HPP_
#define TOPOLOGICAL_SORT_HPP_

#include "include/mcm/graph/graph.hpp"
#include "include/mcm/algorithms/is_dag.hpp"
#include <set>
#include <vector>
#include <algorithm>
#include <stack>
#include "include/mcm/compiler/compilation_profiler.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"

namespace mv
{
    template <typename NodeIterator>
    struct OpItComparatorTemplate2
    {
        bool operator()(NodeIterator lhs, NodeIterator rhs) const
        {
            return (*lhs) < (*rhs);
        }
    };

    template <typename T_node, typename T_edge, typename OpIteratorComp>
    void visit(typename graph<T_node, T_edge>::node_list_iterator root, std::set<typename graph<T_node, T_edge>::node_list_iterator, OpIteratorComp>& unmarkedNodes, std::vector<typename graph<T_node, T_edge>::node_list_iterator>& toReturn, graph<T_node, T_edge>& g)
    {
        if(unmarkedNodes.find(root) == unmarkedNodes.end())
            return;

        std::vector<typename graph<T_node, T_edge>::node_list_iterator> sortedNbrs;
        for(auto neighbour = root->leftmost_child(); neighbour != g.node_end(); ++neighbour)
            sortedNbrs.push_back(neighbour);

        for (auto nbr: sortedNbrs)
            visit(nbr, unmarkedNodes, toReturn, g);

        unmarkedNodes.erase(root);
        toReturn.push_back(root);
    }

    // NOTE: This graph non member function works only on DAGs
    template <typename T_node, typename T_edge>
    std::vector<typename graph<T_node, T_edge>::node_list_iterator> topologicalSort(graph<T_node, T_edge>& g)
    {
        MV_PROFILED_FUNCTION(MV_PROFILE_ALGO)
        typedef typename graph<T_node, T_edge>::node_list_iterator iterator_t;

        // [Track number: S#47419]
        // if(!isDAG(g))
        //     throw RuntimeError("Algorithm", "Trying to execute topologicalSort on a graph that is not a DAG");

        std::vector<iterator_t> toReturn;
        std::stack<iterator_t> stk;
        std::set<iterator_t, OpItComparatorTemplate2<iterator_t>> unvisitedNodes;
        std::set<iterator_t, OpItComparatorTemplate2<iterator_t>> inResult;
        for(auto node = g.node_begin(); node != g.node_end(); ++node)
            unvisitedNodes.insert(node);
        toReturn.reserve(unvisitedNodes.size());

        while(!unvisitedNodes.empty()) {
            stk.push(*unvisitedNodes.begin());

            while(!stk.empty()) {
                iterator_t v = stk.top();
                unvisitedNodes.erase(v);
                bool isFinish = true;
                for(auto neighbour = v->leftmost_child(); neighbour != g.node_end(); ++neighbour) {
                    if (unvisitedNodes.find(neighbour) != unvisitedNodes.end()) {
                        stk.push(neighbour);
                        isFinish = false;
                    }
                }
                if(isFinish) {
                    stk.pop();
                    if (inResult.find(v) == inResult.end()) {
                        inResult.insert(v);
                        toReturn.push_back(v);
                    }
                }
            }
        }
        if(toReturn.size() != g.node_size())
            throw RuntimeError("Algorithm", "topologicalSort not visited all nodes");

        std::reverse(toReturn.begin(), toReturn.end());
        return toReturn;
    }
}

#endif
