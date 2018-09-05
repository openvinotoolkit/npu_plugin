#ifndef DIJKSTRA_HPP_
#define DIJKSTRA_HPP_

#include "include/mcm/graph/graph.hpp"
#include "include/mcm/graph/stl_allocator.hpp"
#include <iostream>
#include <queue>
#include <vector>
#include <functional>
#include <set>
#include <algorithm>

namespace mv
{

    template <typename T>
    struct HeapContent
    {
        T id;
        int distance;

        friend bool operator<(const HeapContent<T>& a, const HeapContent<T>& b)
        {
            return a.distance < b.distance;
        }
    };

    //In dijkstraRT the initial graph consists of only one node. A rule to generate new neighbours must be passed
    //This rule specifies as well the exit condition (no more neighbours)
    template <typename NodeValue>
    std::vector<NodeValue> dijkstraRT(NodeValue source, NodeValue target, std::function<std::vector<NodeValue>(NodeValue)> generateNeighbours, std::function<int(NodeValue, NodeValue)> computeCost)
    {
         // Auxiliary data structures
         std::map<NodeValue, int> distances;
         std::map<NodeValue, NodeValue> previous;
         std::set<NodeValue> seen;
         std::set<NodeValue> generatedNodes;
         std::priority_queue<HeapContent<NodeValue>> minHeap;

         // Vector with paths to return
         std::vector<NodeValue> toReturn;

         // Inserting the source into heap and graph
         distances[source] = 0;
         previous[source] = source;
         HeapContent<NodeValue> source_heap = {source, distances[source]};
         minHeap.push(source_heap);
         generatedNodes.insert(source);

         while(!minHeap.empty())
         {
            HeapContent<NodeValue> top = minHeap.top();
            NodeValue u = top.id;
            minHeap.pop();
            if(seen.count(u))
                continue;
            seen.insert(u);
            if(u == target)
                break;
            std::vector<NodeValue> neighbours = generateNeighbours(u);
            int degree = neighbours.size();
            if(degree == 0)
                continue;
            for(int i = 0; i < degree; ++i)
            {
                NodeValue v = neighbours[i];
                int cost_u_v = computeCost(u, v);
                if(cost_u_v > 0) //solutions with infinite cost are marked with -1
                {
                    int distance = distances[u] + cost_u_v;
                    if(generatedNodes.count(v))
                    {
                        if(distance < distances[v])
                        {
                            distances[v] = distance;
                            previous[v] = u;
                            HeapContent<NodeValue> toPush = {v, distances[v]};
                            minHeap.push(toPush);
                        }
                    }
                    else
                    {
                        generatedNodes.insert(v);
                        distances[v] = distance;
                        previous[v] = u;
                        HeapContent<NodeValue> toPush = {v, distances[v]};
                        minHeap.push(toPush);
                    }
                }
            }
         }

         for(auto mapIt = previous.find(target); mapIt->first != source; mapIt = previous.find(mapIt->second))
            toReturn.push_back(mapIt->first);

         toReturn.push_back(source);

         std::reverse(toReturn.begin(), toReturn.end());

         return toReturn;
    }

}

#endif // DIJKSTRA_HPP_
