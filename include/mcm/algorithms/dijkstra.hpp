#ifndef DIJKSTRA_HPP_
#define DIJKSTRA_HPP_

#include "include/mcm/graph/graph.hpp"
#include <iostream>
#include <queue>
#include <vector>
#include <functional>
#include <set>
#include <algorithm>
#include <map>

namespace mv
{

    template <typename T, typename D>
    struct HeapContent
    {
        T id;
        D distance;

        friend bool operator<(const HeapContent<T, D>& a, const HeapContent<T, D>& b)
        {
            return a.distance < b.distance;
        }

        friend bool operator>(const HeapContent<T, D>& a, const HeapContent<T, D>& b)
        {
            return a.distance > b.distance;
        }
    };


    template<typename T, typename D>
    struct DijkstraReturnValue
    {
        std::vector<T> nodes;
        std::vector<D> distances;
    };

    //In dijkstraRT the initial graph consists of only one node. A rule to generate new neighbours must be passed
    //This rule specifies as well the exit condition (no more neighbours)
    template <typename NodeValue, typename DistanceValue>
    DijkstraReturnValue<NodeValue, DistanceValue> dijkstraRT(NodeValue source, NodeValue target, std::function<std::vector<NodeValue>(NodeValue)> generateNeighbours, std::function<DistanceValue(NodeValue, NodeValue)> computeCost)
    {
         // Auxiliary data structures
         std::map<NodeValue, DistanceValue> distances;
         std::map<NodeValue, NodeValue> previous;
         std::set<NodeValue> seen;
         std::set<NodeValue> generatedNodes;
         std::priority_queue<HeapContent<NodeValue, DistanceValue>, std::vector<HeapContent<NodeValue, DistanceValue>>, std::greater<HeapContent<NodeValue, DistanceValue>>> minHeap;
         DistanceValue zeroCost(0);

         // Variable to return
         DijkstraReturnValue<NodeValue, DistanceValue> toReturn;

         // Inserting the source into heap and graph
         distances[source] = zeroCost;
         previous[source] = source;
         HeapContent<NodeValue, DistanceValue> source_heap = {source, distances[source]};
         minHeap.push(source_heap);
         generatedNodes.insert(source);

         while(!minHeap.empty())
         {
            HeapContent<NodeValue, DistanceValue> top = minHeap.top();
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
                DistanceValue cost_u_v = computeCost(u, v);
                if(cost_u_v > zeroCost) //solutions with infinite cost are marked with -1
                {
                    DistanceValue distance = distances[u] + cost_u_v;
                    if(generatedNodes.count(v))
                    {
                        if(distance < distances[v])
                        {
                            distances[v] = distance;
                            previous[v] = u;
                            HeapContent<NodeValue, DistanceValue> toPush = {v, distances[v]};
                            minHeap.push(toPush);
                        }
                    }
                    else
                    {
                        generatedNodes.insert(v);
                        distances[v] = distance;
                        previous[v] = u;
                        HeapContent<NodeValue, DistanceValue> toPush = {v, distances[v]};
                        minHeap.push(toPush);
                    }
                }
            }
         }

         for(auto mapIt = previous.find(target); mapIt->first != source; mapIt = previous.find(mapIt->second))
         {
            toReturn.nodes.push_back(mapIt->first);
            toReturn.distances.push_back(distances[mapIt->first]);
         }

         toReturn.nodes.push_back(source);

         std::reverse(toReturn.nodes.begin(), toReturn.nodes.end());
         std::reverse(toReturn.distances.begin(), toReturn.distances.end());

         return toReturn;
    }

}

#endif // DIJKSTRA_HPP_
