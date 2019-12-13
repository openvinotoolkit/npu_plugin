#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/graph/tensor_interference_graph.hpp"
#include "include/mcm/target/kmb/interference_graph_ordering_strategy.hpp"
#include "include/mcm/algorithms/edge_exists.hpp"
#include <limits.h>


static void tensorGraphColoringFnc(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&passDesc, mv::Element&);

namespace mv
{

    namespace pass
    {
        MV_REGISTER_PASS(TensorGraphColoring)
        .setFunc(tensorGraphColoringFnc)
        .setDescription(
            "Tensor graph coloring implementation used in memory allocation algorithm"
        );
    }
}

struct OrderingStrategyClassHash
{
    template <typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};


bool sortbyWeightAsc(const mv::TensorInterferenceGraphNode &a, const mv::TensorInterferenceGraphNode &b)
{
    return (a.weight < b.weight);
}

bool sortbyNeighborWeightAsc(const mv::TensorInterferenceGraphNode &a, const mv::TensorInterferenceGraphNode &b)
{
    return (a.neighborsWeight < b.neighborsWeight);
}

const std::unordered_map<mv::OrderingStrategy,
    mv::OrderingStrategyFunc , OrderingStrategyClassHash> interferenceGraphNodeSorters_=
{
    {
        mv::OrderingStrategy::IG_RANDOM_ORDER, [](mv::TensorInterferenceGraph& g)->std::vector<mv::TensorInterferenceGraphNode>
        {
            std::vector<mv::TensorInterferenceGraphNode> res;
            for(auto itr = g.node_begin(); itr != g.node_end(); ++itr)
            {
                res.push_back((*itr));
            }
            std::random_shuffle ( res.begin(), res.end() );
            return res;
        }
    },
    {
        mv::OrderingStrategy::IG_LARGEST_FIRST_ORDER, [](mv::TensorInterferenceGraph& g)->std::vector<mv::TensorInterferenceGraphNode>
        {
            std::vector<mv::TensorInterferenceGraphNode> nodeWeights;
            for(auto itr = g.node_begin(); itr != g.node_end(); ++itr)
            {
                nodeWeights.push_back(*itr);
            }
            sort(nodeWeights.begin(), nodeWeights.end(), sortbyWeightAsc);
            std::reverse(nodeWeights.begin(),nodeWeights.end());
            return nodeWeights;
        }
    },
    {
        mv::OrderingStrategy::IG_SMALLEST_FIRST_ORDER, [](mv::TensorInterferenceGraph& g)->std::vector<mv::TensorInterferenceGraphNode>
        {
            std::vector<mv::TensorInterferenceGraphNode> nodeWeights;
            for(auto itr = g.node_begin(); itr != g.node_end(); ++itr)
            {
                nodeWeights.push_back(*itr);
            }
            sort(nodeWeights.begin(), nodeWeights.end(), sortbyWeightAsc);
            return nodeWeights;
        }
    },
    {
        mv::OrderingStrategy::IG_LARGEST_NEIGHBORS_FIRST, [](mv::TensorInterferenceGraph& g)->std::vector<mv::TensorInterferenceGraphNode>
        {
            std::vector<mv::TensorInterferenceGraphNode> nodeWeights;
            for(auto itr = g.node_begin(); itr != g.node_end(); ++itr)
            {
                nodeWeights.push_back(*itr);
                //(*itr).print();
            }
            sort(nodeWeights.begin(), nodeWeights.end(), sortbyNeighborWeightAsc);
            std::reverse(nodeWeights.begin(),nodeWeights.end());
            return nodeWeights;
        }
    },
    {
        mv::OrderingStrategy::IG_SMALLEST_NEIGHBORS_FIRST, [](mv::TensorInterferenceGraph& g)->std::vector<mv::TensorInterferenceGraphNode>
        {
            std::vector<mv::TensorInterferenceGraphNode> nodeWeights;
            for(auto itr = g.node_begin(); itr != g.node_end(); ++itr)
            {
                nodeWeights.push_back(*itr);
            }
            sort(nodeWeights.begin(), nodeWeights.end(), sortbyNeighborWeightAsc);
            return nodeWeights;
        }
    },
};

bool isNodeSimplificable(mv::TensorInterferenceGraph::node_list_iterator& ni, long long memorySize)
{
    //std::cout << " name " << (*ni).name << " weight " << (*ni).weight << " ne_Weight " << (*ni).neighborsWeight << std::endl;
    return ((*ni).weight <= (memorySize - (*ni).neighborsWeight - (*ni).weight) );
}

std::string maxWeightedNeighbors(mv::TensorInterferenceGraph& g)
{
    auto orderingFunc = interferenceGraphNodeSorters_.at(mv::OrderingStrategy::IG_LARGEST_NEIGHBORS_FIRST);
    auto orderedNodes = orderingFunc(g);

    //In case of a tie, return the node with min weight to minimize the spill cost
    auto itr = orderedNodes.begin();
    auto max = (*itr).neighborsWeight;
    auto minWeight = (*itr).weight;
    auto selectedNode = (*itr).name;
    itr++;
    while (itr != orderedNodes.end() && (*itr).neighborsWeight == max)
    {
        if ((*itr).weight < minWeight)
        {
            selectedNode = (*itr).name;
            minWeight = (*itr).weight;
        }
        itr++;
    }

    return selectedNode;
}

void updateWeights(mv::TensorInterferenceGraph& g, mv::TensorInterferenceGraph::node_list_iterator ni, mv::OrderingStrategy orderStrategy)
{
    auto parentWeight = (*ni).weight;

    for (mv::TensorInterferenceGraph::node_child_iterator itc(ni); itc != g.node_end(); ++itc)
    {
        if (!(*itc).isSelected)
            (*itc).neighborsWeight -= parentWeight;
    }

    //update the neighbors weight so it will be last in the ordering
    if (orderStrategy == mv::OrderingStrategy::IG_LARGEST_NEIGHBORS_FIRST)
        (*ni).neighborsWeight = 0;
    else
    {
        (*ni).neighborsWeight = std::numeric_limits<size_t>::infinity();
    }

    (*ni).isSelected = true; //instead of removing it, just mark it as selected
    //g.node_erase(ni);
}

std::queue<std::string> aggressiveSimplify(const mv::pass::PassEntry& pass, mv::TensorInterferenceGraph& g, long long memorySize, mv::OrderingStrategy orderStrategy)
{
    //mv::TensorInterferenceGraph gCopy(g);
    // Limiting the strategies we support to IG_LARGEST_NEIGHBORS_FIRST or IG_SMALLEST_NEIGHBORS_FIRST, to avoid
    // creating a copy of the graph and removing nodes from it, since these are very heavy on performance and take a lot of time for large graphs.
    // This can be extended to support the other strategies if needed by adding more fields for each node.

    if (orderStrategy != mv::OrderingStrategy::IG_LARGEST_NEIGHBORS_FIRST &&
         orderStrategy != mv::OrderingStrategy::IG_SMALLEST_NEIGHBORS_FIRST)
            throw mv::ArgumentError("aggressiveSimplify", "orderStrategy", "", "aggressiveSimplify currently only supports IG_LARGEST_NEIGHBORS_FIRST and IG_SMALLEST_NEIGHBORS_FIRST!");

    std::queue<std::string> agNodeOrder;
    std::vector<mv::TensorInterferenceGraphNode> orderedNodes;
    auto orderingFunc = interferenceGraphNodeSorters_.at(orderStrategy);

    auto n = g.node_size();
    while(n > 0)
    {
        orderedNodes = orderingFunc(g);
        auto itr = orderedNodes.begin();
        auto ni = g.node_find(*itr);
        if (orderStrategy != mv::OrderingStrategy::IG_LARGEST_NEIGHBORS_FIRST && !isNodeSimplificable(ni, memorySize))
        {
            //std::cout << "potentialSpill" <<std::endl;
            auto potentialSpillOrderingFunc = interferenceGraphNodeSorters_.at(mv::OrderingStrategy::IG_LARGEST_NEIGHBORS_FIRST);
            orderedNodes = potentialSpillOrderingFunc(g);
            ni = g.node_find(*orderedNodes.begin());
        }
        pass.log(mv::Logger::MessageType::Debug, "aggressiveSimplify picked " + (*ni).name + " nodes left " + std::to_string(n));

        agNodeOrder.push((*ni).name);
        updateWeights(g, ni, orderStrategy);
        n--;
    }
    return agNodeOrder;
}

void printASOrder(std::queue<std::string> order, std::string name)
{
    std::cout << " printing aggressive simplify for " << name << ":" << std::endl;
    while (!order.empty())
    {
        std::string node = order.front();
        order.pop();
        std::cout << node << std::endl;
    }
    std::cout << " ========================================" << std::endl;
}

std::list<std::string> getColoredNeighbors(mv::TensorInterferenceGraph& g, mv::TensorInterferenceGraph::node_list_iterator ni)
{
    std::list<std::string> coloredNeighbors;
    for (mv::TensorInterferenceGraph::node_parent_iterator itr(ni); itr != g.node_end(); ++itr)
    {
        if ((*itr).isColored)
            coloredNeighbors.push_back((*itr).name);
    }
    return coloredNeighbors;
}

std::vector<std::pair<size_t, size_t>> mergeIntervals(std::vector<std::pair<size_t, size_t>> intervals)
{
    size_t m = 0;

    for (size_t i = 1; i < intervals.size(); i++)
    {
        if (intervals[i].first > intervals[m].second)
        {
            m++;
            intervals[m] = intervals[i];
        }
        else
        {
            intervals[m].second = std::max(intervals[i].second, intervals[m].second);
        }

    }
    std::vector<std::pair<size_t, size_t>> mergedIntervals(intervals.begin(), intervals.begin() + m + 1);
    return mergedIntervals;
}

std::vector<std::pair<size_t, size_t>> calculateGaps(std::list<std::string>& coloredNeighbors, mv::TensorInterferenceGraph& g, long long memorySize, bool addRightMost = true)
{
    std::vector<std::pair<size_t, size_t>> gaps;

    if (coloredNeighbors.size() == 0)
    {
        gaps.push_back(std::pair<size_t, size_t>(0, memorySize));
        return gaps;
    }

    size_t maxHeight = 0;
    std::vector<std::pair<size_t, size_t>> intervals;
    for (auto it = coloredNeighbors.begin(); it != coloredNeighbors.end(); it++)
    {
        auto coloredNode = g.node_find(*it);
        intervals.push_back(std::pair<size_t, size_t>((*coloredNode).address, (*coloredNode).height));
        if ((*coloredNode).height > maxHeight)
            maxHeight = (*coloredNode).height;
    }

    sort(intervals.begin(), intervals.end(), [](const std::pair<size_t, size_t> & a, const std::pair<size_t, size_t> & b)
        {
            if (a.first < b.first)
                return true;
            if (a.first == b.first)
                return (a.second <= b.second);
            return false;
        });

    //merge intervals
    intervals = std::move(mergeIntervals(intervals));

    size_t currentAddress = 0;
    size_t gap;

    for (auto it = intervals.begin(); it != intervals.end(); it++)
    {
        gap = it->first - currentAddress;
        if (gap > 0)
            gaps.push_back(std::pair<size_t, size_t>(currentAddress, gap));
        currentAddress = it->second;
    }

    //add rightmost gap
    if (addRightMost)
    {
        gap = memorySize - maxHeight;
        if (gap > 0)
            gaps.push_back(std::pair<size_t, size_t>(maxHeight, gap));
    }
    return gaps;
}
void assignOrientation(mv::graph<std::string, int>& directedGraph, mv::TensorInterferenceGraph::node_list_iterator& ni,
    std::list<std::string>& coloredNeighbors, mv::TensorInterferenceGraph& g, std::size_t& directedGraphEdgeMaxId)
{
    auto currentNode = directedGraph.node_find((*ni).name);
    for (auto it = coloredNeighbors.begin(); it != coloredNeighbors.end(); it++)
    {
        auto neighbor = g.node_find(*it);
        auto dn = directedGraph.node_find((*neighbor).name);

        if ((*ni).address < (*neighbor).address)
        {
            directedGraph.edge_insert(dn, currentNode, directedGraphEdgeMaxId++);
            //std::cout << "\tassignOrientation::adding edge from" << *dn << " -> " << *currentNode << std::endl;
        }
        else
        {
            directedGraph.edge_insert(currentNode, dn, directedGraphEdgeMaxId++);
            //std::cout << "\tassignOrientation::adding edge from" << *currentNode  << " -> "  << *dn << std::endl;
        }

    }

    //collect redundent edges of predecessors and successors of colored neighbors in DAG
    //POC removes edges when detecting one, and dont collect then remove, that might leave some edges (since it's order dependant)
    std::vector<int> redundantEdges;
    for (auto it = coloredNeighbors.begin(); it != coloredNeighbors.end(); it++)
    {
        mv::graph<std::string, int>::node_list_iterator dn = directedGraph.node_find(*it);
        for (mv::graph<std::string, int>::node_parent_iterator parentIt(dn); parentIt != directedGraph.node_end(); ++parentIt)
        {
            //for each parent collect map of children to edges
            std::map<std::string, mv::graph<std::string, int>::edge_sibling_iterator> sinkMap;
            for(auto e = parentIt->leftmost_output(); e != directedGraph.edge_end(); ++e)
                sinkMap.emplace(*e->sink(), e);
            //now check if any of successors of  current neighbor is in the map
            for (mv::graph<std::string, int>::node_child_iterator childIt(dn); childIt != directedGraph.node_end(); ++childIt)
            {
                auto edgeEntry = sinkMap.find(*childIt);
                if (edgeEntry != sinkMap.end())
                {
                    redundantEdges.push_back(*edgeEntry->second);
                }
            }
        }
    }

    //remove those edges
    for (auto it = redundantEdges.begin(); it != redundantEdges.end(); it++)
    {
        auto e = directedGraph.edge_find((*it));
        if  (e != directedGraph.edge_end())
            directedGraph.edge_erase(e);
    }
}

std::size_t updateNodeAddress(std::size_t startAddress, mv::TensorInterferenceGraph::node_list_iterator& ni, size_t chromaticNumber,
    std::list<std::string>& coloredNeighbors, mv::graph<std::string, int>& directedGraph, mv::TensorInterferenceGraph& g, size_t& directedGraphMaxEdgeId)
{
    (*ni).address = startAddress;
    (*ni).height = startAddress + (*ni).weight;
    (*ni).isColored = true;
    assignOrientation(directedGraph, ni, coloredNeighbors, g, directedGraphMaxEdgeId);

    return std::max(chromaticNumber, (*ni).height);
}

std::size_t maxHeight(mv::TensorInterferenceGraph::node_list_iterator& ni, mv::graph<std::string, int>& directedGraph, mv::TensorInterferenceGraph& g)
{
    auto directedNode = directedGraph.node_find((*ni).name);
    if (directedNode->parents_size() == 0)
        return (*ni).height;

    size_t currMaxHeight = 0;
    for (mv::graph<std::string, int>::node_parent_iterator parentIt(directedNode); parentIt != directedGraph.node_end(); ++parentIt)
    {
        auto nj = g.node_find(*parentIt);
        auto currHeight = maxHeight(nj, directedGraph, g);
        if (currHeight > currMaxHeight)
            currMaxHeight = currHeight;
    }
    return currMaxHeight;
}

std::size_t tryInsertion(std::pair<size_t, size_t>& gap, mv::TensorInterferenceGraph::node_list_iterator& ni,
    std::list<std::string>& coloredNeighbors, mv::graph<std::string, int>& directedGraph, mv::TensorInterferenceGraph& g)
{
    auto startAddress = gap.first;
    auto size = gap.second;
    auto gapDelta = std::max((*ni).weight - size, (size_t)0);

    size_t currMaxHeight = 0;

    for (auto it = coloredNeighbors.begin(); it != coloredNeighbors.end(); it++)
    {
        auto neighbor = g.node_find(*it);
        if ((*neighbor).address > startAddress)
        {
            auto neighborMax = maxHeight(neighbor, directedGraph, g);
            if (neighborMax > currMaxHeight)
                currMaxHeight = neighborMax;
        }
    }
    //is it possible that currMaxHeight stays 0??
    return currMaxHeight + gapDelta;
}

std::size_t updateHeights(mv::TensorInterferenceGraph::node_list_iterator& ni, mv::graph<std::string, int>& directedGraph,
    mv::TensorInterferenceGraph& g, size_t chromaticNumber = 0)
{
    size_t maxChromaticNumber = chromaticNumber;
    auto directedNode = directedGraph.node_find((*ni).name);
    for (mv::graph<std::string, int>::node_parent_iterator parentIt(directedNode); parentIt != directedGraph.node_end(); ++parentIt)
    {
        auto parentNodeIt = g.node_find(*parentIt);
        (*parentNodeIt).address = std::max((*parentNodeIt).address, (*ni).height);
        (*parentNodeIt).height = (*parentNodeIt).address + (*parentNodeIt).weight;

        chromaticNumber = std::max(chromaticNumber, (*parentNodeIt).height);
        auto currChromaticnumber = updateHeights(parentNodeIt, directedGraph, g, chromaticNumber);
        if (currChromaticnumber > maxChromaticNumber)
            maxChromaticNumber = currChromaticnumber;
    }
    return maxChromaticNumber;
}

void printGraph(std::string name, mv::graph<std::string, int>& g)
{
     // Nodes list
    std::cout << "Printing Graph: " << name << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Nodes list: " << std::endl;
    for (auto it = g.node_begin(); it != g.node_end(); ++it)
        std::cout << (*it) << " " << std::endl;

    std::cout << std::endl;

     // Edges list
    std::cout << "Edges list: " << std::endl;
    for (auto it = g.edge_begin(); it != g.edge_end(); ++it)
        std::cout << " EDGE: " << *it << " Source " << (*it->source()) <<  " sink " << (*it->sink()) << std::endl;

    std::cout << std::endl;
    std::cout << "=========================================================" << std::endl;

}

std::size_t bestFitSelect(std::string& name, mv::TensorInterferenceGraph& g, long long memorySize, size_t chromaticNumber,
    mv::graph<std::string, int>& directedGraph, std::size_t& directedGraphMaxEdgeId)
{
    auto ni = g.node_find(name);
    auto coloredNeighbors = std::move(getColoredNeighbors(g, ni));
    coloredNeighbors.sort();

    auto gaps = calculateGaps(coloredNeighbors, g, memorySize);

    for (auto itr = gaps.begin(); itr != gaps.end(); itr++)
    {
        if (itr->second > (*ni).weight)
        {
            chromaticNumber = updateNodeAddress(itr->first, ni, chromaticNumber, coloredNeighbors, directedGraph, g, directedGraphMaxEdgeId);
            return chromaticNumber;
        }
    }

    //no gap big enough
    if (gaps.size() == 1)
        //Actual Spill will be handled in runtime
        //eventually, the exception is just to indicate that currently it's not handled
        throw mv::ArgumentError("bestFitSelect", "gaps size", "", "trying to allocate " + name + " gap size == 1");

    auto lastgap = gaps.end();
    lastgap--;
    size_t insertionChromaticNumbersMin = ULONG_MAX;
    size_t index = 0;
    size_t currChromaticNumber;
    for (auto itr = gaps.begin(); itr != lastgap; itr++)
    {
        currChromaticNumber = tryInsertion(*itr, ni, coloredNeighbors, directedGraph, g);
        if (currChromaticNumber < insertionChromaticNumbersMin)
        {
            insertionChromaticNumbersMin = currChromaticNumber;
            index = (itr - gaps.begin());
        }
    }

    if (insertionChromaticNumbersMin > memorySize)
        //Actual Spill will be handled in runtime
        //eventually, the exception is just to indicate that currently it's not handled
        throw mv::ArgumentError("bestFitSelect", "insertionChromaticNumbersMin > memorySize", "", "trying to allocate " + name + std::to_string(memorySize) + " < " + std::to_string(insertionChromaticNumbersMin));

    chromaticNumber = updateNodeAddress(gaps[index].first, ni, chromaticNumber, coloredNeighbors, directedGraph, g, directedGraphMaxEdgeId);
    chromaticNumber = updateHeights(ni, directedGraph, g);
    return chromaticNumber;
}


size_t bestFitMemoryAllocation(const mv::pass::PassEntry& pass, mv::ComputationModel& model,
                            std::queue<std::string>& order,
                            mv::TensorInterferenceGraph& g,
                            long long memorySize,
                            bool isHeap)
{
    size_t chromaticNumber = 0;
    std::size_t directedGraphEdgeMaxId = 0;

    //build orientation assignment dag
    mv::graph<std::string, int> directedGraph;
    for (auto ni = g.node_begin(); ni != g.node_end(); ++ni)
    {
        directedGraph.node_insert((*ni).name);
    }

    while (!order.empty())
    {
        std::string node = order.front();
        order.pop();
        chromaticNumber = bestFitSelect(node, g, memorySize, chromaticNumber, directedGraph, directedGraphEdgeMaxId);
        //printGraph("BestFitDirectedGraph", directedGraph);

    }
    //printGraph("BestFitDirectedGraphFinal", directedGraph);

    mv::DataModel dm(model);
    //set address in tensors
    size_t maxHeight = 0;
    for (mv::TensorInterferenceGraph::node_list_iterator it = g.node_begin(); it != g.node_end(); ++it)
    {
        auto t = model.getTensor((*it).name);
        t->setAddress((*it).address); //still needed to set sparsityMap and storageElement addresses
        if ((*it).height > maxHeight)
            maxHeight = (*it).height;
        auto tensorAllocatorName = t->get<std::set<std::string>>("allocators").begin();
        auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
        mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, t); // 0 is the only stage for now, but this will probably change in the future
        tensorBufferIt->setOffset((*it).address);
    }
    if (isHeap && maxHeight != chromaticNumber)
        throw mv::ArgumentError("Graph Coloring produced overlap!", "", "chromaticNumber is not correct!!", "chromaticNumber " + std::to_string(chromaticNumber) + " maxHeight " + std::to_string(maxHeight));
    ///test address allocation dont overlap:
    for (mv::TensorInterferenceGraph::node_list_iterator ni = g.node_begin(); ni != g.node_end(); ++ni)
    {
        //std::cout << "testing node " << (*ni).name << std::endl;
        for (mv::TensorInterferenceGraph::node_parent_iterator itr(ni); itr != g.node_end(); ++itr)
        {
            auto coloredNode = itr;
            if ((*ni).address < (*itr).address && (*itr).address < (*ni).height)
                throw mv::ArgumentError("Graph Coloring produced overlap!", "", "", " Overlap is between " + (*ni).name  + " and " + (*itr).name );
        }
    }

    return chromaticNumber;
}

void tensorGraphColoringFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element& passDesc, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    pass.log(mv::Logger::MessageType::Debug, "Graph Coloring Started");

    mv::OpModel om(model);

    auto memDefs = target.memoryDefs();
    auto globalConfigParams = model.getGlobalConfigParams();

    pass.log(mv::Logger::MessageType::Debug, "MemoryDefs ");
    for (auto i = memDefs.begin(); i != memDefs.end(); i++)
        pass.log(mv::Logger::MessageType::Debug, ""+ i->first + " size " + std::to_string(i->second.size) +  " alignment " +  std::to_string(i->second.alignment));

//    auto memsize = memDefs.find("VPU_DDR_BSS")->second.size;
//    auto alignment = 16; //memDefs.find("VPU_DDR_BSS")->second.alignment; //TODO for now POC uses 16 for all memory
//    mv::TensorInterferenceGraph ddr_bss_g(model, alignment,
//            [](const mv::Data::TensorIterator& t) -> bool
//            {
//                return (t->isPopulated());
//            },
//            [](const mv::Data::OpListIterator& t) -> bool
//            {
//                return (t->getOpType() == "DMATask");
//            },
//            true);
//    auto agOrder = aggressiveSimplify(ddr_bss_g, memsize, mv::OrderingStrategy::IG_LARGEST_NEIGHBORS_FIRST);
//    //printASOrder(agOrder, "DDR_BSS");

//    bestFitMemoryAllocation(model, agOrder, ddr_bss_g, memsize);
//    //ddr_bss_g.drawGraph("ddr_bss_memory");
    auto alignment = 16; //memDefs.find("VPU_DDR_Heap")->second.alignment;//TODO for now POC uses 16 for all memory
    pass.log(mv::Logger::MessageType::Info, " Generating Heap Tig");
    mv::TensorInterferenceGraph ddr_heap_g(pass, model, alignment,
            [](const mv::Data::TensorIterator& t) -> bool
            {
                return (!t->isPopulated() &&
                    t->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::DDR);
            },
            [](const mv::Data::OpListIterator& t) -> bool
            {

                if (t->getOpType() == "DMATask" &&
                    t->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::DDR)
                    return true;
                if (t->getOpType() == "UPATask")
                    return true;
                return false;
            },
            [](const mv::Data::OpListIterator& opIterator) -> bool
            {
                auto opType = opIterator->getOpType();
                if (opType == "Deallocate")
                {
                    //Deallocate is a LeonTask
                    auto location = opIterator->get<mv::Tensor::MemoryLocation>("Location");
                    return (location == mv::Tensor::MemoryLocation::DDR);
                }
                if (opType == "UPATask")
                    return true;

                return false;
            },
            false);
    auto memsize = memDefs.find("VPU_DDR_Heap")->second.size;

    auto agOrder = aggressiveSimplify(pass, ddr_heap_g, memsize, mv::OrderingStrategy::IG_LARGEST_NEIGHBORS_FIRST);
    //printASOrder(agOrder, "DDR_HEAP");
    bool isHeap = true;
    size_t maxMemoryUsed = bestFitMemoryAllocation(pass, model, agOrder, ddr_heap_g, memsize, isHeap);
    globalConfigParams->set<int>("DDRScratch", maxMemoryUsed);
    if(passDesc.hasAttr("heapOutput"))
        ddr_heap_g.drawGraph(passDesc.get<std::string>("heapOutput"));

    alignment = 16; //memDefs.find("VPU_CMX_NN")->second.alignment;//TODO for now POC uses 16 for all memory
    pass.log(mv::Logger::MessageType::Info, " generating cmx TIG");
    mv::TensorInterferenceGraph nncmx_g(pass, model, alignment, nullptr, nullptr,
    [](const mv::Data::OpListIterator& opIterator) -> bool
    {
        auto opType = opIterator->getOpType();
        if (opType == "Deallocate")
        {
            //Deallocate is a LeonTask
            auto location = opIterator->get<mv::Tensor::MemoryLocation>("Location");
            return (location == mv::Tensor::MemoryLocation::NNCMX);
        }
        //TODO: it might be that we dont need this condition, dellocate is what we are looking for
        // removing the below condition generated the same mcm blob for a small network
        if (opType  == "DMATask" &&
            (opIterator->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::NNCMX2DDR ||
            opIterator->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::NNCMX2UPACMX))
            return true;

        return false;
    },
    false, true);

    memsize = globalConfigParams->get<unsigned>("totalCmx");
    pass.log(mv::Logger::MessageType::Info, " Calling AggressiveSimplify");

    agOrder = aggressiveSimplify(pass, nncmx_g, memsize, mv::OrderingStrategy::IG_LARGEST_NEIGHBORS_FIRST);
    //printASOrder(agOrder, "NNCMX");
    pass.log(mv::Logger::MessageType::Info, " Calling bestFitMemoryAllocation");
    bestFitMemoryAllocation(pass, model, agOrder, nncmx_g, memsize, !isHeap);
    pass.log(mv::Logger::MessageType::Info, " Calling DrawGraph");
    if(passDesc.hasAttr("cmxOutput"))
        nncmx_g.drawGraph(passDesc.get<std::string>("cmxOutput"));

    pass.log(mv::Logger::MessageType::Debug, "Graph Coloring Ended");
}
