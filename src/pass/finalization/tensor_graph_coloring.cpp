#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/graph/tensor_interference_graph.hpp"
#include "include/mcm/target/keembay/interference_graph_ordering_strategy.hpp"
#include "include/mcm/algorithms/edge_exists.hpp"
#include <limits.h>


static void tensorGraphColoringFnc(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

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

void removeNodeAndUpdateWeights(mv::TensorInterferenceGraph& g, mv::TensorInterferenceGraph::node_list_iterator ni)
{
    auto parentWeight = (*ni).weight;
    for (mv::TensorInterferenceGraph::node_child_iterator itc(ni); itc != g.node_end(); ++itc)
    {
        (*itc).neighborsWeight -= parentWeight;
    }

    g.node_erase(ni);
}

std::queue<std::string> aggressiveSimplify(mv::TensorInterferenceGraph& g, long long memorySize, mv::OrderingStrategy orderStrategy)
{
    //create a copy of g since we are going to delete nodes from it
    mv::TensorInterferenceGraph gCopy(g);
    std::queue<std::string> agNodeOrder;
    std::vector<mv::TensorInterferenceGraphNode> orderedNodes;
    auto orderingFunc = interferenceGraphNodeSorters_.at(orderStrategy);

    while(gCopy.node_size() > 0)
    {
        orderedNodes = orderingFunc(gCopy);
        auto itr = orderedNodes.begin();
        auto ni = gCopy.node_find(*itr);
        if (orderStrategy != mv::OrderingStrategy::IG_LARGEST_NEIGHBORS_FIRST && !isNodeSimplificable(ni, memorySize))
        {
            //std::cout << "potentialSpill" <<std::endl;
            auto potentialSpillOrderingFunc = interferenceGraphNodeSorters_.at(mv::OrderingStrategy::IG_LARGEST_NEIGHBORS_FIRST);
            orderedNodes = potentialSpillOrderingFunc(gCopy);
            ni = gCopy.node_find(*orderedNodes.begin());
        }

        agNodeOrder.push((*ni).name);
        removeNodeAndUpdateWeights(gCopy, ni);
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
    std::list<std::string>& coloredNeighbors, mv::TensorInterferenceGraph& g)
{
    auto currentNode = directedGraph.node_find((*ni).name);
    auto edgeIdx = directedGraph.edge_size();
    for (auto it = coloredNeighbors.begin(); it != coloredNeighbors.end(); it++)
    {
        auto neighbor = g.node_find(*it);
        auto dn = directedGraph.node_find((*neighbor).name);

        if ((*ni).address < (*neighbor).address)
        {
            directedGraph.edge_insert(dn, currentNode, edgeIdx++);
            //std::cout << "\tassignOrientation::adding edge from" << *dn << " -> " << *currentNode << std::endl;
        }
        else
        {
            directedGraph.edge_insert(currentNode, dn, edgeIdx++);
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
    std::list<std::string>& coloredNeighbors, mv::graph<std::string, int>& directedGraph, mv::TensorInterferenceGraph& g)
{
    (*ni).address = startAddress;
    (*ni).height = startAddress + (*ni).weight;
    (*ni).isColored = true;
    assignOrientation(directedGraph, ni, coloredNeighbors, g);

    return std::max(chromaticNumber, (*ni).height);
}

std::size_t maxHeight(mv::TensorInterferenceGraph::node_list_iterator& ni, mv::graph<std::string, int>& directedGraph, mv::TensorInterferenceGraph& g)
{
    auto directedNode = directedGraph.node_find((*ni).name);
    if (ni->parents_size() == 0)
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
    auto gapDelta = (*ni).weight - size;

    size_t currMaxHeight = 0;

    for (auto it = coloredNeighbors.begin(); it != coloredNeighbors.end(); it++)
    {
        auto neighbor = g.node_find(*it);
        if ((*neighbor).address > startAddress)
        {
            auto dn = directedGraph.node_find((*neighbor).name);
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
        auto parentNode = *parentNodeIt;
        parentNode.address = std::max(parentNode.address, (*ni).height);
        parentNode.height = parentNode.address + parentNode.weight;
        chromaticNumber = std::max(chromaticNumber, parentNode.height);
        auto currChromaticnumber = updateHeights(parentNodeIt, directedGraph, g, chromaticNumber);
        if (currChromaticnumber > maxChromaticNumber)
            maxChromaticNumber = currChromaticnumber;
    }
    return maxChromaticNumber;
}

std::size_t bestFitSelect(std::string& name, mv::TensorInterferenceGraph& g, long long memorySize, size_t chromaticNumber,
    mv::graph<std::string, int>& directedGraph)
{
    auto ni = g.node_find(name);
    auto coloredNeighbors = std::move(getColoredNeighbors(g, ni));
    coloredNeighbors.sort();

    auto gaps = calculateGaps(coloredNeighbors, g, memorySize);

    for (auto itr = gaps.begin(); itr != gaps.end(); itr++)
    {
        if (itr->second > (*ni).weight)
        {
            chromaticNumber = updateNodeAddress(itr->first, ni, chromaticNumber, coloredNeighbors, directedGraph, g);
            return chromaticNumber;
        }
    }

    //no gap big enough
    if (gaps.size() == 1)
        //Actual Spill will be handled in runtime
        //eventually, the exception is just to indicate that currently it's not handled
        throw mv::ArgumentError("bestFitSelect", "gaps size", "", "TODO Implement Actual Spill Routine");

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
        throw mv::ArgumentError("bestFitSelect", "gaps size", "", "TODO Implement Actual Spill Routine");

    chromaticNumber = updateNodeAddress(gaps[index].first, ni, chromaticNumber, coloredNeighbors, directedGraph, g);
    chromaticNumber = updateHeights(ni, directedGraph, g);
    return chromaticNumber;
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

void bestFitMemoryAllocation(mv::ComputationModel& model, std::queue<std::string>& order, mv::TensorInterferenceGraph& g, long long memorySize)
{
    size_t chromaticNumber = 0;
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
        chromaticNumber = bestFitSelect(node, g, memorySize, chromaticNumber, directedGraph);
        //printGraph("BestFitDirectedGraph", directedGraph);

    }
    //printGraph("BestFitDirectedGraphFinal", directedGraph);

    mv::DataModel dm(model);
    //set address in tensors
    for (mv::TensorInterferenceGraph::node_dfs_iterator it = g.node_begin(); it != g.node_end(); ++it)
    {
        auto t = model.getTensor((*it).name);
        t->setAddress((*it).address); //still needed to set sparsityMap and storageElement addresses
        auto tensorAllocatorName = t->get<std::set<std::string>>("allocators").begin();
        auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
        mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, t); // 0 is the only stage for now, but this will probably change in the future
        tensorBufferIt->setOffset((*it).address);
    }

}

void tensorGraphColoringFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::json::Object&)
{
    pass.log(mv::Logger::MessageType::Debug, "Graph Coloring Started");

    mv::OpModel om(model);

    auto memDefs = target.memoryDefs();
    auto globalConfigParams = model.getGlobalConfigParams();

    pass.log(mv::Logger::MessageType::Debug, "MemoryDefs ");
    for (auto i = memDefs.begin(); i != memDefs.end(); i++)
        pass.log(mv::Logger::MessageType::Debug, ""+ i->first + " size " + std::to_string(i->second.size) +  " alignment " +  std::to_string(i->second.alignment));

    //Collect all input/output tensor names
    /*for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        std::cout << " Layer: " << opIterator->getName() ;
        if (opIterator->getOpType() == "DMATask")
           std::cout << "\t\t DMA direction" << opIterator->get<mv::DmaDirection>("direction");
        std::cout << std::endl;
    }*/
    auto memsize = memDefs.find("VPU_DDR_BSS")->second.size;
    auto alignment = 16; //memDefs.find("VPU_DDR_BSS")->second.alignment; //TODO for now POC uses 16 for all memory
    mv::TensorInterferenceGraph ddr_bss_g(model, alignment,
            [](const mv::Data::TensorIterator& t) -> bool
            {
                return (t->isPopulated());
            },
            [](const mv::Data::OpListIterator& t) -> bool
            {
                return (t->getOpType() == "DMATask");
            },
            true);
    auto agOrder = aggressiveSimplify(ddr_bss_g, memsize, mv::OrderingStrategy::IG_LARGEST_NEIGHBORS_FIRST);
    //printASOrder(agOrder, "DDR_BSS");

    bestFitMemoryAllocation(model, agOrder, ddr_bss_g, memsize);
    //ddr_bss_g.drawGraph("ddr_bss_memory");

    mv::TensorInterferenceGraph ddr_heap_g(model, alignment,
            [](const mv::Data::TensorIterator& t) -> bool
            {
                return (!t->isPopulated());
            },
            [](const mv::Data::OpListIterator& t) -> bool
            {
                return (t->getOpType() == "DMATask");
            },
            false);
    memsize = memDefs.find("VPU_DDR_Heap")->second.size;
    alignment = 16; //memDefs.find("VPU_DDR_Heap")->second.alignment;//TODO for now POC uses 16 for all memory
    agOrder = aggressiveSimplify(ddr_heap_g, memsize, mv::OrderingStrategy::IG_LARGEST_NEIGHBORS_FIRST);
    //printASOrder(agOrder, "DDR_HEAP");
    bestFitMemoryAllocation(model, agOrder, ddr_heap_g, memsize);
    //ddr_heap_g.drawGraph("ddr_heap_memory");

    mv::TensorInterferenceGraph nncmx_g(model, alignment, nullptr, nullptr, false, true);
    memsize = globalConfigParams->get<unsigned>("cmx");
    alignment = 16; //memDefs.find("VPU_CMX_NN")->second.alignment;//TODO for now POC uses 16 for all memory
    agOrder = aggressiveSimplify(nncmx_g, memsize, mv::OrderingStrategy::IG_LARGEST_NEIGHBORS_FIRST);
    //printASOrder(agOrder, "NNCMX");
    bestFitMemoryAllocation(model, agOrder, nncmx_g, memsize);
    //nncmx_g.drawGraph("nncmx_memory");

    pass.log(mv::Logger::MessageType::Debug, "Graph Coloring Ended");
}
