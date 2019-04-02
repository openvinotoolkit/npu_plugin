#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/graph/tensor_interference_graph.hpp"
#include "include/mcm/target/keembay/interference_graph_ordering_strategy.hpp"


static void graphColoringFnc(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {
        MV_REGISTER_PASS(GraphColoring)
        .setFunc(graphColoringFnc)
        .setDescription(
            "graph coloring implmentation used in memory allocation algorithm"
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
bool sortbyWeightDesc(const mv::TensorInterferenceGraphNode &a, const mv::TensorInterferenceGraphNode &b)
{
    return (a.weight >= b.weight);
}

bool sortbyNeighborWeightAsc(const mv::TensorInterferenceGraphNode &a, const mv::TensorInterferenceGraphNode &b)
{
    return (a.neighborsWeight < b.neighborsWeight);
}

bool sortbyNeighborWeightDesc(const mv::TensorInterferenceGraphNode &a, const mv::TensorInterferenceGraphNode &b)
{
    return (a.neighborsWeight >= b.neighborsWeight);
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
            sort(nodeWeights.begin(), nodeWeights.end(), sortbyWeightDesc);
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
            }
            sort(nodeWeights.begin(), nodeWeights.end(), sortbyNeighborWeightDesc);
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
    return ((*ni).weight <= (memorySize - (*ni).neighborsWeight));
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

std::stack<std::string> aggressiveSimplify(mv::TensorInterferenceGraph& g, long long memorySize, mv::OrderingStrategy orderStrategy)
{
    //create a copy of g since we are going to delete nodes from it
    mv::TensorInterferenceGraph gCopy(g);
    std::stack<std::string> agNodeOrder;
    std::vector<mv::TensorInterferenceGraphNode> orderedNodes;
    auto orderingFunc = interferenceGraphNodeSorters_.at(orderStrategy);

    bool continueSimplify = true;

    while (continueSimplify)
    {
        continueSimplify = false;
        orderedNodes = orderingFunc(gCopy);

        for (auto itr = orderedNodes.begin(); itr != orderedNodes.end(); itr++)
        {
            //std::cout << "checking " << *itr << std::endl;
            auto ni = gCopy.node_find(*itr);
            if (isNodeSimplificable(ni, memorySize))
            {
                agNodeOrder.push((*itr).name);
                removeNodeAndUpdateWeights(gCopy, ni);
                continueSimplify = true;
            }

            if (gCopy.node_size() > 0 && !continueSimplify)
            {
                auto node = maxWeightedNeighbors(gCopy);
                agNodeOrder.push(node);
                auto nj = gCopy.node_find(node);
                removeNodeAndUpdateWeights(gCopy, nj);
                continueSimplify = true;
            }
        }
    }
    return agNodeOrder;
}

void printASOrder(std::stack<std::string> order, std::string name)
{
    std::cout << " printing aggressive simplify for " << name << ":" << std::endl;
    while (!order.empty())
    {
        std::string node = order.top();
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

std::list<std::pair<size_t, size_t>> calculateGaps(std::list<std::string>& coloredNeighbors, mv::TensorInterferenceGraph& g, long long memorySize, bool addRightMost = true)
{
    std::list<std::pair<size_t, size_t>> gaps;

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
    for (auto it = coloredNeighbors.begin(); it != coloredNeighbors.end(); it++)
    {
        auto neighbor = g.node_find(*it);
        auto dn = directedGraph.node_find((*neighbor).name);

        if ((*ni).address < (*neighbor).address)
        {
            directedGraph.edge_insert(dn, currentNode, 0);
        }
        else
        {
            directedGraph.edge_insert(currentNode, dn, 0);
        }

    }

    //TODO remove edges...?
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

    ///TODO - no gap big enough
    return chromaticNumber;
}

void bestFitMemoryAllocation(std::stack<std::string>& order, mv::TensorInterferenceGraph& g, long long memorySize)
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
        std::string node = order.top();
        order.pop();
        chromaticNumber = bestFitSelect(node, g, memorySize, chromaticNumber, directedGraph);
    }

    //TODO now update original tensors with address
}

void graphColoringFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::json::Object&)
{
    pass.log(mv::Logger::MessageType::Debug, "Graph Coloring Started");

    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto memDefs = target.memoryDefs();

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
    auto alignment = memDefs.find("VPU_DDR_BSS")->second.alignment;
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
    //ddr_bss_g.printGraph("DDR_BSS");
    ddr_bss_g.drawGraph("ddr_bss");
    auto agOrder = aggressiveSimplify(ddr_bss_g, memsize, mv::OrderingStrategy::IG_SMALLEST_NEIGHBORS_FIRST);
    printASOrder(agOrder, "DDR_BSS");

    bestFitMemoryAllocation(agOrder, ddr_bss_g, memsize);
    ddr_bss_g.drawGraph("ddr_bss_memory");

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
    //ddr_heap_g.printGraph("DDR_HEAP");
    ddr_heap_g.drawGraph("ddr_heap");
    memsize = memDefs.find("VPU_DDR_Heap")->second.size;
    alignment = memDefs.find("VPU_DDR_Heap")->second.alignment;
    agOrder = aggressiveSimplify(ddr_heap_g, memsize, mv::OrderingStrategy::IG_SMALLEST_NEIGHBORS_FIRST);
    printASOrder(agOrder, "DDR_HEAP");
    bestFitMemoryAllocation(agOrder, ddr_heap_g, memsize);
    ddr_heap_g.drawGraph("ddr_heap_memory");

    mv::TensorInterferenceGraph nncmx_g(model, alignment, nullptr, nullptr, false);
    memsize = memDefs.find("VPU_CMX_NN")->second.size + memDefs.find("VPU_CMX_UPA")->second.size;
    alignment = memDefs.find("VPU_CMX_NN")->second.alignment;
    agOrder = aggressiveSimplify(nncmx_g, memsize, mv::OrderingStrategy::IG_SMALLEST_NEIGHBORS_FIRST);
    //nncmx_g.printGraph("NNCMX");
    nncmx_g.drawGraph("nncmx");
    printASOrder(agOrder, "NNCMX");
    bestFitMemoryAllocation(agOrder, nncmx_g, memsize);
    nncmx_g.drawGraph("nncmx_memory");

    pass.log(mv::Logger::MessageType::Debug, "Graph Coloring Ended");
}
