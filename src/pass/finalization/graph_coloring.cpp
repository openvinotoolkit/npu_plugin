#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/graph/graph.hpp"
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

struct InterferenceGraphNode
{
    std::string name;
    size_t weight;
    size_t neighborsWeight;

    InterferenceGraphNode(std::string name_) : name(name_) {

    }

    bool operator==(const InterferenceGraphNode& rhs) const
    {
        return (name == rhs.name);
    }
};

using InterferenceGraph = mv::graph<InterferenceGraphNode, int>;
using TensorIteratorFilter = std::function<bool(const mv::Data::TensorIterator& t)>;
using OpIteratorFilter = std::function<bool(mv::Data::OpListIterator& t)>;
using OrderingStrategyFunc = std::function<std::vector<InterferenceGraphNode>(InterferenceGraph& g)>;

struct OrderingStrategyClassHash
{
    template <typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};


bool sortbyWeightAsc(const InterferenceGraphNode &a, const InterferenceGraphNode &b)
{
    return (a.weight < b.weight);
}
bool sortbyWeightDesc(const InterferenceGraphNode &a, const InterferenceGraphNode &b)
{
    return (a.weight >= b.weight);
}

bool sortbyNeighborWeightAsc(const InterferenceGraphNode &a, const InterferenceGraphNode &b)
{
    return (a.neighborsWeight < b.neighborsWeight);
}

bool sortbyNeighborWeightDesc(const InterferenceGraphNode &a, const InterferenceGraphNode &b)
{
    return (a.neighborsWeight >= b.neighborsWeight);
}

std::size_t getNeighborsWeight(InterferenceGraph& g, mv::ComputationModel& model, std::string& node, std::size_t alignment)
{
    size_t totalWeights = 0;
    //Since we use a directed graph to represent undirected IG, every parent node is also a child node
    // so no need to go through both
    auto ni = g.node_find(node);
    /*
    //add children
    for (InterferenceGraph::node_child_iterator itr(ni); itr != g.node_end(); ++itr)
    {
        totalWeights += model.getTensor(*itr)->computeTotalSize(alignment);
    }*/
    //add parents
    for (InterferenceGraph::node_parent_iterator itr(ni); itr != g.node_end(); ++itr)
    {
        totalWeights += model.getTensor((*itr).name)->computeTotalSize(alignment);
    }
    return totalWeights;
}

const std::unordered_map<mv::OrderingStrategy,
    OrderingStrategyFunc , OrderingStrategyClassHash> interferenceGraphNodeSorters_=
{
    {
        mv::OrderingStrategy::IG_RANDOM_ORDER, [](InterferenceGraph& g)->std::vector<InterferenceGraphNode>
        {
            std::vector<InterferenceGraphNode> res;
            for(auto itr = g.node_begin(); itr != g.node_end(); ++itr)
            {
                res.push_back((*itr));
            }
            std::random_shuffle ( res.begin(), res.end() );
            return res;
        }
    },
    {
        mv::OrderingStrategy::IG_LARGEST_FIRST_ORDER, [](InterferenceGraph& g)->std::vector<InterferenceGraphNode>
        {
            std::vector<InterferenceGraphNode> nodeWeights;
            for(auto itr = g.node_begin(); itr != g.node_end(); ++itr)
            {
                nodeWeights.push_back(*itr);
            }
            sort(nodeWeights.begin(), nodeWeights.end(), sortbyWeightDesc);
            return nodeWeights;
        }
    },
    {
        mv::OrderingStrategy::IG_SMALLEST_FIRST_ORDER, [](InterferenceGraph& g)->std::vector<InterferenceGraphNode>
        {
            std::vector<InterferenceGraphNode> nodeWeights;
            for(auto itr = g.node_begin(); itr != g.node_end(); ++itr)
            {
                nodeWeights.push_back(*itr);
            }
            sort(nodeWeights.begin(), nodeWeights.end(), sortbyWeightAsc);
            return nodeWeights;
        }
    },
    {
        mv::OrderingStrategy::IG_LARGEST_NEIGHBORS_FIRST, [](InterferenceGraph& g)->std::vector<InterferenceGraphNode>
        {
            std::vector<InterferenceGraphNode> nodeWeights;
            for(auto itr = g.node_begin(); itr != g.node_end(); ++itr)
            {
                nodeWeights.push_back(*itr);
            }
            sort(nodeWeights.begin(), nodeWeights.end(), sortbyNeighborWeightDesc);
            return nodeWeights;
        }
    },
    {
        mv::OrderingStrategy::IG_SMALLEST_NEIGHBORS_FIRST, [](InterferenceGraph& g)->std::vector<InterferenceGraphNode>
        {
            std::vector<InterferenceGraphNode> nodeWeights;
            for(auto itr = g.node_begin(); itr != g.node_end(); ++itr)
            {
                nodeWeights.push_back(*itr);
            }
            sort(nodeWeights.begin(), nodeWeights.end(), sortbyNeighborWeightAsc);
            return nodeWeights;
        }
    },
};


bool pathExists(const mv::Data::OpListIterator& source, const mv::Data::OpListIterator& sink, const mv::Data::OpListIterator end)
{
    for (mv::Data::OpDFSIterator it(source); it != end; ++it)
    {
        if (*it == *sink)
            return true;
    }
    return false;
}

std::string getTensorTopMaster(const mv::Data::TensorIterator& t, mv::ComputationModel& model)
{
    if (t->hasAttr("master"))
    {
        auto master = model.getTensor(t->get<std::string>("master"));
        return getTensorTopMaster(master, model);
    }
    return t->getName();
}

std::set<std::string> getTaskTopTensors(const std::vector<mv::Data::TensorIterator>& tensorList, mv::ComputationModel& model, const TensorIteratorFilter& tensorFilter)
{
    std::set<std::string> topTensors;

    for (unsigned i = 0; i < tensorList.size(); i++)
    {
        std::string name = getTensorTopMaster(tensorList[i], model);
        if (!tensorFilter || tensorFilter(model.getTensor(name)))
        {
            topTensors.insert(name);
        }
    }
    return topTensors;
}

bool isTensorInTopNames(const std::vector<mv::Data::TensorIterator>& tensorList, mv::ComputationModel& model,
    const std::string tensorName)
{
    for (unsigned i = 0; i < tensorList.size(); i++)
    {
        std::string name = getTensorTopMaster(tensorList[i], model);
        if (name == tensorName)
        {
            return true;
        }
    }
    return false;
}

std::set<std::string> getTensorNames(mv::ComputationModel& model, const TensorIteratorFilter& tensorFilter, const OpIteratorFilter& taskFilter)
{
    mv::OpModel om(model);
    std::set<std::string> tensorNames;

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if (!taskFilter || taskFilter(opIterator))
        {
            std::set<std::string> temp = getTaskTopTensors(opIterator->getInputTensor(), model, tensorFilter);
            tensorNames.insert(temp.begin(), temp.end());
            temp = getTaskTopTensors(opIterator->getOutputTensor(), model, tensorFilter);
            tensorNames.insert(temp.begin(), temp.end());
        }
    }
    return tensorNames;
}

void buildCompleteGraph(InterferenceGraph& g, std::set<std::string> tensorNames)
{
    for (std::set<std::string>::const_iterator name = tensorNames.begin( ); name != tensorNames.end( ); ++name)
    {
        g.node_insert(InterferenceGraphNode(*name));
    }

    int nodeId = 0;
    for (std::set<std::string>::const_iterator src = tensorNames.begin( ); src != tensorNames.end( ); ++src)
    {
        auto ni = g.node_find(*src);
        //since we are directed graph need to create a->b and b->a, so we go through all combinations
        for (std::set<std::string>::const_iterator target = tensorNames.begin( ); target != tensorNames.end( ); ++target)
        {
            if (src != target)
                g.edge_insert(ni, g.node_find(*target), nodeId++);
        }
    }
}

bool checkNodesAreNeighbors(InterferenceGraph&g, InterferenceGraph::node_list_iterator& n1, InterferenceGraph::node_list_iterator& n2)
{
    //check children
    for (InterferenceGraph::node_child_iterator it(n1); it != g.node_end(); ++it)
    {
        if (it == n2)
            return true;
    }
    //check parents
    for (InterferenceGraph::node_parent_iterator it(n1); it != g.node_end(); ++it)
    {
        if (it == n2)
            return true;
    }
    return false;
}

bool isSinkNode(mv::Data::OpListIterator& opIterator)
{
    auto opType = opIterator->getOpType();
    if (opType == "Deallocate") //Deallocate is a LeonTask
        return true;
    if (opType  == "DMATask" &&
        (opIterator->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::CMX2DDR ||
         opIterator->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::CMX2UPA))
        return true;

    return false;
}

bool checkNodeInterference(mv::ComputationModel& model, const std::string& tensor1, const std::string& tensor2)
{
    //returns true if tensor2 is already deallocated when tesnor1 is allocated
    mv::OpModel om(model);
    std::set<std::string> sourceNodeNames;
    std::set<std::string> sinkNodeNames;
    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if (isTensorInTopNames(opIterator->getOutputTensor(), model, tensor1))
            sourceNodeNames.insert(opIterator->getName());
        if (isTensorInTopNames(opIterator->getInputTensor(), model, tensor2) && isSinkNode(opIterator))
            sinkNodeNames.insert(opIterator->getName());

        //Check if there's a path from any node in sink to any node in source, if yes return true
        for (std::set<std::string>::const_iterator src = sinkNodeNames.begin( ); src != sinkNodeNames.end( ); ++src)
        {
            for (std::set<std::string>::const_iterator target = sourceNodeNames.begin( ); target != sourceNodeNames.end( ); ++target)
            {
                if (pathExists(om.getOp(*src), om.getOp(*target), om.opEnd()))
                    return true;
            }
        }

    }
    return false;
}

void drawGraph(InterferenceGraph &g, std::string outputFile, mv::ComputationModel& model)
{
    std::ofstream ostream;

    ostream.open(outputFile, std::ios::trunc | std::ios::out);
    ostream << "digraph G {\n\tgraph [splines=spline]\n";

    for (auto it = g.node_begin(); it != g.node_end(); ++it)
    {
        auto name = (*it).name;
        std::string nodeDef = "\t\"" + name + "\" [shape=box,";
        nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + name + "</B></FONT></TD></TR>";
        nodeDef += "<TR><TD ALIGN=\"CENTER\"><FONT POINT-SIZE=\"11.0\"> size: " + std::to_string(model.getTensor(name)->computeTotalSize(16))  + "</FONT></TD></TR>";
        nodeDef += "</TABLE>>";
        ostream << nodeDef << "];\n";
    }

    for (auto it = g.edge_begin(); it != g.edge_end(); ++it)
    {
        std::string edgeDef = "\t\"" + (*it->source()).name + "\" -> \"" +  (*it->sink()).name + "\"";
        ostream << edgeDef << "\n";
    }
    ostream << "}\n";
    ostream.close();
}

void genIntereferenceGraph(InterferenceGraph& g, mv::ComputationModel& model , const TensorIteratorFilter& tensorFilter,
    const OpIteratorFilter& taskFilter)
{
    InterferenceGraph directed_g;
    std::set<std::string> inputTensorNames;
    std::set<std::string> outputTensorNames;
    std::set<std::string> nodeNames;
    int nodeId = 0;

    mv::OpModel om(model);

    //Collect all input/output tensor names
    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if (!taskFilter || taskFilter(opIterator))
        {
            inputTensorNames = getTaskTopTensors(opIterator->getInputTensor(), model, tensorFilter);

            for (std::set<std::string>::const_iterator name = inputTensorNames.begin( ); name != inputTensorNames.end( ); ++name)
            {
                auto res = nodeNames.insert(*name);
                if (res.second)
                {//if we dont have it already
                    directed_g.node_insert(*name);
                    g.node_insert(*name);
                }
            }

            outputTensorNames = getTaskTopTensors(opIterator->getOutputTensor(), model, tensorFilter);

            for (std::set<std::string>::const_iterator name = outputTensorNames.begin( ); name != outputTensorNames.end( ); ++name)
            {
                auto res = nodeNames.insert(*name);
                if (res.second)
                {//if we dont have it already
                    directed_g.node_insert(*name);
                    g.node_insert(*name);
                }
            }

            //Add the obvious edges
            for (std::set<std::string>::const_iterator src = inputTensorNames.begin( ); src != inputTensorNames.end( ); ++src)
            {
                auto ni = g.node_find(*src);
                auto directed_ni = directed_g.node_find(*src);
                for (std::set<std::string>::const_iterator target = outputTensorNames.begin( ); target != outputTensorNames.end( ); ++target)
                {
                    if (*src != *target)
                    {
                        auto nj = g.node_find(*target);
                        auto directed_nj = directed_g.node_find(*target);
                        g.edge_insert(ni, nj, 2*nodeId);
                        g.edge_insert(nj, ni, 2*nodeId+1); //since we are directed graph need to create a->b and b->a
                        directed_g.edge_insert(directed_ni, directed_nj, nodeId);
                        nodeId++;
                    }
                }
            }

        }
    }

    //for each 2 nodes, if they are not yet connected (neighbors) in the undirected graph
    // and dont have a path from one to the other in the directed graph, then check if they
    // exist in memory at the same time
    for (std::set<std::string>::const_iterator source = nodeNames.begin( ); source != nodeNames.end( ); ++source)
    {
        auto ni = g.node_find(*source);
        auto directed_ni = directed_g.node_find(*source);

        for (std::set<std::string>::const_iterator target = source; target != nodeNames.end( ); ++target)
        {
            auto nj = g.node_find(*target);
            auto directed_nj = directed_g.node_find(*target);
            if (source != target && !checkNodesAreNeighbors(g, ni, nj) && !mv::pathExists<InterferenceGraphNode, int>( directed_ni, directed_nj, directed_g.node_end()) &&
                !mv::pathExists<InterferenceGraphNode, int>(directed_nj, directed_ni, directed_g.node_end()))
            {
                if (!checkNodeInterference(model, *source, *target) || !checkNodeInterference(model, *target, *source))
                {
                    g.edge_insert(ni, nj, 2*nodeId);
                    g.edge_insert(nj, ni, 2*nodeId+1);
                    nodeId++;
                }
            }
        }
    }

}

void printGraph(const InterferenceGraph& g, std::string name)
{
     // Nodes list
    std::cout << "Printing Graph: " << name << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Nodes list: " << std::endl;
    for (auto it = g.node_begin(); it != g.node_end(); ++it)
        std::cout << (*it).name << " " << std::endl;

    std::cout << std::endl;

     // Edges list
    std::cout << "Edges list: " << std::endl;
    for (auto it = g.edge_begin(); it != g.edge_end(); ++it)
        std::cout << " EDGE: " << *it << " Source " << (*it->source()).name <<  " sink " << (*it->sink()).name << std::endl;

    std::cout << std::endl;
    std::cout << "=========================================================" << std::endl;

}

bool edgeExists(InterferenceGraph::node_parent_iterator p, InterferenceGraph::node_child_iterator c, InterferenceGraph& g)
{
    for (InterferenceGraph::node_child_iterator itc(p); itc != g.node_end(); ++itc)
        if (*itc == *c)
            return true;
    return false;
}

void cleanupDMATensorNodes(InterferenceGraph& g)
{
    std::vector<std::string> nodesToRemove;
    for (InterferenceGraph::node_dfs_iterator it = g.node_begin(); it != g.node_end(); ++it)
    {
        if ((*it).name.rfind("DMATask", 0) == 0)
        {
            nodesToRemove.push_back((*it).name);
        }
    }

    //before deleting the edge, connect parents with childs
    int nodeIdx = g.edge_size();
    for (size_t i = 0; i< nodesToRemove.size(); i++)
    {
        auto ni = g.node_find(nodesToRemove[i]);
        for (InterferenceGraph::node_parent_iterator itp(ni); itp != g.node_end(); ++itp)
        {
            for (InterferenceGraph::node_child_iterator itc(ni); itc != g.node_end(); ++itc)
            {
                if (itp != itc && !edgeExists(itp, itc, g))
                {
                    g.edge_insert(itp, itc, nodeIdx++);
                }
            }
        }
        g.node_erase(ni);
    }
}

void  addWeightsToInterferenceGraph(mv::ComputationModel& model, InterferenceGraph g, std::size_t alignment)
{
    for (InterferenceGraph::node_dfs_iterator it = g.node_begin(); it != g.node_end(); ++it)
    {
        (*it).weight = model.getTensor((*it).name)->computeTotalSize(alignment);
        (*it).neighborsWeight = getNeighborsWeight(g, model, (*it).name, alignment);
    }
}

InterferenceGraph buildInterferenceGraph(mv::ComputationModel& model, std::size_t alignment, const TensorIteratorFilter& tensorFilter = nullptr,
    const OpIteratorFilter& taskFilter = nullptr, bool isCompleteTig = false)
{

    InterferenceGraph g;
    if (isCompleteTig)
    {
        buildCompleteGraph(g, getTensorNames(model, tensorFilter, taskFilter));
    }
    else
    {
        genIntereferenceGraph(g, model , tensorFilter, taskFilter);
    }
    cleanupDMATensorNodes(g);
    addWeightsToInterferenceGraph(model, g, alignment);
    return g;
}

//TODO add copy constructor to graph?
InterferenceGraph createGraphCopy(InterferenceGraph& g)
{
    InterferenceGraph copyG;

    for (auto it = g.node_begin(); it != g.node_end(); ++it)
        copyG.node_insert(*it);

    for (auto it = g.edge_begin(); it != g.edge_end(); ++it)
    {
        auto source = copyG.node_find(*it->source());
        auto sink = copyG.node_find(*it->sink());
        copyG.edge_insert(source, sink, *it);
    }

    return copyG;
}

bool isNodeSimplificable(InterferenceGraph::node_list_iterator ni, long long memorySize)
{
    //std::cout << " name " << (*ni).name << " weight " << (*ni).weight << " ne_Weight " << (*ni).neighborsWeight << std::endl;
    return ((*ni).weight <= (memorySize - (*ni).neighborsWeight));
}

std::string maxWeightedNeighbors(InterferenceGraph& g)
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

void removeNodeAndUpdateWeights(InterferenceGraph& g, InterferenceGraph::node_list_iterator ni)
{
    auto parentWeight = (*ni).weight;
    for (InterferenceGraph::node_child_iterator itc(ni); itc != g.node_end(); ++itc)
    {
        (*itc).neighborsWeight -= parentWeight;
    }

    g.node_erase(ni);
}

std::stack<std::string> aggressiveSimplify(InterferenceGraph& g, long long memorySize, mv::OrderingStrategy orderStrategy)
{
    //create a copy of g since we are going to delete nodes from it
    InterferenceGraph gCopy = createGraphCopy(g);
    std::stack<std::string> agNodeOrder;
    std::vector<InterferenceGraphNode> orderedNodes;
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
    InterferenceGraph ddr_bss_g = buildInterferenceGraph(model, alignment,
            [](const mv::Data::TensorIterator& t) -> bool
            {
                return (t->isPopulated() || t->hasAttr("slave"));
            },
            [](const mv::Data::OpListIterator& t) -> bool
            {
                return (t->getOpType() == "DMATask");
            },
            true);
    //printGraph(ddr_bss_g, "DDR_BSS");
    drawGraph(ddr_bss_g, "ddr_bss.dot", model);
    system("dot -Tpng ddr_bss.dot -o ddr_bss.png");
    auto agOrder = aggressiveSimplify(ddr_bss_g, memsize, mv::OrderingStrategy::IG_SMALLEST_NEIGHBORS_FIRST);
    printASOrder(agOrder, "DDR_BSS");
    InterferenceGraph ddr_heap_g = buildInterferenceGraph(model, alignment,
            [](const mv::Data::TensorIterator& t) -> bool
            {
                return (!t->isPopulated() && !t->hasAttr("slave"));
            },
            [](const mv::Data::OpListIterator& t) -> bool
            {
                return (t->getOpType() == "DMATask");
            },
            false);
    //printGraph(ddr_heap_g, "DDR_HEAP");
    //drawGraph(ddr_heap_g, "ddr_heap.dot", model);
    //system("dot -Tpng ddr_heap.dot -o ddr_heap.png");
    memsize = memDefs.find("VPU_DDR_Heap")->second.size;
    alignment = memDefs.find("VPU_DDR_Heap")->second.alignment;
    agOrder = aggressiveSimplify(ddr_heap_g, memsize, mv::OrderingStrategy::IG_SMALLEST_NEIGHBORS_FIRST);
    printASOrder(agOrder, "DDR_HEAP");

    InterferenceGraph nncmx_g = buildInterferenceGraph(model, alignment, nullptr, nullptr, false);
    memsize = memDefs.find("VPU_CMX_NN")->second.size + memDefs.find("VPU_CMX_UPA")->second.size;
    alignment = memDefs.find("VPU_CMX_NN")->second.alignment;
    agOrder = aggressiveSimplify(nncmx_g, memsize, mv::OrderingStrategy::IG_SMALLEST_NEIGHBORS_FIRST);
    //printGraph(nncmx_g, "NNCMX");
    //drawGraph(nncmx_g, "nncmx.dot", model);
    //system("dot -Tpng nncmx.dot -o nncmx.png");
    printASOrder(agOrder, "NNCMX");
    pass.log(mv::Logger::MessageType::Debug, "Graph Coloring Ended");
}
