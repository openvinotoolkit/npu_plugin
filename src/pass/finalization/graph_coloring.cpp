#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/graph/graph.hpp"


static void graphColoringFnc(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(graphColoring)
        .setFunc(graphColoringFnc)
        .setDescription(
            "graph coloring implmentation used in memory allocation algorithm"
        );
    }
}

using InterferenceGraph = mv::graph<std::string, int>;
using TensorIteratorFilter = std::function<bool(const mv::Data::TensorIterator& t)>;
using OpIteratorFilter = std::function<bool(mv::Data::OpListIterator& t)>;

bool pathExists(const InterferenceGraph::node_list_iterator& source, const InterferenceGraph::node_list_iterator& target,
    const InterferenceGraph::node_list_iterator& end)
{
    for (InterferenceGraph::node_dfs_iterator it(source); it != end; ++it)
    {
        //std::cout << " it " << *it <<  " tn " << *tn << std::endl;
        if (*it == *target)
            return true;
    }
    return false;
}

bool pathExists(const mv::Data::OpListIterator& source, const mv::Data::OpListIterator& sink, const mv::Data::OpListIterator end)
{
    for (mv::Data::OpDFSIterator it(source); it != end; ++it)
    {
        //std::cout << " it " << *it <<  " tn " << *tn << std::endl;
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

            //TODO Parameters?? aren't they already in inputs
        }
    }
    return tensorNames;
}

void buildCompleteGraph(InterferenceGraph& g, std::set<std::string> tensorNames)
{
    for (std::set<std::string>::const_iterator name = tensorNames.begin( ); name != tensorNames.end( ); ++name)
    {
        g.node_insert(*name);
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
        (opIterator->get<mv::DmaDirection>("direction") == mv::DmaDirectionEnum::CMX2DDR /*TODO add OR CMX2UPA*/))
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
        //pathExists(mv::Data::OpDFSIterator& source,  mv::Data::OpDFSIterator& sink, mv::Data::OpDFSIterator end)
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

            std::cout << "Collecting output Tensors " << std::endl;
            outputTensorNames = getTaskTopTensors(opIterator->getOutputTensor(), model, tensorFilter);

            for (std::set<std::string>::const_iterator name = inputTensorNames.begin( ); name != inputTensorNames.end( ); ++name)
            {
                std::cout << "Adding Input Tensor Node:  " << *name <<  std::endl;
                auto res = nodeNames.insert(*name);
                if (res.second)
                {//if we dont have it already
                    directed_g.node_insert(*name);
                    g.node_insert(*name);
                }
            }
            //Add them as nodes
            for (std::set<std::string>::const_iterator name = outputTensorNames.begin( ); name != outputTensorNames.end( ); ++name)
            {
                std::cout << "Adding Output Tensor Node:  " << *name <<  std::endl;
                auto res = nodeNames.insert(*name);
                if (res.second)
                {//if we dont have it already
                    directed_g.node_insert(*name);
                    g.node_insert(*name);
                }
            }
            //TODO again adding parameters?

            //Add the obvious edges
            for (std::set<std::string>::const_iterator src = inputTensorNames.begin( ); src != inputTensorNames.end( ); ++src)
            {
                auto ni = g.node_find(*src);
                auto directed_ni = directed_g.node_find(*src);
                for (std::set<std::string>::const_iterator target = outputTensorNames.begin( ); target != outputTensorNames.end( ); ++target)
                {
                    if (*src != *target)
                    {
                        std::cout << " Adding edge :  " << *src << " To " << *target <<  std::endl;
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
            if (source != target && !checkNodesAreNeighbors(g, ni, nj) && !pathExists( directed_ni, directed_nj, directed_g.node_end()) &&
                !pathExists(directed_nj, directed_ni, directed_g.node_end()))
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

InterferenceGraph buildInterferenceGraph(mv::ComputationModel& model, const TensorIteratorFilter& tensorFilter = nullptr,
    const OpIteratorFilter& taskFilter = nullptr, bool isCompleteTig = false)
{
    InterferenceGraph g;
    if (isCompleteTig)
    {
        buildCompleteGraph(g, getTensorNames(model, tensorFilter, taskFilter));
    }
    else
    {
        genIntereferenceGraph( g, model , tensorFilter, taskFilter);
    }
    return g;
}

void printGraph(const InterferenceGraph& g, std::string name)
{
     // Nodes list
    std::cout << "Printing Graph: " << name << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Nodes list: " << std::endl;
    for (auto it = g.node_begin(); it != g.node_end(); ++it)
    {
        std::cout << *it << " " << std::endl;
    }
    std::cout << std::endl;

     // Edges list
    std::cout << "Edges list: " << std::endl;
    for (auto it = g.edge_begin(); it != g.edge_end(); ++it)
    {
        std::cout << " EDGE: " << *it << " Source " << *(it->source()) <<  " sink " << *(it->sink()) << std::endl;
    }
    std::cout << std::endl;
    std::cout << "=========================================================" << std::endl;

}

void graphColoringFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    pass.log(mv::Logger::MessageType::Debug, "Graph Coloring Started");

    mv::OpModel om(model);
    mv::DataModel dm(model);


    InterferenceGraph ddr_bss_g = buildInterferenceGraph(model,
            [](const mv::Data::TensorIterator& t) -> bool
            {
                return (t->isPopulated());
            },
            [](const mv::Data::OpListIterator& t) -> bool
            {
                return (t->getOpType() == "DMATask");
            },
            true);
    printGraph(ddr_bss_g, "DDR_BSS");

    InterferenceGraph ddr_heap_g = buildInterferenceGraph(model,
            [](const mv::Data::TensorIterator& t) -> bool
            {
                return (!t->isPopulated());
            },
            [](const mv::Data::OpListIterator& t) -> bool
            {
                return (t->getOpType() == "DMATask");
            },
            false);
    printGraph(ddr_heap_g, "DDR_HEAP");

    pass.log(mv::Logger::MessageType::Debug, "Graph Coloring Ended");
}
