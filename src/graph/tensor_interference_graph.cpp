#include "include/mcm/graph/tensor_interference_graph.hpp"
#include "include/mcm/algorithms/path_exists.hpp"
#include "include/mcm/algorithms/edge_exists.hpp"
#include <fstream>
#include <iostream>
#include "include/mcm/base/exception/argument_error.hpp"

mv::TensorInterferenceGraph::TensorInterferenceGraph(const mv::TensorInterferenceGraph& g) : graph<TensorInterferenceGraphNode, int>()
{

    for (auto it = g.node_begin(); it != g.node_end(); ++it)
        this->node_insert(*it);

    for (auto it = g.edge_begin(); it != g.edge_end(); ++it)
    {
        auto source = this->node_find(*it->source());
        auto sink = this->node_find(*it->sink());
        this->edge_insert(source, sink, *it);
    }
}

std::string mv::TensorInterferenceGraph::getTensorTopMaster_(const mv::Data::TensorIterator& t, mv::ComputationModel& model)
{
    if (t->hasAttr("master"))
    {
        auto master = model.getTensor(t->get<std::string>("master"));
        return getTensorTopMaster_(master, model);
    }
    return t->getName();
}

std::set<std::string> mv::TensorInterferenceGraph::getTaskTopTensors_(const std::vector<mv::Data::TensorIterator>& tensorList, mv::ComputationModel& model, const mv::TensorIteratorFilter& tensorFilter)
{
    std::set<std::string> topTensors;

    for (unsigned i = 0; i < tensorList.size(); i++)
    {
        std::string name = getTensorTopMaster_(tensorList[i], model);
        if (!tensorFilter || tensorFilter(model.getTensor(name)))
        {
            topTensors.insert(name);
        }
    }
    return topTensors;
}

bool mv::TensorInterferenceGraph::checkNodesAreNeighbors_(mv::TensorInterferenceGraph::node_list_iterator& n1, mv::TensorInterferenceGraph::node_list_iterator& n2)
{
    //check children
    for (mv::TensorInterferenceGraph::node_child_iterator it(n1); it != this->node_end(); ++it)
    {
        if (it == n2)
            return true;
    }
    //check parents
    for (mv::TensorInterferenceGraph::node_parent_iterator it(n1); it != this->node_end(); ++it)
    {
        if (it == n2)
            return true;
    }
    return false;
}

bool mv::TensorInterferenceGraph::isTensorInTopNames_(const std::vector<mv::Data::TensorIterator>& tensorList, mv::ComputationModel& model,
    const std::string tensorName)
{
    for (unsigned i = 0; i < tensorList.size(); i++)
    {
        std::string name = getTensorTopMaster_(tensorList[i], model);
        if (name == tensorName)
        {
            return true;
        }
    }
    return false;
}

bool mv::TensorInterferenceGraph::isSinkNode_(mv::Data::OpListIterator& opIterator)
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

bool mv::TensorInterferenceGraph::checkNodeInterference_(mv::ComputationModel& model, const std::string& tensor1, const std::string& tensor2)
{
    //returns true if tensor2 is already deallocated when tesnor1 is allocated
    mv::OpModel om(model);
    std::set<std::string> sourceNodeNames;
    std::set<std::string> sinkNodeNames;
    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if (isTensorInTopNames_(opIterator->getOutputTensor(), model, tensor1))
            sourceNodeNames.insert(opIterator->getName());
        if (isTensorInTopNames_(opIterator->getInputTensor(), model, tensor2) && isSinkNode_(opIterator))
            sinkNodeNames.insert(opIterator->getName());

        //Check if there's a path from any node in sink to any node in source, if yes return true
        for (std::set<std::string>::const_iterator src = sinkNodeNames.begin( ); src != sinkNodeNames.end( ); ++src)
        {
            for (std::set<std::string>::const_iterator target = sourceNodeNames.begin( ); target != sourceNodeNames.end( ); ++target)
            {
                if (om.pathExists(om.getOp(*src), om.getOp(*target)))
                    return true;
            }
        }

    }
    return false;
}

std::set<std::string> mv::TensorInterferenceGraph::getTensorNames_(mv::ComputationModel& model, const mv::TensorIteratorFilter& tensorFilter, const mv::OpIteratorFilter& taskFilter)
{
    mv::OpModel om(model);
    std::set<std::string> tensorNames;

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if (!taskFilter || taskFilter(opIterator))
        {
            std::set<std::string> temp = getTaskTopTensors_(opIterator->getInputTensor(), model, tensorFilter);
            tensorNames.insert(temp.begin(), temp.end());
            temp = getTaskTopTensors_(opIterator->getOutputTensor(), model, tensorFilter);
            tensorNames.insert(temp.begin(), temp.end());
        }
    }
    return tensorNames;
}


void mv::TensorInterferenceGraph::cleanupDMATensorNodes_()
{
    std::vector<std::string> nodesToRemove;
    for (mv::TensorInterferenceGraph::node_dfs_iterator it = this->node_begin(); it != this->node_end(); ++it)
    {
        if ((*it).name.rfind("DMA", 0) == 0)
        {
            nodesToRemove.push_back((*it).name);
        }
    }

    //before deleting the edge, connect parents with childs
    int nodeIdx = this->edge_size();
    for (size_t i = 0; i< nodesToRemove.size(); i++)
    {
        auto ni = this->node_find(nodesToRemove[i]);
        for (mv::TensorInterferenceGraph::node_parent_iterator itp(ni); itp != this->node_end(); ++itp)
        {
            for (mv::TensorInterferenceGraph::node_child_iterator itc(ni); itc != this->node_end(); ++itc)
            {
                if (itp != itc && !mv::edgeExists(*this, itp, itc))
                {
                    this->edge_insert(itp, itc, nodeIdx++);
                }
            }
        }
        this->node_erase(ni);
    }
}

std::size_t  mv::TensorInterferenceGraph::getNeighborsWeight_(mv::ComputationModel& model, std::string& inode, std::size_t alignment)
{
    size_t totalWeights = 0;
    //Since we use a directed graph to represent undirected IG, every parent node is also a child node
    // so no need to go through both
    auto ni = this->node_find(inode);

    //add parents
    for (mv::TensorInterferenceGraph::node_parent_iterator itr(ni); itr != this->node_end(); ++itr)
    {
        totalWeights += model.getTensor((*itr).name)->computeTotalSize(alignment);
    }
    return totalWeights;
}

void  mv::TensorInterferenceGraph::addWeightsToInterferenceGraph_(mv::ComputationModel& model, std::size_t alignment)
{
    for (mv::TensorInterferenceGraph::node_dfs_iterator it = this->node_begin(); it != this->node_end(); ++it)
    {
        (*it).weight = model.getTensor((*it).name)->computeTotalSize(alignment);
        (*it).neighborsWeight = getNeighborsWeight_(model, (*it).name, alignment);
    }
}

mv::TensorInterferenceGraph::TensorInterferenceGraph(mv::ComputationModel& model, std::size_t alignment, const mv::TensorIteratorFilter& tensorFilter,
    const mv::OpIteratorFilter& taskFilter, bool isCompleteTig)
{

    if (isCompleteTig)
    {
        buildCompleteGraph_(getTensorNames_(model, tensorFilter, taskFilter));
    }
    else
    {
        genIntereferenceGraph_(model , tensorFilter, taskFilter);
    }
    cleanupDMATensorNodes_();
    addWeightsToInterferenceGraph_(model, alignment);
}


void mv::TensorInterferenceGraph::buildCompleteGraph_(std::set<std::string> tensorNames)
{
    for (std::set<std::string>::const_iterator name = tensorNames.begin( ); name != tensorNames.end( ); ++name)
    {
        this->node_insert(mv::TensorInterferenceGraphNode(*name));
    }

    int nodeId = 0;
    for (std::set<std::string>::const_iterator src = tensorNames.begin( ); src != tensorNames.end( ); ++src)
    {
        auto ni = this->node_find(*src);
        //since we are directed graph need to create a->b and b->a, so we go through all combinations
        for (std::set<std::string>::const_iterator target = tensorNames.begin( ); target != tensorNames.end( ); ++target)
        {
            if (src != target)
            this->  edge_insert(ni, this->node_find(*target), nodeId++);
        }
    }
}

void mv::TensorInterferenceGraph::genIntereferenceGraph_(mv::ComputationModel& model , const mv::TensorIteratorFilter& tensorFilter,
    const mv::OpIteratorFilter& taskFilter)// : graph<InterferenceGraphNode, int>()
{
    mv::TensorInterferenceGraph directed_g;
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
            inputTensorNames = getTaskTopTensors_(opIterator->getInputTensor(), model, tensorFilter);

            for (std::set<std::string>::const_iterator name = inputTensorNames.begin( ); name != inputTensorNames.end( ); ++name)
            {
                auto res = nodeNames.insert(*name);
                if (res.second)
                {//if we dont have it already
                    directed_g.node_insert(*name);
                    this->node_insert(*name);
                }
            }

            outputTensorNames = getTaskTopTensors_(opIterator->getOutputTensor(), model, tensorFilter);

            for (std::set<std::string>::const_iterator name = outputTensorNames.begin( ); name != outputTensorNames.end( ); ++name)
            {
                auto res = nodeNames.insert(*name);
                if (res.second)
                {//if we dont have it already
                    directed_g.node_insert(*name);
                    this->node_insert(*name);
                }
            }

            //Add the obvious edges
            for (std::set<std::string>::const_iterator src = inputTensorNames.begin( ); src != inputTensorNames.end( ); ++src)
            {
                auto ni = this->node_find(*src);
                auto directed_ni = directed_g.node_find(*src);
                for (std::set<std::string>::const_iterator target = outputTensorNames.begin( ); target != outputTensorNames.end( ); ++target)
                {
                    if (*src != *target)
                    {
                        auto nj = this->node_find(*target);
                        auto directed_nj = directed_g.node_find(*target);
                        this->edge_insert(ni, nj, 2*nodeId);
                        this->edge_insert(nj, ni, 2*nodeId+1); //since we are directed graph need to create a->b and b->a
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
        auto ni = this->node_find(*source);
        auto directed_ni = directed_g.node_find(*source);

        for (std::set<std::string>::const_iterator target = source; target != nodeNames.end( ); ++target)
        {
            auto nj = this->node_find(*target);
            auto directed_nj = directed_g.node_find(*target);
            if (source != target && !checkNodesAreNeighbors_(ni, nj) && !mv::pathExists(directed_g, directed_ni, directed_nj) &&
                !mv::pathExists(directed_g, directed_nj, directed_ni))
            {
                if (!checkNodeInterference_(model, *source, *target) || !checkNodeInterference_(model, *target, *source))
                {
                    this->edge_insert(ni, nj, 2*nodeId);
                    this->edge_insert(nj, ni, 2*nodeId+1);
                    nodeId++;
                }
            }
        }
    }

}

void mv::TensorInterferenceGraph::drawGraph(std::string outputFileName)
{
    std::ofstream ostream;

    ostream.open(outputFileName + ".dot", std::ios::trunc | std::ios::out);
    ostream << "digraph G {\n\tgraph [splines=spline]\n";

    for (auto it = this->node_begin(); it != this->node_end(); ++it)
    {
        auto name = (*it).name;
        std::string nodeDef = "\t\"" + name + "\" [shape=box,";
        nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + name + "</B></FONT></TD></TR>";
        nodeDef += "<TR><TD ALIGN=\"CENTER\"><FONT POINT-SIZE=\"11.0\"> size: " + std::to_string((*it).weight) + "</FONT></TD></TR>";
        nodeDef += "</TABLE>>";
        ostream << nodeDef << "];\n";
    }

    for (auto it = this->edge_begin(); it != this->edge_end(); ++it)
    {
        std::string edgeDef = "\t\"" + (*it->source()).name + "\" -> \"" +  (*it->sink()).name + "\"";
        ostream << edgeDef << "\n";
    }
    ostream << "}\n";
    ostream.close();
    std::string dotToPngCmd = "dot -Tpng " + outputFileName +".dot" + " -o " + outputFileName + ".png";
    system(dotToPngCmd.c_str());
}

void mv::TensorInterferenceGraph::printGraph(std::string name)
{
     // Nodes list
    std::cout << "Printing Graph: " << name << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Nodes list: " << std::endl;
    for (auto it = this->node_begin(); it != this->node_end(); ++it)
        std::cout << (*it).name << " " << std::endl;

    std::cout << std::endl;

     // Edges list
    std::cout << "Edges list: " << std::endl;
    for (auto it = this->edge_begin(); it != this->edge_end(); ++it)
        std::cout << " EDGE: " << *it << " Source " << (*it->source()).name <<  " sink " << (*it->sink()).name << std::endl;

    std::cout << std::endl;
    std::cout << "=========================================================" << std::endl;

}