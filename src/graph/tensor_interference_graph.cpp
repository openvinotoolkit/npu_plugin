#include "include/mcm/graph/tensor_interference_graph.hpp"
#include "include/mcm/algorithms/path_exists.hpp"
#include "include/mcm/algorithms/edge_exists.hpp"
#include <fstream>
#include <iostream>
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/computation/model/control_model.hpp"


mv::TensorInterferenceGraph::TensorInterferenceGraph(const mv::TensorInterferenceGraph& g) : graph<TensorInterferenceGraphNode, int>()
{

    for (auto it = g.node_begin(); it != g.node_end(); ++it)
    {
        auto newIt = this->node_insert(*it);
        nodeIteratorsMap_.insert(std::make_pair((*it).name, newIt));
    }

    for (auto it = g.edge_begin(); it != g.edge_end(); ++it)
    {
        auto source = nodeIteratorsMap_.find((*it->source()).name)->second;
        auto sink = nodeIteratorsMap_.find((*it->sink()).name)->second;

        this->edge_insert(source, sink, *it);
    }
}
std::string mv::TensorInterferenceGraph::getTensorTopMaster_(const mv::Data::TensorIterator& t,  mv::DataModel& dm)
{
    auto foundIt = topMasterMap_.find(t->getName());

    if (foundIt != topMasterMap_.end())
        return foundIt->second;

    auto tensorAllocatorName = t->get<std::set<std::string>>("allocators").begin();
    auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
    mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, t); // 0 is the only stage for now, but this will probably change in the future
    auto masterTensor = tensorAllocator.getTopMasterBuffer(tensorBufferIt);
    std::pair<std::string,std::string> newEntry (t->getName(),(*masterTensor)->getData()->getName());
    auto res = topMasterMap_.insert(newEntry);
    if (!res.second)
        throw mv::ArgumentError("getTensorTopMaster_", "adding already existing entry!", t->getName(), "");

    return newEntry.second;
}

std::unordered_set<std::string> mv::TensorInterferenceGraph::getTaskTopTensors_(const std::vector<mv::Data::TensorIterator>& tensorList,
    mv::ComputationModel& model, mv::DataModel& dm, const mv::TensorIteratorFilter& tensorFilter, bool isDMA)
{
    std::unordered_set<std::string> topTensors;

    for (unsigned i = 0; i < tensorList.size(); i++)
    {

        bool isCMXTensor = checkIsCMXTensor_(tensorList[i]);
        if ((isDMA && isCMXTensor) || (!isDMA && !isCMXTensor))
        {
            std::string name = getTensorTopMaster_(tensorList[i], dm);
            if (!tensorFilter || tensorFilter(model.getTensor(name)))
            {
                topTensors.insert(name);
            }
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
    return false;
}

bool mv::TensorInterferenceGraph::isTensorInTopNames_(const std::vector<mv::Data::TensorIterator>& tensorList, mv::DataModel& model,
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

bool mv::TensorInterferenceGraph::checkNodesDontInterfere_(std::unordered_set<std::string>& sourceNodeNames, std::unordered_set<std::string>& sinkNodeNames)
{
    //Check if there's a path from any node in sink to any node in source, if yes return true
    for (std::unordered_set<std::string>::const_iterator src = sinkNodeNames.begin( ); src != sinkNodeNames.end( ); ++src)
    {
        for (std::unordered_set<std::string>::const_iterator target = sourceNodeNames.begin( ); target != sourceNodeNames.end( ); ++target)
        {
            auto pathExists = cmTransitiveClosureSet_.find(std::make_pair(*src, *target));
            if (pathExists == cmTransitiveClosureSet_.end())
                return false;
        }

    }
    return true;
}

std::set<std::string> mv::TensorInterferenceGraph::getTensorNames_(mv::ComputationModel& model, const mv::TensorIteratorFilter& tensorFilter,
    const mv::OpIteratorFilter& taskFilter, bool isDMA)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    std::set<std::string> tensorNames;

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if (!taskFilter || taskFilter(opIterator))
        {
            std::unordered_set<std::string> temp = getTaskTopTensors_(opIterator->getInputTensor(), model, dm, tensorFilter, isDMA);
            tensorNames.insert(temp.begin(), temp.end());

            temp = getTaskTopTensors_(opIterator->getOutputTensor(), model, dm, tensorFilter, isDMA);
            tensorNames.insert(temp.begin(), temp.end());
        }
    }
    return tensorNames;
}

std::size_t  mv::TensorInterferenceGraph::getNeighborsWeight_(std::string& inode)
{
    size_t totalWeights = 0;
    //Since we use a directed graph to represent undirected IG, every parent node is also a child node
    // so no need to go through both
    auto ni = nodeIteratorsMap_.find(inode)->second;

    //add parents
    for (mv::TensorInterferenceGraph::node_parent_iterator itr(ni); itr != this->node_end(); ++itr)
    {
        totalWeights += (*itr).weight;
    }
    return totalWeights;
}

void  mv::TensorInterferenceGraph::addWeightsToInterferenceGraph_(const mv::pass::PassEntry& pass, mv::ComputationModel& model, std::size_t alignment)
{
    pass.log(mv::Logger::MessageType::Info, " \tcalc weights for nodes");

    for (mv::TensorInterferenceGraph::node_list_iterator it = this->node_begin(); it != this->node_end(); ++it)
    {
        auto tensor = model.getTensor((*it).name);
        auto tensorMemoryLocation = tensor->get<mv::Tensor::MemoryLocation>("Location");
        // DDR does not need getClusterSize
        if(tensorMemoryLocation == mv::Tensor::MemoryLocation::NNCMX)
            (*it).weight = tensor->getClusterSize(alignment);
        else
            (*it).weight = tensor->computeTotalSize();
    }

    pass.log(mv::Logger::MessageType::Info, " \tcalc neighbors weights");
    for (mv::TensorInterferenceGraph::node_list_iterator it = this->node_begin(); it != this->node_end(); ++it)
        (*it).neighborsWeight = getNeighborsWeight_((*it).name) + (*it).weight;

}

mv::TensorInterferenceGraph::TensorInterferenceGraph(const mv::pass::PassEntry& pass, mv::ComputationModel& model, std::size_t alignment, const mv::TensorIteratorFilter& tensorFilter,
    const mv::OpIteratorFilter& taskFilter, const SinkOpIteratorFilter& sinkFilter, bool isCompleteTig, bool isDMA)
{

    if (isCompleteTig)
    {
        buildCompleteGraph_(getTensorNames_(model, tensorFilter, taskFilter, isDMA));
    }
    else
    {
        pass.log(mv::Logger::MessageType::Info, " calling genIntereferenceGraph_");

        genIntereferenceGraph_(pass, model , tensorFilter, taskFilter, sinkFilter, isDMA);
    }
    pass.log(mv::Logger::MessageType::Info, " calling addWeightsToInterferenceGraph_");
    addWeightsToInterferenceGraph_(pass, model, alignment);
}


void mv::TensorInterferenceGraph::buildCompleteGraph_(std::set<std::string> tensorNames)
{
    for (std::set<std::string>::const_iterator name = tensorNames.begin( ); name != tensorNames.end( ); ++name)
    {
        auto newIt = this->node_insert(mv::TensorInterferenceGraphNode(*name));
        nodeIteratorsMap_.insert(std::make_pair(*name, newIt));

    }

    int nodeId = 0;
    for (std::set<std::string>::const_iterator src = tensorNames.begin( ); src != tensorNames.end( ); ++src)
    {
        auto ni = nodeIteratorsMap_.find(*src)->second;

        //since we are directed graph need to create a->b and b->a, so we go through all combinations
        for (std::set<std::string>::const_iterator target = tensorNames.begin( ); target != tensorNames.end( ); ++target)
        {
            if (src != target)
                this->edge_insert(ni, nodeIteratorsMap_.find(*target)->second, nodeId++);
        }
    }
}

bool mv::TensorInterferenceGraph::checkIsCMXTensor_(const Data::TensorIterator tensorIt)
{
    if (tensorIt->hasAttr("allocators"))
    {
        auto allocators = tensorIt->get<std::set<std::string>>("allocators");
        if (allocators.count("VPU_CMX_NN") != 0)
            return true;
    }
    else
    {
        throw mv::ArgumentError("checkIsCMXTensor_", "no allocators for tensor", tensorIt->getName(), "no allocators for tensor");
        return false;
    }

    return false;
}

void mv::TensorInterferenceGraph::cmTransitiveClosureHelper_(mv::OpModel& om, mv::ControlModel& cm, std::string source, std::string target)
{
    //Todo maybe add only if source/target are in nodenames?
    cmTransitiveClosureSet_.insert(std::make_pair(source, target));

    //for all childs of target, add them to reachable nodes from source
    for (mv::Control::OpChildIterator it(cm.switchContext(om.getOp(target))); it != cm.opEnd(); ++it)
    {
        //only if we haven't explored this path before lets explore it
        if (cmTransitiveClosureSet_.find(std::make_pair(source, it->getName())) == cmTransitiveClosureSet_.end())
            cmTransitiveClosureHelper_(om, cm, source, it->getName());
    }
}

void mv::TensorInterferenceGraph::cmTransitiveClosure_(mv::ComputationModel& model)
{
    mv::OpModel om(model);
    mv::ControlModel cm(om);

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
        cmTransitiveClosureHelper_(om, cm, opIterator->getName(), opIterator->getName()); //node is always reachable from itself
}

void mv::TensorInterferenceGraph::genIntereferenceGraph_(const mv::pass::PassEntry& pass, mv::ComputationModel& model , const mv::TensorIteratorFilter& tensorFilter,
    const mv::OpIteratorFilter& taskFilter, const mv::OpIteratorFilter& sinkFilter, bool isDMA)
{
    std::unordered_set<std::string> inputTensorNames;

    std::unordered_set<std::string> outputTensorNames;
    std::unordered_set<std::string> nodeNames;
    int nodeId = 0;

    mv::OpModel om(model);
    mv::DataModel dm(model);

    std::unordered_set<std::pair<std::string, std::string>, pair_hash> addedEdges;
    //Collect all input/output tensor names
    pass.log(mv::Logger::MessageType::Info, "\t collecting nodes and  obvious edges");

    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        if (!taskFilter || taskFilter(opIterator))
        {
            inputTensorNames = getTaskTopTensors_(opIterator->getInputTensor(), model, dm, tensorFilter, isDMA);

            for (std::unordered_set<std::string>::const_iterator name = inputTensorNames.begin( ); name != inputTensorNames.end( ); ++name)
            {
                auto res = nodeNames.insert(*name);
                if (res.second)
                {//if we dont have it already
                    auto newIt = this->node_insert(*name);
                    nodeIteratorsMap_.insert(std::make_pair(*name , newIt));
                }
            }

            outputTensorNames = getTaskTopTensors_(opIterator->getOutputTensor(), model, dm, tensorFilter, isDMA);

            for (std::unordered_set<std::string>::const_iterator name = outputTensorNames.begin( ); name != outputTensorNames.end( ); ++name)
            {
                auto res = nodeNames.insert(*name);
                if (res.second)
                {//if we dont have it already
                    auto newIt = this->node_insert(*name);
                    nodeIteratorsMap_.insert(std::make_pair(*name , newIt));
                }
            }

            if (inputTensorNames.size() == 0 || outputTensorNames.size() == 0)
                continue;
            //Add the obvious edges
            for (std::unordered_set<std::string>::const_iterator src = inputTensorNames.begin( ); src != inputTensorNames.end( ); ++src)
            {
                auto ni = nodeIteratorsMap_.find(*src)->second;

                for (std::unordered_set<std::string>::const_iterator target = outputTensorNames.begin( ); target != outputTensorNames.end( ); ++target)
                {
                    if (*src != *target)
                    {
                        auto nj = nodeIteratorsMap_.find(*target)->second;

                        auto inserted = addedEdges.insert(std::make_pair(*src, *target));
                        if (inserted.second)
                        {
                            this->edge_insert(ni, nj, 2*nodeId);
                            this->edge_insert(nj, ni, 2*nodeId+1); //since we are directed graph need to create a->b and b->a
                            nodeId++;
                        }
                    }
                }
            }

        }
    }

    pass.log(mv::Logger::MessageType::Info, "\t creating source/sink maps");

    //for each 2 nodes, if they are not yet connected (neighbors) in the undirected graph
    // and dont have a path from one to the other in the directed graph, then check if they
    // exist in memory at the same time
    std::unordered_map<std::string, std::unordered_set<std::string>> sourceNodesMap;
    std::unordered_map<std::string, std::unordered_set<std::string>> sinkNodesMap;
    for (std::unordered_set<std::string>::const_iterator source = nodeNames.begin( ); source != nodeNames.end( ); ++source)
    {
        std::unordered_set<std::string> sourceNodeNames;
        std::unordered_set<std::string> sinkNodeNames;
        for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
        {
            if (!opIterator->hasTypeTrait("executable"))
                continue;
            if (isTensorInTopNames_(opIterator->getOutputTensor(), dm, *source))
                sourceNodeNames.insert(opIterator->getName());
            if (isTensorInTopNames_(opIterator->getInputTensor(), dm, *source) && sinkFilter(opIterator))
                sinkNodeNames.insert(opIterator->getName());
        }
        sourceNodesMap.insert(std::pair<std::string,std::unordered_set<std::string>>(*source, sourceNodeNames));
        sinkNodesMap.insert(std::pair<std::string,std::unordered_set<std::string>>(*source, sinkNodeNames));
    }
    //create transitive closure for all nodes in graph
    cmTransitiveClosure_(model);

    pass.log(mv::Logger::MessageType::Info, "\t adding more edges (current Number of edges) " + std::to_string(nodeId));

    for (std::unordered_set<std::string>::const_iterator source = nodeNames.begin( ); source != nodeNames.end( ); ++source)
    {
        auto ni = nodeIteratorsMap_.find(*source)->second;

        for (std::unordered_set<std::string>::const_iterator target = source; target != nodeNames.end( ); ++target)
        {
            auto nj = nodeIteratorsMap_.find(*target)->second;

            if (source != target && !checkNodesAreNeighbors_(ni, nj))
            {
                if (!checkNodesDontInterfere_(sourceNodesMap[*source], sinkNodesMap[*target]) && !checkNodesDontInterfere_(sourceNodesMap[*target], sinkNodesMap[*source]))
                {
                    this->edge_insert(ni, nj, 2*nodeId);
                    this->edge_insert(nj, ni, 2*nodeId+1);
                    nodeId++;
                }
            }
        }
    }
    pass.log(mv::Logger::MessageType::Info, "\t Done - adding more edges (current Number of edges now) " + std::to_string(nodeId));

}

void mv::TensorInterferenceGraph::drawGraph(std::string outputFileName)
{
    std::ofstream ostream;

    ostream.open(outputFileName, std::ios::trunc | std::ios::out);
    ostream << "graph G {\n\tgraph [splines=spline]\n";

    for (auto it = this->node_begin(); it != this->node_end(); ++it)
    {
        auto name = (*it).name;
        std::string nodeDef = "\t\"" + name + "\" [shape=box,";
        nodeDef += " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\"><TR><TD ALIGN=\"CENTER\" COLSPAN=\"2\"><FONT POINT-SIZE=\"14.0\"><B>" + name + "</B></FONT></TD></TR>";
        nodeDef += "<TR><TD ALIGN=\"CENTER\"><FONT POINT-SIZE=\"11.0\"> size: " + std::to_string((*it).weight) + "</FONT></TD></TR>";
        nodeDef += "<TR><TD ALIGN=\"CENTER\"><FONT POINT-SIZE=\"11.0\"> address: " + std::to_string((*it).address) + "</FONT></TD></TR>";
        nodeDef += "</TABLE>>";
        ostream << nodeDef << "];\n";
    }
    typedef std::pair<std::string, std::string> DotEdge;
    std::set<DotEdge> existingEdges;
    for (auto it = this->edge_begin(); it != this->edge_end(); ++it)
    {
        DotEdge e1;
        e1.first = (*it->sink()).name;
        e1.second = (*it->source()).name;
        auto ret = existingEdges.insert(e1);
        e1.second = (*it->sink()).name;
        e1.first = (*it->source()).name;
        auto ret2 = existingEdges.insert(e1);
        if (ret.second == true && ret2.second == true)
        {
            std::string edgeDef = "\t\"" + (*it->source()).name + "\" -- \"" +  (*it->sink()).name + "\"";
            ostream << edgeDef << "\n";
        }
    }
    ostream << "}\n";
    ostream.close();
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
