#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/op/ops_headers.hpp"

mv::allocator mv::ComputationModel::allocator_;
mv::DefaultLogger mv::ComputationModel::defaultLogger_;
mv::Logger &mv::ComputationModel::logger_ = mv::ComputationModel::defaultLogger_;


mv::ComputationModel::ComputationModel(Logger::VerboseLevel verboseLevel, bool logTime) :
opsGraph_(allocator_.make_owner<computation_graph>(computation_graph())),
dataGraph_(opsGraph_->get_first()),
controlGraph_(opsGraph_->get_second()),
flowTensors_(allocator_.make_owner<map<string, allocator::owner_ptr<Tensor>>>(map<string, allocator::owner_ptr<Tensor>>())),
tensorsSources_(allocator_.make_owner<map<string, Data::OpListIterator>>(map<string, Data::OpListIterator>())),
groups_(allocator_.make_owner<map<string, allocator::owner_ptr<ComputationGroup>>>(map<string, allocator::owner_ptr<ComputationGroup>>())),
stages_(allocator_.make_owner<map<unsigned_type, allocator::owner_ptr<ComputationStage>>>(map<unsigned_type, allocator::owner_ptr<ComputationStage>>())),
memoryAllocators_(allocator_.make_owner<map<string, allocator::owner_ptr<MemoryAllocator>>>(map<string, allocator::owner_ptr<MemoryAllocator>>())),
opsCounter_(allocator_.make_owner<map<OpType, unsigned>>(map<OpType, unsigned>())),
dataOpEnd_(new Data::OpListIterator(dataGraph_.node_end())),
dataFlowEnd_(new Data::FlowListIterator(dataGraph_.edge_end())),
controlOpEnd_(new Control::OpListIterator(controlGraph_.node_end())),
controlFlowEnd_(new Control::FlowListIterator(controlGraph_.edge_end())),
input_(new Data::OpListIterator(*dataOpEnd_)),
output_(new Data::OpListIterator(*dataOpEnd_))
/*dataOpEnd_(std::make_shared<Data::OpListIterator>(dataGraph_.node_end())),
dataFlowEnd_(std::make_shared<Data::FlowListIterator>(dataGraph_.edge_end())),
controlOpEnd_(std::make_shared<Control::OpListIterator>(controlGraph_.node_end())),
controlFlowEnd_(std::make_shared<Control::FlowListIterator>(controlGraph_.edge_end())),
input_(std::make_shared<Data::OpListIterator>(dataOpEnd_)),
output_(std::make_shared<Data::OpListIterator>(dataOpEnd_)),
lastOp_(std::make_shared<Control::OpListIterator>()),
defaultControlFlow_(std::make_shared<bool>(defaultControlFlow))*/
{
    logger_.setVerboseLevel(verboseLevel);
    logger_.setLogTime(logTime);
}

void mv::ComputationModel::addOutputTensorsJson(Data::OpListIterator insertedOp)
{
    unsigned numOutputs = insertedOp->outputSlots();
    for(unsigned j = 0; j < numOutputs; j++)
    {
        string output_string("output"+std::to_string(j));
        auto output_tensor = findTensor_(insertedOp->getAttr(output_string).getContent<string>());
        insertedOp->setOutputTensor(output_tensor, j);
    }
}

void mv::ComputationModel::addInputTensorsJson(Data::OpListIterator insertedOp)
{
    unsigned numInputs = insertedOp->inputSlots();
    for(unsigned j = 0; j < numInputs; j++)
    {
        string input_string("input"+std::to_string(j));
        auto input_tensor = findTensor_(insertedOp->getAttr(input_string).getContent<string>());
        insertedOp->setInputTensor(input_tensor, j);
    }
}

mv::Data::OpListIterator mv::ComputationModel::addNodeFromJson(mv::json::Value& node)
{
    Attribute opTypeAttr = mv::Attribute::JsonAttributeFactory(node["attributes"]["opType"]);
    OpType opType = opTypeAttr.getContent<OpType>();
    mv::Data::OpListIterator toReturn;

    //Parenthesis are necessary because a scope has to be created
    switch (opType)
    {
        case OpType::Input:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::Input>(node));
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Output:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::Output>(node));
            addInputTensorsJson(toReturn);
            break;
        case OpType::Constant:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::Constant>(node));
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Conv2D:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::Conv2D>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::MatMul:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::MatMul>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::MaxPool2D:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::MaxPool2D>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::AvgPool2D:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::AvgPool2D>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Concat:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::Concat>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::ReLU:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::ReLU>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Softmax:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::Softmax>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Scale:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::Scale>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::BatchNorm:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::BatchNorm>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Add:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::Add>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Subtract:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::Subtract>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Multiply:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::Multiply>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Divide:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::Divide>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::FullyConnected:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::FullyConnected>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Reshape:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::Reshape>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Bias:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::Bias>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
    }
    return toReturn;

}

void mv::ComputationModel::addDataFlowFromJson(mv::json::Value& data_flow, std::map<string, mv::Data::OpListIterator>& addedOperations)
{
    mv::DataFlow d(data_flow);
    string source = d.getAttr("sourceOp").getContent<string>();
    string target = d.getAttr("sinkOp").getContent<string>();

    dataGraph_.edge_insert(addedOperations[source], addedOperations[target], d);
}

void mv::ComputationModel::addGroupFromJson(mv::json::Value& group)
{

}

void mv::ComputationModel::addControlFlowFromJson(mv::json::Value& control_flow, std::map<string, mv::Data::OpListIterator>& addedOperations)
{
    mv::ControlFlow d(control_flow);
    string source = d.getAttr("sourceOp").getContent<string>();
    string target = d.getAttr("sinkOp").getContent<string>();

    controlGraph_.edge_insert(opsGraph_->get_second_iterator(addedOperations[source]), opsGraph_->get_second_iterator(addedOperations[target]), d);
}

mv::ComputationModel::ComputationModel(mv::json::Value &model, Logger::VerboseLevel verboseLevel, bool logTime):
    ComputationModel(verboseLevel, logTime)
{
    //Utility structure to store Name -> Operation iterator mapping
    std::map<string, Data::OpListIterator> addedOperations;

    // All the structures have been initialized by the traditional constructor, time to fill them.
    // TENSORS
    mv::json::Value tensors = model["tensors"];
    for(unsigned i = 0; i < tensors.size(); ++i)
    {
        Tensor currentTensor(tensors[i]);
        flowTensors_->emplace(currentTensor.getName(), currentTensor);
    }

    // OPERATIONS/NODES
    mv::json::Value nodes = model["graph"]["nodes"];
    for(unsigned i = 0; i < nodes.size(); ++i)
    {
        auto addedOp = addNodeFromJson(nodes[i]);
        addedOperations[addedOp->getName()] = addedOp;
    }

    // TENSOR SOURCES
    mv::json::Value tensorSources = model["source_ops"];
    std::vector<string> tensorKeys(tensorSources.getKeys());
    for(unsigned i = 0; i < tensorKeys.size(); ++i)
    {
        tensorsSources_->emplace(tensorKeys[i], addedOperations[tensorKeys[i]]);
    }

    // DATA FLOWS
    mv::json::Value data_flows = model["graph"]["data_flows"];
    for(unsigned i = 0; i < data_flows.size(); ++i)
    {
        addDataFlowFromJson(data_flows[i], addedOperations);
    }

    // CONTROL FLOWS
    mv::json::Value control_flows = model["graph"]["control_flows"];
    for(unsigned i = 0; i < control_flows.size(); ++i)
    {
        addControlFlowFromJson(control_flows[i], addedOperations);
    }

    // GROUPS
    mv::json::Value groups = model["groups"];
    for(unsigned i = 0; i < groups.size(); ++i)
    {
        addGroupFromJson(groups[i]);
    }

    // OPS COUNTERS
    mv::json::Value counters = model["operations_counters"];
    std::vector<string> counterKeys(counters.getKeys());
    for(unsigned i = 0; i < counterKeys.size(); ++i)
    {
        mv::json::Value v(counterKeys[i]);
        opsCounter_->emplace(mv::Jsonable::constructOpTypeFromJson(v), mv::Jsonable::constructUnsignedTypeFromJson(counters[counterKeys[i]]));
    }
}


mv::ComputationModel::ComputationModel(const ComputationModel &other) :
opsGraph_(other.opsGraph_),
dataGraph_(other.dataGraph_),
controlGraph_(other.controlGraph_),
flowTensors_(other.flowTensors_),
tensorsSources_(other.tensorsSources_),
groups_(other.groups_),
stages_(other.stages_),
memoryAllocators_(other.memoryAllocators_),
opsCounter_(other.opsCounter_),
dataOpEnd_(other.dataOpEnd_),
dataFlowEnd_(other.dataFlowEnd_),
controlOpEnd_(other.controlOpEnd_),
controlFlowEnd_(other.controlFlowEnd_),
input_(other.input_),
output_(other.output_)
{

}

mv::ComputationModel::~ComputationModel()
{
    /*
    delete dataOpEnd_;
    delete dataFlowEnd_;
    delete controlOpEnd_;
    delete controlFlowEnd_;
    delete input_;
    delete output_;
    delete lastOp_;
    delete defaultControlFlow_;*/
}

bool mv::ComputationModel::isValid() const
{
    return !dataGraph_.disjoint() && *input_ != *dataOpEnd_ && *output_ != *dataOpEnd_;
}

bool mv::ComputationModel::isValid(const Data::TensorIterator &it) const
{

    if (!it)
        return false;

    if (flowTensors_->find(it->getName()) != flowTensors_->end())
        return true;
    return false;
}

bool mv::ComputationModel::isValid(const Data::OpListIterator &it) const
{
    
    if (!it)
        return false;

    if (dataGraph_.node_find(it) != dataGraph_.node_end())
        return true;
    return false;
}

bool mv::ComputationModel::isValid(const Control::OpListIterator &it) const
{
    
    if (!it)
        return false;

    if (controlGraph_.node_find(it) != controlGraph_.node_end())
        return true;
    return false;

}

bool mv::ComputationModel::isValid(const Data::FlowListIterator &it) const
{
    if (!it)
        return false;

    if (dataGraph_.edge_find(it) != dataGraph_.edge_end())
        return true;
    return false;
} 

bool mv::ComputationModel::isValid(const Control::FlowListIterator &it) const
{
    if (!it)
        return false;

    if (controlGraph_.edge_find(it) != controlGraph_.edge_end())
        return true;
    return false;
} 

mv::GroupContext::GroupIterator mv::ComputationModel::addGroup(const string &name)
{
    
    if (getGroup(name) == groupEnd())
    {
        
        auto result = groups_->emplace(name, allocator_.make_owner<ComputationGroup>(name));
        if (result.second)
        {
            logger_.log(Logger::MessageType::MessageInfo, "Defined " + result.first->second->toString());
            return result.first;
        }
        return groupEnd();
        
    }

    return groupEnd();

}

bool mv::ComputationModel::hasGroup(const string &name)
{

    if (getGroup(name) != groupEnd())
    {
        return true;
    }

    return false;

}

mv::GroupContext::GroupIterator mv::ComputationModel::getGroup(const string &name)
{
    auto group = groups_->find(name);
    if (group != groups_->end())
        return group;
    return groupEnd();
}


mv::GroupContext::MemberIterator mv::ComputationModel::addGroupElement_(allocator::owner_ptr<ComputationElement> element, GroupContext::GroupIterator &group)
{
    if (group != groupEnd())
    {
        auto result = group->insert(element);
        if (result != group->end())
        {
            logger_.log(Logger::MessageType::MessageInfo, "Appended new member '" + (*result)->getName() + "' to group '" + group->getName() + "'");
            return result;
        }
    }

    return group->end();
    
}

bool mv::ComputationModel::removeGroupElement_(allocator::access_ptr<ComputationElement> element, mv::GroupContext::GroupIterator &group)
{

    if (group != groupEnd())
    {

        GroupContext::MemberIterator it = group->find(*element);

        if (it != memberEnd(group))
        {
            return group->erase(it);
        }

    }

    return false;

}

bool mv::ComputationModel::checkOpsStages_() const
{

    if (*input_ == *dataOpEnd_)
        return false;
    
    for (auto opIt = *input_; opIt != *dataOpEnd_; ++opIt)
    {
        if (!opIt->hasAttr("stage") && opIt->getAttr("executable").getContent<bool>())
            return false;
    }

    return true;
    
}

mv::Control::StageIterator mv::ComputationModel::addStage_()
{   

    auto it = stages_->emplace(stages_->size(), allocator_.make_owner<ComputationStage>(stages_->size()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + it.first->second->toString());
    return it.first;
    
}

bool mv::ComputationModel::addToStage_(Control::StageIterator &stage, Data::OpListIterator &op)
{
    if (stage)
    {
        allocator::owner_ptr<ComputationOp> ptr = op;
        auto result = stage->insert(ptr);

        if (result != stage->end())
        {
            logger_.log(Logger::MessageType::MessageInfo, "Appended new member '" + (*result)->getName() + "' to stage " + (*result)->getAttr("stage").getContentStr());
            return true;
        }
    }

    return false;
}

mv::Data::TensorIterator mv::ComputationModel::defineOutputTensor_(Data::OpListIterator source, byte_type outputIdx)
{
    if (!source)
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define an output tensor - invalid source op");
        return Data::TensorIterator();
    }

    auto tensorDef = source->getOutputDef(outputIdx);

    if (flowTensors_->find(tensorDef.getName()) == flowTensors_->end())
    {
        // TODO: handle failure
        auto result = flowTensors_->emplace(tensorDef.getName(), allocator_.make_owner<Tensor>(tensorDef));
        tensorsSources_->emplace(tensorDef.getName(), source);
        logger_.log(Logger::MessageType::MessageInfo, "Defined " + result.first->second->toString());
        return result.first;
    }
    else
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define an output tensor - tensor already defined");
        return Data::TensorIterator();
    }
}

mv::Data::TensorIterator mv::ComputationModel::findTensor_(const string &name)
{
    auto it = flowTensors_->find(name);

    if (it != flowTensors_->end())
        return it;

    return Data::TensorIterator();

}

mv::Data::OpListIterator mv::ComputationModel::findSourceOp_(Data::TensorIterator &tensor)
{

    if (tensor == tensorEnd())
        return Data::OpListIterator();

    auto it = tensorsSources_->find(tensor->getName());

    if (it != tensorsSources_->end())
        return it->second;

    return Data::OpListIterator();

}

mv::GroupContext::MemberIterator mv::ComputationModel::addGroupElement(GroupContext::GroupIterator &element, GroupContext::GroupIterator &group)
{
    allocator::owner_ptr<ComputationGroup> ptr = element;
    return addGroupElement_(ptr, group);
}

bool mv::ComputationModel::removeGroupElement(GroupContext::GroupIterator &element, GroupContext::GroupIterator &group)
{
    allocator::owner_ptr<ComputationGroup> ptr = element;
    return removeGroupElement_(ptr, group);
}

mv::GroupContext::GroupIterator mv::ComputationModel::groupBegin()
{
    return groups_->begin();
}

mv::GroupContext::GroupIterator mv::ComputationModel::groupEnd()
{
    //return GroupContext::GroupIterator();
    return groups_->end();
}

mv::GroupContext::MemberIterator mv::ComputationModel::memberBegin(GroupContext::GroupIterator &group)
{

    if (group != groupEnd())
    {
        return group->begin();
    }
    
    //return memberEnd(group);
    return GroupContext::MemberIterator();

}

mv::GroupContext::MemberIterator mv::ComputationModel::memberEnd(GroupContext::GroupIterator &group)
{
    
    if (group != groupEnd())
    {
        return group->end();
    }

    return GroupContext::MemberIterator();

}

mv::Data::TensorIterator mv::ComputationModel::tensorBegin() const
{
    return flowTensors_->begin();
}

mv::Data::TensorIterator mv::ComputationModel::tensorEnd() const
{
    //return Data::TensorIterator();
    return flowTensors_->end();
}

void mv::ComputationModel::clear()
{
    flowTensors_->clear();
    tensorsSources_->clear();
    groups_->clear();
    stages_->clear();
    memoryAllocators_->clear();
    opsCounter_->clear();
    *dataOpEnd_ = dataGraph_.node_end();
    *dataFlowEnd_ = dataGraph_.edge_end();
    *controlOpEnd_ = controlGraph_.node_end();
    *controlFlowEnd_ = controlGraph_.edge_end();
    *input_ = *dataOpEnd_;
    *output_ = *dataOpEnd_;
    dataGraph_.clear();
    controlGraph_.clear();
}

mv::Logger &mv::ComputationModel::logger()
{
    return logger_;
}

void mv::ComputationModel::setLogger(Logger &logger)
{
    logger_ = logger;
}
mv::json::Value mv::ComputationModel::toJsonValue() const
{
    mv::json::Object computationModel;
    mv::json::Object graph;
    mv::json::Array nodes;
    mv::json::Array data_flows;
    mv::json::Array control_flows;
    mv::json::Array tensors;
    mv::json::Array groups;
    mv::json::Array stages;
    mv::json::Object memory_allocators;
    mv::json::Object sourceOps;
    mv::json::Object opsCounters;

    //Tensors and source operations
    for (auto tensorIt = flowTensors_->begin(); tensorIt != flowTensors_->end(); ++tensorIt)
        tensors.append(mv::Jsonable::toJsonValue(*(tensorIt->second)));

    //Nodes and operation counters
    for (auto opIt = dataGraph_.node_begin(); opIt != dataGraph_.node_end(); ++opIt)
        nodes.append(mv::Jsonable::toJsonValue(**opIt));

    //Data flows
    for (auto dataIt = dataGraph_.edge_begin(); dataIt != dataGraph_.edge_end(); ++dataIt)
        data_flows.append(mv::Jsonable::toJsonValue(**dataIt));

    //Control flows
    for (auto controlIt = controlGraph_.edge_begin(); controlIt != controlGraph_.edge_end(); ++controlIt)
        control_flows.append(mv::Jsonable::toJsonValue(**controlIt));

    //Groups
    for (auto groupIt = groups_->begin(); groupIt != groups_->end(); ++groupIt)
        groups.append(mv::Jsonable::toJsonValue(*groupIt->second));

    //Deploying stages (Waiting for Stanislaw proper implementation)
    //for (auto stagesIt = stages_->begin(); stagesIt != stages_->end(); ++stagesIt)
        //stages.append(mv::Jsonable::toJsonValue(*stagesIt->second));

    //Operations counters
    for (auto opsCounterIt = opsCounter_->begin(); opsCounterIt != opsCounter_->end(); ++opsCounterIt)
        opsCounters[opsStrings.at(opsCounterIt->first)] = mv::Jsonable::toJsonValue(opsCounterIt->second);

    //Source ops counters: NOTE: why there are some null pointers?
    for (auto sourceOpsIt = tensorsSources_->begin(); sourceOpsIt != tensorsSources_->end(); ++sourceOpsIt)
        sourceOps[sourceOpsIt->first] = mv::Jsonable::toJsonValue(sourceOpsIt->second);

    //Memory Allocators
    for (auto memoryAllocatorIt = memoryAllocators_->begin(); memoryAllocatorIt != memoryAllocators_->end(); ++memoryAllocatorIt)
        memory_allocators[memoryAllocatorIt->first] = mv::Jsonable::toJsonValue(*memoryAllocatorIt->second);

    graph["nodes"] = mv::json::Value(nodes);
    graph["data_flows"] = mv::json::Value(data_flows);
    graph["control_flows"] = mv::json::Value(control_flows);
    computationModel["graph"] = graph;
    computationModel["tensors"] = tensors;
    computationModel["groups"] = groups;
    computationModel["stages"] = stages;
    computationModel["source_ops"] = sourceOps;
    computationModel["memory_allocators"] = memory_allocators;
    computationModel["operations_counters"] = opsCounters;
    return mv::json::Value(computationModel);
}
