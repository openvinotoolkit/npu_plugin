#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/op/ops_headers.hpp"

mv::ComputationModel::ComputationModel(const std::string& name) :
name_(name),
opsGraph_(std::make_shared<computation_graph>(computation_graph())),
dataGraph_(opsGraph_->get_first()),
controlGraph_(opsGraph_->get_second()),
flowTensors_(std::make_shared<std::map<std::string, std::shared_ptr<Tensor>>>(std::map<std::string, std::shared_ptr<Tensor>>())),
tensorsSources_(std::make_shared<std::map<std::string, Data::OpListIterator>>(std::map<std::string, Data::OpListIterator>())),
groups_(std::make_shared<std::map<std::string, std::shared_ptr<ComputationGroup>>>(std::map<std::string, std::shared_ptr<ComputationGroup>>())),
stages_(std::make_shared<std::map<std::size_t, std::shared_ptr<ComputationStage>>>(std::map<std::size_t, std::shared_ptr<ComputationStage>>())),
memoryAllocators_(std::make_shared<std::map<std::string, std::shared_ptr<MemoryAllocator>>>(std::map<std::string, std::shared_ptr<MemoryAllocator>>())),
opsCounter_(std::make_shared<std::map<OpType, std::size_t>>(std::map<OpType, std::size_t>())),
dataOpEnd_(std::make_shared<Data::OpListIterator>(dataGraph_.node_end())),
dataFlowEnd_(std::make_shared<Data::FlowListIterator>(dataGraph_.edge_end())),
controlOpEnd_(std::make_shared<Control::OpListIterator>(controlGraph_.node_end())),
controlFlowEnd_(std::make_shared<Control::FlowListIterator>(controlGraph_.edge_end())),
input_(std::make_shared<Data::OpListIterator>(dataGraph_.node_end())),
output_(std::make_shared<Data::OpListIterator>(dataGraph_.node_end())),
binary_("test",0)
{
    log(Logger::MessageType::MessageInfo, "Initialized");
}

/*void mv::ComputationModel::addOutputTensorsJson(Data::OpListIterator insertedOp)
{
    std::size_t numOutputs = insertedOp->outputSlots();
    for(std::size_t j = 0; j < numOutputs; j++)
    {
        std::string output_string("output"+std::to_string(j));
        auto output_tensor = findTensor_(insertedOp->getAttr(output_string).getContent<std::string>());
        insertedOp->setOutputTensor(output_tensor, j);
    }
}

void mv::ComputationModel::addInputTensorsJson(Data::OpListIterator insertedOp)
{
    std::size_t numInputs = insertedOp->inputSlots();
    for(std::size_t j = 0; j < numInputs; j++)
    {
        std::string input_string("input"+std::to_string(j));
        auto input_tensor = findTensor_(insertedOp->getAttr(input_string).getContent<std::string>());
        insertedOp->setInputTensor(input_tensor, j);
    }
}

mv::Data::OpListIterator mv::ComputationModel::addNodeFromJson(json::Value& node)
{
    Attribute opTypeAttr = Attribute::JsonAttributeFactory(node["attributes"]["opType"]);
    OpType opType = opTypeAttr.getContent<OpType>();
    Data::OpListIterator toReturn;

    //Parenthesis are necessary because a scope has to be created
    switch ((unsigned short)opType)
    {
        case OpType::Input:
            toReturn = dataGraph_.node_insert(std::make_shared<op::Input>(node));
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Output:
            toReturn = dataGraph_.node_insert(std::make_shared<op::Output>(node));
            addInputTensorsJson(toReturn);
            break;
        case OpType::Constant:
            toReturn = dataGraph_.node_insert(std::make_shared<op::Constant>(node));
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Conv2D:
            toReturn = dataGraph_.node_insert(std::make_shared<op::Conv2D>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::MatMul:
            toReturn = dataGraph_.node_insert(std::make_shared<op::MatMul>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::MaxPool2D:
            toReturn = dataGraph_.node_insert(std::make_shared<op::MaxPool2D>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::AvgPool2D:
            toReturn = dataGraph_.node_insert(std::make_shared<op::AvgPool2D>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Concat:
            toReturn = dataGraph_.node_insert(std::make_shared<op::Concat>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::ReLU:
            toReturn = dataGraph_.node_insert(std::make_shared<op::ReLU>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::PReLU:
            toReturn = dataGraph_.node_insert(allocator_.make_owner<mv::op::PReLU>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Softmax:
            toReturn = dataGraph_.node_insert(std::make_shared<op::Softmax>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Scale:
            toReturn = dataGraph_.node_insert(std::make_shared<op::Scale>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::BatchNorm:
            toReturn = dataGraph_.node_insert(std::make_shared<op::BatchNorm>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Add:
            toReturn = dataGraph_.node_insert(std::make_shared<op::Add>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Subtract:
            toReturn = dataGraph_.node_insert(std::make_shared<op::Subtract>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Multiply:
            toReturn = dataGraph_.node_insert(std::make_shared<op::Multiply>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Divide:
            toReturn = dataGraph_.node_insert(std::make_shared<op::Divide>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::FullyConnected:
            toReturn = dataGraph_.node_insert(std::make_shared<op::FullyConnected>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Reshape:
            toReturn = dataGraph_.node_insert(std::make_shared<op::Reshape>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Bias:
            toReturn = dataGraph_.node_insert(std::make_shared<op::Bias>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
        case OpType::Conversion:
            toReturn = dataGraph_.node_insert(std::make_shared<op::Conversion>(node));
            addInputTensorsJson(toReturn);
            addOutputTensorsJson(toReturn);
            break;
    }
    return toReturn;

}

mv::Data::FlowListIterator mv::ComputationModel::addDataFlowFromJson(json::Value& data_flow, std::map<std::string, Data::OpListIterator>& addedOperations)
{

    auto d = std::make_shared<DataFlow>(data_flow, findTensor_(constructStringFromJson(data_flow["tensor"])));
    std::string source = d->getAttr("sourceOp").getContent<std::string>();
    std::string target = d->getAttr("sinkOp").getContent<std::string>();

    return dataGraph_.edge_insert(addedOperations[source], addedOperations[target], d);

}

mv::Control::FlowListIterator mv::ComputationModel::addControlFlowFromJson(json::Value& control_flow, std::map<std::string, Data::OpListIterator>& addedOperations)
{
    auto d = std::make_shared<ControlFlow>(control_flow);
    std::string source = d->getAttr("sourceOp").getContent<std::string>();
    std::string target = d->getAttr("sinkOp").getContent<std::string>();

    return controlGraph_.edge_insert(opsGraph_->get_second_iterator(addedOperations[source]), opsGraph_->get_second_iterator(addedOperations[target]), d);
}

mv::ComputationModel::ComputationModel(json::Value &model) :
    ComputationModel()
{

    //COMPUTATIONAL GROUPS
    json::Value groups = model["computation_groups"];
    for(std::size_t i = 0; i < groups.size(); ++i)
    {
        auto currentGroup = std::make_shared<ComputationGroup>(groups[i]);
        groups_->emplace(currentGroup->getName(), currentGroup);
    }

    for(auto currentGroupIt = groups_->begin(); currentGroupIt != groups_->end(); ++currentGroupIt)
        handleGroupsForAddedElement<ComputationGroup, mv::GroupContext::GroupIterator>(currentGroupIt);

    // TENSORSconst std::string& name
    json::Valuconst std::string& name["tensors"];
    for(std::sconst std::string& nametensors.size(); ++i)
    {
        auto currentTensor = std::make_shared<Tensor>(tensors[i]);
        auto addedTensor = flowTensors_->emplace(currentTensor->getName(), currentTensor);
        handleGroupsForAddedElement<Tensor, Data::TensorIterator>(addedTensor.first);
    }

    // OPERATIONS/NODES and OPS COUNTERS

    // Utility structure to store Name -> Operation iterator mapping
    std::map<std::string, Data::OpListIterator> addedOperations;

    json::Value nodes = model["graph"]["nodes"];
    for(std::size_t i = 0; i < nodes.size(); ++i)
    {
        auto addedOp = addNodeFromJson(nodes[i]);

        if (addedOp->getOpType() == OpType::Input)
        {
            delete input_;
            input_ = new Data::OpListIterator(addedOp);
        }

        if (addedOp->getOpType() == OpType::Output)
        {
            delete output_;
            output_ = new Data::OpListIterator(addedOp);
        }

        addedOperations[addedOp->getName()] = addedOp;

        if(opsCounter_->find(addedOp->getOpType()) == opsCounter_->end())
            opsCounter_->emplace(addedOp->getOpType(), 1);
        else
            ++(opsCounter_->at(addedOp->getOpType()));

        handleGroupsForAddedElement<ComputationOp, Data::OpListIterator>(addedOp);
    }

    // TENSOR SOURCES
    json::Value tensorSources = model["source_ops"];
    std::vector<std::string> tensorKeys(tensorSources.getKeys());
    for(std::size_t i = 0; i < tensorKeys.size(); ++i)
    {
        std::string sourceOpName(mv::Jsonable::constructStringFromJson(tensorSources[tensorKeys[i]]));
        Data::OpListIterator sourceOp = addedOperations[sourceOpName];
        tensorsSources_->emplace(tensorKeys[i], sourceOp);
    }

    // DATA FLOWS
    json::Value data_flows = model["graph"]["data_flows"];
    for(std::size_t i = 0; i < data_flows.size(); ++i)
    {
        auto addedDataFlow = addDataFlowFromJson(data_flows[i], addedOperations);
        handleGroupsForAddedElement<DataFlow, Data::FlowListIterator>(addedDataFlow);
    }

    // CONTROL FLOWS
    json::Value control_flows = model["graph"]["control_flows"];
    for(std::size_t i = 0; i < control_flows.size(); ++i)
    {
        auto addedControlFlow = addControlFlowFromJson(control_flows[i], addedOperations);
        handleGroupsForAddedElement<ControlFlow, Control::FlowListIterator>(addedControlFlow);
    }

    // MEMORY ALLOCATORS

    // STAGES
}*/


mv::ComputationModel::ComputationModel(ComputationModel &other) :
name_(other.name_),
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
output_(other.output_),
binary_("test2",0)
{
    log(Logger::MessageType::MessageInfo, "Bound");
}

mv::ComputationModel::~ComputationModel()
{
    log(Logger::MessageType::MessageInfo, "Deleted");
}

bool mv::ComputationModel::isValid() const
{
    return !dataGraph_.disjoint() && *input_ != *dataOpEnd_ && *output_ != *dataOpEnd_;
}

bool mv::ComputationModel::isValid(const Data::TensorIterator &it) const
{

    if (it == tensorEnd())
        return false;
    if (flowTensors_->find(it->getName()) != flowTensors_->end())
        return true;
    return false;

}

bool mv::ComputationModel::isValid(const Data::OpListIterator &it) const
{
    
    if (it == *dataOpEnd_)
        return false;
    if (dataGraph_.node_find(it) != dataGraph_.node_end())
        return true;
    return false;

}

bool mv::ComputationModel::isValid(const Control::OpListIterator &it) const
{
    
    if (it == *controlOpEnd_)
        return false;
    if (controlGraph_.node_find(it) != controlGraph_.node_end())
        return true;
    return false;

}

bool mv::ComputationModel::isValid(const Data::FlowListIterator &it) const
{
    if (it == *dataFlowEnd_)
        return false;
    if (dataGraph_.edge_find(it) != dataGraph_.edge_end())
        return true;
    return false;
}

bool mv::ComputationModel::isValid(const Control::FlowListIterator &it) const
{
    if (it == *controlFlowEnd_)
        return false;
    if (controlGraph_.edge_find(it) != controlGraph_.edge_end())
        return true;
    return false;
}

mv::GroupContext::GroupIterator mv::ComputationModel::addGroup(const std::string &name)
{

    if (getGroup(name) == groupEnd())
    {
        
        auto result = groups_->emplace(name, std::make_shared<ComputationGroup>(name));
        if (result.second)
        {
            log(Logger::MessageType::MessageInfo, "Defined " + result.first->second->toString());
            return result.first;
        }
        return groupEnd();

    }

    return groupEnd();

}

bool mv::ComputationModel::hasGroup(const std::string &name)
{

    if (getGroup(name) != groupEnd())
    {
        return true;
    }

    return false;

}

mv::GroupContext::GroupIterator mv::ComputationModel::getGroup(const std::string &name)
{
    auto group = groups_->find(name);
    if (group != groups_->end())
        return group;
    return groupEnd();
}


mv::GroupContext::MemberIterator mv::ComputationModel::addGroupElement_(std::shared_ptr<Element> element, GroupContext::GroupIterator &group)
{
    if (group != groupEnd())
    {
        auto result = group->insert(element);
        if (result != group->end())
        {
            log(Logger::MessageType::MessageInfo, "Appended new member '" + result->lock()->getName() + "' to group '" + 
                group->getName() + "'");
            return result;
        }
    }

    return group->end();

}

bool mv::ComputationModel::removeGroupElement_(std::weak_ptr<Element> element, mv::GroupContext::GroupIterator &group)
{

    if (group != groupEnd())
    {

        GroupContext::MemberIterator it = group->find(*element.lock());

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
        if (!opIt->hasAttr("stage") && opIt->get<bool>("executable"))
            return false;
    }

    return true;

}

mv::Control::StageIterator mv::ComputationModel::addStage_()
{

    auto it = stages_->emplace(stages_->size(), std::make_shared<ComputationStage>(stages_->size()));
    log(Logger::MessageType::MessageInfo, "Defined " + it.first->second->toString());
    return it.first;

}

bool mv::ComputationModel::addToStage_(Control::StageIterator &stage, Data::OpListIterator &op)
{
    if (stage)
    {
        std::shared_ptr<ComputationOp> ptr = op;
        auto result = stage->insert(ptr);

        if (result != stage->end())
        {
            log(Logger::MessageType::MessageInfo, "Appended new member '" + result->lock()->getName() + "' to stage " + 
                std::to_string(result->lock()->get<std::size_t>("stage")));
            return true;
        }
    }

    return false;

}

mv::Data::TensorIterator mv::ComputationModel::defineOutputTensor_(Data::OpListIterator source, short unsigned outputIdx)
{
    if (!isValid(source))
        throw ArgumentError(*this, "Source op", "null", "Undefined source op");

    auto tensorDef = source->getOutputDef(outputIdx);

    if (flowTensors_->find(tensorDef.getName()) == flowTensors_->end())
    {
        auto result = flowTensors_->emplace(tensorDef.getName(), std::make_shared<Tensor>(tensorDef));
        tensorsSources_->emplace(tensorDef.getName(), source);
        log(Logger::MessageType::MessageInfo, "Defined " + result.first->second->toString());
        return result.first;
    }

    throw(ArgumentError(*this, "Tensor name", tensorDef.getName(), "Duplicated tensor identifier"));
    
}

mv::Data::TensorIterator mv::ComputationModel::findTensor_(const std::string &name)
{
    auto it = flowTensors_->find(name);

    if (it == flowTensors_->end())
        throw ArgumentError(*this, "tensor name", name, "Attempt of finding an undefined tensor");

    return it;

}

mv::Data::OpListIterator mv::ComputationModel::findSourceOp_(Data::TensorIterator &tensor)
{

    if (tensor == tensorEnd())
        throw ArgumentError(*this, "tensor iterator", "end", "Attempt of finding a source op of an invalid tensor");

    auto it = tensorsSources_->find(tensor->getName());

    if (it == tensorsSources_->end())
        throw ArgumentError(*this, "tensor", "invalid", "Attempt of finding a source op of a tensor that does not belong to the model");

    return it->second;

}

mv::GroupContext::MemberIterator mv::ComputationModel::addGroupElement(GroupContext::GroupIterator &element, GroupContext::GroupIterator &group)
{
    std::shared_ptr<ComputationGroup> ptr = element;
    return addGroupElement_(ptr, group);
}

bool mv::ComputationModel::removeGroupElement(GroupContext::GroupIterator &element, GroupContext::GroupIterator &group)
{
    std::shared_ptr<ComputationGroup> ptr = element;
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
    return flowTensors_->end();
}

int mv::ComputationModel::getBinarySize()
{
    int retval = binary_.getSize() ;
    return retval ;
}

bool mv::ComputationModel::getBinaryBuffer(std::string newName, int newSize)
{
    bool retval = binary_.getBuffer(newName, newSize) ;
    return retval ;
}


void mv::ComputationModel::clear()
{
    flowTensors_->clear();
    tensorsSources_->clear();
    groups_->clear();
    stages_->clear();
    memoryAllocators_->clear();
    opsCounter_->clear();
    dataGraph_.clear();
    controlGraph_.clear();
    *dataOpEnd_ = dataGraph_.node_end();
    *dataFlowEnd_ = dataGraph_.edge_end();
    *controlOpEnd_ = controlGraph_.node_end();
    *controlFlowEnd_ = controlGraph_.edge_end();
    *input_ = dataGraph_.node_end();
    *output_ = dataGraph_.node_end();
}

//NOTE: Populated tensors dumping are handled in json pass.
/*mv::json::Value mv::ComputationModel::toJsonValue() const
{
    json::Object computationModel;
    json::Object graph;
    json::Array nodes;
    json::Array data_flows;
    json::Array control_flows;
    json::Array tensors;
    json::Array groups;
    json::Array stages;
    json::Object memory_allocators;
    json::Object sourceOps;
    json::Object opsCounters;

    bool hasPopulatedTensors = false;

    //Groups
    for (auto groupIt = groups_->begin(); groupIt != groups_->end(); ++groupIt)
        groups.append(Jsonable::toJsonValue(*groupIt->second));

    //Tensors and source operations
    for (auto tensorIt = flowTensors_->begin(); tensorIt != flowTensors_->end(); ++tensorIt)
        tensors.append(Jsonable::toJsonValue(*(tensorIt->second)));

    for (auto tensorIt = flowTensors_->begin(); tensorIt != flowTensors_->end(); ++tensorIt)
    {
        if(tensorIt->second->isPopulated())
        {
            hasPopulatedTensors = true;
            break;
        }
    }

    //Nodes and operation counters
    for (auto opIt = dataGraph_.node_begin(); opIt != dataGraph_.node_end(); ++opIt)
        nodes.append(Jsonable::toJsonValue(**opIt));

    //Data flows
    for (auto dataIt = dataGraph_.edge_begin(); dataIt != dataGraph_.edge_end(); ++dataIt)
        data_flows.append(Jsonable::toJsonValue(**dataIt));

    //Control flows
    for (auto controlIt = controlGraph_.edge_begin(); controlIt != controlGraph_.edge_end(); ++controlIt)
        control_flows.append(Jsonable::toJsonValue(**controlIt));

    //Deploying stages (Waiting for Stanislaw proper implementation)
    //for (auto stagesIt = stages_->begin(); stagesIt != stages_->end(); ++stagesIt)
        //stages.append(mv::Jsonable::toJsonValue(*stagesIt->second));

    //Operations counters
    for (auto opsCounterIt = opsCounter_->begin(); opsCounterIt != opsCounter_->end(); ++opsCounterIt)
        opsCounters[opsStrings.at(opsCounterIt->first)] = Jsonable::toJsonValue(opsCounterIt->second);

    //Source ops counters.
    for (auto sourceOpsIt = tensorsSources_->begin(); sourceOpsIt != tensorsSources_->end(); ++sourceOpsIt)
        sourceOps[sourceOpsIt->first] = Jsonable::toJsonValue(sourceOpsIt->second->getName());

    //Memory Allocators
    for (auto memoryAllocatorIt = memoryAllocators_->begin(); memoryAllocatorIt != memoryAllocators_->end(); ++memoryAllocatorIt)
        memory_allocators[memoryAllocatorIt->first] = Jsonable::toJsonValue(*memoryAllocatorIt->second);

    graph["nodes"] = json::Value(nodes);
    graph["data_flows"] = json::Value(data_flows);
    graph["control_flows"] = json::Value(control_flows);
    computationModel["graph"] = graph;
    computationModel["tensors"] = tensors;
    computationModel["computation_groups"] = groups;
    computationModel["stages"] = stages;
    computationModel["source_ops"] = sourceOps;
    computationModel["memory_allocators"] = memory_allocators;
    computationModel["operations_counters"] = opsCounters;
    computationModel["has_populated_tensors"] = Jsonable::toJsonValue(hasPopulatedTensors);
    return json::Value(computationModel);

}*/

std::string mv::ComputationModel::getLogID() const
{
    return "Model " + name_;
}
