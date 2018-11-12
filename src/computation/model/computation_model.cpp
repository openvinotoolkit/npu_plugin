#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/op/op.hpp"

mv::ComputationModel::ComputationModel(const std::string& name) :
name_(name),
opsGraph_(std::make_shared<computation_graph>(computation_graph())),
dataGraph_(opsGraph_->get_first()),
controlGraph_(opsGraph_->get_second()),
ops_(std::make_shared<std::unordered_map<std::string, Data::OpListIterator>>()),
dataFlows_(std::make_shared<std::unordered_map<std::string, Data::FlowListIterator>>()),
controlFlows_(std::make_shared<std::unordered_map<std::string, Control::FlowListIterator>>()),
tensors_(std::make_shared<std::map<std::string, std::shared_ptr<Tensor>>>()),
groups_(std::make_shared<std::map<std::string, std::shared_ptr<Group>>>()),
stages_(std::make_shared<std::map<std::size_t, std::shared_ptr<Stage>>>()),
memoryAllocators_(std::make_shared<std::map<std::string, std::shared_ptr<MemoryAllocator>>>()),
opsInstanceCounter_(std::make_shared<std::map<std::string, std::size_t>>()),
opsIndexCounter_(std::make_shared<std::map<std::string, std::size_t>>()),
dataOpEnd_(std::make_shared<Data::OpListIterator>(dataGraph_.node_end())),
dataFlowEnd_(std::make_shared<Data::FlowListIterator>(dataGraph_.edge_end())),
controlOpEnd_(std::make_shared<Control::OpListIterator>(controlGraph_.node_end())),
controlFlowEnd_(std::make_shared<Control::FlowListIterator>(controlGraph_.edge_end())),
input_(std::make_shared<Data::OpListIterator>(dataGraph_.node_end())),
output_(std::make_shared<Data::OpListIterator>(dataGraph_.node_end())),
selfRef_(*this)
{
    
}

mv::ComputationModel::ComputationModel(ComputationModel &other) :
name_(other.name_),
opsGraph_(other.opsGraph_),
dataGraph_(other.dataGraph_),
controlGraph_(other.controlGraph_),
ops_(other.ops_),
dataFlows_(other.dataFlows_),
controlFlows_(other.controlFlows_),
tensors_(other.tensors_),
groups_(other.groups_),
stages_(other.stages_),
memoryAllocators_(other.memoryAllocators_),
opsInstanceCounter_(other.opsInstanceCounter_),
opsIndexCounter_(other.opsIndexCounter_),
dataOpEnd_(other.dataOpEnd_),
dataFlowEnd_(other.dataFlowEnd_),
controlOpEnd_(other.controlOpEnd_),
controlFlowEnd_(other.controlFlowEnd_),
input_(other.input_),
output_(other.output_),
selfRef_(other.selfRef_)
{
    
}

mv::ComputationModel::~ComputationModel()
{
    
}

void mv::ComputationModel::incrementOpsIndexCounter_(const std::string& opType)
{
    if (opsIndexCounter_->find(opType) == opsIndexCounter_->end())
        opsIndexCounter_->emplace(opType, 1);
    else
        ++opsIndexCounter_->at(opType);
}

void mv::ComputationModel::incrementOpsInstanceCounter_(const std::string& opType)
{
    if (opsInstanceCounter_->find(opType) == opsInstanceCounter_->end())
        opsInstanceCounter_->emplace(opType, 0);
    else
        ++opsInstanceCounter_->at(opType);
}

void mv::ComputationModel::decrementOpsInstanceCounter_(const std::string& opType)
{
    if (opsInstanceCounter_->find(opType) != opsInstanceCounter_->end())
        if (opsInstanceCounter_->at(opType) > 0)
            --opsInstanceCounter_->at(opType);
}

bool mv::ComputationModel::isValid() const
{
    return !dataGraph_.disjoint() && *input_ != *dataOpEnd_ && *output_ != *dataOpEnd_;
}

bool mv::ComputationModel::isValid(Data::TensorIterator it) const
{

    if (it == tensorEnd())
        return false;
    if (tensors_->find(it->getName()) != tensors_->end())
        return true;
    return false;

}

bool mv::ComputationModel::isValid(Data::OpListIterator it) const
{
    
    if (it == *dataOpEnd_)
        return false;
    if (ops_->find(it->getName()) != ops_->end())
        return true;
    return false;

}

bool mv::ComputationModel::isValid(Control::OpListIterator it) const
{
    
    if (it == *controlOpEnd_)
        return false;
    if (ops_->find(it->getName()) != ops_->end())
        return true;
    return false;

}

bool mv::ComputationModel::isValid(Data::FlowListIterator it) const
{
    if (it == *dataFlowEnd_)
        return false;
    if (dataFlows_->find(it->getName()) != dataFlows_->end())
        return true;
    return false;
}

bool mv::ComputationModel::isValid(Control::FlowListIterator it) const 
{
    if (it == *controlFlowEnd_)
        return false;
    if (controlFlows_->find(it->getName()) != controlFlows_->end())
        return true;
    return false;
}

bool mv::ComputationModel::isValid(GroupIterator it) const
{
    if (it == GroupIterator(groups_->end()))
        return false;
    if (groups_->find(it->getName()) != groups_->end())
        return true;
    return false;
}

bool mv::ComputationModel::isValid(Control::StageIterator it) const
{
    if (it == Control::StageIterator(stages_->end()))
        return false;
    if (stages_->find(it->get<std::size_t>("idx")) != stages_->end())
        return true;
    return false;
}

mv::GroupIterator mv::ComputationModel::addGroup(const std::string &name)
{

    if (getGroup(name) == groupEnd())
    {
        
        auto result = groups_->emplace(name, std::make_shared<Group>(*this, name));
        if (result.second)
        {
            log(Logger::MessageType::Info, "Defined " + result.first->second->toString());
            return result.first;
        }
        return groupEnd();

    }

    return groupEnd();

}

bool mv::ComputationModel::hasGroup(const std::string &name)
{

    if (getGroup(name) != groupEnd())
        return true;

    return false;

}

mv::GroupIterator mv::ComputationModel::getGroup(const std::string &name)
{
    auto group = groups_->find(name);
    if (group != groups_->end())
        return group;
    return groupEnd();
}

void mv::ComputationModel::addGroupElement(GroupIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while including group as element of another group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while including group as element of another group");

    group->include(element);
}

void mv::ComputationModel::removeGroupElement(GroupIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid",
            "Invalid iterator passed while excluding group that is an element of another group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid",
            "Invalid iterator passed while excluding group that is an element of another group");
    group->exclude(element);
}

mv::GroupIterator mv::ComputationModel::groupBegin()
{
    return groups_->begin();
}

mv::GroupIterator mv::ComputationModel::groupEnd()
{
    //return GroupContext::GroupIterator();
    return groups_->end();
}

mv::Data::TensorIterator mv::ComputationModel::tensorBegin() const
{
    return tensors_->begin();
}

mv::Data::TensorIterator mv::ComputationModel::tensorEnd() const
{
    return tensors_->end();
}

mv::Data::TensorIterator mv::ComputationModel::getTensor(const std::string& name)
{

    auto it = tensors_->find(name);

    if (it == tensors_->end())
        throw ArgumentError(*this, "tensor name", name, "Attempt of finding an undefined tensor");

    return it;

}

mv::Data::OpListIterator mv::ComputationModel::getOp(const std::string& name)
{

    auto it = ops_->find(name);

    if (it == ops_->end())
        throw ArgumentError(*this, "tensor name", name, "Attempt of finding an undefined tensor");

    return it->second;

}

mv::Data::FlowListIterator mv::ComputationModel::getDataFlow(const std::string& name)
{

    auto it = dataFlows_->find(name);

    if (it == dataFlows_->end())
        throw ArgumentError(*this, "tensor name", name, "Attempt of finding an undefined tensor");

    return it->second;

}

mv::Control::FlowListIterator mv::ComputationModel::getControlFlow(const std::string& name)
{

    auto it = controlFlows_->find(name);

    if (it == controlFlows_->end())
        throw ArgumentError(*this, "tensor name", name, "Attempt of finding an undefined tensor");

    return it->second;

}

void mv::ComputationModel::clear()
{
    ops_->clear();
    dataFlows_->clear();
    controlFlows_->clear();
    tensors_->clear();
    groups_->clear();
    stages_->clear();
    memoryAllocators_->clear();
    opsInstanceCounter_->clear();
    opsIndexCounter_->clear();
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
    for (auto tensorIt = tensors_->begin(); tensorIt != tensors_->end(); ++tensorIt)
        tensors.append(Jsonable::toJsonValue(*(tensorIt->second)));

    for (auto tensorIt = tensors_->begin(); tensorIt != tensors_->end(); ++tensorIt)
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

std::reference_wrapper<mv::ComputationModel> mv::ComputationModel::getRef()
{
    return selfRef_;
}

std::string mv::ComputationModel::getLogID() const
{
    return "Model:" + name_;
}

std::string mv::ComputationModel::getName() const
{
    return name_;
}