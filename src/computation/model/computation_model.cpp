#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/op/op.hpp"

mv::ComputationModel::ComputationModel(const std::string& name) :
name_(name),
opsGraph_(std::make_shared<conjoined_graph<Op, DataFlow, ControlFlow>>()),
binary_(std::make_shared<mv::RuntimeBinary>()),
dataGraph_(opsGraph_->get_first()),
controlGraph_(opsGraph_->get_second()),
globalConfigParams_(std::make_shared<mv::Element>("GlobalConfigParams")),
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
binary_(other.binary_),
dataGraph_(other.dataGraph_),
controlGraph_(other.controlGraph_),
globalConfigParams_(other.globalConfigParams_),
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
        opsInstanceCounter_->emplace(opType, 1);
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
//    bool isDataGraphDisjoint = dataGraph_.disjoint();
//    bool isDataInputValid = *input_ != *dataOpEnd_;
//    bool isDataOutputValid = *output_ != *dataOpEnd_;
//    if(isDataGraphDisjoint)
//    {
//        //If the dataGraph is disjoint it means that there is an Op
//        //not connected to anything. Since Op can also represent Tasks unrelated
//        //to tensors.
//        //ASSUMPTION: If the dataGraph is disjoint the controlGraph must not be disjoint.
//        bool isControlGraphDisjoint = controlGraph_.disjoint();
//        if(isControlGraphDisjoint)
//            return false;
//    }
//    //We get to this point in two circumstances
//    //1) The dataGraph is Disjoint but the controlGraph is not
//    //2) The dataGraph is not disjoint.
//    //At this point we just have to check input and output validity
//    return isDataInputValid && isDataOutputValid;

    return true;
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

bool mv::ComputationModel::checkOp(const std::string& name)
{
    auto it = ops_->find(name);

    if (it == ops_->end())
        return false;
    return true;
}

mv::Data::OpListIterator mv::ComputationModel::getOp(const std::string& name)
{
    auto it = ops_->find(name);

    if (it == ops_->end())
        throw ArgumentError(*this, "tensor name", name, "Attempt of finding an undefined tensor");

    return it->second;
}

// NOTE: Complexity is linear in the number of operations in the graph. Can we do better without an additional
// data strucuture?
std::vector<mv::Data::OpListIterator> mv::ComputationModel::getOps(const std::string &opType)
{
    std::vector<mv::Data::OpListIterator> toReturn;
    for(auto opPairIt = ops_->begin(); opPairIt != ops_->end(); ++opPairIt)
        if(opPairIt->second->getOpType() == opType)
            toReturn.push_back(opPairIt->second);
    return toReturn;
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

std::shared_ptr<mv::RuntimeBinary>  mv::ComputationModel::allocateBinaryBuffer(std::string newName, std::size_t newSize)
{
    if (binary_->getBuffer(newName, newSize))
    {   
        return binary_ ;
    }
    return nullptr ;
}

std::shared_ptr<mv::RuntimeBinary>  mv::ComputationModel::allocateBinaryBuffer(std::size_t newSize)
{
   if (binary_->getBuffer(newSize))
    {
        return binary_ ;
    }
    return nullptr ;
}

std::shared_ptr<mv::RuntimeBinary>  mv::ComputationModel::getBinaryBuffer()
{
    return binary_ ;
}

mv::json::Array mv::ComputationModel::dataFlowToJSON() const
{
    json::Array data_flows;
    for (auto dataIt = dataGraph_.edge_begin(); dataIt != dataGraph_.edge_end(); ++dataIt)
        data_flows.append((*dataIt).toJSON());
    //    std::cout<<"Data Flows in computation model: "<<data_flows.stringify()<<std::endl;
    return data_flows;
}

mv::json::Array mv::ComputationModel::controlFlowToJSON() const
{
    json::Array control_flow;
    for (auto controlIt = controlGraph_.edge_begin(); controlIt != controlGraph_.edge_end(); ++controlIt)
        control_flow.append((*controlIt).toJSON());
    std::cout << "control Flows in computation model: " << control_flow.stringify() << std::endl;
    return control_flow;
}

//Nodes and operation counters
mv::json::Array mv::ComputationModel::opsToJSON() const
{
    json::Array nodes;
    for (auto opIt = dataGraph_.node_begin(); opIt != dataGraph_.node_end(); ++opIt)
        nodes.append((*opIt).toJSON());
    return nodes;
}

//Operation Index counters
mv::json::Object mv::ComputationModel::opsIndexCounterToJSON() const
{
    json::Object opsIndCounters;
    for (auto opsCounterIt = opsIndexCounter_->begin(); opsCounterIt != opsIndexCounter_->end(); ++opsCounterIt)
        opsIndCounters[(opsCounterIt->first)] = json::Value(static_cast<double>(opsCounterIt->second));
    return opsIndCounters;
}

//Operation Instance Counters
mv::json::Object mv::ComputationModel::opsInstanceCounterToJSON() const
{
    json::Object opsInsCounters;
    for (auto opsInsCounterIt = opsInstanceCounter_->begin(); opsInsCounterIt != opsInstanceCounter_->end(); ++opsInsCounterIt)
        opsInsCounters[opsInsCounterIt->first] = json::Value(static_cast<double>(opsInsCounterIt->second));
    return opsInsCounters;
}

//stages toJSON
mv::json::Array mv::ComputationModel::stagesToJSON() const
{
    json::Array stages;
    for (auto stagesIt = stages_->begin(); stagesIt != stages_->end(); ++stagesIt)
        stages.append((*stagesIt->second).toJSON());
    return stages;
}

// tensors to JSON
mv::json::Array mv::ComputationModel::tensorsToJSON() const
{
    json::Array tensors;
    for (auto tensorIt = tensors_->begin(); tensorIt != tensors_->end(); ++tensorIt)
        tensors.append((*tensorIt->second).toJSON());
    return tensors;
}

//groups to JSON
mv::json::Array mv::ComputationModel::groupsToJSON() const
{
    json::Array groups;
    for (auto groupIt = groups_->begin(); groupIt != groups_->end(); ++groupIt)
        groups.append((*groupIt->second).toJSON());
    return groups;
}

//has Populated Tensors --> JSON
bool mv::ComputationModel::hasPopulatedTensorsToJSON() const
{
    bool hasPopulatedTensors = false;
    for (auto tensorIt = tensors_->begin(); tensorIt != tensors_->end(); ++tensorIt)
    {
        if (tensorIt->second->isPopulated())
        {
            hasPopulatedTensors = true;
            break;
        }
    }
    return hasPopulatedTensors;
}
/*
//has to implement memoryAllocatorsToJSON. Waiting as per Stanislaw's sugestion as code refactor planned will affect this
mv::json::Object mv::ComputationModel::memoryAllocatorsToJSON() const
{
   json::Object memory_allocators;
    for (auto memoryAllocatorIt = memoryAllocators_->begin(); memoryAllocatorIt != memoryAllocators_->end(); ++memoryAllocatorIt)
        memory_allocators[memoryAllocatorIt->first] = Jsonable::toJsonValue(*memoryAllocatorIt->second);
   return memory_allocators;
}
//has to implement the sourceOpsToJSON. Not 100% what the source Ops mean here, given nodes already captured ops details
mv::json::Object mv::ComputationModel::sourceOpsToJSON() const
{
    json::Object source_ops;

}
*/
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

mv::json::Value mv::ComputationModel::toJSON() const
{
    json::Object computationModel;
    json::Object graph;

    graph["nodes"] = json::Value(opsToJSON());
    graph["data_flows"] = json::Value(dataFlowToJSON());
    graph["control_flows"] = json::Value(controlFlowToJSON());
    computationModel["graph"] = graph;
    computationModel["tensors"] = json::Value(tensorsToJSON());
    computationModel["computation_groups"] = json::Value(groupsToJSON());
    computationModel["stages"] = json::Value(stagesToJSON());
    //    computationModel["source_ops"] = sourceOps;
    //    computationModel["memory_allocators"] = memory_allocators;
    computationModel["operations_Instance_counters"] = opsInstanceCounterToJSON();
    computationModel["operations_Index_counters"] = opsIndexCounterToJSON();
    computationModel["has_populated_tensors"] = json::Value(hasPopulatedTensorsToJSON());
    return json::Value(computationModel);
}
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

std::shared_ptr<mv::Element>mv::ComputationModel::getGlobalConfigParams() const
{
    return globalConfigParams_;
}

void mv::ComputationModel::setGlobalConfigParams(mv::Element& element)
{
    *globalConfigParams_ = element;
}
