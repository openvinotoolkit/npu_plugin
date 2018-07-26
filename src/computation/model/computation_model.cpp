#include "include/mcm/computation/model/computation_model.hpp"

mv::allocator mv::ComputationModel::allocator_;
mv::DefaultLogger mv::ComputationModel::defaultLogger_;
mv::Logger &mv::ComputationModel::logger_ = mv::ComputationModel::defaultLogger_;


mv::ComputationModel::ComputationModel(Logger::VerboseLevel verboseLevel, bool logTime, bool defaultControlFlow) :
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
output_(new Data::OpListIterator(*dataOpEnd_)),
lastOp_(new Control::OpListIterator()),
defaultControlFlow_(new bool(defaultControlFlow))
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
output_(other.output_),
lastOp_(other.lastOp_),
defaultControlFlow_(other.defaultControlFlow_)
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
    return !dataGraph_.disjoint() && *input_ != *dataOpEnd_ && *output_ != *dataOpEnd_ && checkOpsStages_();
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
        auto result = group->addElement(element);
        if (result != group->end())
        {
            logger_.log(Logger::MessageType::MessageInfo, "Appended new member '" + (*result)->getName() + "' to group '" + group->getName() + "'");
            return result;
        }
    }

    return group->end();
    
}

bool mv::ComputationModel::removeGroupElement_(allocator::owner_ptr<ComputationElement> element, mv::GroupContext::GroupIterator &group)
{

    if (group != groupEnd())
    {

        GroupContext::MemberIterator it = group->find(element);

        if (it != memberEnd(group))
        {
            return group->removeElement(it);
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
        auto result = stage->addElement(ptr);

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
    *lastOp_ = controlGraph_.node_end();
    dataGraph_.clear();
    controlGraph_.clear();
}

void mv::ComputationModel::disableDefaultControlFlow()
{
    *defaultControlFlow_ = false;
}

bool mv::ComputationModel::enableDefaultControlFlow(Control::OpListIterator lastOp)
{
    
    if (!isValid(lastOp))
        return false;

    *lastOp_ = lastOp;
    *defaultControlFlow_ = true;

    return true;

}

bool mv::ComputationModel::enableDefaultControlFlow(Data::OpListIterator lastOp)
{
    return enableDefaultControlFlow(opsGraph_->get_second_iterator(lastOp));
}

mv::Logger &mv::ComputationModel::logger()
{

    return logger_;

}

bool mv::ComputationModel::getDefaultControlFlow() const
{
    return defaultControlFlow_;
}


void mv::ComputationModel::setLogger(Logger &logger)
{
    logger_ = logger;
}
