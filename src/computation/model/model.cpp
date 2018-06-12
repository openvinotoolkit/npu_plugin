#include "include/fathom/computation/model/model.hpp"

mv::allocator mv::ComputationModel::allocator_;
mv::DefaultLogger mv::ComputationModel::defaultLogger_;
mv::Logger &mv::ComputationModel::logger_ = mv::ComputationModel::defaultLogger_;


mv::ComputationModel::ComputationModel(Logger::VerboseLevel verboseLevel, bool logTime) :
opsGraph_(allocator_.make_owner<computation_graph>(computation_graph())),
dataGraph_(opsGraph_->get_first()),
controlGraph_(opsGraph_->get_second()),
flowTensors_(allocator_.make_owner<map<string, allocator::owner_ptr<Tensor>>>(map<string, allocator::owner_ptr<Tensor>>())),
tensorsSources_(allocator_.make_owner<map<string, DataContext::OpListIterator>>(map<string, DataContext::OpListIterator>())),
groups_(allocator_.make_owner<map<string, allocator::owner_ptr<ComputationGroup>>>(map<string, allocator::owner_ptr<ComputationGroup>>())),
stages_(allocator_.make_owner<map<unsigned_type, allocator::owner_ptr<ComputationStage>>>(map<unsigned_type, allocator::owner_ptr<ComputationStage>>())),
memoryAllocators_(allocator_.make_owner<map<string, allocator::owner_ptr<MemoryAllocator>>>(map<string, allocator::owner_ptr<MemoryAllocator>>())),
dataOpEnd_(dataGraph_.node_end()),
dataFlowEnd_(dataGraph_.edge_end()),
controlOpEnd_(controlGraph_.node_end()),
controlFlowEnd_(controlGraph_.edge_end()),
tensorEnd_()
{
    logger_.setVerboseLevel(verboseLevel);
    logger_.setLogTime(logTime);
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
input_(other.input_),
output_(other.output_),
lastOp_(other.lastOp_),
dataOpEnd_(other.dataOpEnd_),
dataFlowEnd_(other.dataFlowEnd_),
controlOpEnd_(other.controlOpEnd_),
controlFlowEnd_(other.controlFlowEnd_)
{

}

mv::ComputationModel::~ComputationModel()
{

}

bool mv::ComputationModel::isValid() const
{
    return !dataGraph_.disjoint() && input_ != dataOpEnd_ && output_ != dataOpEnd_ && checkOpsStages_();
}

mv::GroupContext::GroupIterator mv::ComputationModel::addGroup(const string &name)
{
    
    if (getGroup(name) == groupEnd())
    {
        
        auto result = groups_->emplace(name, allocator_.make_owner<ComputationGroup>(name));
        if (result.second)
        {
            logger_.log(Logger::MessageType::MessageInfo, "Defined " + (*result.first)->toString());
            return result.first;
        }
        return groups_->end();
        
    }

    return groups_->end();

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

    return groups_->find(name);

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

    if (input_ == dataOpEnd_)
        return false;
    
    for (auto opIt = input_; opIt != dataOpEnd_; ++opIt)
    {
        if (!opIt->hasAttr("stage") && opIt->getAttr("executable").getContent<bool>())
            return false;
    }

    return true;
    
}

mv::ControlContext::StageIterator mv::ComputationModel::addStage_()
{   

    auto it = stages_->emplace(stages_->size(), allocator_.make_owner<ComputationStage>(stages_->size()));
    logger_.log(Logger::MessageType::MessageInfo, "Defined " + it.first->second->toString());
    return it.first;
    
}

bool mv::ComputationModel::addToStage_(ControlContext::StageIterator &stage, DataContext::OpListIterator &op)
{
    ControlContext::StageIterator stageEnd(stages_->end());
    if (stage != stageEnd)
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

mv::DataContext::TensorIterator mv::ComputationModel::defineOutputTensor_(DataContext::OpListIterator &source)
{
    if (source == dataOpEnd_)
    {
        logger_.log(Logger::MessageType::MessageError, "Unable to define an output tensor - invalid source op");
        return DataContext::TensorIterator();
    }

    auto tensorDef = source->getOutputDef();

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
        return DataContext::TensorIterator();
    }
}

mv::DataContext::TensorIterator mv::ComputationModel::findTensor_(const string &name)
{
    auto it = flowTensors_->find(name);

    if (it != flowTensors_->end())
        return it;

    return DataContext::TensorIterator();

}

mv::DataContext::OpListIterator mv::ComputationModel::findSourceOp_(DataContext::TensorIterator &tensor)
{

    if (tensor == tensorEnd_)
        return DataContext::OpListIterator();

    auto it = tensorsSources_->find(tensor->getName());

    if (it != tensorsSources_->end())
        return it->second;

    return DataContext::OpListIterator();

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
    return groups_->end();
}

mv::GroupContext::MemberIterator mv::ComputationModel::memberBegin(GroupContext::GroupIterator &group)
{

    if (group != groupEnd())
    {
        return group->begin();
    }
    
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

mv::Logger &mv::ComputationModel::logger()
{

    return logger_;

}

void mv::ComputationModel::setLogger(Logger &logger)
{
    logger_ = logger;
}