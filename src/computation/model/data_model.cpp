#include "include/mcm/computation/model/data_model.hpp"

/*mv::DataModel::DataModel(Logger::VerboseLevel verboseLevel, bool logTime) :
ComputationModel(verboseLevel, logTime)
{

}

mv::DataModel::DataModel(Logger &logger) :
ComputationModel(logger)
{

}*/

mv::DataModel::DataModel(const ComputationModel &other) :
ComputationModel(other)
{

}

mv::Data::OpListIterator mv::DataModel::switchContext(Control::OpListIterator &other)
{
    return opsGraph_->get_first_iterator(other);
}

mv::Data::FlowSiblingIterator mv::DataModel::getInputFlow()
{
    return input_.leftmostOutput();
}

mv::Data::FlowSiblingIterator mv::DataModel::getOutputFlow()
{
    return output_.leftmostInput();
}

mv::Data::FlowListIterator mv::DataModel::flowBegin()
{
    return dataGraph_.edge_begin();
}

mv::Data::FlowListIterator mv::DataModel::flowEnd()
{
    return dataFlowEnd_;
}

mv::GroupContext::MemberIterator mv::DataModel::addGroupElement(Data::FlowListIterator &element, GroupContext::GroupIterator &group)
{
    allocator::owner_ptr<DataFlow> ptr = element;
    return addGroupElement_(ptr, group);
}

bool mv::DataModel::removeGroupElement(Data::FlowListIterator &element, GroupContext::GroupIterator &group)
{
    allocator::owner_ptr<DataFlow> ptr = element;
    return removeGroupElement_(ptr, group);
}

mv::Data::TensorIterator mv::DataModel::findTensor(string name)
{

    return ComputationModel::findTensor_(name);

}

unsigned mv::DataModel::tensorsCount() const
{
    return flowTensors_->size();
}

bool mv::DataModel::addAllocator(const string &name, size_type maxSize)
{
    auto result = memoryAllocators_->emplace(name, allocator_.make_owner<MemoryAllocator>(name, maxSize));
    if (result.second)
    {
        logger_.log(Logger::MessageType::MessageInfo, "Defined " + result.first->second->toString());
        return true;
    }
    return false;
}

bool mv::DataModel::allocateTensor(const string &allocatorName, Control::StageIterator &stage, Data::TensorIterator &tensor)
{
    if (memoryAllocators_->find(allocatorName) != memoryAllocators_->end())
    {
        if ((*memoryAllocators_)[allocatorName]->allocate(*tensor, stage->getAttr("idx").getContent<unsigned_type>()))
        {
            logger_.log(Logger::MessageType::MessageInfo, "Allocated memory for '" + tensor->getName() + "' using " + (*memoryAllocators_)[allocatorName]->toString());
            return true;
        }
        else
        {
            logger_.log(Logger::MessageType::MessageWarning, "Unable to allocate '" + tensor->getName() + "' (of size " + Printable::toString(tensor->getShape().totalSize()) + ") using " + (*memoryAllocators_)[allocatorName]->toString());
        }

    }

    return false;

}