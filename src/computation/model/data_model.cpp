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

mv::Data::OpListIterator mv::DataModel::switchContext(Control::OpListIterator other)
{
    return opsGraph_->get_first_iterator(other);
}

mv::Data::FlowSiblingIterator mv::DataModel::getInputFlow()
{
    return input_->leftmostOutput();
}

mv::Data::FlowSiblingIterator mv::DataModel::getOutputFlow()
{
    return output_->leftmostInput();
}

mv::Data::FlowListIterator mv::DataModel::flowBegin()
{
    return dataGraph_.edge_begin();
}

mv::Data::FlowListIterator mv::DataModel::flowEnd()
{
    return *dataFlowEnd_;
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

bool mv::DataModel::addAllocator(const string &name, std::size_t size, Order order)
{
    auto result = memoryAllocators_->emplace(name, allocator_.make_owner<MemoryAllocator>(name, size, order));
    if (result.second)
    {
        logger_.log(Logger::MessageType::MessageInfo, "Defined " + result.first->second->toString());
        return true;
    }
    return false;
}

mv::Data::BufferIterator mv::DataModel::allocateTensor(const string &allocatorName, Control::StageIterator &stage,
    Data::TensorIterator &tensor, mv::dynamic_vector<size_t> pad)
{

    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError("allocatorName", allocatorName, "Undefined allocator");

    unsigned stageIdx = stage->getAttr("idx").getContent<unsigned_type>();
    if(pad.size() == 0)
        pad = mv::dynamic_vector<size_t>(tensor->getShape().ndims(), 0);
    auto buf = (*memoryAllocators_)[allocatorName]->allocate(tensor, stageIdx, pad);
    if (buf != (*memoryAllocators_)[allocatorName]->bufferEnd(stageIdx))
    {
        addAttr(tensor, "allocator", Attribute(AttrType::StringType, allocatorName));
        logger_.log(Logger::MessageType::MessageInfo, "Allocated memory for '" + tensor->getName() + "' using " +
            (*memoryAllocators_)[allocatorName]->toString());
        return buf;
    }

    logger_.log(Logger::MessageType::MessageWarning, "Unable to allocate '" + tensor->getName() + "' (of size " +
        Printable::toString(tensor->getShape().totalSize()) + ") using " + (*memoryAllocators_)[allocatorName]->toString());
    return buf;

}

bool mv::DataModel::deallocateTensor(const string &allocatorName, Control::StageIterator &stage, Data::TensorIterator &tensor)
{

    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError("allocatorName", allocatorName, "Undefined allocator");

    unsigned stageIdx = stage->getAttr("idx").getContent<unsigned_type>();
    return (*memoryAllocators_)[allocatorName]->deallocate(tensor, stageIdx);

}

void mv::DataModel::deallocateAll(const string &allocatorName, Control::StageIterator &stage)
{

    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError("allocatorName", allocatorName, "Undefined allocator");

    unsigned stageIdx = stage->getAttr("idx").getContent<unsigned_type>();
    (*memoryAllocators_)[allocatorName]->deallocateAll(stageIdx);

}

mv::Data::BufferIterator mv::DataModel::bufferBegin(const string &allocatorName, Control::StageIterator &stage)
{
    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError("allocatorName", allocatorName, "Undefined allocator");

    unsigned stageIdx = stage->getAttr("idx").getContent<unsigned_type>();
    return (*memoryAllocators_)[allocatorName]->bufferBegin(stageIdx);
}

mv::Data::BufferIterator mv::DataModel::bufferEnd(const string &allocatorName, Control::StageIterator &stage)
{
    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError("allocatorName", allocatorName, "Undefined allocator");

    unsigned stageIdx = stage->getAttr("idx").getContent<unsigned_type>();
    return (*memoryAllocators_)[allocatorName]->bufferEnd(stageIdx);
}

mv::Data::BufferIterator mv::DataModel::getBuffer(const string &allocatorName, Control::StageIterator &stage, Data::TensorIterator tensor)
{
    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError("allocatorName", allocatorName, "Undefined allocator");

    unsigned stageIdx = stage->getAttr("idx").getContent<unsigned_type>();
    return (*memoryAllocators_)[allocatorName]->getBuffer(stageIdx, tensor);
}

bool mv::DataModel::hasAllocator(const string& name)
{

    if (memoryAllocators_->find(name) != memoryAllocators_->end())
        return true;

    return false;

}   

bool mv::DataModel::addAttr(Data::TensorIterator tensor, const string& name, const Attribute& attr)
{

    return tensor->addAttr(name, attr);

}
