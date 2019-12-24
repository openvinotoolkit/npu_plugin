#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/op/op.hpp"

mv::DataModel::DataModel(ComputationModel& other) :
ComputationModel(other)
{
    log(Logger::MessageType::Debug, "Bound");
}

mv::DataModel::~DataModel()
{
    log(Logger::MessageType::Debug, "Deleted");
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

void mv::DataModel::addGroupElement(Data::FlowListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while including data flow to a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while including data flow to a group");

    group->include(element);
}

void mv::DataModel::addGroupElement(Data::TensorIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while including tensor to a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while including tensor to a group");

    group->include(element);
}

void mv::DataModel::removeGroupElement(Data::FlowListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while excluding data flow from a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while excluding data flow from a group");
    group->exclude(element);
}

void mv::DataModel::removeGroupElement(Data::TensorIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while excluding tensor from a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while excluding tensor from a group");
    group->exclude(element);
}

mv::Data::TensorIterator mv::DataModel::defineTensor(const std::string& name, const Shape& shape, DType dType, Order order)
{

    if (tensors_->find(name) != tensors_->end())
        throw ArgumentError(*this, "Tensor::name", name, "Duplicated");

    auto result = tensors_->emplace(name, std::make_shared<Tensor>(name, shape, dType, order));
    log(Logger::MessageType::Debug, "Defined " + result.first->second->toString());
    return result.first;

}

mv::Data::TensorIterator mv::DataModel::defineTensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<double>& data)
{

    if (tensors_->find(name) != tensors_->end())
        throw ArgumentError(*this, "Tensor::name", name, "Duplicated");

    auto result = tensors_->emplace(name, std::make_shared<Tensor>(name, shape, dType, order, data));
    log(Logger::MessageType::Debug, "Defined " + result.first->second->toString());
    return result.first;

}

mv::Data::TensorIterator mv::DataModel::defineTensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<int64_t>& data)
{

    if (tensors_->find(name) != tensors_->end())
        throw ArgumentError(*this, "Tensor::name", name, "Duplicated");

    auto result = tensors_->emplace(name, std::make_shared<Tensor>(name, shape, dType, order, data));
    log(Logger::MessageType::Debug, "Defined " + result.first->second->toString());
    return result.first;

}

mv::Data::TensorIterator mv::DataModel::defineTensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<mv::DataElement>& data)
{

    if (tensors_->find(name) != tensors_->end())
        throw ArgumentError(*this, "Tensor::name", name, "Duplicated");

    auto result = tensors_->emplace(name, std::make_shared<Tensor>(name, shape, dType, order, data));
    log(Logger::MessageType::Debug, "Defined " + result.first->second->toString());
    return result.first;

}

mv::Data::TensorIterator mv::DataModel::defineTensor(const Tensor& tensor)
{

    if (tensors_->find(tensor.getName()) != tensors_->end())
        throw ArgumentError(*this, "Tensor::name", tensor.getName(), "Duplicated");

    auto result = tensors_->emplace(tensor.getName(), std::make_shared<Tensor>(tensor));
    log(Logger::MessageType::Debug, "Defined " + result.first->second->toString());
    return result.first;
}

mv::Data::TensorIterator mv::DataModel::defineTensor(std::shared_ptr<Tensor> tensor)
{
    if (tensors_->find(tensor->getName()) != tensors_->end())
        throw ArgumentError(*this, "Tensor::name", tensor->getName(), "Duplicated");

    auto result = tensors_->emplace(tensor->getName(), tensor);
    log(Logger::MessageType::Debug, "Defined " + result.first->second->toString());
    return result.first;
}

void mv::DataModel::undefineTensor(const std::string& name)
{

    auto it = getTensor(name);

    if (it == tensorEnd())
        throw ArgumentError(*this, "tensor:name", name, "Undefined");


    if (it->hasAttr("sourceOp"))
        throw ArgumentError(*this, "tensor:name", name, "Unable to delete a tensor that is an output of op " + it->get<std::string>("sourceOp")
            + ", this is possible only by removing op itself");

    log(Logger::MessageType::Debug, "Removed " + it->toString());

    tensors_->erase(name);

}

void mv::DataModel::undefineTensor(Data::TensorIterator tensor)
{
    undefineTensor(tensor->getName());
}

std::size_t mv::DataModel::tensorsCount() const
{
    return tensors_->size();
}

bool mv::DataModel::addAllocator(const std::string& name, std::size_t size, std::size_t alignment)
{
    auto result = memoryAllocators_->emplace(name, std::make_shared<MemoryAllocator>(name, size, alignment));
    if (result.second)
    {
        log(Logger::MessageType::Debug, "Defined " + result.first->second->toString());
        return true;
    }
    return false;
}

mv::Data::BufferIterator mv::DataModel::allocateTensor(const std::string& allocatorName, Control::StageIterator& stage,
    Data::TensorIterator& tensor)
{

    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError(*this, "allocatorName", allocatorName, "Undefined allocator");

    unsigned stageIdx = stage->getIdx();
    auto buf = (*memoryAllocators_)[allocatorName]->allocate(tensor, stageIdx);

    if (buf != (*memoryAllocators_)[allocatorName]->bufferEnd(stageIdx))
    {
        log(Logger::MessageType::Debug, "Allocated memory for '" + tensor->getName() + "' using " +
            (*memoryAllocators_)[allocatorName]->toString());
        return buf;
    }

    log(Logger::MessageType::Warning, "Unable to allocate '" + tensor->getName() + "' (of size " +
        std::to_string(tensor->getShape().totalSize()) + ") using " + (*memoryAllocators_)[allocatorName]->toString());

    return buf;

}

mv::Data::BufferIterator mv::DataModel::allocateTensor(const std::string& allocatorName, Data::BufferIterator buffer,
    Data::TensorIterator tensor, const std::vector<std::size_t>& leftPadding, const std::vector<std::size_t>& rightPadding)
{

    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError(*this, "allocatorName", allocatorName, "Undefined allocator");

    auto buf = (*memoryAllocators_)[allocatorName]->allocate(tensor, buffer, leftPadding, rightPadding);

    if (buf != (*memoryAllocators_)[allocatorName]->bufferEnd(buffer->getStage()))
    {
        log(Logger::MessageType::Debug, "Allocated memory for '" + tensor->getName() + "' using " +
            (*memoryAllocators_)[allocatorName]->toString());
        return buf;
    }

    log(Logger::MessageType::Warning, "Unable to allocate '" + tensor->getName() + "' (of size " +
        std::to_string(tensor->getShape().totalSize()) + ") using " + (*memoryAllocators_)[allocatorName]->toString());
    return buf;

}

mv::Data::BufferIterator mv::DataModel::moveTensor(const std::string& allocatorName, Data::BufferIterator slaveBuffer, Data::BufferIterator masterBuffer,
    const std::vector<std::size_t>& leftPadding, const std::vector<std::size_t>& rightPadding)
{

    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError(*this, "allocatorName", allocatorName, "Undefined allocator");

    auto buf = (*memoryAllocators_)[allocatorName]->move(slaveBuffer, masterBuffer, leftPadding, rightPadding);

    if (buf != (*memoryAllocators_)[allocatorName]->bufferEnd(slaveBuffer->getStage()))
    {
        log(Logger::MessageType::Debug, "Moved tensor " + (*buf)->getData()->getName() + "' using " +
            (*memoryAllocators_)[allocatorName]->toString());
        return buf;
    }

    log(Logger::MessageType::Warning, "Unable to move '" + (*buf)->getData()->getName() + "' (of size " +
        std::to_string((*buf)->getData()->getShape().totalSize()) + ") using " + (*memoryAllocators_)[allocatorName]->toString());

    return buf;

}

void mv::DataModel::padLeft(const std::string& allocatorName, Data::BufferIterator buffer, const std::vector<std::size_t>& padding)
{

    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError(*this, "allocatorName", allocatorName, "Undefined allocator");

    (*memoryAllocators_)[allocatorName]->padLeft(buffer, padding);

}

void mv::DataModel::padRight(const std::string& allocatorName, Data::BufferIterator buffer, const std::vector<std::size_t>& padding)
{

    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError(*this, "allocatorName", allocatorName, "Undefined allocator");

    (*memoryAllocators_)[allocatorName]->padRight(buffer, padding);

}

bool mv::DataModel::deallocateTensor(const std::string& allocatorName, Control::StageIterator& stage, Data::TensorIterator& tensor)
{

    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError(*this, "allocatorName", allocatorName, "Undefined allocator");

    unsigned stageIdx = stage->getIdx();
    auto result = (*memoryAllocators_)[allocatorName]->deallocate(tensor, stageIdx);

    if (result)
    {
        log(Logger::MessageType::Debug, "Unallocated memory for '" + tensor->getName() + "' in '" +
            allocatorName + "'");
        return true;
    }

    log(Logger::MessageType::Debug, "Unable to unallocated memory for '" + tensor->getName() + "' in '" +
            allocatorName + "'");

    return false;

}

void mv::DataModel::deallocateAll(const std::string& allocatorName, Control::StageIterator& stage)
{

    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError(*this, "allocatorName", allocatorName, "Undefined allocator");

    unsigned stageIdx = stage->getIdx();
    (*memoryAllocators_)[allocatorName]->deallocateAll(stageIdx);

}


bool mv::DataModel::iterable(const std::string& allocatorName, Control::StageIterator& stage)
{

    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError(*this, "allocatorName", allocatorName, "Undefined allocator");

    unsigned stageIdx = stage->getIdx();
    return (*memoryAllocators_)[allocatorName]->iterable(stageIdx);
}

mv::Data::BufferIterator mv::DataModel::bufferBegin(const std::string& allocatorName, Control::StageIterator& stage)
{
    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError(*this, "allocatorName", allocatorName, "Undefined allocator");

    unsigned stageIdx = stage->getIdx();
    return (*memoryAllocators_)[allocatorName]->bufferBegin(stageIdx);
}

mv::Data::BufferIterator mv::DataModel::bufferEnd(const std::string& allocatorName, Control::StageIterator& stage)
{
    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError(*this, "allocatorName", allocatorName, "Undefined allocator");

    unsigned stageIdx = stage->getIdx();
    return (*memoryAllocators_)[allocatorName]->bufferEnd(stageIdx);
}

mv::Data::BufferIterator mv::DataModel::getBuffer(const std::string& allocatorName, Control::StageIterator& stage, Data::TensorIterator tensor)
{
    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError(*this, "allocatorName", allocatorName, "Undefined allocator");

    unsigned stageIdx = stage->getIdx();
    return (*memoryAllocators_)[allocatorName]->getBuffer(stageIdx, tensor);
}

bool mv::DataModel::hasAllocator(const std::string& name)
{

    if (memoryAllocators_->find(name) != memoryAllocators_->end())
        return true;

    return false;

}

const mv::MemoryAllocator& mv::DataModel::getAllocator(const std::string& allocatorName)
{
    if (memoryAllocators_->find(allocatorName) == memoryAllocators_->end())
        throw ArgumentError(*this, "allocatorName", allocatorName, "Undefined allocator");

    return *(*memoryAllocators_)[allocatorName];
}

unsigned long long mv::DataModel::populatedTotalSize() const
{

    long long result = 0;
    for (auto it = tensorBegin(); it != tensorEnd(); ++it)
        if (it->isPopulated())
            result += it->computeTotalSize();
    return result;
}

unsigned long long mv::DataModel::unpopulatedTotalSize() const
{
    long long result = 0;
    for (auto it = tensorBegin(); it != tensorEnd(); ++it)
        if (!it->isPopulated())
            result += it->computeTotalSize();
    return result;
}

std::string mv::DataModel::getLogID() const
{
    return "DataModel:" + name_;
}
