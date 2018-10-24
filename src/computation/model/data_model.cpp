#include "include/mcm/computation/model/data_model.hpp"

/*mv::DataModel::DataModel(Logger::VerboseLevel verboseLevel, bool logTime) :
ComputationModel(verboseLevel, logTime)
{

}

mv::DataModel::DataModel(Logger& logger) :
ComputationModel(logger)
{

}*/

mv::DataModel::DataModel(ComputationModel& other) :
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

mv::GroupContext::MemberIterator mv::DataModel::addGroupElement(Data::FlowListIterator& element, GroupContext::GroupIterator& group)
{
    std::shared_ptr<DataFlow> ptr = element;
    return addGroupElement_(ptr, group);
}

bool mv::DataModel::removeGroupElement(Data::FlowListIterator& element, GroupContext::GroupIterator& group)
{
    std::shared_ptr<DataFlow> ptr = element;
    return removeGroupElement_(ptr, group);
}

mv::Data::TensorIterator mv::DataModel::defineTensor(const std::string& name, const Shape& shape, DType dType, Order order)
{

    if (flowTensors_->find(name) != flowTensors_->end())
        throw ArgumentError(*this, "Tensor::name", name, "Attempt of duplication of an upopulated tensor name during the creation");

    auto result = flowTensors_->emplace(name, std::make_shared<Tensor>(name, shape, dType, order));
    log(Logger::MessageType::Info, "Defined " + result.first->second->toString());
    return result.first;

}

mv::Data::TensorIterator mv::DataModel::defineTensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<double>& data)
{

    if (flowTensors_->find(name) != flowTensors_->end())
        throw ArgumentError(*this, "Tensor::name", name, "Attempt of duplication of a populated tensor name during the creation");

    auto result = flowTensors_->emplace(name, std::make_shared<Tensor>(name, shape, dType, order, data));
    log(Logger::MessageType::Info, "Defined " + result.first->second->toString());
    return result.first;

}

mv::Data::TensorIterator mv::DataModel::defineTensor(const Tensor& tensor)
{

    if (flowTensors_->find(tensor.getName()) != flowTensors_->end())
        throw ArgumentError(*this, "Tensor::name", tensor.getName(), "Attempt of duplication of a tensor name during the copy creation");

    auto result = flowTensors_->emplace(tensor.getName(), std::make_shared<Tensor>(tensor));
    log(Logger::MessageType::Info, "Defined " + result.first->second->toString());
    return result.first;

}

bool mv::DataModel::undefineTensor(const std::string& name)
{

    if (flowTensors_->find(name) == flowTensors_->end())
    {
        log(Logger::MessageType::Error, "Unable to remove unexisting tensor " + name +
            " from the computation model");
        return false;
    }

    auto tensorSource = tensorsSources_->find(name);
    if (tensorSource != tensorsSources_->end())
    {
        log(Logger::MessageType::Error, "Unable to remove the tensor " + name +
            " that is an output of the operation " + tensorSource->second->getName() + " - source "
            "operation has to be removed to achieve this");
        return false;
    }

    flowTensors_->erase(name);
    return true;

}

mv::Data::TensorIterator mv::DataModel::findTensor(const std::string& name)
{

    return ComputationModel::findTensor_(name);

}

unsigned mv::DataModel::tensorsCount() const
{
    return flowTensors_->size();
}

bool mv::DataModel::addAllocator(const std::string& name, std::size_t size, std::size_t alignment, std::size_t dataTypeSize)
{
    auto result = memoryAllocators_->emplace(name, std::make_shared<MemoryAllocator>(name, size, alignment, dataTypeSize));
    if (result.second)
    {
        log(Logger::MessageType::Info, "Defined " + result.first->second->toString());
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
        log(Logger::MessageType::Info, "Allocated memory for '" + tensor->getName() + "' using " +
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
        log(Logger::MessageType::Info, "Allocated memory for '" + tensor->getName() + "' using " +
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
        log(Logger::MessageType::Info, "Moved tensor " + (*buf)->getData()->getName() + "' using " +
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
        log(Logger::MessageType::Info, "Unallocated memory for '" + tensor->getName() + "' in '" +
            allocatorName + "'");
        return true;
    }

    log(Logger::MessageType::Info, "Unable to unallocated memory for '" + tensor->getName() + "' in '" +
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

std::string mv::DataModel::getLogID() const
{
    return "DataModel:" + name_;
}