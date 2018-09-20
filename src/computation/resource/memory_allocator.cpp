#include "include/mcm/computation/resource/memory_allocator.hpp"
#include <iostream>

std::size_t mv::MemoryAllocator::MemoryBuffer::getOffset() const
{
    return offset;
}

std::size_t mv::MemoryAllocator::MemoryBuffer::getSize() const
{
    return size;
}

const std::deque<size_t>& mv::MemoryAllocator::MemoryBuffer::getStrides() const
{
    return strides;
}

std::size_t mv::MemoryAllocator::MemoryBuffer::getBlockSize() const
{
    return blockSize;
}

mv::Data::TensorIterator mv::MemoryAllocator::MemoryBuffer::getData() const
{
    return data;
}

std::size_t mv::MemoryAllocator::MemoryBuffer::getStage() const
{
    return stage;
}

const std::vector<std::size_t>& mv::MemoryAllocator::MemoryBuffer::getLeftPad() const
{
    return leftPad;
}

const std::vector<std::size_t>& mv::MemoryAllocator::MemoryBuffer::getRightPad() const
{
    return rightPad;
}

mv::MemoryAllocator::BufferIterator mv::MemoryAllocator::MemoryBuffer::getMaster() const
{
    return masterBuffer;
}

const std::vector<mv::MemoryAllocator::BufferIterator>& mv::MemoryAllocator::MemoryBuffer::getSlaves() const
{
    return slaveBuffers;
}

bool mv::MemoryAllocator::MemoryBuffer::operator<(const MemoryBuffer& other) const
{

    if (offset < other.offset)
        return true;

    if (size < other.size)
        return true;

    return false;

}

bool mv::MemoryAllocator::MemoryBuffer::operator==(const MemoryBuffer& other) const
{
    return offset == other.offset && size == other.size && strides == other.strides;
}

std::string mv::MemoryAllocator::MemoryBuffer::toString(bool printValues) const
{

    std::string res =  "data: '" + this->data->getName() + "'; offset: " + std::to_string(this->offset) +
        "; size: " + std::to_string(this->size) + "; block size: " + std::to_string(this->blockSize) + 
        "; block num: " + std::to_string(this->blockNum);

    res += "; strides:";

    for(auto &stride: strides)
        res += " " + std::to_string(stride);

    if (printValues && data->isPopulated())
    {
        res += "\nvalues:\n";

        std::size_t dataIdx = 0, blockIdx = 0;
        for (auto &stride: strides)
        {

            for (std::size_t i = 0; i < stride; ++i)
                res += "X ";

            if (blockIdx < blockNum)
                for (std::size_t i = 0; i < blockSize; ++i)
                    res += std::to_string(data->at(dataIdx++)) + " ";
            
            ++blockIdx;

        }

    }

    return res;

}

void mv::MemoryAllocator::placeBuffers_(unsigned stageIdx, BufferIterator first, BufferIterator last)
{

    if (entries_.find(stageIdx) == entries_.end())
        return;

    if (first == last || first == entries_[stageIdx].end())
        return;

    long long lastOffset = first->second->offset;

    for (auto it = ++first; it != last; ++it)
    {
        // Move only master buffers
        if (it->second->masterBuffer == bufferEnd(stageIdx))
        {
            it->second->offset = lastOffset;
            lastOffset += it->second->size;
            // Align slave buffers
            for (auto itSlave = it->second->slaveBuffers.begin(); itSlave != it->second->slaveBuffers.end(); ++it)
                (*itSlave)->second->offset = it->second->offset;
        }
    }

}

mv::MemoryAllocator::MemoryAllocator(std::string name, std::size_t size, Order order) :
name_(name),
size_(size),
order_(order)
{

}

std::deque<std::size_t> mv::MemoryAllocator::computeStrides_(const std::vector<std::size_t>& leftPadding, 
    const std::vector<std::size_t>& rightPadding, const mv::Shape& shape)
{
    std::deque<std::size_t> leftStrides;
    std::deque<std::size_t> rightStrides;
    computeStrides_(order_.lastContiguousDimensionIndex(shape), shape, leftPadding, rightPadding, leftStrides, rightStrides);
    std::deque<std::size_t> strides;

    strides.push_back(leftStrides.back());
    leftStrides.pop_back();

    for (std::size_t i = 0; i < leftStrides.size(); ++i)
        strides.push_back(leftStrides[i] + rightStrides[i]);
    
    strides.push_back(rightStrides.back());

    return strides;
}

long mv::MemoryAllocator::computeStrides_(std::size_t currentDim, const mv::Shape& shape, const std::vector<std::size_t>& leftPadding, 
    const std::vector<std::size_t>& rightPadding, std::deque<std::size_t>& leftStrides, std::deque<std::size_t>& rightStrides)
{

    if(order_.isFirstContiguousDimensionIndex(shape, currentDim))
    {
        leftStrides.push_back(leftPadding[currentDim]);
        rightStrides.push_back(rightPadding[currentDim]);
        return leftPadding[currentDim] + rightPadding[currentDim] + shape[currentDim];
    }
    
    long newStride;
    for(std::size_t c = 0; c < shape[currentDim]; ++c)
    {
        std::size_t nextDimIdx = order_.previousContiguousDimensionIndex(shape, currentDim);
        newStride = computeStrides_(nextDimIdx, shape, leftPadding, rightPadding, leftStrides, rightStrides);
    }
    //Last stride should be joined (stride definition -> only between two blocks)
    long toAddLeft = leftStrides.back();
    long toAddRight = rightStrides.back();
    leftStrides.pop_back();
    rightStrides.pop_back();
    leftStrides.push_back((leftPadding[currentDim]) * newStride + toAddLeft);
    rightStrides.push_back((rightPadding[currentDim]) * newStride + toAddRight);
    return newStride * (shape[currentDim] + leftPadding[currentDim] + rightPadding[currentDim]);
    
}

mv::MemoryAllocator::BufferIterator mv::MemoryAllocator::allocate(Data::TensorIterator tensor, std::size_t stageIdx)
{

    if (entries_.find(stageIdx) == entries_.end())
        entries_.emplace(stageIdx, std::map<Data::TensorIterator, std::shared_ptr<MemoryBuffer>, TensorIteratorComparator>());

    Shape shape(tensor->getShape());
    
    MemoryBuffer newBuffer;
    newBuffer.offset = 0;
    newBuffer.size = shape.totalSize();
    newBuffer.blockSize = shape[order_.firstContiguousDimensionIndex(shape)];
    newBuffer.blockNum = newBuffer.size / newBuffer.blockSize;
    newBuffer.strides = std::deque<std::size_t>(newBuffer.blockNum - 1);
    newBuffer.data = tensor;
    newBuffer.stage = stageIdx;
    newBuffer.leftPad = std::vector<std::size_t>(shape.ndims());
    newBuffer.rightPad = std::vector<std::size_t>(shape.ndims());
    newBuffer.masterBuffer = bufferEnd(stageIdx);
    newBuffer.slaveBuffers = {};

    std::fill(newBuffer.strides.begin(), newBuffer.strides.end(), 0);
    std::fill(newBuffer.leftPad.begin(), newBuffer.leftPad.end(), 0);
    std::fill(newBuffer.rightPad.begin(), newBuffer.rightPad.end(), 0);

    if (entries_[stageIdx].size() != 0)
        newBuffer.offset = entries_[stageIdx].rbegin()->second->offset + entries_[stageIdx].rbegin()->second->size;

    return entries_[stageIdx].emplace(tensor, std::make_shared<MemoryBuffer>(newBuffer)).first;

}

mv::MemoryAllocator::BufferIterator mv::MemoryAllocator::allocate(Data::TensorIterator tensor, BufferIterator masterBuffer, 
    const std::vector<std::size_t>& leftPadding, const std::vector<std::size_t>& rightPadding)
{

    if (tensor->getDType() != masterBuffer->first->getDType())
        throw ArgumentError(*this, tensor->getName() + "::DType", tensor->getDType().toString(), "Does not match the DType " + 
            masterBuffer->first->getDType().toString() + " of the tensor " + masterBuffer->first->getName() + " already allocated in the given buffer");

    if (tensor->getOrder() != masterBuffer->first->getOrder())
        throw ArgumentError(*this, tensor->getName() + "::Order", tensor->getOrder().toString(), "Does not match the Order " + 
            masterBuffer->first->getOrder().toString() + " of the tensor " + masterBuffer->first->getName() + " already allocated in the given buffer");

    Shape shape(tensor->getShape());
    Shape allocatedShape(masterBuffer->first->getShape());
    
    if (shape.ndims() != allocatedShape.ndims())
        throw ArgumentError(*this, tensor->getName() + "::Shape", tensor->getShape().toString(), "Does not match the dimensionality of the shape " + 
            masterBuffer->first->getShape().toString() + " of the tensor " + masterBuffer->first->getName() + " already allocated in the given buffer");
    
    if (shape.ndims() != leftPadding.size())
        throw ArgumentError(*this, "leftPadding::size", std::to_string(leftPadding.size()), "Does not match the dimensionality of the shape " + 
            shape.toString() + " of the input tensor " + tensor->getName());

    if (shape.ndims() != rightPadding.size())
        throw ArgumentError(*this, "rightPadding::size", std::to_string(rightPadding.size()), "Does not match the dimensionality of the shape " + 
            shape.toString() + " of the input tensor " + tensor->getName());
    
    for (std::size_t i = 0; i < shape.ndims(); ++i)
        if (shape[i] + leftPadding[i] + rightPadding[i] != allocatedShape[i])
            throw ArgumentError(*this, tensor->getName() + "::paddedShape[" + std::to_string(i) + "]", 
                std::to_string(shape[i] + leftPadding[i] + rightPadding[i]), "Does not match the dimension " + std::to_string(allocatedShape[i]) +
                " of the tensor " + masterBuffer->first->getName() + " already allocated in the given buffer");

    auto slaveBuffer = allocate(tensor, masterBuffer->second->stage);
    padLeft(slaveBuffer, leftPadding);
    padRight(slaveBuffer, rightPadding);
    slaveBuffer->second->offset = masterBuffer->second->offset;
    slaveBuffer->second->masterBuffer = masterBuffer;
    masterBuffer->second->slaveBuffers.push_back(slaveBuffer);

    if (masterBuffer->first->isPopulated())
        slaveBuffer->first->bindData(*masterBuffer->first, leftPadding, rightPadding);

    return slaveBuffer;

}

bool mv::MemoryAllocator::deallocate(Data::TensorIterator tensor, std::size_t stageIdx)
{

    if (entries_.find(stageIdx) == entries_.end())
        throw IndexError(*this, stageIdx, "Deallocation of tensor for an undefined stage");

    auto it = entries_[stageIdx].find(tensor);
    if (it != entries_[stageIdx].end())
    {

        auto nextIt = it;
        nextIt++;
        entries_[stageIdx].erase(it);
        placeBuffers_(stageIdx, nextIt, entries_[stageIdx].end());
        return true;
    }

    return false;

}

void mv::MemoryAllocator::deallocateAll(std::size_t stageIdx)
{

    if (entries_.find(stageIdx) == entries_.end())
        throw IndexError(*this, stageIdx, "Deallocation of all tensors for an undefined stage");

    entries_[stageIdx].clear();

}

void mv::MemoryAllocator::padBuffer_(BufferIterator buffer)
{

    Shape shape(buffer->second->data->getShape());

    std::deque<size_t> strides = computeStrides_(buffer->second->leftPad, buffer->second->rightPad, shape);

    buffer->second->strides = strides;
    buffer->second->size = shape.totalSize();

    for (auto& stride : strides)
        buffer->second->size += stride;

    for (auto it = buffer->second->slaveBuffers.begin(); it != buffer->second->slaveBuffers.end(); ++it)
        padBuffer_(*it);

    placeBuffers_(buffer->second->stage, buffer, entries_[buffer->second->stage].end());

}

void mv::MemoryAllocator::padLeft(BufferIterator buffer, const std::vector<std::size_t>& padding)
{

    if (padding.size() != buffer->second->data->getShape().ndims())
        throw ArgumentError(*this, "padding::size", std::to_string(padding.size()), "Does not match the dimensionality of the shape " + 
            buffer->second->data->getShape().toString() + " of the allocated tensor " + buffer->second->data->getName());

    for (std::size_t i = 0; i < buffer->second->leftPad.size(); ++i)
        buffer->second->leftPad[i] += padding[i];

    for (auto it = buffer->second->slaveBuffers.begin(); it != buffer->second->slaveBuffers.end(); ++it)
        for (std::size_t i = 0; i < buffer->second->leftPad.size(); ++i)
            (*it)->second->leftPad[i] += padding[i];
    
    padBuffer_(buffer);

}

void mv::MemoryAllocator::padRight(BufferIterator buffer, const std::vector<std::size_t>& padding)
{
    
    if (padding.size() != buffer->second->data->getShape().ndims())
        throw ArgumentError(*this, "padding::size", std::to_string(padding.size()), "Does not match the dimensionality of the shape " + 
            buffer->second->data->getShape().toString() + " of the allocated tensor " + buffer->second->data->getName());

    for (std::size_t i = 0; i < buffer->second->rightPad.size(); ++i)
        buffer->second->rightPad[i] += padding[i];

    for (auto it = buffer->second->slaveBuffers.begin(); it != buffer->second->slaveBuffers.end(); ++it)
        for (std::size_t i = 0; i < buffer->second->rightPad.size(); ++i)
            (*it)->second->rightPad[i] += padding[i];

    padBuffer_(buffer);
    
}

long long unsigned mv::MemoryAllocator::usedSpace(std::size_t stageIdx) const
{

    if (entries_.find(stageIdx) == entries_.cend())
        throw IndexError(*this, stageIdx, "Check of used space for an undefined stage");

    return entries_.at(stageIdx).rbegin()->second->offset + entries_.at(stageIdx).rbegin()->second->size;

}

long long unsigned mv::MemoryAllocator::freeSpace(std::size_t stageIdx) const
{

    if (entries_.find(stageIdx) == entries_.cend())
        throw IndexError(*this, stageIdx, "Check of free space for an undefined stage");

    long long freeSpaceValue = size_;

    for (auto itEntry = entries_.at(stageIdx).cbegin(); itEntry != entries_.at(stageIdx).cend(); ++itEntry)
    {
        freeSpaceValue -= itEntry->second->size;
    }

    return freeSpaceValue;

}

std::string mv::MemoryAllocator::toString() const
{

    std::string result = "memory allocator '" + name_ + "'";
    for (auto it = entries_.cbegin(); it != entries_.cend(); ++it)
    {

        result += "\nStage '" + std::to_string(it->first) + "'" + "(" + std::to_string(usedSpace(it->first)) + " used " +
            std::to_string(freeSpace(it->first)) + " free " + std::to_string(size_) + " total)";
        for (auto itEntry = it->second.cbegin(); itEntry != it->second.cend(); ++itEntry)
            result += "\n\t" + itEntry->second->toString();

    }

    return result;

}

/*mv::json::Value mv::MemoryAllocator::toJsonValue() const
{


    mv::json::Object obj;

    obj["name"] = name_;
    obj["max_size"] = mv::Jsonable::toJsonValue(maxSize_);
    mv::json::Array states;

    for (auto it = states_.cbegin(); it != states_.cend(); ++it)
    {
        mv::json::Object state;
        state["stage"] = mv::Jsonable::toJsonValue(it->first);
        state["free_space"] = mv::Jsonable::toJsonValue(freeSpace(it->first));

        mv::json::Array memoryBuffers;
        for (auto itEntry = it->second.cbegin(); itEntry != it->second.cend(); ++itEntry)
        {
             mv::json::Object memoryBuffer;
             memoryBuffer["name"] = mv::Jsonable::toJsonValue(itEntry->first);
             memoryBuffer["offset"] =  mv::Jsonable::toJsonValue(itEntry->second.offset);
             memoryBuffer["lenght"] =  mv::Jsonable::toJsonValue(itEntry->second.lenght);
             switch(itEntry->second.memoryLayout)
             {
                 case MemoryLayout::LayoutPlain:
                     memoryBuffer["layout"] = mv::Jsonable::toJsonValue("plain");
                     break;

                 default:
                     memoryBuffer["layout"] = mv::Jsonable::toJsonValue("unknown");
                     break;

             }
             memoryBuffers.append(memoryBuffer);
        }
        state["buffers"] = mv::json::Value(mv::json::Value(memoryBuffers));
        states.append(mv::json::Value(state));
    }

    obj["states"] = mv::json::Value(states);
    return mv::json::Value(obj);

}*/

mv::MemoryAllocator::BufferIterator mv::MemoryAllocator::bufferBegin(std::size_t stageIdx)
{

    auto it = entries_.find(stageIdx);
    if (it == entries_.end())
        throw IndexError(*this, stageIdx, "Getting the buffer begin iterator for an undefined stage");
    return it->second.begin();

}

mv::MemoryAllocator::BufferIterator mv::MemoryAllocator::bufferEnd(std::size_t stageIdx)
{

    auto it = entries_.find(stageIdx);
    if (it == entries_.end())
        throw IndexError(*this, stageIdx, "Getting the buffer end iterator for an undefined stage");
    return it->second.end();

}

mv::MemoryAllocator::BufferIterator mv::MemoryAllocator::getBuffer(std::size_t stageIdx, Data::TensorIterator tensor)
{

    auto it = entries_.find(stageIdx);
    if (it == entries_.end())
        throw IndexError(*this, stageIdx, "Finding a buffer iterator for an undefined stage");
    return it->second.find(tensor);

}

std::string mv::MemoryAllocator::getLogID() const
{
    return "Memory allocator " + name_;
}
