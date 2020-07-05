#include "include/mcm/computation/resource/memory_allocator.hpp"
#include <iostream>

mv::MemoryAllocator::MemoryBuffer::MemoryBuffer() :
id(0),
offset(0),
size(0),
blockSize(0),
blockNum(0),
postAlign(0),
stage(0),
dataTypeSize(1)
{

}

mv::MemoryAllocator::MemoryBuffer::MemoryBuffer(const MemoryBuffer& other) :
id(other.id),
offset(other.offset),
size(other.size),
strides(other.strides),
blockSize(other.blockSize),
blockNum(other.blockNum),
postAlign(other.postAlign),
data(other.data),
stage(other.stage),
leftPad(other.leftPad),
rightPad(other.rightPad),
masterBuffer(other.masterBuffer),
slaveBuffers(other.slaveBuffers),
dataTypeSize(other.dataTypeSize)
{

}

mv::MemoryAllocator::MemoryBuffer& mv::MemoryAllocator::MemoryBuffer::operator=(const MemoryBuffer& other)
{
    id = other.id;
    offset = other.offset;
    size = other.size;
    strides = other.strides;
    blockSize = other.blockSize;
    blockNum = other.blockNum;
    postAlign = other.postAlign;
    data = other.data;
    stage = other.stage;
    leftPad = other.leftPad;
    rightPad = other.rightPad;
    masterBuffer = other.masterBuffer;
    slaveBuffers = other.slaveBuffers;
    dataTypeSize = other.dataTypeSize;
    return *this;
}

void mv::MemoryAllocator::MemoryBuffer::setOffset(std::size_t offset_)
{
    offset = offset_;
    for (auto itSlave = slaveBuffers.begin(); itSlave != slaveBuffers.end(); ++itSlave)
        (**itSlave)->setOffset(offset);
}

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

std::size_t mv::MemoryAllocator::MemoryBuffer::getBlockNum() const
{
    return blockNum;
}

std::size_t mv::MemoryAllocator::MemoryBuffer::getPostAlign() const
{
    return postAlign;
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

std::size_t mv::MemoryAllocator::MemoryBuffer::getDataTypeSize() const
{
    return dataTypeSize;
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
        "; block num: " + std::to_string(this->blockNum) + "; post align: " + std::to_string(this->postAlign);

    res += "; left pad: {";
    for (auto &pad : leftPad)
        res += std::to_string(pad) + ", ";
    res.erase(res.size() - 2, 2);
    res += "}";

    res += "; right pad: {";
    for (auto &pad : rightPad)
        res += std::to_string(pad) + ", ";
    res.erase(res.size() - 2, 2);
    res += "}";

    /*res += "; strides:";

    for(auto &stride : strides)
        res += " " + std::to_string(stride);*/

    if (printValues && data->isPopulated())
    {
        res += "\nvalues:\n";

        std::size_t dataIdx = 0, blockIdx = 0;
        for (auto &stride: strides)
        {

            for (std::size_t i = 0; i < stride / dataTypeSize; ++i)
                res += "X ";

            if (blockIdx < blockNum)
                for (std::size_t i = 0; i < blockSize / dataTypeSize; ++i)
                    res += data->at(dataIdx++) + " ";

            ++blockIdx;

        }

    }

    return res;

}

void mv::MemoryAllocator::placeBuffers_(unsigned stageIdx)
{

    if (entries_.find(stageIdx) == entries_.end())
        return;

    long long lastOffset = 0;

    for (auto it = entries_[stageIdx].begin(); it != entries_[stageIdx].end(); ++it)
    {
        // Move only master buffers
        if ((*it)->masterBuffer == bufferEnd(stageIdx))
        {
            if ((lastOffset + (*it)->size) >= size_ && name_ == "VPU_DDR_Heap")
            {
                log(Logger::MessageType::Warning, "Memory allocator offset wrapping around. offset ("
                                        + std::to_string(lastOffset) + ") + data (" + std::to_string((*it)->size)
                                        + ") will overflow " + name_ + " address range.");
                lastOffset = 0;
            }

            (*it)->offset = lastOffset;
            lastOffset += (*it)->size;

            if (alignment_ > 0)
            {
                if(lastOffset % alignment_ != 0)
                {
                    (*it)->postAlign = alignment_ - lastOffset % alignment_ ;
                    lastOffset += (*it)->postAlign;
                }
            }
            // Align slave buffers
            for (auto itSlave = (*it)->slaveBuffers.begin(); itSlave != (*it)->slaveBuffers.end(); ++itSlave)
                (**itSlave)->offset = (*it)->offset;

        }

    }

}

mv::MemoryAllocator::MemoryAllocator(std::string name, std::size_t size, unsigned short alignment) :
name_(name),
size_(size),
alignment_(alignment),
currentID_(1)
{

}

std::deque<std::size_t> mv::MemoryAllocator::computeStrides_(const Order& order, const std::vector<std::size_t>& leftPadding,
    const std::vector<std::size_t>& rightPadding, const Shape& shape, const unsigned dataTypeSize)
{
    std::deque<std::size_t> leftStrides;
    std::deque<std::size_t> rightStrides;
    computeStrides_(order, order.size() - 1, shape, leftPadding, rightPadding, leftStrides, rightStrides);
    std::deque<std::size_t> strides;

    strides.push_back(leftStrides.back() * dataTypeSize);
    leftStrides.pop_back();

    for (std::size_t i = 0; i < leftStrides.size(); ++i)
        strides.push_back((leftStrides[i] + rightStrides[i]) * dataTypeSize);

    strides.push_back(rightStrides.back() * dataTypeSize);

    return strides;
}

long mv::MemoryAllocator::computeStrides_(const Order& order, std::size_t idx, const Shape& shape, const std::vector<std::size_t>& leftPadding,
    const std::vector<std::size_t>& rightPadding, std::deque<std::size_t>& leftStrides, std::deque<std::size_t>& rightStrides)
{
    std::size_t currentDim = order[idx];
    if(idx == 0)
    {
        leftStrides.push_back(leftPadding[currentDim]);
        rightStrides.push_back(rightPadding[currentDim]);
        return leftPadding[currentDim] + rightPadding[currentDim] + shape[currentDim];
    }

    long newStride = 0;
    for(std::size_t c = 0; c < shape[currentDim]; ++c)
        newStride = computeStrides_(order, idx-1, shape, leftPadding, rightPadding, leftStrides, rightStrides);

    //Last stride should be joined (stride definition -> only between two blocks)
    long toAddLeft = leftStrides.back();
    long toAddRight = rightStrides.back();
    leftStrides.pop_back();
    rightStrides.pop_back();
    leftStrides.push_back((leftPadding[currentDim]) * newStride + toAddLeft);
    rightStrides.push_back((rightPadding[currentDim]) * newStride + toAddRight);
    return newStride * (shape[currentDim] + leftPadding[currentDim] + rightPadding[currentDim]);

}

void mv::MemoryAllocator::moveSlave_(BufferIterator slaveBuffer)
{
    auto masterBuffer = (*slaveBuffer)->masterBuffer;
    if (masterBuffer == bufferEnd((*slaveBuffer)->stage))
        return;

    (*slaveBuffer)->offset = (*masterBuffer)->offset;
    for (auto it = (*slaveBuffer)->slaveBuffers.begin(); it != (*slaveBuffer)->slaveBuffers.end(); ++it)
        moveSlave_(*it);

}

void mv::MemoryAllocator::bindData_(BufferIterator slaveBuffer, bool pad)
{
    auto masterBuffer = (*slaveBuffer)->masterBuffer;
    if (masterBuffer == bufferEnd((*slaveBuffer)->stage))
        return;

    if (pad)
        (*slaveBuffer)->getData()->bindData(*(*masterBuffer)->getData(), (*slaveBuffer)->leftPad, (*slaveBuffer)->rightPad);
    else
        (*slaveBuffer)->getData()->bindData(*(*masterBuffer)->getData());
    for (auto it = (*slaveBuffer)->slaveBuffers.begin(); it != (*slaveBuffer)->slaveBuffers.end(); ++it)
        bindData_(*it, false);

}

mv::MemoryAllocator::BufferIterator mv::MemoryAllocator::allocate(Data::TensorIterator tensor, std::size_t stageIdx)
{
    unsigned dataTypeSize = tensor->getDType().getSizeInBytes();

    if (entries_.find(stageIdx) == entries_.end())
        entries_.emplace(stageIdx, std::set<std::shared_ptr<MemoryBuffer>, BufferOrderComparator>());

    if (!tensor->hasAttr("allocators"))
    {
        Shape shape(tensor->getShape());
        MemoryBuffer newBuffer;
        newBuffer.id = currentID_++;
        newBuffer.offset = 0;
        newBuffer.size = shape.totalSize() * dataTypeSize;
        newBuffer.blockSize = shape[tensor->getOrder()[0]] * dataTypeSize;
        newBuffer.blockNum = newBuffer.size / newBuffer.blockSize;
        newBuffer.postAlign= 0;
        newBuffer.strides = std::deque<std::size_t>(newBuffer.blockNum + 1);
        newBuffer.data = tensor;
        newBuffer.stage = stageIdx;
        newBuffer.leftPad = std::vector<std::size_t>(shape.ndims());
        newBuffer.rightPad = std::vector<std::size_t>(shape.ndims());
        newBuffer.masterBuffer = bufferEnd(stageIdx);
        newBuffer.slaveBuffers = {};
        newBuffer.dataTypeSize = dataTypeSize;

        std::fill(newBuffer.strides.begin(), newBuffer.strides.end(), 0);
        std::fill(newBuffer.leftPad.begin(), newBuffer.leftPad.end(), 0);
        std::fill(newBuffer.rightPad.begin(), newBuffer.rightPad.end(), 0);

        auto buffer = entries_[stageIdx].emplace(std::make_shared<MemoryBuffer>(newBuffer)).first;
        placeBuffers_(stageIdx);

        std::set<std::string> allocatorNames;
        allocatorNames.insert(name_);
        tensor->set<std::set<std::string>>("allocators",allocatorNames);

        return buffer;
    }
    else {

        /*Get attribute allocators*/
        auto& allocators = tensor->get<std::set<std::string>>("allocators");

        /*If tensor  already has an allocator of the same name*/
        if (allocators.count(name_) != 0)
            throw ArgumentError(*this, "tensor", tensor->getName(), "Already has an allocater called " + name_);
        else {

            Shape shape(tensor->getShape());

            MemoryBuffer newBuffer;
            newBuffer.id = currentID_++;
            newBuffer.offset = 0;
            newBuffer.size = shape.totalSize() * dataTypeSize;
            newBuffer.blockSize = shape[tensor->getOrder()[0]] * dataTypeSize;
            newBuffer.blockNum = newBuffer.size / newBuffer.blockSize;
            newBuffer.postAlign= 0;
            newBuffer.strides = std::deque<std::size_t>(newBuffer.blockNum + 1);
            newBuffer.data = tensor;
            newBuffer.stage = stageIdx;
            newBuffer.leftPad = std::vector<std::size_t>(shape.ndims());
            newBuffer.rightPad = std::vector<std::size_t>(shape.ndims());
            newBuffer.masterBuffer = bufferEnd(stageIdx);
            newBuffer.slaveBuffers = {};
            newBuffer.dataTypeSize = dataTypeSize;

            std::fill(newBuffer.strides.begin(), newBuffer.strides.end(), 0);
            std::fill(newBuffer.leftPad.begin(), newBuffer.leftPad.end(), 0);
            std::fill(newBuffer.rightPad.begin(), newBuffer.rightPad.end(), 0);

            auto buffer = entries_[stageIdx].emplace(std::make_shared<MemoryBuffer>(newBuffer)).first;
            placeBuffers_(stageIdx);

            /*Add allocator name*/
            allocators.insert(name_);

            return buffer;

        }
    }
}

mv::MemoryAllocator::BufferIterator mv::MemoryAllocator::allocate(Data::TensorIterator tensor, BufferIterator masterBuffer,
    const std::vector<std::size_t>& leftPadding, const std::vector<std::size_t>& rightPadding)
{

    auto slaveBuffer = allocate(tensor, (*masterBuffer)->getStage());
    move(slaveBuffer, masterBuffer, leftPadding, rightPadding);

    return slaveBuffer;

}
mv::MemoryAllocator::BufferIterator mv::MemoryAllocator::getTopMasterBuffer(mv::MemoryAllocator::BufferIterator t)
{
    auto mt = (*t)->masterBuffer;
    if (*mt != *bufferEnd((*t)->stage))
        return getTopMasterBuffer(mt);
    return t;
}

mv::MemoryAllocator::BufferIterator mv::MemoryAllocator::move(BufferIterator slaveBuffer, BufferIterator masterBuffer,
    const std::vector<std::size_t>& leftPadding, const std::vector<std::size_t>& rightPadding, bool propagate_to_slaves)
{

    auto tensor = (*slaveBuffer)->getData();

    if (tensor->getDType() != (*masterBuffer)->getData()->getDType())
        throw ArgumentError(*this, tensor->getName() + "::DType", tensor->getDType().toString(), "Does not match the DType " +
           (*masterBuffer)->getData()->getDType().toString() + " of the tensor " + (*masterBuffer)->getData()->getName() + " already allocated in the given buffer");

    if (tensor->getOrder() != (*masterBuffer)->getData()->getOrder())
        throw ArgumentError(*this, tensor->getName() + "::Order", tensor->getOrder().toString(), "Does not match the order " +
            (*masterBuffer)->getData()->getOrder().toString() + " of the tensor " + (*masterBuffer)->getData()->getName() + " already allocated in the given buffer");

    Shape shape(tensor->getShape());
    Shape allocatedShape((*masterBuffer)->getData()->getShape());

    if (shape.ndims() != allocatedShape.ndims())
        throw ArgumentError(*this, tensor->getName() + "::Shape", tensor->getShape().toString(), "Does not match the dimensionality of the shape " +
            (*masterBuffer)->getData()->getShape().toString() + " of the tensor " + (*masterBuffer)->getData()->getName() + " already allocated in the given buffer");

    if (shape.ndims() != leftPadding.size())
        throw ArgumentError(*this, "leftPadding::size", std::to_string(leftPadding.size()), "Does not match the dimensionality of the shape " +
            shape.toString() + " of the input tensor " + tensor->getName());

    if (shape.ndims() != rightPadding.size())
        throw ArgumentError(*this, "rightPadding::size", std::to_string(rightPadding.size()), "Does not match the dimensionality of the shape " +
            shape.toString() + " of the input tensor " + tensor->getName());

    // To accomodate implicit ops, allow shape mismatch if totalSize is equal
    auto oldTotal = 0;
    auto newTotal = 0;
    for (std::size_t i = 0; i < shape.ndims(); ++i)
    {
        oldTotal += allocatedShape[i];
        newTotal += shape[i] + leftPadding[i] + rightPadding[i];
    }
    if (oldTotal != newTotal)
    {
        for (std::size_t i = 0; i < shape.ndims(); ++i)
            if (shape[i] + leftPadding[i] + rightPadding[i] != allocatedShape[i])
                throw ArgumentError(*this, tensor->getName() + "::paddedShape[" + std::to_string(i) + "]",
                    std::to_string(shape[i] + leftPadding[i] + rightPadding[i]), "Does not match the dimension " + std::to_string(allocatedShape[i]) +
                    " of the tensor " + (*masterBuffer)->getData()->getName() + " already allocated in the given buffer");
    }

    if ((*slaveBuffer)->masterBuffer != bufferEnd((*slaveBuffer)->stage))
    {
        auto &slaves = (*masterBuffer)->slaveBuffers;
        slaves.erase(std::remove(slaves.begin(), slaves.end(), slaveBuffer), slaves.end());
    }

    (*masterBuffer)->slaveBuffers.push_back(slaveBuffer);

    auto slaveBuf = slaveBuffer;
    while (*slaveBuf != nullptr)
    {
        (*slaveBuf)->offset = (*masterBuffer)->offset;
        (*slaveBuf)->masterBuffer = masterBuffer;
        auto masterTensor = (*masterBuffer)->getData();
        (*slaveBuf)->leftPad = std::vector<std::size_t>(shape.ndims());
        (*slaveBuf)->rightPad = std::vector<std::size_t>(shape.ndims());
        std::fill((*slaveBuf)->strides.begin(), (*slaveBuf)->strides.end(), 0);
        std::fill((*slaveBuf)->leftPad.begin(), (*slaveBuf)->leftPad.end(), 0);
        std::fill((*slaveBuf)->rightPad.begin(), (*slaveBuf)->rightPad.end(), 0);
        padLeft(slaveBuf, leftPadding);
        padRight(slaveBuf, rightPadding);

        if ((propagate_to_slaves == false) || (*slaveBuf)->getSlaves().size() == 0)
            break;

        slaveBuf = (*slaveBuf)->getSlaves()[0];
    }

    if ((*masterBuffer)->getData()->isPopulated())
        bindData_(slaveBuffer, true);

    return slaveBuffer;

}

bool mv::MemoryAllocator::deallocate(Data::TensorIterator tensor, std::size_t stageIdx)
{

    if (entries_.find(stageIdx) == entries_.end())
        throw IndexError(*this, stageIdx, "Deallocation of tensor for an undefined stage");

    auto it = entries_[stageIdx].begin();
    while (it != entries_[stageIdx].end() && (*it)->getData() != tensor)
        ++it;

    if (it != entries_[stageIdx].end())
    {

        entries_[stageIdx].erase(it);
        placeBuffers_(stageIdx);
        tensor->erase("allocators");
        return true;

    }

    return false;

}

void mv::MemoryAllocator::deallocateAll(std::size_t stageIdx)
{

    if (entries_.find(stageIdx) == entries_.end())
        throw IndexError(*this, stageIdx, "Deallocation of all tensors for an undefined stage");

    for (auto it = entries_[stageIdx].begin(); it != entries_[stageIdx].end(); ++it)
        (*it)->getData()->erase("allocators");

    entries_[stageIdx].clear();

}

void mv::MemoryAllocator::padBuffer_(BufferIterator buffer)
{
    unsigned dataTypeSize = (*buffer)->getData()->getDType().getSizeInBytes();

    Shape shape((*buffer)->getData()->getShape());

    std::deque<size_t> strides = computeStrides_((*buffer)->getData()->getOrder(), (*buffer)->leftPad,
        (*buffer)->rightPad, shape, dataTypeSize);

    (*buffer)->strides = strides;
    (*buffer)->size = shape.totalSize() * dataTypeSize;

    for (auto& stride : strides)
        (*buffer)->size += stride;

    for (auto it = (*buffer)->slaveBuffers.begin(); it != (*buffer)->slaveBuffers.end(); ++it)
        padBuffer_(*it);

    placeBuffers_((*buffer)->stage);

}

void mv::MemoryAllocator::padLeft(BufferIterator buffer, const std::vector<std::size_t>& padding)
{

    if (padding.size() != (*buffer)->getData()->getShape().ndims())
        throw ArgumentError(*this, "padding::size", std::to_string(padding.size()), "Does not match the dimensionality of the shape " +
            (*buffer)->getData()->getShape().toString() + " of the allocated tensor " + (*buffer)->getData()->getName());

    for (std::size_t i = 0; i < (*buffer)->leftPad.size(); ++i)
        (*buffer)->leftPad[i] += padding[i];

    for (auto it = (*buffer)->slaveBuffers.begin(); it != (*buffer)->slaveBuffers.end(); ++it)
        for (std::size_t i = 0; i < (*buffer)->leftPad.size(); ++i)
            (**it)->leftPad[i] += padding[i];

    padBuffer_(buffer);

}

void mv::MemoryAllocator::padRight(BufferIterator buffer, const std::vector<std::size_t>& padding)
{

    if (padding.size() != (*buffer)->getData()->getShape().ndims())
        throw ArgumentError(*this, "padding::size", std::to_string(padding.size()), "Does not match the dimensionality of the shape " +
            (*buffer)->getData()->getShape().toString() + " of the allocated tensor " + (*buffer)->getData()->getName());

    for (std::size_t i = 0; i < (*buffer)->rightPad.size(); ++i)
        (*buffer)->rightPad[i] += padding[i];

    for (auto it = (*buffer)->slaveBuffers.begin(); it != (*buffer)->slaveBuffers.end(); ++it)
        for (std::size_t i = 0; i < (*buffer)->rightPad.size(); ++i)
            (**it)->rightPad[i] += padding[i];

    padBuffer_(buffer);

}

long long unsigned mv::MemoryAllocator::usedSpace(std::size_t stageIdx) const
{

    if (entries_.find(stageIdx) == entries_.cend())
        throw IndexError(*this, stageIdx, "Check of used space for an undefined stage");

    return (*entries_.at(stageIdx).rbegin())->offset + (*entries_.at(stageIdx).rbegin())->size;

}

long long unsigned mv::MemoryAllocator::freeSpace(std::size_t stageIdx) const
{

    if (entries_.find(stageIdx) == entries_.cend())
        throw IndexError(*this, stageIdx, "Check of free space for an undefined stage");

    long long freeSpaceValue = size_;

    for (auto itEntry = entries_.at(stageIdx).cbegin(); itEntry != entries_.at(stageIdx).cend(); ++itEntry)
    {
        freeSpaceValue -= (*itEntry)->size;
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
            result += "\n\t" + (*itEntry)->toString();

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

bool mv::MemoryAllocator::iterable(std::size_t stageIdx)
{

    auto it = entries_.find(stageIdx);
    if (it == entries_.end()){
        return false;
    }else{
        return true;
    }
}

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
    auto bufIt = entries_[stageIdx].begin();
    while (bufIt != entries_[stageIdx].end() && (*bufIt)->getData() != tensor)
        ++bufIt;
    return bufIt;

}

std::string mv::MemoryAllocator::getLogID() const
{
    return "MemoryAllocator:" + name_;
}

const std::string& mv::MemoryAllocator::getAllocatorName() const
{
    return name_;
}
