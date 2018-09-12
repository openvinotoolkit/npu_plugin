#include "include/mcm/computation/resource/memory_allocator.hpp"
#include <iostream>

bool mv::MemoryAllocator::MemoryBuffer::operator<(const MemoryBuffer& other) const
{
    if (offset < other.offset)
        return true;

    if (size < other.size)
        return true;

    return false;

}

std::string mv::MemoryAllocator::MemoryBuffer::toString(bool printValues) const
{

    std::string res =  "data: '" + this->data->getName() + "'; offset: " + Printable::toString(this->offset) +
        "; size: " + Printable::toString(this->size) + "; left pad: " + Printable::toString(this->left_pad) +
         + "; right pad: " + Printable::toString(this->right_pad)
         + "; block size: " + Printable::toString(this->block_size) + "; block num: " + Printable::toString(this->block_num);

    res += "; strides:";

    for(size_t stride: this->strides)
        res += " " + std::to_string(stride);

    if (printValues && data->isPopulated())
    {
        res += "\nvalues:\n";

        for (unsigned i = 0; i < left_pad; ++i)
            res += "LP ";

        auto values = data->getData();
        for (unsigned current_block, i = 0; current_block < block_num; ++current_block)
        {
<<<<<<< HEAD
            for(std::size_t j = 0; j < block; j++)
                res += std::to_string(values[i]) + " ";
=======
            for(unsigned j = 0; j < block_size; ++j)
                res += Printable::toString(values[i++]) + " ";
            if(current_block != block_num - 1)
                for(unsigned j = 0; j < strides[current_block]; ++j)
                    res += "X ";
>>>>>>> 3fead80... Changed block variable of MemoryBuffer to block_size + Updated MemoryBuffer toString method to take in account strides.
        }

        for (unsigned i = 0; i < right_pad; ++i)
            res += "RP ";

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

        it->second->offset = lastOffset;
        lastOffset += it->second->size;

    }

}

mv::MemoryAllocator::MemoryAllocator(std::string name, std::size_t size, Order order) :
name_(name),
size_(size),
order_(order)
{

}

long mv::MemoryAllocator::writeStrides(const std::vector<std::size_t>& p, const mv::Shape& s, std::vector<std::size_t>& strides)
{
    return recursiveWriteStrides(order_.lastContiguousDimensionIndex(s), s, p, strides);
}

long mv::MemoryAllocator::recursiveWriteStrides(unsigned i, const mv::Shape& d, const std::vector<std::size_t>& p, std::vector<std::size_t>& strides)
{
    if(order_.isFirstContiguousDimensionIndex(d, i))
    {
        strides.push_back(p[i]);
        return p[i] + d[i];
    }
    else
    {
        long new_stride;
        for(unsigned c = 0; c < d[i]; ++c)
        {
            unsigned next_dim_index = order_.previousContiguousDimensionIndex(d, i);
            new_stride = recursiveWriteStrides(next_dim_index, d, p, strides);
        }
        //Last stride should be joined (stride definition -> only between two blocks)
        long toAdd = strides.back();
        strides.pop_back();
        strides.push_back(p[i] * new_stride + toAdd);
        return new_stride * (d[i] + p[i])                                                                                                                                                                         ;
    }
}

mv::MemoryAllocator::BufferIterator mv::MemoryAllocator::allocate(Data::TensorIterator tensor, 
    std::size_t stageIdx, const std::vector<std::size_t>& paddings)
{

    std::vector<size_t> strides;
    Shape s(tensor->getShape());
    writeStrides(paddings, s, strides);
    //Last stride is actually right padding
    size_t right_pad = strides.back();
    strides.pop_back();
    size_t left_pad = 0; //No real reason to allocate any left pad at the moment
    size_t block = s[order_.firstContiguousDimensionIndex(s)];
    size_t offset = 0;
    size_t size = s.totalSize();
    size_t block_num = size / block;
    for(size_t stride: strides)
        size += stride;
    size += right_pad;
    size += left_pad;
    MemoryBuffer newBuffer = {offset, size, strides, block, block_num, left_pad, right_pad, tensor};

    if (entries_.find(stageIdx) == entries_.end())
        entries_.emplace(stageIdx, std::map<Data::TensorIterator, std::shared_ptr<MemoryBuffer>, TensorIteratorComparator>());
    else
        if (entries_[stageIdx].size() != 0)
            newBuffer.offset = entries_[stageIdx].rbegin()->second->offset + entries_[stageIdx].rbegin()->second->size;

    return entries_[stageIdx].emplace(tensor, std::make_shared<MemoryBuffer>(newBuffer)).first;

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
<<<<<<< HEAD

std::string mv::MemoryAllocator::getLogID() const
{
    return "Memory allocator " + name_;
}
=======
>>>>>>> 3fead80... Changed block variable of MemoryBuffer to block_size + Updated MemoryBuffer toString method to take in account strides.
