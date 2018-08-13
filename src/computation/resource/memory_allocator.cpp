#include "include/mcm/computation/resource/memory_allocator.hpp"
#include "include/mcm/base/order/order_factory.hpp"
#include <iostream>

mv::allocator mv::MemoryAllocator::allocator_;

long mv::MemoryAllocator::recursiveWriteStrides(unsigned i, const mv::dynamic_vector<unsigned>& p, mv::dynamic_vector<unsigned>& strides, const mv::Shape d)
{
    if(order_->isFirstContiguousDimensionIndex(d, i))
    {
        strides.push_back(p[i]);
        return p[i] + d[i];
    }
    else
    {
        long new_stride;
        for(unsigned c = 0; c < d[i]; ++c)
        {
            unsigned next_dim_index = order_->previousContiguousDimensionIndex(d, i);
            new_stride = recursiveWriteStrides(next_dim_index, p, strides, d);
        }
        //Last stride should be joined (stride definition -> only between two blocks)
        long toAdd = strides.back();
        strides.pop_back();
        strides.push_back(p[i] * new_stride + toAdd);
        return new_stride * (d[i] + p[i])                                                                                                                                                                         ;
    }
}

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
        "; size: " + Printable::toString(this->size) + "; pad: " + Printable::toString(this->pad) +
        "; stride: " + Printable::toString(this->stride) + "; block: " + Printable::toString(this->block);

    if (printValues && data->isPopulated())
    {
        res += "\nvalues:\n";
        
        for (std::size_t i = 0; i < this->pad; ++i)
            res += "0 ";

        auto values = data->getData();
        for (std::size_t i = 0; i < values.size(); ++i)
            res += Printable::toString(values[i]) + " ";

        for (std::size_t i = 0; i < this->pad; ++i)
            res += "0 ";
            
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

mv::MemoryAllocator::MemoryAllocator(string name, std::size_t size, Order order) :
name_(name),
size_(size),
order_(mv::OrderFactory::createOrder(order))
{

}

mv::MemoryAllocator::BufferIterator mv::MemoryAllocator::allocate(Data::TensorIterator tensor, unsigned stageIdx, int pad)
{

    std::size_t padValue = (pad > 0) ? static_cast<std::size_t>(pad) : 0;
    MemoryBuffer newBuffer = {0, tensor->getShape().totalSize() + 2 * padValue, 0, tensor->getShape().totalSize(), 
        padValue, tensor};
    
    if (entries_.find(stageIdx) == entries_.end())
        entries_.emplace(stageIdx, std::map<Data::TensorIterator, allocator::owner_ptr<MemoryBuffer>, TensorIteratorComparator>());
    else
        if (entries_[stageIdx].size() != 0)
            newBuffer.offset = entries_[stageIdx].rbegin()->second->offset + entries_[stageIdx].rbegin()->second->size;

    return entries_[stageIdx].emplace(tensor, allocator_.make_owner<MemoryBuffer>(newBuffer)).first;

}

bool mv::MemoryAllocator::deallocate(Data::TensorIterator tensor, unsigned stageIdx)
{

    if (entries_.find(stageIdx) == entries_.end())
        throw ArgumentError("stageIdx", std::to_string(stageIdx), "Attempt of deallocating a tensor using "
            "an undefined stage");
    
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

void mv::MemoryAllocator::deallocateAll(unsigned stageIdx)
{

    if (entries_.find(stageIdx) == entries_.end())
        throw ArgumentError("stageIdx", std::to_string(stageIdx), "Attempt of deallocating tensors from "
            "an undefined stage");

    entries_[stageIdx].clear();

}

long long mv::MemoryAllocator::usedSpace(unsigned stageIdx) const
{

    if (entries_.find(stageIdx) == entries_.cend())
        throw ArgumentError("stageIdx", std::to_string(stageIdx), "Attempt of check used space of a buffer " +
            name_ + " for an undefined stage");

    return entries_.at(stageIdx).rbegin()->second->offset + entries_.at(stageIdx).rbegin()->second->size;

}

long long mv::MemoryAllocator::freeSpace(unsigned stageIdx) const
{
    
    if (entries_.find(stageIdx) == entries_.cend())
        throw ArgumentError("stageIdx", std::to_string(stageIdx), "Attempt of check free space of a buffer " +
            name_ + " for an undefined stage");

    long long freeSpaceValue = size_;
    
    for (auto itEntry = entries_.at(stageIdx).cbegin(); itEntry != entries_.at(stageIdx).cend(); ++itEntry)
    {
        freeSpaceValue -= itEntry->second->size;
    }

    return freeSpaceValue;

}

mv::string mv::MemoryAllocator::toString() const
{
    
    string result = "memory allocator '" + name_ + "'";
    for (auto it = entries_.cbegin(); it != entries_.cend(); ++it)
    {

        result += "\nStage '" + Printable::toString(it->first) + "'" + "(" + Printable::toString(usedSpace(it->first)) + " used " + 
            Printable::toString(freeSpace(it->first)) + " free " + Printable::toString(size_) + " total)";
        for (auto itEntry = it->second.cbegin(); itEntry != it->second.cend(); ++itEntry)
            result += "\n\t" + itEntry->second->toString();

    }

    return result;

}

mv::json::Value mv::MemoryAllocator::toJsonValue() const
{

    
    mv::json::Object obj;

    /*obj["name"] = name_;
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

    obj["states"] = mv::json::Value(states);*/
    return mv::json::Value(obj);

}

mv::MemoryAllocator::BufferIterator mv::MemoryAllocator::bufferBegin(unsigned stageIdx)
{
    auto it = entries_.find(stageIdx);
    if (it == entries_.end())
        throw ArgumentError("stageIdx", std::to_string(stageIdx), "Attempt of getting a begin buffer iterator "
            "for an undefined stage");
    return it->second.begin();
}

mv::MemoryAllocator::BufferIterator mv::MemoryAllocator::bufferEnd(unsigned stageIdx)
{
    auto it = entries_.find(stageIdx);
    if (it == entries_.end())
        throw ArgumentError("stageIdx", std::to_string(stageIdx), "Attempt of getting an end buffer iterator "
            "for an undefined stage");
    return it->second.end();
}

mv::MemoryAllocator::BufferIterator mv::MemoryAllocator::getBuffer(unsigned stageIdx, Data::TensorIterator tensor)
{
    auto it = entries_.find(stageIdx);
    if (it == entries_.end())
        throw ArgumentError("stageIdx", std::to_string(stageIdx), "Attempt of finding a buffer iterator "
            "for an undefined stage");

    return it->second.find(tensor);

}
