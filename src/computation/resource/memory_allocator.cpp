#include "include/mcm/computation/resource/memory_allocator.hpp"

mv::size_type mv::MemoryAllocator::findOffset_(unsigned_type stageIdx)
{

    if (states_.find(stageIdx) != states_.end())
    {
        auto lastItem = states_[stageIdx].rbegin();
        return lastItem->second.offset + lastItem->second.lenght;
    }

    return 0;

}        

mv::MemoryAllocator::MemoryAllocator(string name, size_type maxSize) :
name_(name),
maxSize_(maxSize)
{

}

bool mv::MemoryAllocator::allocate(Tensor &tensor, unsigned_type stageIdx)
{

    auto newOffset = findOffset_(stageIdx);

    if (newOffset + tensor.getShape().totalSize() > maxSize_)
    {
        return false;
    }

    if (states_.find(stageIdx) == states_.end())
    {
        states_.emplace(stageIdx, map<string, MemoryBuffer>());
    }
    
    // No need to unroll, because if stage was not referenced before there can be no
    // already allocated tensor in it
    if (states_[stageIdx].find(tensor.getName()) != states_[stageIdx].end())
        return false;

    MemoryBuffer newBuffer = {newOffset, tensor.getShape().totalSize(), MemoryLayout::LayoutPlain};
    states_[stageIdx].emplace(tensor.getName(), newBuffer);

    return true;

}

bool mv::MemoryAllocator::deallocate(Tensor &tensor, unsigned_type stageIdx)
{

    if (states_.find(stageIdx) != states_.end())
    {
        if (states_[stageIdx].find(tensor.getName()) != states_[stageIdx].end())
        {
            states_[stageIdx].erase(tensor.getName());
            return true;
        }
    }

    return false;
}

bool mv::MemoryAllocator::deallocateAll(unsigned_type stageIdx)
{
    if (states_.find(stageIdx) != states_.end())
    {
        states_.erase(stageIdx);
    }

    return false;
}

mv::size_type mv::MemoryAllocator::freeSpace(unsigned_type stageIdx) const
{
    
    size_type freeSpaceValue = maxSize_;

    if (states_.find(stageIdx) != states_.cend())
    {
    
        for (auto itEntry = states_.at(stageIdx).cbegin(); itEntry != states_.at(stageIdx).cend(); ++itEntry)
        {
            freeSpaceValue -= itEntry->second.lenght;
        }

    }
    return freeSpaceValue;

}

mv::string mv::MemoryAllocator::toString() const
{
    string result = "memory allocator '" + name_ + "'";
    for (auto it = states_.cbegin(); it != states_.cend(); ++it)
    {

        size_type space = freeSpace(it->first);
        result += "\nStage '" + Printable::toString(it->first) + "'" + "(" + Printable::toString(space) + " free " + Printable::toString(maxSize_) + " total)";
        for (auto itEntry = it->second.cbegin(); itEntry != it->second.cend(); ++itEntry)
        {
            
            result += "\n\towner: '" + itEntry->first + "'; offset: " + Printable::toString(itEntry->second.offset) + "; lenght: " + Printable::toString(itEntry->second.lenght) + "; layout: ";
            
            switch(itEntry->second.memoryLayout)
            {
                case MemoryLayout::LayoutPlain:
                    result += "plain";
                    break;
                
                default:
                    result += " unknown";
                    break;

            }

        }

    }

    return result;

}