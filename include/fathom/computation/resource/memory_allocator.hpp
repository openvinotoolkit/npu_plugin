#ifndef MEMORY_ALLOCATOR_HPP_
#define MEMORY_ALLOCATOR_HPP_

#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/tensor/model_tensor.hpp"

namespace mv
{

    class MemoryAllocator : public Printable
    {

    public:

        struct AllocatorOrderComparator
        {

            bool operator()(const allocator::owner_ptr<MemoryAllocator> &lhs, const allocator::owner_ptr<MemoryAllocator> &rhs)
            {
                return lhs->name_ < rhs->name_;
            }

        };

        enum class MemoryLayout
        {
            LayoutPlain
        };

        struct MemoryBuffer
        {
            size_type offset;
            size_type lenght;
            MemoryLayout memoryLayout;
        };

    private:

        string name_;
        size_type maxSize_;
        map<unsigned_type, map<string, MemoryBuffer>> states_;

        size_type findOffset_(unsigned_type stageIdx)
        {

            if (states_.find(stageIdx) != states_.end())
            {
                auto lastItem = states_[stageIdx].rbegin();
                return lastItem->second.offset + lastItem->second.lenght;
            }

            return 0;

        }        

    public:

        MemoryAllocator(string name, size_type maxSize) :
        name_(name),
        maxSize_(maxSize)
        {

        }

        bool allocate(ModelTensor &tensor, unsigned_type stageIdx)
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

            states_[stageIdx].emplace(tensor.getName(), {newOffset, tensor.getShape().totalSize(), MemoryLayout::LayoutPlain});

            return true;

        }

        bool deallocate(ModelTensor &tensor, unsigned_type stageIdx)
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
        
        bool deallocateAll(unsigned_type stageIdx)
        {
            if (states_.find(stageIdx) != states_.end())
            {
                states_.erase(stageIdx);
            }

            return false;
        }

        size_type freeSpace(unsigned_type stageIdx) const
        {
            
            size_type freeSpace = maxSize_;

            if (states_.find(stageIdx) != states_.cend())
            {
            
                for (auto itEntry = states_.at(stageIdx).cbegin(); itEntry != states_.at(stageIdx).cend(); ++itEntry)
                {
                    freeSpace -= itEntry->second.lenght;
                }

            }
            return freeSpace;

        }

        string toString() const
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

    };

}

#endif // MEMORY_ALLOCATOR_HPP_