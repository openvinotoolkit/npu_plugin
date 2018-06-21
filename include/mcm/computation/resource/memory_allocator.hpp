#ifndef MEMORY_ALLOCATOR_HPP_
#define MEMORY_ALLOCATOR_HPP_

#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/computation/tensor/tensor.hpp"

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

        size_type findOffset_(unsigned_type stageIdx);

    public:

        MemoryAllocator(string name, size_type maxSize);
        bool allocate(Tensor &tensor, unsigned_type stageIdx);
        bool deallocate(Tensor &tensor, unsigned_type stageIdx);
        bool deallocateAll(unsigned_type stageIdx);
        size_type freeSpace(unsigned_type stageIdx) const;
        string toString() const;

    };

}

#endif // MEMORY_ALLOCATOR_HPP_