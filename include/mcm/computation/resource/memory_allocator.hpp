#ifndef MEMORY_ALLOCATOR_HPP_
#define MEMORY_ALLOCATOR_HPP_

#include <map>
#include <vector>
#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/base/order/order_factory.hpp"

namespace mv
{

    class MemoryAllocator : public Printable, public Jsonable
    {

        static allocator allocator_;

    public:

        struct AllocatorOrderComparator
        {

            bool operator()(const allocator::owner_ptr<MemoryAllocator> &lhs, const allocator::owner_ptr<MemoryAllocator> &rhs)
            {
                return lhs->name_ < rhs->name_;
            }

        };

        struct TensorIteratorComparator
        {

            bool operator()(const Data::TensorIterator &lhs, const Data::TensorIterator &rhs) const
            {
                return lhs->getName() < rhs->getName();
            }

        };

        struct MemoryBuffer
        {
            /**
             * @brief Value specifing the start location of the buffer relatively to the beginning
             * of the whole memory block specified by an allocator
             */
            std::size_t offset;
            /**
             * @brief Value specifing the size of the buffer, added to the offset represents
             * the end location of the buffer in an allocator
             */
            std::size_t size;
            /**
             * @brief Values specifing the displacement between consequent storage memory blocks owned by the buffer
             */
            mv::dynamic_vector<size_t> strides;
            /**
             * @brief Value specifing the size of the storage memory blocks owned by the buffer
             */
            std::size_t block_size;
            /**
             * @brief Value specifing the size of the storage memory blocks owned by the buffer
             */
            std::size_t block_num;
            /**
             * @brief Value specifing the displacement between the start location of the buffer and the beginning of
             * the first memory storage block
             */
            std::size_t left_pad;
            /**
             * @brief Value specifing the displacement between the end of last memory storage block and the end location of the buffer.
             */
            std::size_t right_pad;
            /**
             * @brief Tensor allocated in the buffer
             */
            Data::TensorIterator data;

            bool operator<(const MemoryBuffer& other) const;
            std::string toString(bool printValues = false) const;
            bool operator==(const MemoryBuffer& other){
                // TODO: Also check length and other things.
                return this->offset == other.offset;
            };

        };

    private:

        friend struct MemoryBuffer;

        /**
         * @brief Allocator's identifier
         */
        string name_;

        /**
         * @brief Total size of the memory block represented by the allocator
         */
        long long size_;

        /**
         * @brief Order of 1-dimensional representations of multidimensional tensors allocated by the allocator
         */
        std::unique_ptr<OrderClass> order_;

        /**
         * @brief Entires representing buffers alllocted by the allocator for each computation stage
         */
        std::map<unsigned, std::map<Data::TensorIterator,  allocator::owner_ptr<MemoryBuffer>, TensorIteratorComparator>> entries_;

        using BufferIterator = std::map<Data::TensorIterator, allocator::owner_ptr<MemoryBuffer>, TensorIteratorComparator>::iterator;

        void placeBuffers_(unsigned stageIdx, BufferIterator first, BufferIterator last);

    public:

        MemoryAllocator(string name, std::size_t size, Order order);
        template <typename VecType>
        mv::MemoryAllocator::BufferIterator allocate(Data::TensorIterator tensor, unsigned stageIdx, const VecType& paddings)
        {
            // TODO: Move this to the cpp.

            mv::dynamic_vector<size_t> strides;
            Shape s(tensor->getShape());
            writeStrides(paddings, s, strides);
            //Last stride is actually right padding
            size_t right_pad = strides.back();
            strides.pop_back();
            size_t left_pad = 0; //No real reason to allocate any left pad at the moment
            size_t block_size = s[order_->firstContiguousDimensionIndex(s)];
            size_t size = s.totalSize();
            size_t block_num = size / block_size;
            size_t offset = 0;
            for(size_t stride: strides)
                size += stride;
            size += right_pad;
            size += left_pad;
            MemoryBuffer newBuffer = {offset, size, strides, block_size, block_num, left_pad, right_pad, tensor};

            if (entries_.find(stageIdx) == entries_.end())
                entries_.emplace(stageIdx, std::map<Data::TensorIterator, allocator::owner_ptr<MemoryBuffer>, TensorIteratorComparator>());
            else
                if (entries_[stageIdx].size() != 0)
                    newBuffer.offset = entries_[stageIdx].rbegin()->second->offset + entries_[stageIdx].rbegin()->second->size;

            return entries_[stageIdx].emplace(tensor, allocator_.make_owner<MemoryBuffer>(newBuffer)).first;
        }
        bool deallocate(Data::TensorIterator tensor, unsigned stageIdx);
        void deallocateAll(unsigned stageIdx);
        long long freeSpace(unsigned stageIdx) const;
        long long usedSpace(unsigned stageIdx) const;
        string toString() const;
        mv::json::Value toJsonValue() const;
        BufferIterator bufferBegin(unsigned stageIdx);
        BufferIterator bufferEnd(unsigned stageIdx);
        BufferIterator getBuffer(unsigned stageIdx, Data::TensorIterator tensor);

        template <typename VecType1, typename VecType2>
        long writeStrides(const VecType1& p, const mv::Shape& s, VecType2& strides)
        {
            return recursiveWriteStrides(order_->lastContiguousDimensionIndex(s), s, p, strides);
        }

        template <typename VecType1, typename VecType2>
        long recursiveWriteStrides(unsigned i, const mv::Shape& d, const VecType1& p, VecType2& strides)
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
                    new_stride = recursiveWriteStrides(next_dim_index, d, p, strides);
                }
                //Last stride should be joined (stride definition -> only between two blocks)
                long toAdd = strides.back();
                strides.pop_back();
                strides.push_back(p[i] * new_stride + toAdd);
                return new_stride * (d[i] + p[i])                                                                                                                                                                         ;
            }
        }
    };

}

#endif // MEMORY_ALLOCATOR_HPP_
