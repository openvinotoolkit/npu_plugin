#ifndef MEMORY_ALLOCATOR_HPP_
#define MEMORY_ALLOCATOR_HPP_

#include <map>
#include <vector>
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/base/exception/index_error.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/tensor/order.hpp"
#include "include/mcm/logger/log_sender.hpp"

namespace mv
{

    class MemoryAllocator : public Printable, public LogSender
    {

    public:

        struct AllocatorOrderComparator
        {

            bool operator()(const std::shared_ptr<MemoryAllocator> &lhs, const std::shared_ptr<MemoryAllocator> &rhs)
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
            std::vector<size_t> strides;
            /**
             * @brief Value specifing the size of the storage memory blocks owned by the buffer
             */
            std::size_t block;
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

        using BufferIterator = std::map<Data::TensorIterator, std::shared_ptr<MemoryBuffer>, TensorIteratorComparator>::iterator;

    private:

        friend struct MemoryBuffer;

        /**
         * @brief Allocator's identifier
         */
        std::string name_;

        /**
         * @brief Total size of the memory block represented by the allocator
         */
        long long unsigned size_;

        /**
         * @brief Order of 1-dimensional representations of multidimensional tensors allocated by the allocator
         */
        Order order_;

        /**
         * @brief Entires representing buffers alllocted by the allocator for each computation stage
         */
        std::map<unsigned, std::map<Data::TensorIterator,  std::shared_ptr<MemoryBuffer>, TensorIteratorComparator>> entries_;

        void placeBuffers_(unsigned stageIdx, BufferIterator first, BufferIterator last);

    public:

        MemoryAllocator(std::string name, std::size_t size, Order order);
        mv::MemoryAllocator::BufferIterator allocate(Data::TensorIterator tensor, std::size_t stageIdx, 
            const std::vector<std::size_t>& paddings);
        bool deallocate(Data::TensorIterator tensor, std::size_t stageIdx);
        void deallocateAll(std::size_t stageIdx);
        long long unsigned freeSpace(std::size_t stageIdx) const;
        long long unsigned usedSpace(std::size_t stageIdx) const;
        std::string toString() const override;
        //mv::json::Value toJSON() const override;
        BufferIterator bufferBegin(std::size_t stageIdx);
        BufferIterator bufferEnd(std::size_t stageIdx);
        BufferIterator getBuffer(std::size_t stageIdx, Data::TensorIterator tensor);
        virtual std::string getLogID() const override; 
        
        long writeStrides(const std::vector<std::size_t>& p, const mv::Shape& s, std::vector<std::size_t>& strides);
        long recursiveWriteStrides(unsigned i, const mv::Shape& d, const std::vector<std::size_t>& p, 
            std::vector<std::size_t>& strides);
    
    };

}

#endif // MEMORY_ALLOCATOR_HPP_
