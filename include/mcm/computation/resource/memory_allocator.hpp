#ifndef MEMORY_ALLOCATOR_HPP_
#define MEMORY_ALLOCATOR_HPP_

#include <map>
#include <vector>
#include <deque>
#include <set>
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/base/exception/index_error.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"
#include "include/mcm/tensor/order/order.hpp"
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

        class MemoryBuffer;

        struct BufferOrderComparator
        {

            bool operator()(const std::shared_ptr<MemoryBuffer> &lhs, const std::shared_ptr<MemoryBuffer> &rhs)
            {
                return lhs->id < rhs->id;
            }

        };

        using BufferIterator = std::set<std::shared_ptr<MemoryBuffer>, BufferOrderComparator>::iterator;

        class MemoryBuffer
        {

            friend class MemoryAllocator;

            /**
             * @brief Unique ID
             */
            std::size_t id;

            /**
             * @brief Value specifing the start location of the buffer relatively to the beginning
             * of the whole memory block specified by an allocator (EXPRESSED IN BYTES)
             */
            std::size_t offset;

            /**
             * @brief Value specifing the size of the buffer, added to the offset represents
             * the end location of the buffer in an allocator (EXPRESSED IN BYTES)
             */
            std::size_t size;

            /**
             * @brief Vector of values specifing lenghts of the gaps between consequent storage memory blocks owned by the buffer. The first
             * element specifies the lenght of the gap between the beginning of the buffer (specified by the offset) and the beginning of the
             * first block of data. The last element specifies the lenght of gap between the end of the last memory block and the end of the
             * buffer (specified by the offest increased by the size). (EXPRESSED IN BYTES)
             */
            std::deque<size_t> strides;

            /**
             * @brief Value specifing the size of the storage memory blocks owned by the buffer (EXPRESSED IN BYTES)
             */
            std::size_t blockSize;

            /**
             * @brief Value specifing the number of the storage memory blocks owned by the buffer
             */
            std::size_t blockNum;

            /**
             * @brief Lenght of trailing, empty block of memory used for aligment (EXPRESSED IN BYTES)
             */
            std::size_t postAlign;

            /**
             * @brief Tensor allocated in the buffer
             */
            Data::TensorIterator data;

            /**
             * @brief Index of the stage for which the buffer is defined
             */
            std::size_t stage;

            /**
             * @brief Left-top padding of the dimensions of the tensor allocted in the buffer (EXPRESSED IN WORDS)
             */
            std::vector<std::size_t> leftPad;

            /**
             * @brief Right-bottom padding of the dimensions of the tensor allocted in the buffer (EXPRESSED IN WORDS)
             */
            std::vector<std::size_t> rightPad;

            /**
             * @brief Iterator pointing to the buffer that owns memory space overallocated by this buffer
             */
            BufferIterator masterBuffer;

            /**
             * @brief List of iterators pointing to buffers that overallocates memory space owned by this buffer
             */
            std::vector<BufferIterator> slaveBuffers;

            /**
             * @brief Size of data type (EXPRESSED IN BYTES)
             */
            std::size_t dataTypeSize;

        public:

            MemoryBuffer();
            MemoryBuffer(const MemoryBuffer& other);
            std::size_t getOffset() const;
            std::size_t getSize() const;
            const std::deque<size_t>& getStrides() const;
            std::size_t getBlockSize() const;
            std::size_t getBlockNum() const;
            std::size_t getPostAlign() const;
            Data::TensorIterator getData() const;
            std::size_t getStage() const;
            const std::vector<std::size_t>& getLeftPad() const;
            const std::vector<std::size_t>& getRightPad() const;
            BufferIterator getMaster() const;
            const std::vector<BufferIterator>& getSlaves() const;
            std::size_t getDataTypeSize() const;
            bool operator<(const MemoryBuffer& other) const;
            bool operator==(const MemoryBuffer& other) const;
            MemoryBuffer& operator=(const MemoryBuffer& other);
            std::string toString(bool printValues = false) const;
            void setOffset(std::size_t off);

        };



    private:

        /**
         * @brief Allocator's identifier
         */
        std::string name_;

        /**
         * @brief Total size of the memory block represented by the allocator
         */
        long long unsigned size_;

        /**
         * @brief Global memory alignment (offset value must be divisible), 0 means none
         */
        unsigned short alignment_;

        /**
         * @brief Current ID for new buffer
         */
        std::size_t currentID_;

        /**
         * @brief Entires representing buffers alllocted by the allocator for each computation stage
         */
        std::map<unsigned, std::set<std::shared_ptr<MemoryBuffer>, BufferOrderComparator>> entries_;

        void placeBuffers_(unsigned stageIdx);
        std::deque<std::size_t> computeStrides_(const Order& order, const std::vector<std::size_t>& leftPadding,
            const std::vector<std::size_t>& rightPadding, const mv::Shape& shape, const unsigned dataTypeSize);
        long computeStrides_(const Order& order, std::size_t idx, const mv::Shape& shape, const std::vector<std::size_t>& leftPadding,
            const std::vector<std::size_t>& rightPadding, std::deque<std::size_t>& leftStrides, std::deque<std::size_t>& rightStrides);
        void padBuffer_(BufferIterator buffer);
        void moveSlave_(BufferIterator slaveBuffer);
        void bindData_(BufferIterator slaveBuffer, bool pad);

    public:

        MemoryAllocator(std::string name, std::size_t size, unsigned short alignment);

        /**
         * @brief Allocate the tensor in a new buffer for the particular stage
         *
         * @param tensor Tensor to be allocated
         * @param stageIdx Stage identifier
         * @return BufferIterator Newly created buffer that contains the tensor
         */
        BufferIterator allocate(Data::TensorIterator tensor, std::size_t stageIdx);

        /**
         * @brief Allocate the tensor in memory space already allocated by another buffer. The input tensor shape padded by values
         * specified by leftPadding and rightPadding has to match the shape of the tensor that is allocated in this buffer. The shape
         * dimensionality, data order and data type of newly allocated tensor has to match the tensor already allocated.
         *
         * @param tensor Tensor to be allocated
         * @param buffer Buffer to be overallocated
         * @param leftPadding Padding (left-top) between tensor to be allocated and the tensor contained by the given buffer - EXPRESSED IN WORDS
         * @param rightPadding Padding (right-bottom) between tensor to be allocated and the tensor contained by the given buffer - EXPRESSED IN WORDS
         * @return BufferIterator Newly created buffer of the same offset and size that overlaps the memory space owned by the input buffer
         */
        BufferIterator allocate(Data::TensorIterator tensor, BufferIterator buffer, const std::vector<std::size_t>& leftPadding,
            const std::vector<std::size_t>& rightPadding);

        /**
         * @brief Moves the given slaveBuffer into masterBuffer, making it its slave. The shape of tensor stored in the slave buffer padded by values
         * specified by leftPadding and rightPadding has to match the shape of the tensor that is allocated in the masterBuffer. The shape
         * dimensionality, data order and data type of the tensor stored in the slaveBuffer has to match the tensor stored in the masterBuffer.
         * Moving will propagated for all slave buffers of slaveBuffer. If slaveBuffer had a master before it will be replaced with masterBuffer (slave
         * buffer will be deleted from its slaveBuffers list). All previously applied paddings will be erased for slaveBuffer.
         *
         * @param slaveBuffer Buffer to be moved into masterBuffer
         * @param masterBuffer Buffer to be overallocated
         * @param leftPadding leftPadding Padding (left-top) between tensor to be allocated and the tensor contained by the given buffer - EXPRESSED IN WORDS
         * @param rightPadding rightPadding Padding (right-bottom) between tensor to be allocated and the tensor contained by the given buffer - EXPRESSED IN WORDS
         * @return BufferIterator Modified slaveBuffer, of the same offset and size that overlaps the memory space owned by the masterBuffer
         */
        BufferIterator move(BufferIterator slaveBuffer, BufferIterator masterBuffer, const std::vector<std::size_t>& leftPadding,
            const std::vector<std::size_t>& rightPadding);
        BufferIterator getTopMasterBuffer(BufferIterator t);

        bool deallocate(Data::TensorIterator tensor, std::size_t stageIdx);
        void deallocateAll(std::size_t stageIdx);

        void padLeft(BufferIterator buffer, const std::vector<std::size_t>& padding);
        void padRight(BufferIterator buffer, const std::vector<std::size_t>& padding);

        long long unsigned freeSpace(std::size_t stageIdx) const;
        long long unsigned usedSpace(std::size_t stageIdx) const;

        std::string toString() const override;
        //mv::json::Value toJSON() const override;

        BufferIterator bufferBegin(std::size_t stageIdx);
        BufferIterator bufferEnd(std::size_t stageIdx);
        BufferIterator getBuffer(std::size_t stageIdx, Data::TensorIterator tensor);

        bool iterable(std::size_t stageIdx);
        virtual std::string getLogID() const override;
        const std::string& getAllocatorName() const;
    };

}

#endif // MEMORY_ALLOCATOR_HPP_
