/*
* {% copyright %}
*/
#ifndef NN_MEMORY_ALLOC_H_
#define NN_MEMORY_ALLOC_H_

#include <nn_memory.h>
#include <nn_math.h>
#include <string.h>
#include <nn_log.h>

namespace nn
{
    namespace memory
    {

    // Round up the size for an allocation to be aligned to max(CACHE_LINE_LENGTH, struct alignment)
    template <typename T>
    size_t aligned_size(size_t count)
    {
        return math::round_up_power_of_2(std::max(NN_CACHE_LINE_LENGTH, alignof(T)), count * sizeof(T));
    }

    // Track allocations from a fixed sized buffer
    // Intentionally does not allow/handle freeing, the allocator can go out of scope and be
    // destroyed while the buffer is still used
    class LinearAllocator
    {
    public:
        LinearAllocator(unsigned char *base, size_t size) :
        ptr_(math::ptr_align_up<NN_CACHE_LINE_LENGTH>(base)),
        remaining_(size)
        {
            intptr_t delta = ptr_ - base;
            if (delta < remaining_)
                remaining_ -= delta;
            else
                remaining_ = 0;
        }

        template <typename T>
        T* allocate()
        {
            return allocate<T>(1);
        }

        template <typename T>
        T* allocate(size_t count)
        {
            size_t alignment = std::max(alignof(T), NN_CACHE_LINE_LENGTH);
            unsigned char *aligned_ptr = (unsigned char*) math::round_up_power_of_2(alignment, (size_t) ptr_);

            intptr_t alignment_delta = aligned_ptr - ptr_;
            int bytes = count * sizeof(T) + alignment_delta;

            if (bytes > remaining_)
            {
                nnLog(MVLOG_FATAL, "Out of space: %u bytes remaining, want %u", remaining_, bytes);
                return nullptr;
            }

            ptr_ = aligned_ptr + count * sizeof(T);
            remaining_ -= bytes;

            return reinterpret_cast<T*>(aligned_ptr);
        }

    private:
        unsigned char *ptr_;
        int remaining_;
    };

    // An allocatable array which is fixed in size at construction but allows some of the STL container operations
    template <typename T>
    class NN_CACHE_ALIGNED FixedVector
    {
    public:
        inline size_t size() const { return size_; }
        inline T* data() { return data_; }
        inline const T* data() const { return data_; }
        inline T& operator[](int index) { return data_[index]; }
        inline const T& operator[](int index) const { return data_[index]; }

        FixedVector() :
        size_(0),
        data_(nullptr)
        {
        }

        FixedVector(size_t size, LinearAllocator &alloc) :
        size_(size),
        data_(alloc.allocate<T>(size_))
        {
        }

        template <typename A>
        FixedVector(const std::vector<T, A> &vec, LinearAllocator &alloc):
        size_(vec.size()),
        data_(alloc.allocate<T>(size_))
        {
            std::uninitialized_copy_n(vec.begin(), size_, data_);
        }

    private:
        size_t size_;
        T* data_;
    };
    }
}

#endif // NN_MEMORY_ALLOC_H_
