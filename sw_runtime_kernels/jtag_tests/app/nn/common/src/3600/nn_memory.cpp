/*
* {% copyright %}
*/
#include "nn_memory.h"
#include <nn_log.h>
#include <assert.h>
#include <new>

using namespace nn;

memory::cache_aligned_t memory::cache_aligned;

namespace
{
    void find_largest_block(size_t upper)
    {
        const size_t tolerance = 1 * 1024;
        size_t largest_size = 0;
        void *largest_block = nullptr;

        for (size_t lower = 0; lower + tolerance < upper;)
        {
            auto size = (upper + lower) / 2;
            void *block = malloc(size);

            if (block)
                largest_block = block, largest_size = lower = size;
            else
                upper = size;

            free(block);
        }

        nnLog(MVLOG_INFO, "Largest heap block @ %p has %u bytes", largest_block, largest_size);
    }
}

void nn::memory::print_heap_stats()
{
    find_largest_block(1 * 1024 * 1024 * 1024);
}

void *memory::cache_aligned_alloc(size_t alignment, size_t size)
{
    if (alignment < NN_CACHE_LINE_LENGTH)
        alignment = NN_CACHE_LINE_LENGTH;

    const size_t larger_size = math::round_up_power_of_2(NN_CACHE_LINE_LENGTH, size);
    void * const p = aligned_alloc(alignment, larger_size);

    nnLog(MVLOG_DEBUG, p ? "Allocated %u bytes at %p. Safe range: [%p-%p)" : "Couldn't allocate %u bytes!",
           size, p, p, reinterpret_cast<char *>(p) + larger_size);

    return p;
}

void memory::cache_aligned_free(void *p)
{
    const bool is_aligned = p == math::ptr_align_down<NN_CACHE_LINE_LENGTH>(p);
    nnLog(MVLOG_DEBUG, "Freeing %s%p", is_aligned ? "" : "non-aligned ", p);

    assert(is_aligned && "Freeing non-aligned pointer");
    free(p);
}

#ifdef NN_USE_MEMORY_MANAGER
#include <memManagerApi.h>

memory::managed_t memory::managed;

void *nn::memory::managed_alloc(size_t alignment, size_t size)
{
    if (alignment < NN_CACHE_LINE_LENGTH)
        alignment = NN_CACHE_LINE_LENGTH;

    size = math::round_up_power_of_2(NN_CACHE_LINE_LENGTH, size);
    void *p = MemMgrAlloc(size, DDR, alignment);

    nnLog(MVLOG_DEBUG,
            p ? "Allocated %u bytes at %p. with %dB alignment"
            : "Couldn't allocate %u bytes!",
            size, p, alignment);

    return p;
}

void nn::memory::managed_free(void *p)
{
    if (p)
    {
        nnLog(MVLOG_DEBUG, "Freeing %p", p);
        MemMgrFree(p);
    }
}

#endif
