/*
* {% copyright %}
*/
#ifndef NN_CACHE_H_
#define NN_CACHE_H_

#include <rtems/rtems/cache.h>
#include <OsCommon.h>
#include <vector>
#include <nn_memory.h>
#include <nn_memory_alloc.h>

namespace nn
{
    namespace cache
    {
        inline void invalidate(const void *p, unsigned int bytes)
        {
            rtems_cache_invalidate_multiple_data_lines(p, bytes);
        }

        inline void flush(const void *p, unsigned int bytes)
        {
            rtems_cache_flush_multiple_data_lines(p, bytes);
        }

        template <typename T>
        void invalidate(const T &t)
        {
            rtems_cache_invalidate_multiple_data_lines(&t, sizeof(T));
        }

        template <typename T>
        void flush(const T &t)
        {
            rtems_cache_flush_multiple_data_lines(&t, sizeof(T));
        }

        template <typename T, typename A>
        void invalidate(const std::vector<T, A> &v)
        {
            rtems_cache_invalidate_multiple_data_lines(v.data(), v.size() * sizeof(T));
        }

        template <typename T, typename A>
        void flush(const std::vector<T, A> &v)
        {
            rtems_cache_flush_multiple_data_lines(v.data(), v.size() * sizeof(T));
        }

        template <typename T>
        void invalidate(const memory::FixedVector<T> &arr)
        {
            rtems_cache_invalidate_multiple_data_lines(arr.data(), arr.size() * sizeof(T));
        }

        template <typename T>
        void flush(const memory::FixedVector<T> &arr)
        {
            rtems_cache_flush_multiple_data_lines(arr.data(), arr.size() * sizeof(T));
        }

        template <typename T>
        void invalidate(const memory::cache_aligned_vector<T> &v)
        {
            rtems_cache_invalidate_multiple_data_lines(v.data(), math::round_up<NN_CACHE_LINE_LENGTH>(v.size() * sizeof(T)));
        }

        template <typename T>
        void flush(const memory::cache_aligned_vector<T> &v)
        {
            rtems_cache_flush_multiple_data_lines(v.data(), math::round_up<NN_CACHE_LINE_LENGTH>(v.size() * sizeof(T)));
        }

#ifdef NN_USE_MEMORY_MANAGER
        template <typename T>
        void invalidate(const memory::shared_vector<T> &v)
        {
            rtems_cache_invalidate_multiple_data_lines(v.data(), math::round_up<NN_CACHE_LINE_LENGTH>(v.size() * sizeof(T)));
        }

        template <typename T>
        void flush(const memory::shared_vector<T> &v)
        {
            rtems_cache_flush_multiple_data_lines(v.data(), math::round_up<NN_CACHE_LINE_LENGTH>(v.size() * sizeof(T)));
        }
#endif
    }
}

#endif /* NN_CACHE_H_ */
