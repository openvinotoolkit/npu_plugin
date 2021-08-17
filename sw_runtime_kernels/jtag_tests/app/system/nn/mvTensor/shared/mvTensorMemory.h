// {% copyright %}
#ifndef MV_TENSOR_MEMORY_H_
#define MV_TENSOR_MEMORY_H_

#include <memory>
#include <vector>
#include <list>
#include "mvTensorConfig.h"
#include "mvTensorUtil.h"

namespace mv
{
    namespace tensor
    {
        namespace memory
        {
            template <typename T>
            struct cache_safe
            {
                union
                {
                    T value;
                    unsigned char padding_[util::round_up<MVTENSOR_CACHE_LINE_LENGTH>(sizeof(T))];
                };
            } __attribute__((aligned(MVTENSOR_CACHE_LINE_LENGTH)));

            struct cache_aligned_t {};
            extern cache_aligned_t cache_aligned;

            void *cache_aligned_new(size_t size);
            void cache_aligned_delete(void *p);

            struct cache_aligned_base
            {
                static void *operator new(size_t size);
                static void *operator new(size_t size, const std::nothrow_t &);
                static void *operator new[](size_t size);
                static void *operator new[](size_t size, const std::nothrow_t &);

                static void operator delete(void *p);
                static void operator delete(void *p, const std::nothrow_t &);
                static void operator delete[](void *p);
                static void operator delete[](void *p, const std::nothrow_t &);
            };

            template <typename T>
            struct cache_aligned_deleter
            {
                void operator ()(T *t)
                {
                    if (t != nullptr)
                    {
                        t->~T();
                        cache_aligned_delete(t);
                    }
                }
            };

            template <typename T>
            struct cache_aligned_deleter<T[]>
            {
                void operator ()(T *t)
                {
                    if (t != nullptr)
                    {
                        cache_aligned_delete(t);
                    }
                }
            };

            // A cache_aligned_deleter<T[]> specialization of cache_aligned_deleter should exist
            // but there's no way to know how many dtors to call there.
            //
            // delete [] t; would not take a (memory::cache_aligned) argument to deallocate using cache_aligned_delete
            // => For POD types don't use [] notation, they have no dtors.
            // => For non-POD types do not store arrays of values in cache_aligned_unique_ptr. Use cache_aligned_vector instead

            template <typename T>
            using cache_aligned_unique_ptr = std::unique_ptr<T, cache_aligned_deleter<T> >;

            template <typename T>
            using cache_aligned_arr_unique_ptr = std::unique_ptr<T[], cache_aligned_deleter<T[]> >;

            template <typename T>
            inline cache_aligned_arr_unique_ptr<T> get_cache_aligned_arr_unique_ptr(size_t size) {
                static_assert(std::is_default_constructible<
                                  typename std::remove_extent<T>::type>::value,
                              "Error: get_cache_aligned_arr_unique_ptr works only for default constructible types");
                static_assert(std::is_pod<typename std::remove_extent<T>::type>::value,
                                "Error: get_cache_aligned_arr_unique_ptr works only for pod types");
                return cache_aligned_arr_unique_ptr<T>(
                        reinterpret_cast<T*>(cache_aligned_new(size * sizeof(T))),
                        cache_aligned_deleter<T[] >()
                        );
            }
            template <typename T>
            class cache_aligned_allocator
            {
            public:
                typedef T value_type;

                typedef T* pointer;
                typedef const T* const_pointer;

                typedef T& reference;
                typedef const T& const_reference;

                typedef std::size_t size_t;
                typedef std::ptrdiff_t difference_type;

                pointer allocate(size_t n, const void * = nullptr)
                {
                    return reinterpret_cast<pointer>(cache_aligned_new(n * sizeof(value_type)));
                }

                void deallocate(pointer p, size_t)
                {
                    cache_aligned_delete(p);
                }
            };

            template <typename T>
            using cache_aligned_vector = std::vector<T, cache_aligned_allocator<T> >;

            template <typename T>
            using cache_aligned_list = std::list<T, cache_aligned_allocator<T> >;
        }
    }
}

void *operator new(size_t size, const mv::tensor::memory::cache_aligned_t &);
void *operator new[](size_t size, const mv::tensor::memory::cache_aligned_t &);

void operator delete(void *p, const mv::tensor::memory::cache_aligned_t &);
void operator delete[](void *p, const mv::tensor::memory::cache_aligned_t &);

#endif // MV_TENSOR_MEMORY_H_
