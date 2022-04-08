/*
* {% copyright %}
*/
#ifndef NN_MEMORY_H_
#define NN_MEMORY_H_

#include <memory>
#include <vector>
#include <nn_math.h>

#define NN_CACHE_LINE_LENGTH (64u)
#define NN_CACHE_ALIGNED alignas(NN_CACHE_LINE_LENGTH)

namespace nn
{
    namespace memory
    {
        namespace helper
        {
            template <void *(new_impl)(std::size_t)>
            void *throwing_new(std::size_t size)
            {
                while (true)
                {
                    void * const p = new_impl(size);

                    if (p != nullptr)
                        return p;

                    if (auto new_handler = std::get_new_handler())
                        new_handler();
                    else
                        exit(-1);
                }
            }

            template <typename T, void (free_impl)(void *)>
            struct deleter
            {
                void operator ()(T *t)
                {
                    if (t != nullptr)
                    {
                        t->~T();
                        free_impl(t);
                    }
                }
            };

            template <typename T, void *(alloc_impl)(std::size_t), void (free_impl)(void *)>
            class allocator
            {
                typedef allocator<T, alloc_impl, free_impl> this_type;

            public:
                typedef T value_type;

                typedef T* pointer;
                typedef const T* const_pointer;

                typedef T& reference;
                typedef const T& const_reference;

                typedef std::size_t size_t;
                typedef std::ptrdiff_t difference_type;

                template< class U > struct rebind { typedef allocator<U, alloc_impl, free_impl> other; };

                pointer allocate(size_t n, const void * = nullptr)
                {
                    return reinterpret_cast<pointer>(helper::throwing_new<alloc_impl>(n * sizeof(value_type)));
                }

                void deallocate(pointer p, size_t)
                {
                    free_impl(p);
                }

                bool operator ==(const this_type &rhs) const
                {
                    return &rhs == this;
                }

                bool operator !=(const this_type &rhs) const
                {
                    return &rhs != this;
                }
            };

            template <void *(alloc_impl)(std::size_t), void (free_impl)(void *)>
            struct base
            {
                static inline void *operator new(size_t size)
                {
                    return helper::throwing_new<alloc_impl>(size);
                }

                static inline void *operator new(size_t size, const std::nothrow_t &)
                {
                    return alloc_impl(size);
                }

                static inline void *operator new[](size_t size)
                {
                    return helper::throwing_new<alloc_impl>(size);
                }

                static inline void *operator new[](size_t size, const std::nothrow_t &)
                {
                    return alloc_impl(size);
                }

                static inline void operator delete(void *p)
                {
                    free_impl(p);
                }

                static inline void operator delete(void *p, const std::nothrow_t &)
                {
                    free_impl(p);
                }

                static inline void operator delete[](void *p)
                {
                    free_impl(p);
                }

                static inline void operator delete[](void *p, const std::nothrow_t &)
                {
                    free_impl(p);
                }
            };
        }

        void print_heap_stats();

        struct cache_aligned_t {};
        extern cache_aligned_t cache_aligned;

        void *cache_aligned_alloc(size_t alignment, size_t size);

        template <unsigned Alignment = NN_CACHE_LINE_LENGTH>
        inline void *cache_aligned_alloc(size_t size)
        {
            static_assert(math::is_power_of_2<Alignment>::value, "Alignment must be a power of 2");
            return cache_aligned_alloc(Alignment, size);
        }

        void cache_aligned_free(void *p);

        typedef helper::base<&cache_aligned_alloc, &cache_aligned_free> cache_aligned_base;

        // A cache_aligned_deleter<T[]> specialization of cache_aligned_deleter should exist
        // but there's no way to know how many dtors to call there.
        //
        // delete [] t; would not take a (memory::cache_aligned) argument to deallocate using cache_aligned_free
        // => For POD types don't use [] notation, they have no dtors.
        // => For non-POD types do not store arrays of values in cache_aligned_unique_ptr. Use cache_aligned_vector instead

        template <typename T>
        using cache_aligned_deleter = helper::deleter<T, &cache_aligned_free>;

        template <typename T>
        using cache_aligned_unique_ptr = std::unique_ptr<T, cache_aligned_deleter<T> >;

        template <typename T>
        using cache_aligned_allocator = helper::allocator<T, &cache_aligned_alloc, &cache_aligned_free>;

        template <typename T>
        using cache_aligned_vector = std::vector<T, cache_aligned_allocator<T> >;

#ifdef NN_USE_MEMORY_MANAGER
        struct shared_t {};
        extern shared_t shared;

        void *shared_alloc(size_t alignment, size_t size);

        template <unsigned Alignment = NN_CACHE_LINE_LENGTH>
        inline void *shared_alloc(size_t size)
        {
            static_assert(math::is_power_of_2<Alignment>::value, "Alignment must be a power of 2");
            return shared_alloc(Alignment, size);
        }

        void shared_free(void *p);

        typedef helper::base<&shared_alloc, &shared_free> shared_base;

        template <typename T>
        using shared_deleter = helper::deleter<T, &shared_free>;

        template <typename T>
        using shared_unique_ptr = std::unique_ptr<T, shared_deleter<T> >;

        template <typename T>
        using shared_allocator = helper::allocator<T, &shared_alloc, &shared_free>;

        template <typename T>
        using shared_vector = std::vector<T, shared_allocator<T> >;
#else
#define shared_t cache_aligned_t
#define shared cache_aligned

#define shared_alloc cache_aligned_alloc
#define shared_free cache_aligned_free

#define shared_base cache_aligned_base

#define shared_deleter cache_aligned_deleter
#define shared_unique_ptr cache_aligned_unique_ptr
#define shared_allocator cache_aligned_allocator
#define shared_vector cache_aligned_vector
#endif
    }
}

inline void *operator new(size_t size, const nn::memory::cache_aligned_t &)
{
    return nn::memory::helper::throwing_new<&nn::memory::cache_aligned_alloc>(size);
}

inline void *operator new[](size_t size, const nn::memory::cache_aligned_t &)
{
    return nn::memory::helper::throwing_new<&nn::memory::cache_aligned_alloc>(size);
}

void operator delete(void *p, const nn::memory::cache_aligned_t &);
void operator delete[](void *p, const nn::memory::cache_aligned_t &);

inline void *operator new(size_t size, const nn::memory::cache_aligned_t &, const std::nothrow_t &)
{
    return nn::memory::cache_aligned_alloc(size);
}

inline void *operator new[](size_t size, const nn::memory::cache_aligned_t &, const std::nothrow_t &)
{
    return nn::memory::cache_aligned_alloc(size);
}

void operator delete(void *p, const nn::memory::cache_aligned_t &, const std::nothrow_t &);
void operator delete[](void *p, const nn::memory::cache_aligned_t &, const std::nothrow_t &);

#ifdef NN_USE_MEMORY_MANAGER
inline void *operator new(size_t size, const nn::memory::shared_t &)
{
    return nn::memory::helper::throwing_new<&nn::memory::shared_alloc>(size);
}

inline void *operator new[](size_t size, const nn::memory::shared_t &)
{
    return nn::memory::helper::throwing_new<&nn::memory::shared_alloc>(size);
}

void operator delete(void *p, const nn::memory::shared_t &);
void operator delete[](void *p, const nn::memory::shared_t &);

inline void *operator new(size_t size, const nn::memory::shared_t &, const std::nothrow_t &)
{
    return nn::memory::shared_alloc(size);
}

inline void *operator new[](size_t size, const nn::memory::shared_t &, const std::nothrow_t &)
{
    return nn::memory::shared_alloc(size);
}

void operator delete(void *p, const nn::memory::shared_t &, const std::nothrow_t &);
void operator delete[](void *p, const nn::memory::shared_t &, const std::nothrow_t &);

#endif

#endif // NN_MEMORY_H_
