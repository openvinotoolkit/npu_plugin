#ifndef STL_DEQUE_HPP_
#define STL_DEQUE_HPP_

#include <deque>
#include "base_allocator.hpp"

namespace mv
{

    template <class T>
    class stl_deque
    {
        
        std::deque<T> stl_deque_;
        const base_allocator &allocator_;

    public:

        stl_deque(const base_allocator &allocator) noexcept : allocator_(allocator) 
        {

        }

        stl_deque(const stl_deque &other):
        stl_deque_(other.stl_deque_),
        allocator_(other.allocator_)
        {

        }

        bool push_front(const T &value) noexcept
        {

            try
            {
                stl_deque_.push_front(value);
                return true;
            }
            catch (std::exception &e)
            {
                return false;
            }

        }

        bool push_back(const T &value) noexcept
        {

            try
            {
                stl_deque_.push_back(value);
                return true;
            }
            catch (std::exception &e)
            {
                return false;
            }

        }

        void pop_front() noexcept
        {
            stl_deque_.pop_front();
        }

        void pop_back() noexcept
        {
            stl_deque_.pop_back();
        }

        T& front() noexcept
        {
            return stl_deque_.front();
        }

        T& back() noexcept
        {
            return stl_deque_.back();
        }

        bool empty() const noexcept
        {
            return stl_deque_.empty();
        }

        void clear() noexcept
        {
            stl_deque_.clear();
        }

        stl_deque& operator=(const stl_deque &other) noexcept
        {
            stl_deque_ = other.stl_deque_;
            allocator_ = other.allocator_;
            return *this;
        }

    };

}

#endif // STL_DEQUE_HPP_