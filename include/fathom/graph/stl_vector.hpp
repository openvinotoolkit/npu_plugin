#ifndef STL_VECTOR_HPP_
#define STL_VECTOR_HPP_

#include <vector>
#include "base_allocator.hpp"

namespace mv
{

    template <class T, class T_allocator, class T_size = uint32_t> 
    class stl_vector
    {

        std::vector<T> stl_vector_;
        const T_allocator allocator_;

    public:

        using iterator = typename std::vector<T>::iterator;
        using reverse_iterator = typename std::vector<T>::reverse_iterator;

        stl_vector()
        {

        }

        stl_vector(T_size size) :
        stl_vector_(size)
        {

        }

        stl_vector(T_size size, const T &value) :
        stl_vector_(size, value)
        {

        }

        stl_vector(const stl_vector &other) :
        stl_vector_(other.stl_vector_),
        allocator_(other.allocator_)
        {

        }

        template <T_size N>
        stl_vector(T (&arr)[N]) : 
        stl_vector_(arr, arr + sizeof(arr) / sizeof(arr[0]))
        {

        }

        stl_vector(T *ptr, T_size size) :
        stl_vector_(ptr, ptr + size)
        {

        }

        iterator insert(iterator pos, const T &value) noexcept
        {

            try
            {
                return stl_vector_.insert(pos, value);
            }
            catch (std::exception &e)
            {
                return stl_vector_.end();
            }

        }

        iterator emplace(iterator pos, const T &value)
        {
            try
            {
                return stl_vector_.emplace(pos, value);
            }
            catch (std::exception &e)
            {
                return stl_vector_.end();
            }
        }

        iterator erase(iterator pos) noexcept
        {

            try
            {
                return stl_vector_.erase(pos);
            }
            catch (std::exception &e)
            {
                return stl_vector_.end();
            }
            
        }

        bool push_back(const T &value) noexcept
        {
            try 
            {
                stl_vector_.push_back(value);
                return true;
            }
            catch (std::exception &e)
            {
                return false;
            }
        }

        void pop_back() noexcept
        {
            stl_vector_.pop_back();
        }

        iterator begin() noexcept
        {
            return stl_vector_.begin();
        }

        iterator end() noexcept
        {
            return stl_vector_.end();
        }

        reverse_iterator rbegin() noexcept
        {
            return stl_vector_.rbegin();
        }

        reverse_iterator rend() noexcept
        {
            return stl_vector_.rend();
        }

        T_size size() const noexcept
        {
            return stl_vector_.size();
        }

        bool empty() const noexcept
        {
            return stl_vector_.empty();
        }

        void clear() noexcept
        {
            stl_vector_.clear();
        }

        stl_vector& operator=(const stl_vector &other) noexcept
        {
            stl_vector_ = other.stl_vector_;
            return *this;
        }

        bool operator==(const stl_vector &other) const noexcept
        {
            return stl_vector_ == other.stl_vector_;
        }

        bool operator!=(const stl_vector &other) const noexcept
        {
            return stl_vector_ != other.stl_vector_;
        }

        T& operator[](T_size pos)
        {
            return stl_vector_[pos];
        }

    };

}

#endif // STL_VECTOR_HPP_