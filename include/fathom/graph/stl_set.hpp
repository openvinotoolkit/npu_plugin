#ifndef STL_SET_HPP_
#define STL_SET_HPP_

#include <set>
#include "pair.hpp"
#include "base_allocator.hpp"

namespace mv
{

    template <class T, class T_comparator> 
    class stl_set
    {

        std::set<T, T_comparator> stl_set_;
        const base_allocator &allocator_;

    public:

        using iterator = typename std::set<T, T_comparator>::iterator;
        using reverse_iterator = typename std::set<T, T_comparator>::reverse_iterator;

        stl_set(const base_allocator &allocator) : allocator_(allocator)
        {

        }

        stl_set(const stl_set &other):
        stl_set_(other.stl_set_),
        allocator_(other.allocator_)
        {

        }

        pair<iterator, bool> insert(const T &value) noexcept
        {

            try
            {
                auto result = stl_set_.insert(value);
                return pair<iterator, bool>(result.first, result.second);
            }
            catch (std::exception &e)
            {
                return pair<iterator, bool>(stl_set_.end(), false);
            }

        }
        
        iterator insert(iterator hint, const T &value) noexcept
        {

            try
            {
                return stl_set_.insert(hint, value);
            }
            catch (std::exception &e)
            {
                return stl_set_.end();
            }

        }

        void erase(const T &pos) noexcept
        {

            try
            {
                stl_set_.erase(pos);
            }
            catch (std::exception &e)
            {

            }
            
        }

        iterator erase(iterator pos) noexcept
        {

            try
            {
                return stl_set_.erase(pos);
            }
            catch (std::exception &e)
            {
                return stl_set_.end();
            }
            
        }

        iterator find(const T& key) noexcept
        {

            try
            {
                return stl_set_.find(key);
            }
            catch (std::exception &e)
            {
                return stl_set_.end();
            }

        }

        iterator begin() const noexcept
        {
            return stl_set_.begin();
        }

        iterator end() const noexcept
        {
            return stl_set_.end();
        }

        reverse_iterator rbegin() const noexcept
        {
            return stl_set_.rbegin();
        }

        reverse_iterator rend() const noexcept
        {
            return stl_set_.rend();
        }

        typename std::set<T>::size_type size() const noexcept
        {
            return stl_set_.size();
        }

        bool empty() const noexcept
        {
            return stl_set_.empty();
        }

        void clear() noexcept
        {
            stl_set_.clear();
        }

        stl_set& operator=(const stl_set &other) noexcept
        {
            stl_set_ = other.stl_set_;
            return *this;
        }

    };

}

#endif // STL_SET_HPP_