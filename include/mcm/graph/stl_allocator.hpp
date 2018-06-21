#ifndef STL_ALLOCATOR_HPP_
#define STL_ALLOCATOR_HPP_

#include "include/mcm/graph/base_allocator.hpp"
#include "stl_smart_ptr.hpp"
#include "stl_set.hpp"
#include "stl_deque.hpp"
#include "stl_map.hpp"
#include "stl_vector.hpp"

namespace mv
{

    class stl_allocator : public base_allocator
    {

    public:


        template <class T>
        using access_ptr = stl_access_ptr<T>;

        template <class T>
        using owner_ptr = stl_owner_ptr<T>;

        template <class T>
        using opaque_ptr = stl_opaque_ptr<T>;

        template <class T, class T_comparator>
        using set = stl_set<T, T_comparator, stl_allocator>;

        template <class T>
        using deque = stl_deque<T>;

        template<class T_key, class T_value>
        using map = stl_map<T_key, T_value, stl_allocator>;

        template<class T>
        using vector = stl_vector<T, stl_allocator>;

        stl_allocator() noexcept
        {

        }

        template <class T, typename... Args>
        owner_ptr<T> make_owner(Args&&... args) const noexcept
        {

            try
            {
                return owner_ptr<T>(args...);
            }
            catch(std::exception &e)
            {
                return owner_ptr<T>();
            }

        }

        template <class T, typename... Args>
        opaque_ptr<T> make_opaque(Args&... args) const noexcept
        {

            try
            {
                return opaque_ptr<T>(args...);
            }
            catch(std::exception &e)
            {
                return opaque_ptr<T>();
            }

        }

        template <class T, class T_comparator>
        owner_ptr<set<T, T_comparator>> make_set() const noexcept
        {
            return make_owner<set<T, T_comparator>>(set<T, T_comparator>());
        }

        template <class T>
        owner_ptr<deque<T>> make_deque() const noexcept
        {
            return make_owner<deque<T>>(*this);
        }

        
    };

}


#endif // STL_ALLOCATOR_HPP_