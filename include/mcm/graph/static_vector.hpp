#ifndef STATIC_VECTOR_HPP_
#define STATIC_VECTOR_HPP_

#include <cstdint>
#include <assert.h>
#include <vector>

namespace mv
{

    template <class T, class T_size, T_size max_length>
    class static_vector  
    {
        T values_[max_length];
        T_size length_;

        template<typename T_value>
        inline void assign_(T_size& idx, T_value value)
        {
            values_[idx++] = value;
        }

        template<typename T_value, typename... Values>
        inline void assign_(T_size& idx, T_value value, Values... values)
        {
            values_[idx++] = value;
            assign_(idx, values...);
        }

    public:

        template<typename... Values>
        static_vector(Values... values) :
        length_(0)
        {
            static_assert(sizeof...(Values) <= max_length, "Number of values exceeds the maximum length of a static vector");
            assign_(length_, values...);
        }

        static_vector(const std::vector<T>& other) :
        length_(other.size())
        {
            assert(length_ < max_length && "Length of a static vector exceeds the maximal value");
            for (unsigned i = 0; i < length_; ++i)
                values_[i] = other[i];
        }

        static_vector(T_size length) :
        length_(length)
        {
            assert(length <= max_length && "Length of a static vector exceeds the maximal value");
        }

        static_vector() :
        length_(0)
        {

        }

        static_vector(const static_vector& other) :
        length_(other.length_)
        {
            for (T_size i = 0; i < other.length_; ++i)
                values_[i] = other.values_[i];
        }

        T_size length() const
        {
            return length_;
        }

        T& at(T_size idx)
        {
            assert(idx < length_ && "Index of value exceeds the number of elements of a static vector");
            return values_[idx];
        }

        const T& at(T_size idx) const
        {
            assert(idx < length_ && "Index of value exceeds the number of elements of a static vector");
            return values_[idx];
        }

        bool push_back(const T& value)
        {

            if (length_ == max_length)
                return false;

            values_[length_++] = value;
            return true;

        }

        void clear()
        {
            length_ = 0;
        }

        bool erase(T_size idx)
        {

            if (idx >= length_)
                return false;

            for (T_size i = idx; i < length_ - 1; ++i)
                values_[i] = values_[i + 1];

            --length_;
            return true;

        }

        T& operator[](T_size idx)
        {
            return at(idx);
        }

        const T& operator[](T_size idx) const
        {
            return at(idx);
        }

        static_vector& operator=(const static_vector& other)
        {
            for (T_size i = 0; i < other.length_; ++i)
                values_[i] = other.values_[i];
            length_ = other.length_;
            return *this;
        }

        bool operator==(const static_vector& other) const
        {
            
            if (length_ != other.length_)
                return false;

            for (T_size i = 0; i < length_; ++i)
                if (values_[i] != other.values_[i])
                    return false;

            return true;

        }

    };

}

#endif // STATIC_VECTOR_HPP_
