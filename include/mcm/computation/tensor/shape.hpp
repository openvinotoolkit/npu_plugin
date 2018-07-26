#ifndef SHAPE_HPP_
#define SHAPE_HPP_

#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/base/printable.hpp"
#include "include/mcm/graph/static_vector.hpp"

namespace mv
{

    class Shape : public Printable
    {

        static_vector<dim_type, byte_type, max_ndims> dims_;

        void addDim(dim_type newDim);

        template<typename... Dims>
        void addDim(dim_type newDim, Dims... newDims)
        {
            dims_.push_back(newDim);
            addDim(newDims...);
        }

    public:

        template<typename... Dims>
        Shape(Dims... dims)
        {
            addDim(dims...);
        }

        Shape(const Shape& other);
        Shape(byte_type n);
        Shape();
        byte_type ndims() const;
        
        unsigned_type totalSize() const;
        dim_type& operator[](int_type ndim);
        const dim_type& operator[](int_type ndim) const;
        Shape& operator=(const Shape& other);
        bool operator==(const Shape& other) const;
        bool operator!=(const Shape& other) const;
        string toString() const;

        static Shape broadcast(const Shape& s1, const Shape& s2);
        static Shape augment(const Shape& s, byte_type ndims);

    };

}

#endif // SHAPE_HPP_
