#ifndef SHAPE_HPP_
#define SHAPE_HPP_

#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/base/printable.hpp"
#include "include/mcm/graph/static_vector.hpp"
#include "include/mcm/base/jsonable.hpp"

namespace mv
{

    class Shape : public Printable, public Jsonable
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

        Shape(json::Value &o);
        Shape(const Shape& other);
        Shape(byte_type n);
        Shape();
        byte_type ndims() const;
        dim_type& dim(byte_type ndim);
        dim_type dim(byte_type ndim) const;
        unsigned_type totalSize() const;
        dim_type& operator[](int_type ndim);
        dim_type operator[](int_type ndim) const;
        Shape& operator=(const Shape& other);
        bool operator==(const Shape& other) const;
        bool operator!=(const Shape& other) const;
        string toString() const;
        mv::json::Value toJsonValue() const;

        static Shape broadcast(const Shape& s1, const Shape& s2);
        static Shape augment(const Shape& s, byte_type ndims);

    };

}

#endif // SHAPE_HPP_
