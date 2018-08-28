#ifndef SHAPE_HPP_
#define SHAPE_HPP_

#include <vector>
#include <initializer_list>
#include "include/mcm/computation/model/types.hpp"
#include "include/mcm/base/printable.hpp"
#include "include/mcm/base/jsonable.hpp"

namespace mv
{

    class Shape : public Printable, public Jsonable
    {

        std::vector<std::size_t> dims_;

    public:

        Shape(std::initializer_list<std::size_t> dims);
        Shape(std::size_t ndims);
        Shape(json::Value &o);
        Shape(const Shape& other);
        Shape();

        std::size_t ndims() const;
        std::size_t totalSize() const;
        std::size_t& operator[](int ndim);
        static Shape broadcast(const Shape& s1, const Shape& s2);
        static Shape augment(const Shape& s, std::size_t ndims);

        const std::size_t& operator[](int ndim) const;
        Shape& operator=(const Shape& other);
        bool operator==(const Shape& other) const;
        bool operator!=(const Shape& other) const;

        std::string toString() const;
        json::Value toJsonValue() const;
        
    };

}

#endif // SHAPE_HPP_
