#ifndef SHAPE_HPP_
#define SHAPE_HPP_

#include <vector>
#include <initializer_list>
#include "include/mcm/base/printable.hpp"
#include "include/mcm/base/exception/shape_error.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

namespace mv
{

    static const size_t IO_BATCH_DIMENSION = 0;
    static const size_t IO_WIDTH_DIMENSION = 1;
    static const size_t IO_HEIGHT_DIMENSION = 2;
    static const size_t IO_CHANNEL_DIMENSION = 3;

    static const size_t KERNEL_WIDTH = 0;
    static const size_t KERNEL_HEIGHT = 1;
    static const size_t KERNEL_INPUT_CHANNELS = 2;
    static const size_t KERNEL_OUTPUT_CHANNELS = 3;

    class Shape : public Printable, public LogSender
    {

        std::vector<std::size_t> dims_;

    public:

        Shape(std::initializer_list<std::size_t> dims);
        Shape(std::vector<std::size_t> dims);
        Shape(std::size_t ndims);
        Shape(const Shape& other);

        std::size_t ndims() const;
        std::size_t totalSize() const;
        std::size_t& operator[](int ndim);
        static Shape broadcast(const Shape& s1, const Shape& s2);
        static Shape augment(const Shape& s, std::size_t ndims);

        const std::size_t& operator[](int ndim) const;
        Shape& operator=(const Shape& other);
        bool operator==(const Shape& other) const;
        bool operator!=(const Shape& other) const;
        operator std::vector<std::size_t>() const;
        operator std::vector<unsigned>() const;

        std::string toString() const override;
        virtual std::string getLogID() const override;

    };

}

#endif // SHAPE_HPP_
