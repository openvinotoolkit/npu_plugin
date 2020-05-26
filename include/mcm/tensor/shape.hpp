#ifndef SHAPE_HPP_
#define SHAPE_HPP_

#include <vector>
#include <initializer_list>
#include "include/mcm/base/printable.hpp"
#include "include/mcm/base/exception/shape_error.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

namespace mv
{

    static const size_t IO_WIDTH_DIMENSION = 0;
    static const size_t IO_HEIGHT_DIMENSION = 1;
    static const size_t IO_CHANNEL_DIMENSION = 2;
    static const size_t IO_BATCH_DIMENSION = 3;

    static const size_t KERNEL_WIDTH = 0;
    static const size_t KERNEL_HEIGHT = 1;
    static const size_t KERNEL_INPUT_CHANNELS = 2;
    static const size_t KERNEL_OUTPUT_CHANNELS = 3;
    static const size_t KERNEL_WEIGHT_SETS = 0;

    static const size_t PADDING_LEFT = 0;
    static const size_t PADDING_RIGHT = 1;
    static const size_t PADDING_TOP = 2;
    static const size_t PADDING_BOT = 3;

    static const size_t STRIDE_HORIZONTAL = 0;
    static const size_t STRIDE_VERTICAL = 1;

    class Shape : public Printable, public LogSender
    {
    private:
        static const std::unordered_map<std::string, std::size_t> axis_;
        std::vector<std::size_t> dims_;

    public:

        static std::size_t getAxis(const std::string& axis);
        Shape(std::initializer_list<std::size_t> dims);
        Shape(std::vector<std::size_t> dims);
        Shape(std::size_t ndims);
        Shape(const Shape& other);
        Shape();

        std::size_t ndims() const;
        std::size_t totalSize() const;
        bool isFlat() const;
        std::size_t& operator[](int ndim);

        static Shape broadcast(const Shape& s1, const Shape& s2);
        static Shape augment(const Shape& s, std::size_t ndims);
        static Shape augment_major(const Shape& s, std::size_t ndims);

        const std::size_t& operator[](int ndim) const;
        const std::size_t& operator[](const std::string& ndim) const;

        Shape& operator=(const Shape& other);
        Shape operator/(const Shape& denum) const;
        Shape operator-(const Shape& subtrahed) const;
        Shape operator*(const Shape& multiplier) const;
        Shape operator+(const Shape& addend) const;

        bool operator==(const Shape& other) const;
        bool operator!=(const Shape& other) const;
        operator std::vector<std::size_t>() const;
        operator std::vector<unsigned>() const;

        std::string toString() const override;
        virtual std::string getLogID() const override;

    };

}

#endif // SHAPE_HPP_
