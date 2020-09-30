#pragma once

#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"
#include "include/mcm/tensor/order/order.hpp"

namespace mv {

class TensorInfo : public Printable, public LogSender {
    mv::Shape shape_{};
    mv::DType type_{};
    mv::Order order_{mv::Order::getZMajorID(4)};

public:
    TensorInfo(Shape shape, DType type, Order order)
        : shape_{std::move(shape)}, type_{std::move(type)}, order_{std::move(order)}
    {}

    TensorInfo() = default;

    const mv::Shape& shape() const { return shape_; }
    const mv::DType& type() const { return type_; }
    const mv::Order& order() const { return order_; }

    std::string toString() const override;
    std::string getLogID() const override;
};

} // namespace mv
