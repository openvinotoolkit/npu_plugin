#ifndef MV_TENSOR_ORDER_HPP_
#define MV_TENSOR_ORDER_HPP_

#include <string>
#include <unordered_map>
#include <functional>
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/base/exception/order_error.hpp"
#include "include/mcm/base/exception/shape_error.hpp"

namespace mv
{

    enum class OrderType
    {
        ColumnMajor,
        ColumnMajorPlanar,
        RowMajor,
        RowMajorPlanar,
        RowInterleaved
    };

    struct OrderTypeHash
    {
        template <typename T>
        std::size_t operator()(T t) const
        {
            return static_cast<std::size_t>(t);
        }
    };

    class Order : public LogSender
    {

    private:

        static const std::unordered_map<OrderType, std::string, OrderTypeHash> orderStrings_;

        static const std::function<int(const Shape&, std::size_t)> colMajPrevContiguousDimIdx_;
        static const std::function<int(const Shape&, std::size_t)> colMajNextContiguousDimIdx_;
        static const std::function<std::size_t(const Shape&)> colMajFirstContiguousDimIdx_;
        static const std::function<std::size_t(const Shape&)> colMajLastContiguousDimIdx_;

        static const std::function<int(const Shape&, std::size_t)> colMajPlanPrevContiguousDimIdx_;
        static const std::function<int(const Shape&, std::size_t)> colMajPlanNextContiguousDimIdx_;
        static const std::function<std::size_t(const Shape&)> colMajPlanFirstContiguousDimIdx_;
        static const std::function<std::size_t(const Shape&)> colMajPlanLastContiguousDimIdx_;

        static const std::function<int(const Shape&, std::size_t)> rowMajPrevContiguousDimIdx_;
        static const std::function<int(const Shape&, std::size_t)> rowMajNextContiguousDimIdx_;
        static const std::function<std::size_t(const Shape&)> rowMajFirstContiguousDimIdx_;
        static const std::function<std::size_t(const Shape&)> rowMajLastContiguousDimIdx_;

        static const std::function<int(const Shape&, std::size_t)> rowMajPlanPrevContiguousDimIdx_;
        static const std::function<int(const Shape&, std::size_t)> rowMajPlanNextContiguousDimIdx_;
        static const std::function<std::size_t(const Shape&)> rowMajPlanFirstContiguousDimIdx_;
        static const std::function<std::size_t(const Shape&)> rowMajPlanLastContiguousDimIdx_;

        static const std::function<int(const Shape&, std::size_t)> RowInterleaved_PrevContiguousDimIdx_;
        static const std::function<int(const Shape&, std::size_t)> RowInterleaved_NextContiguousDimIdx_;
        static const std::function<std::size_t(const Shape&)> RowInterleaved_FirstContiguousDimIdx_;
        static const std::function<std::size_t(const Shape&)> RowInterleaved_LastContiguousDimIdx_;

        std::function<int(const Shape&, std::size_t)> prevContiguousDimIdx_;
        std::function<int(const Shape&, std::size_t)> nextContiguousDimIdx_;
        std::function<std::size_t(const Shape&)> firstContiguousDimIdx_;
        std::function<std::size_t(const Shape&)> lastContiguousDimIdx_;

        void setFuncs_();

        OrderType order_;

    public:

        Order();
        Order(OrderType value);
        Order(const Order& other);
        Order(const std::string& value);

        int previousContiguousDimensionIndex(const Shape& s, std::size_t dim) const;
        int nextContiguousDimensionIndex(const Shape& s, std::size_t dim) const;
        std::size_t firstContiguousDimensionIndex(const Shape &s) const;
        std::size_t lastContiguousDimensionIndex(const Shape &s) const;

        bool isFirstContiguousDimensionIndex(const Shape &s, std::size_t index) const;
        bool isLastContiguousDimensionIndex(const Shape &s, std::size_t index) const;

        std::size_t subToInd(const Shape &s, const std::vector<std::size_t>& sub) const;
        std::vector<std::size_t> indToSub(const Shape &s, std::size_t idx) const;

        std::string toString() const;

        Order& operator=(const Order& other);
        Order& operator=(const OrderType& other);
        bool operator==(const Order& other) const;
        bool operator==(const OrderType& other) const;
        bool operator!=(const Order& other) const;
        bool operator!=(const OrderType& other) const;
        operator OrderType() const;

        std::string getLogID() const override;

    };

}

#endif // MV_TENSOR_ORDER_HPP_