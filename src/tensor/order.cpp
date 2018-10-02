#include "include/mcm/tensor/order.hpp"

const std::unordered_map<mv::OrderType, std::string, mv::OrderTypeHash> mv::Order::orderStrings_ =
{

    {OrderType::ColumnMajor, "ColumnMajor"},
    {OrderType::ColumnMajorPlanar, "ColumnMajorPlanar"},
    {OrderType::RowMajor, "RowMajor"},
    {OrderType::RowMajorPlanar, "RowMajorPlanar"},
    {OrderType::RowInterleaved, "RowInterleaved"}

};

void mv::Order::setFuncs_()
{

    switch (order_)
    {
        case OrderType::ColumnMajor:
            prevContiguousDimIdx_ = colMajPrevContiguousDimIdx_;
            nextContiguousDimIdx_ = colMajNextContiguousDimIdx_;
            firstContiguousDimIdx_ = colMajFirstContiguousDimIdx_;
            lastContiguousDimIdx_ = colMajLastContiguousDimIdx_;
            break;

        case OrderType::ColumnMajorPlanar:
            prevContiguousDimIdx_ = colMajPlanPrevContiguousDimIdx_;
            nextContiguousDimIdx_ = colMajPlanNextContiguousDimIdx_;
            firstContiguousDimIdx_ = colMajPlanFirstContiguousDimIdx_;
            lastContiguousDimIdx_ = colMajPlanLastContiguousDimIdx_;
            break;

        case OrderType::RowMajor:
            prevContiguousDimIdx_ = rowMajPrevContiguousDimIdx_;
            nextContiguousDimIdx_ = rowMajNextContiguousDimIdx_;
            firstContiguousDimIdx_ = rowMajFirstContiguousDimIdx_;
            lastContiguousDimIdx_ = rowMajLastContiguousDimIdx_;
            break;

        case OrderType::RowMajorPlanar:
            prevContiguousDimIdx_ = rowMajPlanPrevContiguousDimIdx_;
            nextContiguousDimIdx_ = rowMajPlanNextContiguousDimIdx_;
            firstContiguousDimIdx_ = rowMajPlanFirstContiguousDimIdx_;
            lastContiguousDimIdx_ = rowMajPlanLastContiguousDimIdx_;
            break;

        case OrderType::RowInterleaved:
            prevContiguousDimIdx_ = RowInterleaved_PrevContiguousDimIdx_;
            nextContiguousDimIdx_ = RowInterleaved_NextContiguousDimIdx_;
            firstContiguousDimIdx_ = RowInterleaved_FirstContiguousDimIdx_;
            lastContiguousDimIdx_ = RowInterleaved_LastContiguousDimIdx_;
            break;

    }

}

mv::Order::Order(OrderType value) :
order_(value)
{
    setFuncs_();
}

mv::Order::Order() :
Order(OrderType::ColumnMajor)
{

}

mv::Order::Order(const Order& other) :
Order(other.order_)
{

}

mv::Order::Order(const std::string& value) :
Order(
        [=]()->OrderType
        {
            for (auto &e : orderStrings_)
                if (e.second == value)
                    return e.first;
            throw OrderError(*this, "Invalid initialization - string value specified as " + value);
        }()
)
{

}

std::size_t mv::Order::subToInd(const Shape &s, const std::vector<std::size_t>& sub) const
{

    if (s.ndims() == 0)
        throw ShapeError(*this, "subToInd: Cannot compute subscripts for 0-dimensional shape");

    if (sub.size() != s.ndims())
        throw ShapeError(*this, "subToInd: Mismatch between subscript vector (length " + std::to_string(sub.size()) +
            ") and number of dimensions in shape (" + std::to_string(s.ndims()) + ")");

    unsigned currentMul = 1;
    unsigned currentResult = 0;

    for (int i = firstContiguousDimIdx_(s); i != -1; i = nextContiguousDimIdx_(s, i))
    {

        if (sub[i] >=  s[i])
            throw ShapeError(*this, "subToInd: Subscript " + std::to_string(sub[i]) + " exceeds the dimension " +
                std::to_string(s[i]));

        currentResult += currentMul * sub[i];
        currentMul *= s[i];

    }

    return currentResult;

}

std::vector<std::size_t> mv::Order::indToSub(const Shape &s, std::size_t idx) const
{

    if (s.ndims() == 0)
        throw ShapeError(*this, "indToSub: Cannot compute subscripts for 0-dimensional shape");

    std::vector<std::size_t> sub(s.ndims());
    sub[firstContiguousDimIdx_(s)] =  idx % s[firstContiguousDimIdx_(s)];
    int offset = -sub[firstContiguousDimIdx_(s)];
    int scale = s[firstContiguousDimIdx_(s)];
    for (int i = nextContiguousDimIdx_(s, firstContiguousDimIdx_(s)); i != -1; i = nextContiguousDimIdx_(s, i))
    {
        sub[i] = (idx + offset) / scale % s[i];
        offset -= sub[i] * scale;
        scale *= s[i];
    }

    return sub;

}

int mv::Order::previousContiguousDimensionIndex(const Shape& s, std::size_t dim) const
{
    return prevContiguousDimIdx_(s, dim);
}

int mv::Order::nextContiguousDimensionIndex(const Shape& s, std::size_t dim) const
{
    return nextContiguousDimIdx_(s, dim);
}

std::size_t mv::Order::firstContiguousDimensionIndex(const Shape &s) const
{
    return firstContiguousDimIdx_(s);
}

std::size_t mv::Order::lastContiguousDimensionIndex(const Shape &s) const
{
    return lastContiguousDimIdx_(s);
}

bool mv::Order::isLastContiguousDimensionIndex(const Shape &s, std::size_t index) const
{
    return index == lastContiguousDimIdx_(s);
}

bool mv::Order::isFirstContiguousDimensionIndex(const Shape &s, std::size_t index) const
{
    return index == firstContiguousDimIdx_(s);
}

std::string mv::Order::toString() const
{
    return orderStrings_.at(order_);
}

mv::Order& mv::Order::operator=(const Order& other)
{
    order_ = other.order_;
    setFuncs_();
    return *this;
}

mv::Order& mv::Order::operator=(const OrderType& other)
{
    order_ = other;
    setFuncs_();
    return *this;
}

bool mv::Order::operator==(const Order &other) const
{
    return order_ == other.order_;
}

bool mv::Order::operator==(const OrderType &other) const
{
    return order_ == other;
}

bool mv::Order::operator!=(const Order &other) const
{
    return !operator==(other);
}

bool mv::Order::operator!=(const OrderType &other) const
{
    return !operator==(other);
}

mv::Order::operator OrderType() const
{
    return order_;
}

std::string mv::Order::getLogID() const
{
    return toString();
}