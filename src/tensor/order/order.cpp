#include "include/mcm/tensor/order/order.hpp"
#include <set>

const std::unordered_map<std::size_t, std::string> mv::Order::rowMajorID_ =
{
    {1, "W"},
    {2, "WH"},
    {3, "WHC"},
    {4, "WHCN"},
    {5, "WHCNT"}
};

const std::unordered_map<std::size_t, std::string> mv::Order::colMajorID_ =
{
    {1, "W"},
    {2, "HW"},
    {3, "CHW"},
    {4, "NCHW"},
    {5, "TNCHW"}
};

const std::unordered_map<std::size_t, std::string> mv::Order::colMajorPlanarID_ =
{
    {1, "W"},
    {2, "WH"},
    {3, "CWH"},
    {4, "NCWH"},
    {5, "TNCWH"}
};

const std::unordered_map<std::size_t, std::string> mv::Order::rowMajorPlanarID_ =
{
    {1, "W"},
    {2, "HW"},
    {3, "HWC"},
    {4, "HWCN"},
    {5, "HWCNT"}
};

const std::unordered_map<std::size_t, std::string> mv::Order::ZMajorID_ =
{
    {4, "NHWC"}
};

mv::Order::Order(const std::string& value)
   :Order([this, value]()->Order
    {

        if(!OrderRegistry::checkOrder(value))
            throw OrderError(*this, "Invalid string passed for Order construction " + value);

        return Order(OrderRegistry::getContVector(value), value);
    }())
{

}

mv::Order::Order(const std::vector<std::size_t>& contVectorParam, const std::string& contVectorStrParam)
    :contVector_(contVectorParam),
     contVectorStr_(contVectorStrParam)
{

}

const std::vector<std::size_t>& mv::Order::getContiguityVector()
{
    return contVector_;
}

std::size_t mv::Order::subToInd(const Shape &s, const std::vector<std::size_t>& sub) const
{

    //No 0-dimensional shapes
    if (s.ndims() == 0)
        throw ShapeError(*this, "subToInd: Cannot compute subscripts for 0-dimensional shape");

    //No shapes bigger than dimension supported
    if (s.ndims() != contVector_.size())
        throw ShapeError(*this, "subToInd: Mismatch between number of dimensions in shape ("
         + std::to_string(s.ndims()) + ") and dimensions supported by this mv::Order " + std::to_string(contVector_.size()));

    //If shape is correct, also sub has to be correct
    if (sub.size() != s.ndims())
        throw ShapeError(*this, "subToInd: Mismatch between subscript vector (length " + std::to_string(sub.size()) +
            ") and number of dimensions in shape (" + std::to_string(s.ndims()) + ")");

    unsigned currentMul = 1;
    unsigned currentResult = 0;

    for (unsigned i = 0; i < contVector_.size(); ++i)
    {

        if (sub[contVector_[i]] >=  s[contVector_[i]])
            throw ShapeError(*this, "subToInd: Subscript " + std::to_string(sub[contVector_[i]]) + " exceeds the dimension " +
                std::to_string(s[contVector_[i]]));

        currentResult += currentMul * sub[contVector_[i]];
        currentMul *= s[contVector_[i]];

    }

    return currentResult;

}

std::vector<std::size_t> mv::Order::indToSub(const Shape &s, std::size_t idx) const
{

    if (s.ndims() == 0)
        throw ShapeError(*this, "indToSub: Cannot compute subscripts for 0-dimensional shape");

    std::vector<std::size_t> sub(s.ndims());
    sub[contVector_[0]] =  idx % s[contVector_[0]];
    int offset = -sub[contVector_[0]];
    int scale = s[contVector_[0]];
    for (unsigned i = 1; i < contVector_.size(); ++i)
    {
        sub[contVector_[i]] = (idx + offset) / scale % s[contVector_[i]];
        offset -= sub[contVector_[i]] * scale;
        scale *= s[contVector_[i]];
    }

    return sub;

}


std::vector<unsigned> mv::Order::computeWordStrides(const Shape &shape) const
{
    unsigned n = shape.ndims();
    std::vector<unsigned> realStrides(n, 1);

    for(unsigned i = 1; i < n; ++i)
        realStrides[contVector_[i]] = realStrides[contVector_[i-1]] * shape[contVector_[i-1]];

    return realStrides;
}


std::vector<unsigned> mv::Order::computeByteStrides(const Shape &s, unsigned dataSize) const
{
    std::vector<unsigned> toReturn(computeWordStrides(s));
    std::transform(toReturn.begin(), toReturn.end(), toReturn.begin(), [dataSize](unsigned n) -> unsigned {return n * dataSize;});
    return toReturn;
}


//Read only access to dimensions
std::size_t mv::Order::operator[](std::size_t idx) const
{
    return contVector_[idx];
}

bool mv::Order::operator!=(const mv::Order& other) const
{
    return contVector_ != other.contVector_;
}

bool mv::Order::operator==(const mv::Order& other) const
{
    return contVector_ == other.contVector_;
}

std::size_t mv::Order::size() const
{
    return contVector_.size();
}

mv::Order::Order(const mv::Order& other)
    :contVector_(other.contVector_),
     contVectorStr_(other.contVectorStr_)
{

}

mv::Order& mv::Order::operator=(const mv::Order& other)
{
    contVector_ = other.contVector_;
    contVectorStr_ = other.contVectorStr_;
    return *this;
}

std::string mv::Order::toString() const
{
    return contVectorStr_;
}

bool mv::Order::isRowMajor()
{
    if(contVectorStr_ == "W")
        return true;
    if(contVectorStr_ == "WH")
        return true;
    if(contVectorStr_ == "WHC")
        return true;
    if(contVectorStr_ == "WHCN")
        return true;
    if(contVectorStr_ == "WHCNT")
        return true;
    return false;
}

bool mv::Order::isColMajor()
{
    if(contVectorStr_ == "W")
        return true;
    if(contVectorStr_ == "HW")
        return true;
    if(contVectorStr_ == "CHW")
        return true;
    if(contVectorStr_ == "NCHW")
        return true;
    if(contVectorStr_ == "TNCHW")
        return true;
    return false;
}

bool mv::Order::isRowMajorPlanar()
{
    if(contVectorStr_ == "W")
        return true;
    if(contVectorStr_ == "HW")
        return true;
    if(contVectorStr_ == "HWC")
        return true;
    if(contVectorStr_ == "HWCN")
        return true;
    if(contVectorStr_ == "HWCNT")
        return true;
    return false;
}

bool mv::Order::isZMajor()
{
    if(contVectorStr_ == "NHWC")
        return true;
    return false;
}

bool mv::Order::isColMajorPlanar()
{
    if(contVectorStr_ == "W")
        return true;
    if(contVectorStr_ == "WH")
        return true;
    if(contVectorStr_ == "CWH")
        return true;
    if(contVectorStr_ == "NCWH")
        return true;
    if(contVectorStr_ == "TNCWH")
        return true;
    return false;
}

bool mv::Order::isRowInterleaved()
{
    if(contVectorStr_ == "HCW")
        return true;
    return false;
}

std::string mv::Order::getLogID() const
{
    return "Order '" + toString() + "'";
}

