#include "include/mcm/target/keembay/dma_direction.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

const std::unordered_map<mv::DmaDirectionEnum, std::string, mv::DmaDirectionEnumHash> mv::DmaDirection::dmaDirectionStrings_ =
{
    {mv::DmaDirectionEnum::CMX2DDR, "CMX2DDR"},
    {mv::DmaDirectionEnum::DDR2CMX, "DDR2CMX"},
    {mv::DmaDirectionEnum::CMX2UPA, "CMX2UPA"},
    {mv::DmaDirectionEnum::UPA2CMX, "UPA2CMX"},

};

mv::DmaDirection::DmaDirection(DmaDirectionEnum value) :
direction_(value)
{

}

mv::DmaDirection::DmaDirection() :
direction_(DmaDirectionEnum::CMX2DDR)
{

}

mv::DmaDirection::DmaDirection(const DmaDirection& other) :
direction_(other.direction_)
{

}

mv::DmaDirection::DmaDirection(const std::string& value)
{

    DmaDirection(
        [=]()->DmaDirection
        {
            for (auto &e : dmaDirectionStrings_)
                if (e.second == value)
                    return e.first;
            throw ArgumentError(*this, "Invalid initialization - string value specified as", value, "Initializer");
        }()
    );

}

std::string mv::DmaDirection::toString() const
{
    return dmaDirectionStrings_.at(*this);
}

mv::DmaDirection& mv::DmaDirection::operator=(const DmaDirection& other)
{
    direction_ = other.direction_;
    return *this;
}

mv::DmaDirection& mv::DmaDirection::operator=(const DmaDirectionEnum& other)
{
    direction_ = other;
    return *this;
}

mv::DmaDirection::operator DmaDirectionEnum() const
{
    return direction_;
}

std::string mv::DmaDirection::getLogID() const
{
    return "Direction:" + toString();
}
