#include "include/mcm/target/kmb/dma_direction.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

const std::unordered_map<mv::DmaDirectionEnum, std::string, mv::DmaDirectionEnumHash> mv::DmaDirection::dmaDirectionStrings_ =
{
    {mv::DmaDirectionEnum::NNCMX2DDR, "NNCMX2DDR"},
    {mv::DmaDirectionEnum::DDR2NNCMX, "DDR2NNCMX"},
    {mv::DmaDirectionEnum::NNCMX2UPACMX, "NNCMX2UPACMX"},
    {mv::DmaDirectionEnum::UPACMX2NNCMX, "UPACMX2NNCMX"},
    {mv::DmaDirectionEnum::DDR2DDR, "DDR2DDR"},
    {mv::DmaDirectionEnum::DDR2UPACMX, "DDR2UPACMX"},
    {mv::DmaDirectionEnum::UPACMX2DDR, "UPACMX2DDR"},
    {mv::DmaDirectionEnum::CSRAM2NNCMX, "CSRAM2NNCMX"}

};

mv::DmaDirection::DmaDirection(DmaDirectionEnum value) :
direction_(value)
{

}

mv::DmaDirection::DmaDirection() :
direction_(DmaDirectionEnum::NNCMX2DDR)
{

}

mv::DmaDirection::DmaDirection(const DmaDirection& other) :
direction_(other.direction_)
{

}

mv::DmaDirection::DmaDirection(const std::string& value)
{

//    DmaDirection(
//        [=]()->DmaDirection
//        {
//            for (auto &e : dmaDirectionStrings_)
//                if (e.second == value)
//                    return e.first;
//            throw ArgumentError(*this, "Invalid initialization - string value specified as", value, "Initializer");
//        }()
//    );
    bool found = false;
    for ( auto &e : dmaDirectionStrings_)
    {
        if(e.second == value)
        {
            direction_ = e.first;
            found = true;
        }
    }
    if(found == false)
        throw ArgumentError(*this, "Invalid initialization - string value specified as", value, "Initializer");
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
