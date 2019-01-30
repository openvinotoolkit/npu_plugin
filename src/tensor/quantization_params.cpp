#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

mv::QuantizationParams::QuantizationParams(int64_t zp, float scale,
    float min, float max):
        zero_point_(zp),
        scale_(scale),
        min_(min),
        max_(max)
{
    if (max < min)
        throw ArgumentError("QuantizationParams", "Quantization min max params", "max",
            "Smaller than min " + std::to_string(min_));
}

mv::QuantizationParams::QuantizationParams(const QuantizationParams& other):
        zero_point_(other.zero_point_),
        scale_(other.scale_),
        min_(other.min_),
        max_(other.max_)
{

}

bool mv::QuantizationParams::operator==(const mv::QuantizationParams& other) const
{
    return (zero_point_ == other.zero_point_ && scale_ == other.scale_ &&
        min_ == other.min_ && max_ == other.max_);
}

void mv::QuantizationParams::setMin(float min)
{
    if (min > max_)
        throw ArgumentError("QuantizationParams", "Quantization min max params", "max",
            "Smaller than min " + std::to_string(min));
    min_ = min;
}

void mv::QuantizationParams::setMax(float max)
{
    if (min_ > max)
        throw ArgumentError("QuantizationParams", "Quantization min max params", "max",
            "Smaller than min " + std::to_string(min_));
    max_ = max;
}

std::string mv::QuantizationParams::getLogID() const
{
    return "QuantizationParams";
}

std::string mv::QuantizationParams:: toString() const
{
    std::string output("{");

    output += "zero_point: " + std::to_string(zero_point_);
    output += ", scale: " + std::to_string(scale_);
    output += ", min: " + std::to_string(min_);
    output += ", max: " + std::to_string(max_);
    output += "}";

    return output;
}