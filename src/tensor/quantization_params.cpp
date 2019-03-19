#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

mv::QuantizationParams::QuantizationParams(const json::Value& content) : Element(content)
{

}
mv::QuantizationParams::QuantizationParams(std::vector<unsigned> zp, std::vector<double> scale,
    std::vector<double> min, std::vector<double> max): Element("quantizationParams")
{
    size_t size = zp.size();
    if (size != scale.size() || size != min.size() || size != max.size())
        throw ArgumentError("QuantizationParams", "Quantization params size", "",
            "Sizes of the different params don't match");

    for (size_t i = 0; i < size; i++)
        if (max[i] < min[i])
            throw ArgumentError("QuantizationParams", "Quantization min max params", "max",
                " Smaller than min " + std::to_string(min[i]));

    set<std::vector<unsigned>>("zeroPoint", zp);
    set<std::vector<double>>("scale", scale);
    set<std::vector<double>>("min", min);
    set<std::vector<double>>("max", max);
}

unsigned mv::QuantizationParams::getZeroPoint(const size_t channel) const
{
    std::vector<unsigned> zeroPoint = get<std::vector<unsigned>>("zeroPoint");
    if (zeroPoint.size() == 1)
        return zeroPoint[0];
    if (channel >= zeroPoint.size())
        throw ArgumentError("QuantizationParams", "channel", std::to_string(channel),
            "Invalid index: channel is greater than zeroPoint vector");
    return zeroPoint[channel];
}
std::string mv::QuantizationParams::getLogID() const
{
    return "QuantizationParams: " + getName();
}

std::string mv::QuantizationParams:: toString() const
{
    return getLogID() + Element::attrsToString_();
}