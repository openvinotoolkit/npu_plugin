#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include <numeric>

mv::QuantizationParams::QuantizationParams(const json::Value& content) : Element(content)
{

}
mv::QuantizationParams::QuantizationParams(
    const std::vector<int64_t>& zp,
    const std::vector<double>& scale,
    const std::vector<double>& min,
    const std::vector<double>& max)
    :Element("quantParams")
{
    set<std::vector<int64_t>>("zeroPoint", zp);
    set<std::vector<double>>("scale", scale);
    set<std::vector<double>>("min", min);
    set<std::vector<double>>("max", max);

    if (scale.size())
    {
        std::vector<unsigned> shiftDefaut(scale.size(), 0);
        std::vector<unsigned> multDefaut(scale.size(), 1);
        set<std::vector<unsigned>>("shift", shiftDefaut);
        set<std::vector<unsigned>>("mult", multDefaut);
        set<signed>("postShift", 0);
    }

}

mv::QuantizationParams::QuantizationParams(
    const std::vector<int64_t>& zp,
    const std::vector<double>& scale,
    const std::vector<double>& min,
    const std::vector<double>& max,
    const std::vector <unsigned>& shift,
    const std::vector<unsigned>& mult):
    QuantizationParams(zp, scale, min, max)
{
    set<std::vector<unsigned>>("shift", shift);
    set<std::vector<unsigned>>("mult", mult);
    set<signed>("postShift", 0);
}

mv::QuantizationParams::QuantizationParams(
    const std::vector<int64_t>& zp,
    const std::vector<double>& scale,
    const std::vector<double>& min,
    const std::vector<double>& max,
    const std::vector <unsigned>& shift,
    const std::vector<unsigned>& mult,
    const signed postShift):
    QuantizationParams(zp, scale, min, max, shift, mult)
{
    set<signed>("postShift", postShift);
}

void mv::QuantizationParams::quantize(std::vector<unsigned> shift, std::vector<unsigned> mult)
{
    set<std::vector<unsigned>>("shift", shift);
    set<std::vector<unsigned>>("mult", mult);
}

void mv::QuantizationParams::setScale(std::vector<double> scale_)
{
    set<std::vector<double>>("scale", scale_);
}

void mv::QuantizationParams::setZeroPoint(std::vector<int64_t> zeroPoint_)
{
    set<std::vector<int64_t>>("zeroPoint", zeroPoint_);
}

void mv::QuantizationParams::setPostShift(signed postShift_)
{
    set<signed>("postShift", postShift_);
}

int64_t mv::QuantizationParams::getZeroPoint(const size_t channel) const
{
    std::vector<int64_t> zeroPoint = get<std::vector<int64_t>>("zeroPoint");
    if (zeroPoint.size() == 1)
        return zeroPoint[0];
    if (channel >= zeroPoint.size())
        throw ArgumentError("quantParams", "channel", std::to_string(channel),
            "Invalid index: channel is greater than zeroPoint vector");
    return zeroPoint[channel];
}
std::string mv::QuantizationParams::getLogID() const
{
    return "QuantizationParams: " + getName();
}

std::string mv::QuantizationParams::toString() const
{
    return getLogID() + Element::attrsToString_();
}

bool mv::QuantizationParams::isEmpty() const
{
    return get<std::vector<int64_t>>("zeroPoint").size() == 0 &&
           get<std::vector<double>>("scale").size() == 0 &&
           get<std::vector<double>>("min").size() == 0 &&
           get<std::vector<double>>("max").size() == 0;
}

bool mv::QuantizationParams::isNeutral() const
{
    const auto& scales = get<std::vector<double>>("scale");
    bool isScaleNeutral = std::all_of(scales.begin(), scales.end(), [](double x) {
                                          return std::abs(1.0f - x) <= 0.01f;
                                      });
    const auto& zeroPoints = get<std::vector<int64_t>>("zeroPoint");
    bool isZeroPointNeutral = std::accumulate(zeroPoints.begin(), zeroPoints.end(), 0) == 0;
    return isZeroPointNeutral && isScaleNeutral;
}

bool mv::QuantizationParams::infinitelimits() const
{
    bool is_infinite = false;
    if (hasAttr("min") && hasAttr("max"))
    {
        for (std::size_t vec_size = 0; vec_size <= get<std::vector<double>>("min").size(); vec_size++)
        {
            if (get<std::vector<double>>("min").size() == 0)
            {
                is_infinite = true;
                break;
            }
            if (std::isinf(get<std::vector<double>>("min")[vec_size])
                    || std::isinf(get<std::vector<double>>("max")[vec_size]))
            {
                is_infinite = true;
                break;
            }
        }
    }
    else
    {
        is_infinite = true;
    }
    return is_infinite;
}

bool mv::QuantizationParams::isScalePerTensor() const{
    return get<std::vector<double>>("scale").size() == 1;
}

//TODO: isn't it too performance consuming?
double mv::QuantizationParams::getScale(const size_t channel) const {
    auto scales = get<std::vector<double>>("scale");
    if (scales.size() == 1)
        return scales[0];
    if (channel >= scales.size())
        throw ArgumentError("quantParams", "channel", std::to_string(channel),
                            "Invalid index: channel is greater than scales vector");
    return scales[channel];
}

mv::QuantizationParams mv::QuantizationParams::empty() {
    return {{}, {}, {}, {}};
}

mv::QuantizationParams mv::QuantizationParams::initial() {
    static double inf = std::numeric_limits<double>::infinity();
    return {{0}, {1}, {-inf}, {inf}};
}
