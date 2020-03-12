#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

mv::QuantizationParams::QuantizationParams(const json::Value& content) : Element(content)
{

}
mv::QuantizationParams::QuantizationParams(const std::vector<int64_t>& zp, const std::vector<double>& scale, const std::vector<double>& min, const std::vector<double>& max)
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
    }

}

//mv::QuantizationParams& mv::QuantizationParams::operator=(const mv::QuantizationParams& quantObject)
//{
//    set<std::vector<int64_t>>("zeroPoint", quantObject.get<std::vector<int64_t>>("zeroPoint"));
//    set<std::vector<double>>("scale", quantObject.get<std::vector<double>>("scale"));
//    set<std::vector<double>>("min", quantObject.get<std::vector<double>>("min"));
//    set<std::vector<double>>("max", quantObject.get<std::vector<double>>("max"));
//    if (quantObject.get<std::vector<double>>("scale").size())
//    {
//        std::vector<unsigned> shiftDefaut(quantObject.get<std::vector<double>>("scale").size(), 0);
//        std::vector<unsigned> multDefaut(quantObject.get<std::vector<double>>("scale").size(), 1);
//        set<std::vector<unsigned>>("shift", shiftDefaut);
//        set<std::vector<unsigned>>("mult", multDefaut);
//    }
//    return *this;
//}


mv::QuantizationParams::QuantizationParams(const std::vector<int64_t>& zp, const std::vector<double>& scale, const std::vector<double>& min, const std::vector<double>& max, const std::vector <unsigned>& shift, const std::vector<unsigned>& mult):
    QuantizationParams(zp, scale, min, max)
{
    set<std::vector<unsigned>>("shift", shift);
    set<std::vector<unsigned>>("mult", mult);
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

std::string mv::QuantizationParams:: toString() const
{
    return getLogID() + Element::attrsToString_();
}

bool mv::QuantizationParams:: isEmpty() const
{
    bool is_empty = false;
    if (get<std::vector<int64_t>>("zeroPoint").size() + get<std::vector<double>>("scale").size() + get<std::vector<double>>("min").size() + get<std::vector<double>>("max").size() == 0)
        is_empty = true;
    return is_empty;
}

bool mv::QuantizationParams:: isNeutral() const
{
    bool is_neutral = false;
    bool zero_point_neutral = false;
    bool scale_neutral = true;
    int64_t sum_of_elems = std::accumulate(get<std::vector<int64_t>>("zeroPoint").begin(),
                                           get<std::vector<int64_t>>("zeroPoint").end(), 0);
    if (sum_of_elems == 0)
        zero_point_neutral = true;
    std::vector<double> neutral_scale(get<std::vector<double>>("scale").size(), 1.0f);
    std::vector<double> absRelativeErrorScale;
    for (std::size_t i =0; i < get<std::vector<double>>("scale").size(); i ++)
        absRelativeErrorScale.push_back(std::abs(get<std::vector<double>>("scale")[i] - neutral_scale[i]));

    for (auto it = absRelativeErrorScale.begin(); it != absRelativeErrorScale.end(); it++)
    {
        if (*it > 0.01f)
            scale_neutral = false;
    }

    is_neutral = (zero_point_neutral&&scale_neutral);
    return is_neutral;
}

bool mv::QuantizationParams:: infinitelimits() const
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


