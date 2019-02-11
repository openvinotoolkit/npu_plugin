#include "include/mcm/target/keembay/ppe_fixed_function.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

mv::PPEFixedFunction::PPEFixedFunction(unsigned low_clamp, unsigned high_clamp)
    : low_clamp_(low_clamp),
      high_clamp_(high_clamp)
{

}

void mv::PPEFixedFunction::addLayer(PpeLayerType layer)
{
    layers_.push_back(layer);
}

unsigned mv::PPEFixedFunction::getLowClamp() const
{
    return low_clamp_;
}

unsigned mv::PPEFixedFunction::getHighClamp() const
{
    return high_clamp_;
}

const std::vector<mv::PpeLayerType>& mv::PPEFixedFunction::getLayers() const
{
    return layers_;
}

std::string mv::PPEFixedFunction::toString() const
{
    std::string output = "";

    output += "Low clamp " + std::to_string(low_clamp_) + "\n";
    output += "High clamp " + std::to_string(high_clamp_) + "\n";
    for(auto layer : layers_)
        output += "PPELayerType " + layer.toString() + "\n";
    
    return output;
}

std::string mv::PPEFixedFunction::getLogID() const
{
    return "PPEFixedFunction:" + toString();
}
