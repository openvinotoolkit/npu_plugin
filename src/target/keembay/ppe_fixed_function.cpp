#include "include/mcm/target/keembay/ppe_fixed_function.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

mv::PPEFixedFunction::PPEFixedFunction(int low_clamp, int high_clamp)
    : lowClamp_(low_clamp),
      highClamp_(high_clamp),
      layers_(std::vector<PPELayerType>())
{

}

void mv::PPEFixedFunction::addLayer(PPELayerType layer)
{
    layers_.push_back(layer);
}

int mv::PPEFixedFunction::getLowClamp() const
{
    return lowClamp_;
}

int mv::PPEFixedFunction::getHighClamp() const
{
    return highClamp_;
}

const std::vector<mv::PPELayerType>& mv::PPEFixedFunction::getLayers() const
{
    return layers_;
}

std::string mv::PPEFixedFunction::toString() const
{
    std::string output = "";

    output += "Low clamp " + std::to_string(lowClamp_) + "\n";
    output += "High clamp " + std::to_string(highClamp_) + "\n";
    for(auto layer : layers_)
        output += "PPELayerType " + layer.toString() + "\n";
    
    return output;
}

std::string mv::PPEFixedFunction::getLogID() const
{
    return "PPEFixedFunction:" + toString();
}
