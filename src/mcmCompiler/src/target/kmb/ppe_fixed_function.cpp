#include "include/mcm/target/kmb/ppe_fixed_function.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

mv::PPEFixedFunction::PPEFixedFunction(int8_t lRelumult, uint8_t lRelushift, int low_clamp, int high_clamp)
    : lowClamp_(low_clamp),
      highClamp_(high_clamp),
      reluMult_(lRelumult),
      reluShift_(lRelushift),
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

int8_t mv::PPEFixedFunction::getLReluMult() const
{
    return reluMult_;
}

uint8_t mv::PPEFixedFunction::getLReluShift() const
{
    return reluShift_;
}

const std::vector<mv::PPELayerType>& mv::PPEFixedFunction::getLayers() const
{
    return layers_;
}

void mv::PPEFixedFunction::setLowClamp(int lowClamp)
{
    lowClamp_ = lowClamp;
}

void mv::PPEFixedFunction::setHighClamp(int highClamp)
{
    highClamp_ = highClamp;
}

void mv::PPEFixedFunction::setLReluMult(int8_t lRelumult)
{
    reluMult_ = lRelumult;
}

void mv::PPEFixedFunction::setLReluShift(uint8_t lRelushift)
{
    reluShift_ = lRelushift;
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
