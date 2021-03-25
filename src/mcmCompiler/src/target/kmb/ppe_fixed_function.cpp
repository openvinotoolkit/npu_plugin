#include "include/mcm/target/kmb/ppe_fixed_function.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

mv::PPEFixedFunction::PPEFixedFunction(int32_t lRelumult, uint8_t lRelushift, int lowClamp, int highClamp)
    : lowClamp_(lowClamp),
      highClamp_(highClamp),
      reluShift_(lRelushift),
      layers_(std::vector<PPELayerType>())
{
    reluMult_ = { lRelumult };
}

mv::PPEFixedFunction::PPEFixedFunction(const std::vector<int32_t>& lRelumult, uint8_t lRelushift, int lowClamp, int highClamp)
    : lowClamp_( lowClamp ),
    highClamp_( highClamp ),
    reluShift_( lRelushift ),
    layers_( std::vector<PPELayerType>() )
{
    if (lRelumult.size() == 0)
    {
        reluMult_ = {1};
    }
    else
    {
        reluMult_ = lRelumult;
    }
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

int32_t mv::PPEFixedFunction::getLReluMult() const
{
    // Note that LReluMult is a deprecated attribute and left for compatibility
    // No chance to be a empty vector.
    return ((reluMult_.size() != 0)?  reluMult_[0]: 1);
}

const std::vector<int32_t>& mv::PPEFixedFunction::getLReluMults() const
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

void mv::PPEFixedFunction::setLReluMult(int32_t lRelumult)
{
    // Note: no empty check as the default size of reluMult is 1
    if(reluMult_.size() != 1U) 
    {
        reluMult_.resize(1);
    }

    reluMult_[0] = lRelumult;
}

void mv::PPEFixedFunction::setLReluMults(const std::vector<int32_t>& lRelumult )
{
    if (lRelumult.size() == 0U)
    {
        // to keep the default reluMult
        reluMult_ = {1};
    }
    else
    {
        reluMult_ = lRelumult;
    }
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
