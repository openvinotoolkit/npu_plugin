#include "include/mcm/target/target_descriptor.hpp"

std::string mv::TargetDescriptor::toString(Target target)
{
    switch (target)
    {

        case Target::ma2480:
            return "ma2480";
        
        default:
            return "unknown";

    }
}

mv::Target mv::TargetDescriptor::toTarget(const std::string& str)
{
    if (str == "ma2480")
        return Target::ma2480;
    
    return Target::Unknown;
}  

mv::DType mv::TargetDescriptor::toDType(const std::string& str)
{
    if (str == "fp16")
        return DType::Float;
    
    return DType::Unknown;
}

mv::Order mv::TargetDescriptor::toOrder(const std::string& str)
{
    // TODO update
    if (str == "planar")
        return Order::LastDimMajor;
    
    return Order::Unknown;
}

mv::TargetDescriptor::TargetDescriptor() :
target_(Target::Unknown),
globalDType_(DType::Unknown),
globalOrder_(Order::Unknown)
{

}

bool mv::TargetDescriptor::load(const std::string& filePath)
{

    JSONTextParser parser(jsonParserBufferLenght_);

    try 
    {

        json::Value jsonRoot;
        json::Object jsonDescriptor;
        parser.parseFile(filePath, jsonRoot);
        
        if (jsonRoot.valueType() != json::JSONType::Object)
            return false;
        else
            jsonDescriptor = jsonRoot.get<json::Object>();

        if (!jsonDescriptor.hasKey("target"))
            return false;
        else
        {
            target_ = toTarget(jsonDescriptor["target"].get<std::string>());
            if (target_ == Target::Unknown)
                return false;
        }

    }
    catch (ParsingError& e)
    {
        
    }

    return false;

}

bool mv::TargetDescriptor::save(const std::string& filePath)
{
    return false;
}