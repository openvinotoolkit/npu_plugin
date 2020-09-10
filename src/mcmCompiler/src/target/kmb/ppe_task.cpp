#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

mv::PPETask::PPETask(const json::Value& content)
    :Element(content)
{

}

mv::PPETask::PPETask(const PPEFixedFunction& fixedFunction)
    :Element("PPETask")
{
    set<PPEFixedFunction>("fixedFunction", fixedFunction);
}

std::string mv::PPETask::toString() const
{
    std::string output = "";

    if(hasAttr("scaleData"))
        output += getScaleData()->toString();

    output += getFixedFunction().toString();
    
    return output;
}

std::string mv::PPETask::getLogID() const
{
    return "PPETask:" + toString();
}
