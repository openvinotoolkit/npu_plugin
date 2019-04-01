#include "include/mcm/target/keembay/ppe_task.hpp"
#include "include/mcm/base/exception/argument_error.hpp"

mv::PPETask::PPETask()
{

}

mv::PPETask::PPETask(const std::string& value)
{

}

mv::PPETask::~PPETask()
{

}

std::string mv::PPETask::toString() const
{
    std::string output = "";
    
    return output;
}

std::string mv::PPETask::getLogID() const
{
    return "PPETask:" + toString();
}
