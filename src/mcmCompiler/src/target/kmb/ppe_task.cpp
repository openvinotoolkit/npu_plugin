//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
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

    output += getFixedFunction().toString();
    
    return output;
}

std::string mv::PPETask::getLogID() const
{
    return "PPETask:" + toString();
}
