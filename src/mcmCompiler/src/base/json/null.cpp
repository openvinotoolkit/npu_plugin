//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/base/json/null.hpp"

mv::json::Null::Null()
{

}

std::string mv::json::Null::stringify() const
{
    return "null";
}

std::string mv::json::Null::stringifyPretty() const
{
    return stringify();
}

std::string mv::json::Null::getLogID() const
{
    return "json::Null";
}