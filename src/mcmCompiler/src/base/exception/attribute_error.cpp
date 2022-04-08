//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/base/exception/attribute_error.hpp"

mv::AttributeError::AttributeError(const LogSender& sender, const std::string& whatArg) :
LoggedError(sender, "AttributeError: " + whatArg)
{

}

mv::AttributeError::AttributeError(const std::string& senderID, const std::string& whatArg) :
LoggedError(senderID, "AttributeError: " + whatArg)
{

}