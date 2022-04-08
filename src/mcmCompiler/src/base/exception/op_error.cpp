//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/base/exception/op_error.hpp"

mv::OpError::OpError(const LogSender& sender, const std::string& whatArg) :
LoggedError(sender, "OpError: " + whatArg)
{

}

mv::OpError::OpError(const std::string& senderID, const std::string& whatArg) :
LoggedError(senderID, "OpError: " + whatArg)
{

}