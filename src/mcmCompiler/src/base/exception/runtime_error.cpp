//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/base/exception/runtime_error.hpp"

mv::RuntimeError::RuntimeError(const LogSender& sender, const std::string& whatArg) :
LoggedError(sender, "RuntimeError: " + whatArg)
{

}

mv::RuntimeError::RuntimeError(const std::string& senderID, const std::string& whatArg) :
LoggedError(senderID, "RuntimeError: " + whatArg)
{

}