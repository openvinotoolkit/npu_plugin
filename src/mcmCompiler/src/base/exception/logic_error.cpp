//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/base/exception/logic_error.hpp"

mv::LogicError::LogicError(const LogSender& sender, const std::string& whatArg) :
LoggedError(sender, "LogicError: " + whatArg)
{

}