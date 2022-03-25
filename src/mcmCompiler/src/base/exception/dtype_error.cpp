//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/base/exception/dtype_error.hpp"

mv::DTypeError::DTypeError(const LogSender& sender, const std::string& whatArg) :
LoggedError(sender, "DTypeError: " + whatArg)
{

}
mv::DTypeError::DTypeError(const std::string& senderID, const std::string& whatArg) :
LoggedError(senderID, "DTypeError: " + whatArg)
{

}
