//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/base/exception/value_error.hpp"

mv::ValueError::ValueError(const LogSender& sender, const std::string& whatArg) :
LoggedError(sender, "ValueError: " + whatArg)
{

}