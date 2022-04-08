//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/base/exception/shape_error.hpp"

mv::ShapeError::ShapeError(const LogSender& sender, const std::string& whatArg) :
LoggedError(sender, "ShapeError: " + whatArg)
{

}