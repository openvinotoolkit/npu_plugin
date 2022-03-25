//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#ifndef MV_VALUE_ERROR_HPP_
#define MV_VALUE_ERROR_HPP_

#include "include/mcm/base/exception/logged_error.hpp"

namespace mv
{

    class ValueError : public LoggedError
    {

    public:

        explicit ValueError(const LogSender& sender, const std::string& whatArg);

    };

}

#endif // MV_VALUE_ERROR_HPP_