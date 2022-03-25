//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#ifndef BINARIZABLE_HPP
#define BINARIZABLE_HPP

#include "include/mcm/base/json/json.hpp"

namespace mv
{

    class Binarizable
    {

    public:

        virtual ~Binarizable() = 0;
        virtual std::vector<uint8_t> toBinary() const = 0;

    };

}

#endif
