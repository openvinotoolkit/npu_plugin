//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#ifndef JSONABLE_HPP
#define JSONABLE_HPP

#include "include/mcm/base/json/json.hpp"

namespace mv
{

    class Jsonable
    {  

    public:

        virtual ~Jsonable() = 0;
        virtual json::Value toJSON() const = 0;

    };

}

#endif
