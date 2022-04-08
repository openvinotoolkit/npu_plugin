//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/utils/helpers.hpp"

void mv::utils::releaseFile(FILE* ptr) {
    if(ptr) {
        fclose(ptr);
    }
}
