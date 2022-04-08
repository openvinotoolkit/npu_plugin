//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#ifdef _WIN32

#include <stdarg.h>
#include <stdio.h>
#include "logging.hpp"

int ms_vasprintf(char** ptr, const char* format, va_list ap) {
    int len = _vscprintf(format, ap) + 1;
    *ptr = (char*)malloc(len);
    if (*ptr)
        return vsprintf_s(*ptr, len, format, ap);
    else
        return -1;
}

#endif  // WIN32
