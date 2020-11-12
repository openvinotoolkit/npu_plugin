///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
///
/// @file      secure_functions.h
///
/// @brief     Secure version of memcpy.
///

#include <errno.h>
#include <stdint.h>

#ifndef __SECURE_FUNCTIONS_H__
#define __SECURE_FUNCTIONS_H__

inline static int memcpy_s(void * dest, size_t destsz, const void * const src, size_t count)
{
    if (dest == NULL) return EINVAL; // dest should not be a NULL ptr
    if (destsz > SIZE_MAX) return ERANGE;
    if (count > SIZE_MAX) return ERANGE;
    if (destsz < count) { memset(dest, 0, destsz); return ERANGE; }
    if (src == NULL) { memset(dest, 0, destsz); return EINVAL; } // src should not be a NULL ptr

    memcpy(dest, src, count);
    return 0;
}

#endif /* __SECURE_FUNCTIONS_H__ */
