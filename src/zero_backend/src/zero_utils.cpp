//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "zero_utils.h"

#include <string>

#include "ze_api.h"

namespace vpux {
namespace zeroUtils {

std::string result_to_string(const ze_result_t result) {
    std::string as_string = {};

    switch (result) {
    case ZE_RESULT_SUCCESS:
        as_string = "ZE_RESULT_SUCCESS";
        break;
    case ZE_RESULT_NOT_READY:
        as_string = "ZE_RESULT_NOT_READY";
        break;
    case ZE_RESULT_ERROR_DEVICE_LOST:
        as_string = "ZE_RESULT_ERROR_DEVICE_LOST";
        break;
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
        as_string = "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
        break;
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
        as_string = "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
        break;
    case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
        as_string = "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
        break;
    case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
        as_string = "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
        break;
    case ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET:
        as_string = "ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET";
        break;
    case ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
        as_string = "ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE";
        break;
    case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
        as_string = "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
        break;
    case ZE_RESULT_ERROR_NOT_AVAILABLE:
        as_string = "ZE_RESULT_ERROR_NOT_AVAILABLE";
        break;
    case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
        as_string = "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
        break;
    case ZE_RESULT_ERROR_UNINITIALIZED:
        as_string = "ZE_RESULT_ERROR_UNINITIALIZED";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
        as_string = "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
        as_string = "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
        break;
    case ZE_RESULT_ERROR_INVALID_ARGUMENT:
        as_string = "ZE_RESULT_ERROR_INVALID_ARGUMENT";
        break;
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
        as_string = "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
        break;
    case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
        as_string = "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
        break;
    case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
        as_string = "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
        break;
    case ZE_RESULT_ERROR_INVALID_SIZE:
        as_string = "ZE_RESULT_ERROR_INVALID_SIZE";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
        as_string = "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
        as_string = "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
        break;
    case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
        as_string = "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
        break;
    case ZE_RESULT_ERROR_INVALID_ENUMERATION:
        as_string = "ZE_RESULT_ERROR_INVALID_ENUMERATION";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
        as_string = "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
        as_string = "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
        break;
    case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
        as_string = "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
        break;
    case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
        as_string = "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
        break;
    case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
        as_string = "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
        break;
    case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
        as_string = "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
        break;
    case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
        as_string = "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
        break;
    case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
        as_string = "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
        break;
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
        as_string = "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
        break;
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
        as_string = "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
        break;
    case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
        as_string = "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
        break;
    case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
        as_string = "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
        break;
    case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
        as_string = "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
        break;
    case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
        as_string = "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
        break;
    case ZE_RESULT_ERROR_UNKNOWN:
        as_string = "ZE_RESULT_ERROR_UNKNOWN";
        break;
    case ZE_RESULT_FORCE_UINT32:
        as_string = "ZE_RESULT_FORCE_UINT32";
        break;
    default:
        as_string = "ze_result_t Unrecognized";
        break;
    };

    return as_string;
}

}  // namespace zeroUtils
}  // namespace vpux
