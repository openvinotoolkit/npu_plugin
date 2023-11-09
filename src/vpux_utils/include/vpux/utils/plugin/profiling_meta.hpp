//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <schema/graphfile_generated.h>
#include <schema/profiling_generated.h>

#include <stdint.h>
#include <cstddef>

namespace vpux {
namespace profiling {

std::vector<std::string> splitBySeparator(const std::string& s, char separator);

bool isElfBinary(const uint8_t* data, size_t size);

const MVCNN::GraphFile* getGraphFileVerified(const uint8_t* buffer, size_t size);

const ProfilingFB::ProfilingMeta* getProfilingSectionMeta(const uint8_t* blobData, size_t blobSize);

const uint8_t* getProfilingSectionPtr(const uint8_t* blobData, size_t blobSize);

}  // namespace profiling
}  // namespace vpux
