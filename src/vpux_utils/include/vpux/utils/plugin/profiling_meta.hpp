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

constexpr uint32_t PROFILING_SECTION_ENCODING = 1;  // Profiling metadata encoded in FB format

constexpr uint32_t PROFILING_METADATA_VERSION_MAJOR = 2;  // Initial major version of FB schema

constexpr uint32_t PROFILING_METADATA_VERSION_MINOR = 0;  // Initial minor version of FB schema

// To avoid another level of buffers nesting, schema versions is declared as first fields in FB schema, so any
// subsequent changes won't break binary compatibility between parser and data. This assert is needed to prevent reorder
// of fields during development
static_assert(ProfilingFB::ProfilingMeta::FlatBuffersVTableOffset::VT_MAJORVERSION == 4,
              "Schema major version isn't first field in FB metadata. Fix it or change major version");
static_assert(ProfilingFB::ProfilingMeta::FlatBuffersVTableOffset::VT_MINORVERSION == 6,
              "Schema minor version isn't second field in FB metadata. Fix it or change major version");

// The layout is:
// +----------------------------+-----------------------
// | SECTION_ENCODING: uint32_t |  DATA...
// +----------------------------+-----------------------
uint32_t getProfilingSectionEncoding(const uint8_t* data, size_t size);

// Creates profiling metadata section. Profiling metadata is stored in flatbuffer(FB) object, where first
// fields are major/minor versions. FB object is aligned by 8 bytes boundary
// +----------------------------+---------------------+------------
// | SECTION_ENCODING: uint32_t |  DATA_LEN: uint32_t | DATA....
// +----------------------------+---------------------+------------
std::vector<uint8_t> constructProfilingSectionWithHeader(flatbuffers::DetachedBuffer rawMetadataFb);

std::vector<std::string> splitBySeparator(const std::string& s, char separator);

bool isElfBinary(const uint8_t* data, size_t size);

const MVCNN::GraphFile* getGraphFileVerified(const uint8_t* buffer, size_t size);

const ProfilingFB::ProfilingMeta* getProfilingSectionMeta(const uint8_t* blobData, size_t blobSize);

const uint8_t* getProfilingSectionPtr(const uint8_t* blobData, size_t blobSize);

}  // namespace profiling
}  // namespace vpux
