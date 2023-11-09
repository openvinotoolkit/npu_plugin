//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/plugin/profiling_meta.hpp"

#include <vpux_elf/accessor.hpp>
#include <vpux_elf/reader.hpp>

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/profiling.hpp"

#include <sstream>

namespace {

using SchemaAndPtrType = std::pair<const ProfilingFB::ProfilingMeta*, const uint8_t*>;

const SchemaAndPtrType getProfilingMetaBufferVerified(const uint8_t* buffer, size_t size) {
    auto verifier = flatbuffers::Verifier(buffer, size);
    VPUX_THROW_UNLESS(ProfilingFB::VerifyProfilingMetaBuffer(verifier), "Cannot verify profiling metadata integrity");
    return {ProfilingFB::GetProfilingMeta(buffer), buffer};
}

const SchemaAndPtrType getProfilingSectionDataElf(const uint8_t* blobData, size_t blobSize) {
    VPUX_THROW_WHEN(blobData == nullptr, "Empty input data");

    elf::ElfDDRAccessManager elfAccess(blobData, blobSize);
    elf::Reader<elf::ELF_Bitness::Elf64> reader(&elfAccess);
    for (size_t i = 0; i < reader.getSectionsNum(); ++i) {
        const auto section = reader.getSectionNoData(i);
        const std::string secName = section.getName();
        if (secName == ".profiling") {
            const uint8_t* profMetaSection = section.getData<uint8_t>();
            const size_t sectionSize = section.getEntriesNum();
            return getProfilingMetaBufferVerified(profMetaSection, sectionSize);
        }
    }
    VPUX_THROW("Cannot find .profiling section");
}

const SchemaAndPtrType getProfilingSectionDataGf(const uint8_t* blobData, size_t blobSize) {
    VPUX_THROW_WHEN(blobData == nullptr, "Empty input data");

    const auto graphFile = vpux::profiling::getGraphFileVerified(blobData, blobSize);
    const auto profilingOutputs = graphFile->header()->profiling_output();
    VPUX_THROW_UNLESS(profilingOutputs, "Blob does not contain profiling information");
    VPUX_THROW_UNLESS(profilingOutputs->size() == 1, "Blob must contain exactly one profiling output");

    const auto profHeader = profilingOutputs->Get(0)->name()->str();
    VPUX_THROW_WHEN(profHeader != vpux::PROFILING_OUTPUT_NAME,
                    "Invalid profiling output name. Must be '{0}', but got '{1}'", vpux::PROFILING_OUTPUT_NAME,
                    profHeader);

    const auto binaryData = graphFile->binary_data();
    VPUX_THROW_WHEN(binaryData == nullptr, "Empty binary data");
    const auto numBinaries = binaryData->size();
    VPUX_THROW_UNLESS(numBinaries > 0, "Invalid number of binaries: {0}", numBinaries);
    const size_t profBinIndex = numBinaries - 1;

    const auto profilingSchema = binaryData->Get(profBinIndex);
    VPUX_THROW_WHEN(profilingSchema == nullptr, "Empty profiling binary data");

    const auto profilingMetaContent = profilingSchema->data()->Data();
    const auto profBinSize = profilingSchema->length();
    return getProfilingMetaBufferVerified(profilingMetaContent, profBinSize);
}
SchemaAndPtrType getProfilingSectionDataImpl(const uint8_t* blobData, size_t blobSize) {
    if (vpux::profiling::isElfBinary(blobData, blobSize)) {
        return getProfilingSectionDataElf(blobData, blobSize);
    }
    return getProfilingSectionDataGf(blobData, blobSize);
}

};  // namespace

namespace vpux {
namespace profiling {

std::vector<std::string> splitBySeparator(const std::string& s, char separator) {
    std::istringstream iss(s);
    std::string part;
    std::vector<std::string> parts;
    while (std::getline(iss, part, separator)) {
        parts.push_back(part);
    }
    return parts;
}

bool isElfBinary(const uint8_t* data, size_t size) {
    VPUX_THROW_WHEN(data == nullptr, "Empty input data");
    VPUX_THROW_WHEN(size < 4, "File is too short");
    return elf::utils::checkELFMagic(data);
}

const MVCNN::GraphFile* getGraphFileVerified(const uint8_t* buffer, size_t size) {
    auto verifier = flatbuffers::Verifier(buffer, size);
    VPUX_THROW_UNLESS(MVCNN::VerifyGraphFileBuffer(verifier), "Cannot verify blob integrity");
    return MVCNN::GetGraphFile(buffer);
}

const ProfilingFB::ProfilingMeta* getProfilingSectionMeta(const uint8_t* blobData, size_t blobSize) {
    return getProfilingSectionDataImpl(blobData, blobSize).first;
}

const uint8_t* getProfilingSectionPtr(const uint8_t* blobData, size_t blobSize) {
    return getProfilingSectionDataImpl(blobData, blobSize).second;
}

}  // namespace profiling
}  // namespace vpux
