//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/attributes/enums.hpp"

#include "vpux/utils/core/string_ref.hpp"

namespace vpux {
namespace VPUIP {

template <ArchKind kind>
struct ArchTraits;

//
// KMB
//

template <>
struct ArchTraits<ArchKind::KMB> final {
    static StringRef getArchGenName();

    template <PhysicalProcessor kind>
    struct ProcessorTraits;

    template <DMAEngine kind>
    struct DMATraits;

    template <PhysicalMemory kind>
    struct MemoryTraits;
};

template <>
struct ArchTraits<ArchKind::KMB>::ProcessorTraits<PhysicalProcessor::ARM> final {
    static constexpr int32_t COUNT = 1;
};

template <>
struct ArchTraits<ArchKind::KMB>::ProcessorTraits<PhysicalProcessor::Leon_RT> final {
    static constexpr int32_t COUNT = 1;
};

template <>
struct ArchTraits<ArchKind::KMB>::ProcessorTraits<PhysicalProcessor::Leon_NN> final {
    static constexpr int32_t COUNT = 1;
};

template <>
struct ArchTraits<ArchKind::KMB>::ProcessorTraits<PhysicalProcessor::SHAVE_UPA> final {
    static constexpr int32_t COUNT = 16;
};

template <>
struct ArchTraits<ArchKind::KMB>::ProcessorTraits<PhysicalProcessor::SHAVE_NN> final {
    static constexpr int32_t COUNT = 20;
};

template <>
struct ArchTraits<ArchKind::KMB>::ProcessorTraits<PhysicalProcessor::NCE_Cluster> final {
    static constexpr int32_t COUNT = 4;
};

template <>
struct ArchTraits<ArchKind::KMB>::ProcessorTraits<PhysicalProcessor::NCE_PerClusterDPU> final {
    static constexpr int32_t COUNT = 5;
};

template <>
struct ArchTraits<ArchKind::KMB>::DMATraits<DMAEngine::UPA> final {
    static constexpr int32_t COUNT = 1;
};

template <>
struct ArchTraits<ArchKind::KMB>::DMATraits<DMAEngine::NN> final {
    static constexpr int32_t COUNT = 1;
};

template <>
struct ArchTraits<ArchKind::KMB>::MemoryTraits<PhysicalMemory::DDR> final {
    static constexpr int32_t CLUSTERS_COUNT = 1;
    static constexpr int32_t CLUSTER_SIZE_KB = 32760;
    static constexpr float DERATE_FACTOR = 0.6f;  // Derate factor for bandwidth (due to MMU/NoC loss)
    static constexpr int32_t BANDWIDTH = 8;       // Bandwidth in Bytes/cycles
};

template <>
struct ArchTraits<ArchKind::KMB>::MemoryTraits<PhysicalMemory::CSRAM> final {
    static constexpr int32_t CLUSTERS_COUNT = 0;
    static constexpr int32_t CLUSTER_SIZE_KB = 0;
    static constexpr float DERATE_FACTOR = 0.85f;  // Derate factor for bandwidth
    static constexpr int32_t BANDWIDTH = 64;       // Bandwidth in Bytes/cycles
};

template <>
struct ArchTraits<ArchKind::KMB>::MemoryTraits<PhysicalMemory::CMX_UPA> final {
    static constexpr int32_t CLUSTERS_COUNT = 1;
    static constexpr int32_t CLUSTER_SIZE_KB = 4096;
    static constexpr float DERATE_FACTOR = 0.85f;  // Derate factor for bandwidth
    static constexpr int32_t BANDWIDTH = 16;       // Bandwidth in Bytes/cycles
};

template <>
struct ArchTraits<ArchKind::KMB>::MemoryTraits<PhysicalMemory::CMX_NN> final {
    static constexpr int32_t CLUSTERS_COUNT = 4;
    static constexpr int32_t CLUSTER_SIZE_KB = 1024;
    static constexpr float DERATE_FACTOR = 1.0f;  // Derate factor for bandwidth
    static constexpr int32_t BANDWIDTH = 32;      // Bandwidth in Bytes/cycles
};

//
// Run-time information
//

int32_t getProcessorUnitCount(ArchKind arch, PhysicalProcessor kind);

int32_t getDmaEngineCount(ArchKind arch, DMAEngine kind);

int32_t getMemoryClustersCount(ArchKind arch, PhysicalMemory kind);
int32_t getMemoryClusterSizeKB(ArchKind arch, PhysicalMemory kind);
float getMemoryDerateFactor(ArchKind arch, PhysicalMemory kind);
int32_t getMemoryBandwidth(ArchKind arch, PhysicalMemory kind);

}  // namespace VPUIP
}  // namespace vpux
