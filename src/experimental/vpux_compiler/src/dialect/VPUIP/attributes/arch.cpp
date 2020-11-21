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

#include "vpux/compiler/dialect/VPUIP/attributes/arch.hpp"

using namespace vpux;

//
// KMB
//

StringRef vpux::VPUIP::ArchTraits<VPUIP::ArchKind::KMB>::getArchGenName() {
    return "VPU 2.0";
}

//
// Run-time information
//

int32_t vpux::VPUIP::getProcessorUnitCount(ArchKind arch, PhysicalProcessor kind) {
#define CASE(_arch_, _proc_)        \
    case PhysicalProcessor::_proc_: \
        return ArchTraits<ArchKind::_arch_>::ProcessorTraits<PhysicalProcessor::_proc_>::COUNT

    switch (arch) {
    case ArchKind::KMB:
        switch (kind) {
            CASE(KMB, ARM);
            CASE(KMB, Leon_RT);
            CASE(KMB, Leon_NN);
            CASE(KMB, SHAVE_UPA);
            CASE(KMB, SHAVE_NN);
            CASE(KMB, NCE_Cluster);
            CASE(KMB, NCE_PerClusterDPU);
        default:
            return 0;
        }
    default:
        return 0;
    }

#undef CASE
}

int32_t vpux::VPUIP::getDmaEngineCount(ArchKind arch, DMAEngine kind) {
#define CASE(_arch_, _dma_) \
    case DMAEngine::_dma_:  \
        return ArchTraits<ArchKind::_arch_>::DMATraits<DMAEngine::_dma_>::COUNT

    switch (arch) {
    case ArchKind::KMB:
        switch (kind) {
            CASE(KMB, UPA);
            CASE(KMB, NN);
        default:
            return 0;
        }
    default:
        return 0;
    }

#undef CASE
}

int32_t vpux::VPUIP::getMemoryClustersCount(ArchKind arch, PhysicalMemory kind) {
#define CASE(_arch_, _mem_)     \
    case PhysicalMemory::_mem_: \
        return ArchTraits<ArchKind::_arch_>::MemoryTraits<PhysicalMemory::_mem_>::CLUSTERS_COUNT

    switch (arch) {
    case ArchKind::KMB:
        switch (kind) {
            CASE(KMB, DDR);
            CASE(KMB, CSRAM);
            CASE(KMB, CMX_UPA);
            CASE(KMB, CMX_NN);
        default:
            return 0;
        }
    default:
        return 0;
    }

#undef CASE
}

int32_t vpux::VPUIP::getMemoryClusterSizeKB(ArchKind arch, PhysicalMemory kind) {
#define CASE(_arch_, _mem_)     \
    case PhysicalMemory::_mem_: \
        return ArchTraits<ArchKind::_arch_>::MemoryTraits<PhysicalMemory::_mem_>::CLUSTER_SIZE_KB

    switch (arch) {
    case ArchKind::KMB:
        switch (kind) {
            CASE(KMB, DDR);
            CASE(KMB, CSRAM);
            CASE(KMB, CMX_UPA);
            CASE(KMB, CMX_NN);
        default:
            return 0;
        }
    default:
        return 0;
    }

#undef CASE
}

float vpux::VPUIP::getMemoryDerateFactor(ArchKind arch, PhysicalMemory kind) {
#define CASE(_arch_, _mem_)     \
    case PhysicalMemory::_mem_: \
        return ArchTraits<ArchKind::_arch_>::MemoryTraits<PhysicalMemory::_mem_>::DERATE_FACTOR

    switch (arch) {
    case ArchKind::KMB:
        switch (kind) {
            CASE(KMB, DDR);
            CASE(KMB, CSRAM);
            CASE(KMB, CMX_UPA);
            CASE(KMB, CMX_NN);
        default:
            return 0.0f;
        }
    default:
        return 0.0f;
    }

#undef CASE
}

int32_t vpux::VPUIP::getMemoryBandwidth(ArchKind arch, PhysicalMemory kind) {
#define CASE(_arch_, _mem_)     \
    case PhysicalMemory::_mem_: \
        return ArchTraits<ArchKind::_arch_>::MemoryTraits<PhysicalMemory::_mem_>::BANDWIDTH

    switch (arch) {
    case ArchKind::KMB:
        switch (kind) {
            CASE(KMB, DDR);
            CASE(KMB, CSRAM);
            CASE(KMB, CMX_UPA);
            CASE(KMB, CMX_NN);
        default:
            return 0;
        }
    default:
        return 0;
    }

#undef CASE
}
