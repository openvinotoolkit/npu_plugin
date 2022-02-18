//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <vpux_elf/types/symbol_entry.hpp>
#include <vpux_elf/utils/error.hpp>
#include <vpux_elf/utils/log.hpp>
#include <vpux_loader/vpux_loader.hpp>

using namespace elf;

namespace vpux {
namespace binutils {
class FlatHexBufferManager : public BufferManager {
public:
    FlatHexBufferManager(uint32_t startAddr, size_t size, uint8_t* const buffer)
                : m_startAddr(startAddr), m_totalSize(size), m_buffer(buffer), m_tracker(m_buffer) {

    }

    DeviceBuffer allocate(size_t alignment, size_t size) override {
        if (!m_buffer) {
            VPUX_THROW("Failed to allocate overall buffer of size {0}", size);
            return elf::DeviceBuffer(nullptr, 0, 0);
        }

        m_tracker = align_up<uint8_t>(m_tracker, alignment);
        uint8_t* start = m_tracker;
        m_tracker += size;

        if (m_tracker >= m_buffer + m_totalSize) {
            VPUX_THROW(
                    "Failed to allocate required buff of size {0} alignment {1} . Exceeding total device buffer space",
                    size, alignment);
            return elf::DeviceBuffer(nullptr, 0, 0);
        }

        return elf::DeviceBuffer(start, vpuWindow(start), size);
    }

    void deallocate(elf::DeviceBuffer& devBuffer) override {
        (void)devBuffer;
    }

    size_t copy(elf::DeviceBuffer& to, const uint8_t* from, size_t count) {
        memcpy(to.cpu_addr(), from, count);
        return count;
    }

    uint8_t* buffer() const {
        return m_buffer;
    }
    size_t size() const {
        return m_tracker - m_buffer;
    }

    uint32_t vpuBaseAddr() const {
        return m_startAddr;
    }

protected:
    template <typename T>
    bool isPowerOfTwo(T val) {
        return val && ((val & (val - 1)) == 0);
    }

    template <typename T>
    T* align_up(const T* val, const size_t to) {
        VPUX_ELF_THROW_UNLESS(isPowerOfTwo(to), " VPU only supports power of 2 alignments {0}", to);
        std::uintptr_t intPtr = reinterpret_cast<std::uintptr_t>(val);
        std::uintptr_t alignedAddr = (intPtr + to - 1) & ~(to - 1);
        return reinterpret_cast<T*>(alignedAddr);
    }

    uint32_t vpuWindow(uint8_t const* const addr) const {
        return m_startAddr + static_cast<uint32_t>(addr - m_buffer);
    }

    uint32_t const m_startAddr;
    size_t const m_totalSize;
    uint8_t* const m_buffer;
    uint8_t* m_tracker;
};

class OwningFlatHexBufferManager : public FlatHexBufferManager {
public:
    OwningFlatHexBufferManager(uint32_t startAddr, size_t size)
                : FlatHexBufferManager(startAddr, size, new uint8_t[size]) {
                }

    ~OwningFlatHexBufferManager() {
        delete[] m_buffer;
    }
};
// TODO(EISW-23975): This beautiful piece of code contains the "special symtab" that normally needs to be queried from
// the runtime. In IMDemo example we have a similar class that constructs this symTab based on data from the
// InferenceRuntimeService. Since we cannot include that in kmb-plugin, we will occasionally manually check the values,
// and update them here Hopefully if we solve the riddle of integration between KmbPlugin and vpuip_2, then we will not
// have to resort to magical solutions (This is my wish to SantaClaus this year :) ) This ticket will not totally solve
// the problem, but will greatly reduce the hack-ishness of this solution

class HardCodedSymtabToCluster0 {
private:
    static constexpr size_t SPECIAL_SYMTAB_SIZE = 7;  // I counted!!!! Twice!!
    elf::SymbolEntry symTab_[SPECIAL_SYMTAB_SIZE];

public:
    HardCodedSymtabToCluster0(): symTab_() {
        for (size_t i = 0; i < SPECIAL_SYMTAB_SIZE; ++i) {
            symTab_[i].st_info = static_cast<unsigned char>(elf64STInfo(STB_GLOBAL, STT_OBJECT));
            symTab_[i].st_other = STV_DEFAULT;
            symTab_[i].st_shndx = 0;
            symTab_[i].st_name = 0;
        }

        symTab_[VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR].st_value = 0x2e014000;
        symTab_[VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR].st_size = 2097152;

        symTab_[VPU_NNRD_SYM_RTM_IVAR].st_value = 0x2e004000;
        symTab_[VPU_NNRD_SYM_RTM_IVAR].st_size = 64;

        symTab_[VPU_NNRD_SYM_RTM_ACT].st_value = 0;
        symTab_[VPU_NNRD_SYM_RTM_ACT].st_size = 0;

        symTab_[VPU_NNRD_SYM_RTM_DMA0].st_value = 0x2e1f8000;
        symTab_[VPU_NNRD_SYM_RTM_DMA0].st_size = 64;

        symTab_[VPU_NNRD_SYM_RTM_DMA1].st_value = 0x2e1fc000;
        symTab_[VPU_NNRD_SYM_RTM_DMA1].st_size = 64;

        symTab_[VPU_NNRD_SYM_FIFO_BASE].st_value = 0x0;
        symTab_[VPU_NNRD_SYM_FIFO_BASE].st_size = 0;

        symTab_[VPU_NNRD_SYM_BARRIERS_START].st_value = 0;
        symTab_[VPU_NNRD_SYM_BARRIERS_START].st_size = 0;
    }

    const elf::details::ArrayRef<SymbolEntry> symTab() const {
        return elf::details::ArrayRef<SymbolEntry>(symTab_, SPECIAL_SYMTAB_SIZE);
    }
};

struct HexMappedInferenceEntry {
    uint32_t elfEntryPtr;
    uint32_t totalSize;
    uint32_t inputsPtr;
    uint32_t inputSizesPtr;
    uint32_t inputsCount;
    uint32_t outputsPtr;
    uint32_t outputSizesPtr;
    uint32_t outputsCount;
};

}  // namespace binutils
}  // namespace vpux
