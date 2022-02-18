#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <utility>

#include <movisim_interface.h>

#include <vpux/utils/core/error.hpp>
#include "runner/movisim/binding.hpp"

#include "vp_logger.hpp"

namespace vpux {
namespace movisim {

#define MB(_val_) (static_cast<uint64_t>(_val_) * 1024 * 1024)

class Simulator final {
public:
    static const uint64_t VPU_DEFAULT_MEMORY_SIZE;
    static const uint64_t VPU_DEFAULT_BASE_MEMORY_ADDRESS;

    enum SimArch { ARCH_KMB, ARCH_MTL };
    static const std::map<SimArch, std::string> m_archStringMap;

    Simulator(llvm::StringLiteral instanceName, SimArch arch, Logger& logger);
    Simulator(const Simulator&) = delete;
    Simulator(Simulator&&) = delete;

    Simulator& operator=(const Simulator&) = delete;
    Simulator& operator=(Simulator&&) = delete;

    ~Simulator() {
        m_interface.exit();
    }

    void loadFile(const std::string& path);

    void expectOutput(std::string value) {
        m_vpInterface.addExpectedResult(std::move(value));
    }
    bool waitForExpectedOutputs(const std::chrono::seconds timeout);

    void start() {
        m_interface.run(0);
    }
    void reset() {
        m_interface.reset(0);
    }
    void stop() {
        m_interface.stop();
    }
    void resume() {
        m_vpInterface.m_executionMutex.lock();
        m_interface.resume();
    }

    uint8_t* vpuWindow(const uint64_t vpuAddr) const {
        return m_vpInterface.vpuWindow(vpuAddr);
    }

    template <typename T>
    void read(const std::uint32_t startAddress, T* const location, const std::uint32_t size);
    template <typename T>
    void write(std::uint32_t& startAddress, const T* const data, const std::uint32_t size);
    template <typename T>
    void write(const std::uint32_t& startAddress, const T* const data, const std::uint32_t size);
    template <typename T>
    void write(std::uint32_t& startAddress, const T& object);

private:
    static constexpr std::uint32_t m_chunkSize = 8;
    static constexpr std::uint32_t m_maximumWriteSize = 31;

    llvm::StringLiteral m_instanceName;
    Logger m_logger;
    SimArch m_arch;
    mv::emu::runner::movisim::Binding m_binding;
    mv::emu::runner::movisim::Interface& m_interface;
    VpuAccessInterface m_vpInterface;
};

template <typename T>
void Simulator::write(std::uint32_t& startAddress, const T* const data, const std::uint32_t size) {
    const std::uint32_t sizeInBytes = size * sizeof(T);
    const std::uint32_t prePadding = startAddress % m_chunkSize;
    const std::uint32_t postPadding = m_chunkSize - (startAddress + sizeInBytes) % m_chunkSize;

    std::vector<char> bytes(prePadding + sizeInBytes + postPadding);

    read<char>(startAddress - prePadding, bytes.data(), bytes.size());
    std::copy_n(reinterpret_cast<const char*>(data), sizeInBytes, std::next(bytes.begin(), prePadding));

    auto xx = m_chunkSize;
    VPUX_THROW_UNLESS(bytes.size() % m_chunkSize == 0, "Invalid chunk size RW {0} {1}",bytes.size(), xx);
    static_assert(m_chunkSize <= m_maximumWriteSize, "chunk size larger than maximum write size");

    const std::uint32_t paddedStartAddress = startAddress - prePadding;
    for (std::uint32_t offset = 0; offset < bytes.size(); offset += m_chunkSize)
        m_interface.movisimMemWrite(paddedStartAddress + offset, m_chunkSize, bytes.data() + offset);
    startAddress += sizeInBytes;
}

template <typename T>
void Simulator::read(const std::uint32_t startAddress, T* const location, const std::uint32_t size) {
    std::size_t cycles;
    const std::uint32_t sizeInBytes = size * sizeof(T);
    const std::uint32_t prePadding = startAddress % m_chunkSize;
    const std::uint32_t postPadding = m_chunkSize - (startAddress + sizeInBytes) % m_chunkSize;

    std::vector<char> bytes(prePadding + sizeInBytes + postPadding);
    m_interface.movisimMemRead(startAddress - prePadding, bytes.size(), bytes.data(), cycles);
    std::copy_n(reinterpret_cast<const T*>(bytes.data() + prePadding), size, location);
}

template <typename T>
void Simulator::write(const std::uint32_t& startAddress, const T* const data, const std::uint32_t size) {
    std::uint32_t mutableStartAddress = startAddress;
    write<T>(mutableStartAddress, data, size);
}

template <typename T>
void Simulator::write(std::uint32_t& startAddress, const T& object) {
    write<T>(startAddress, &object, 1);
}

}  // namespace movisim
}  // namespace vpux
