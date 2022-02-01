#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>

#include <vpux/utils/core/logger.hpp>

#include <movisim_interface.h>

namespace vpux {
namespace movisim {

class VpuAccessInterface final : public ::movisim::VpInterface {
public:
    using moviLog = ::movisim::MovisimLogLevel;

    static const std::map<::movisim::MovisimLogLevel, vpux::LogLevel> moviSimToVpuxLog;
    static const std::map<vpux::LogLevel, uint32_t> vpuxLogToMovisim;

    VpuAccessInterface(llvm::StringLiteral instanceName, Logger logger, uint64_t vpuBaseAddr, uint64_t vpuMemSize)
            : m_instanceName(instanceName),
              m_logger(logger),
              m_expectedResults{},
              m_vpuMemory(new uint8_t[vpuMemSize]),
              m_vpuBaseAddr(vpuBaseAddr) {
        bzero(m_vpuMemory.get(), vpuMemSize);
    }

    // callback function called by movisim on Prints and log requests
    void vpLogMessage(const ::movisim::MovisimLogLevel logLevel, const char* str) final;

    bool vpMemRead(uint64_t, uint32_t, void*, uint32_t, uint32_t, bool&, uint32_t) final;
    bool vpMemWrite(uint64_t, uint32_t, const void*, uint32_t, uint32_t, bool&, uint32_t) final;
    bool vpSignal(const std::uint64_t, const std::uint32_t, bool) final;
    bool vpRequestMemoryPointer(std::uint64_t&, void**, std::uint32_t&) final;

    // poll-based mechanism to wait for a specific output string emitted by the logger
    void addExpectedResult(std::string output);
    bool waitForResults(const std::chrono::seconds timeout);

private:
    llvm::StringLiteral m_instanceName;
    Logger m_logger;
    std::unordered_set<std::string> m_expectedResults;

    std::unique_ptr<uint8_t> m_vpuMemory;
    const uint64_t m_vpuBaseAddr;
};

}  // namespace movisim
}  // namespace vpux
