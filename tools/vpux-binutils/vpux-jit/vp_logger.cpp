#include "vp_logger.hpp"

#include <iostream>
#include <string>
#include <thread>

#include <vpux/utils/core/error.hpp>

namespace vpux {
namespace movisim {

// movisim has maskFlags based log setting, not level-based. Theese maps should be the glue layer
const std::map<::movisim::MovisimLogLevel, vpux::LogLevel> VpuAccessInterface::moviSimToVpuxLog = {
        {moviLog::MOVISIM_NONE, vpux::LogLevel::None},      {moviLog::MOVISIM_FATAL, vpux::LogLevel::Fatal},
        {moviLog::MOVISIM_ERROR, vpux::LogLevel::Error},    {moviLog::MOVISIM_WARNING, vpux::LogLevel::Warning},
        {moviLog::MOVISIM_PIPEPRINT, vpux::LogLevel::Info}, {moviLog::MOVISIM_INFO, vpux::LogLevel::Debug}};

const std::map<vpux::LogLevel, uint32_t> VpuAccessInterface::vpuxLogToMovisim = {
        {vpux::LogLevel::None, moviLog::MOVISIM_NONE},
        {vpux::LogLevel::Fatal, moviLog::MOVISIM_FATAL},
        {vpux::LogLevel::Error, moviLog::MOVISIM_FATAL | moviLog::MOVISIM_ERROR},
        {vpux::LogLevel::Warning, moviLog::MOVISIM_FATAL | moviLog::MOVISIM_ERROR | moviLog::MOVISIM_WARNING},
        {vpux::LogLevel::Info,
         moviLog::MOVISIM_FATAL | moviLog::MOVISIM_ERROR | moviLog::MOVISIM_WARNING | moviLog::MOVISIM_PIPEPRINT},
        {vpux::LogLevel::Debug, moviLog::MOVISIM_FATAL | moviLog::MOVISIM_ERROR | moviLog::MOVISIM_WARNING |
                                        moviLog::MOVISIM_PIPEPRINT | moviLog::MOVISIM_INFO},
        {vpux::LogLevel::Trace, moviLog::MOVISIM_FATAL | moviLog::MOVISIM_ERROR | moviLog::MOVISIM_WARNING |
                                        moviLog::MOVISIM_PIPEPRINT | moviLog::MOVISIM_INFO}};

using Clock = std::conditional<std::chrono::high_resolution_clock::is_steady, std::chrono::high_resolution_clock,
                               std::chrono::steady_clock>::type;

inline static int memcpy_safe(void* dest, size_t destSize, const void* const src, size_t count) {
    VPUX_THROW_WHEN(dest == nullptr, "MoviSim VPU Interface requested R/W to NULL");
    VPUX_THROW_WHEN(destSize > std::numeric_limits<size_t>::max(),
                    "MoviSize VPU Interface requested R/W exceeding allocated memory");
    VPUX_THROW_WHEN(count > std::numeric_limits<size_t>::max(),
                    "MoviSize VPU Interface requested R/W exceeding allocated memory");
    VPUX_THROW_WHEN(count > destSize, "MoviSize VPU Interface requested R/W out of bounds");
    VPUX_THROW_WHEN(src == nullptr, "Movisim VPU Interface requested R/W to NULL");

    // Copying shall not take place between regions that overlap.
    if (((dest > src) && (dest < (static_cast<const char*>(src) + count))) ||
        ((src > dest) && (src < (static_cast<char*>(dest) + destSize)))) {
        VPUX_THROW("MoviSim VPU Interface requested overlapping R/W");
    }

    memcpy(dest, src, count);
    return 0;
}

void VpuAccessInterface::vpLogMessage(const ::movisim::MovisimLogLevel logLevel, const char* message) {
    // if (logLevel <= m_logLevel || logLevel == LogLevel::MOVISIM_PIPEPRINT)

    // std::cout << "moviSimLog |" << logLevel << "| " << message << '\n';
    // std::cout << message << std::endl;

    auto logMapEntry = moviSimToVpuxLog.find(logLevel);
    VPUX_THROW_WHEN(logMapEntry == moviSimToVpuxLog.end(), "Could not find corresponding VPUX Log level to {0}",
                    logLevel);
    auto vpuxLogLevel = logMapEntry->second;

    m_logger.addEntry(vpuxLogLevel, "VPIFace{0} : {1}", logLevel, message);

    if (logLevel == ::movisim::MovisimLogLevel::MOVISIM_PIPEPRINT) {
        m_expectedResults.erase(message);
    }
}

void VpuAccessInterface::addExpectedResult(std::string output) {
    m_expectedResults.emplace(std::move(output));
    return;
}

bool VpuAccessInterface::waitForResults(const std::chrono::seconds timeout) {
    auto startTime = Clock::now();

    while (true) {
        if (m_expectedResults.empty())
            return true;

        auto elapsed = Clock::now() - startTime;
        if (elapsed > timeout) {
            m_logger.error("MoviSim application {0} failed (timeout: {1} seconds)", m_instanceName, elapsed.count());
            return false;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

bool VpuAccessInterface::vpMemRead(const uint64_t address, const uint32_t size, void* buffer, uint32_t streamID,
                                   uint32_t subStreamID, bool& subStreamIDValid, uint32_t busId) {
    void* relativeAddress = (void*)(m_vpuMemory.get() + (address - m_vpuBaseAddr));
    m_logger.trace("Movisim VpuMemInterface Read {0:x}||{1:x} {2:x} {3}", address, relativeAddress, buffer, size);

    memcpy_safe(buffer, size, relativeAddress, size);
    // std::cout << "VpvpMemRead- " << std::hex << address<<" - "<< std::hex << *((uint64_t*)buffer) << " -size:" <<
    // std::hex << size << std::endl;
    // checkExpectedMemAccesses(address);
    return true;
}

bool VpuAccessInterface::vpMemWrite(const uint64_t address, const uint32_t size, const void* buffer, uint32_t streamID,
                                    uint32_t subStreamID, bool& subStreamIDValid, uint32_t busId) {
    void* relativeAddress = (void*)(m_vpuMemory.get() + (address - m_vpuBaseAddr));

    m_logger.trace("Movisim VpuMemInterface Write {0:x}||{1:x} {2:x} {3} {4}", address, relativeAddress, buffer, size);
    memcpy_safe(relativeAddress, size, buffer, size);
    /*
    std::cout << "vpMemWrite address: " << std::hex << address
        << ", data: " << std::hex << *((uint64_t*)buffer) << ", size:"<< std::hex << size
        << ", streamID: " << std::hex << streamID << ", subStreamID:" << std::hex << subStreamID
        << ", busId:" << std::hex << busId
        << std::endl;
        */
    // checkExpectedMemAccesses(address);
    return true;
}

bool VpuAccessInterface::vpSignal(const uint64_t core_id, const uint32_t int_id, bool state) {
    // std::cout << "vpAccess::vpSignal | " << core_id << " | " << std::hex << int_id << std::dec << " | " << state <<
    // std::endl; ext_interrupt newint(core_id, int_id, state); checkInterrupts(newint);

    m_logger.warning("Movisim requested signal??? {0} {1} {2}", core_id, int_id, state);
    return false;
}

bool VpuAccessInterface::vpRequestMemoryPointer(uint64_t& address, void** memPtr, uint32_t& size) {
    // m_logger.warning("Movisim requested memoryPtr??? {0:x} {1:x} {2}", address, memPtr, size);
    return false;
    // address -= (address % 4);
    // size = 0x10000;
    // *memPtr = malloc(0x10000 * sizeof(uint32_t));
    // //*memPtr = memory;
    // return true;
}

}  // namespace movisim
}  // namespace vpux
