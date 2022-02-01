#include "simulator.hpp"

#include <array>
#include <cstddef>
#include <cstdlib>

namespace vpux {
namespace movisim {

const uint64_t Simulator::VPU_DEFAULT_BASE_MEMORY_ADDRESS = 0x80000000;
const uint64_t Simulator::VPU_DEFAULT_MEMORY_SIZE = 0x80000000;

const std::map<Simulator::SimArch, std::string> Simulator::m_archStringMap = {{Simulator::ARCH_KMB, "-cv:ma2x9x"},
                                                                              {Simulator::ARCH_MTL, "-cv:3720xx"}};

Simulator::Simulator(llvm::StringLiteral instanceName, SimArch arch, Logger& logger)
        : m_instanceName(instanceName),
          m_logger(logger.nest(m_instanceName, 4)),
          m_arch(arch),
          m_binding{},
          m_interface{m_binding.interface()},
          m_vpInterface(instanceName, m_logger.nest(m_instanceName, 4), VPU_DEFAULT_BASE_MEMORY_ADDRESS,
                        VPU_DEFAULT_MEMORY_SIZE) {
    m_interface.setVpInterface(&m_vpInterface);

    std::vector<const char*> args{nullptr};

    auto moviLogLevel = VpuAccessInterface::vpuxLogToMovisim.find(m_logger.level())->second;
    // Will always enable pipePrint for now. Need it for polling of app-end;
    m_interface.setVerbosity(moviLogLevel | static_cast<uint32_t>(VpuAccessInterface::moviLog::MOVISIM_PIPEPRINT));

    m_logger.debug("Setting verbosity of MoviSim service to {0}",
                   moviLogLevel | static_cast<uint32_t>(VpuAccessInterface::moviLog::MOVISIM_PIPEPRINT));
    if (m_logger.level() < vpux::LogLevel::Debug) {
        args.emplace_back("-nodasm");
    }

    args.emplace_back(m_archStringMap.find(m_arch)->second.data());

    m_interface.initialize(args.size(), const_cast<char**>(args.data()),
                           const_cast<char*>(MV_TOOLS_PATH "/linux64/bin"));

    m_logger.trace("Movisim  setting external memAdrrRange: {0} size {1}", VPU_DEFAULT_BASE_MEMORY_ADDRESS,
                   VPU_DEFAULT_MEMORY_SIZE);
    m_interface.setExternalMemoryAddressRange(VPU_DEFAULT_BASE_MEMORY_ADDRESS, VPU_DEFAULT_MEMORY_SIZE);
}

void Simulator::loadFile(const std::string& path) {
    std::string coreName;

    if (m_arch == ARCH_KMB) {
        coreName = "LRT0:";
    } else if (m_arch == ARCH_MTL) {
        coreName = "LRT:";
    }

    const std::string arg = coreName + path;
    char* argPtr = const_cast<char*>(arg.data());

    m_logger.trace("Movisim loading file {0}", arg);

    m_interface.loadFiles(1, &argPtr);
}

bool Simulator::waitForExpectedOutputs(const std::chrono::seconds timeout) {
    return m_vpInterface.waitForResults(timeout);
}

}  // namespace movisim
}  // namespace vpux
