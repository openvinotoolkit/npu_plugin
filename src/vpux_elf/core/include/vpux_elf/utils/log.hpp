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

#ifndef __VPUX_ELF_LOG_H__
#define __VPUX_ELF_LOG_H__

#define VPUX_ELF_FATAL_LEVEL 1
#define VPUX_ELF_ERROR_LEVEL 2
#define VPUX_ELF_WARN_LEVEL 3
#define VPUX_ELF_INFO_LEVEL 4
#define VPUX_ELF_TRACE_LEVEL 5
#define VPUX_ELF_DEBUG_LEVEL 6

#define VPUX_ELF_LOG_DEFAULT_LEVEL 6

#ifndef VPUX_ELF_LOG_LEVEL
#define VPUX_ELF_LOG_LEVEL VPUX_ELF_LOG_DEFAULT_LEVEL
#endif

#ifndef VPUX_ELF_ENABLE_LOGGING
#define VPUX_ELF_ENABLE_LOGGING 1
#endif

#ifdef __leon__
#include <mvLog.h>
#define VPUX_ELF_FATAL MVLOG_FATAL
#define VPUX_ELF_ERROR MVLOG_ERROR
#define VPUX_ELF_WARN MVLOG_WARN
#define VPUX_ELF_INFO MVLOG_INFO
#define VPUX_ELF_TRACE MVLOG_DEBUG
#define VPUX_ELF_DEBUG MVLOG_DEBUG

#define vpuxElfLogLevelSet(__lvl__) mvLogLevelSet(__lvl__)

#define vpuxElfLogFunc(__ELF_LOG_LEVEL__, ...) mvLog(__ELF_LOG_LEVEL__, __VA_ARGS__)
#else
#include "vpux/utils/core/logger.hpp"
#include "cstdio"

namespace elf {
namespace details {
class ElfLogger {
public:
    static vpux::Logger& instance();
private:
    ElfLogger();

    vpux::Logger m_elfLogger;
};

} // namespace details
} // namespace elf

#define vpuxElfLogLevelSet(__lvl__) elf::details::ElfLogger::instance().setLevel(static_cast<vpux::LogLevel>(__lvl__))

#define VPUX_ELF_FATAL elf::details::ElfLogger::instance().fatal
#define VPUX_ELF_ERROR elf::details::ElfLogger::instance().error
#define VPUX_ELF_WARN  elf::details::ElfLogger::instance().warning
#define VPUX_ELF_INFO  elf::details::ElfLogger::instance().info
#define VPUX_ELF_TRACE elf::details::ElfLogger::instance().trace
#define VPUX_ELF_DEBUG elf::details::ElfLogger::instance().debug

#define vpuxElfLogFunc(__ELF_LOG_LEVEL__, ...)      \
    do {                                            \
        char prntBuf[1024];                         \
        std::snprintf(prntBuf, 1024, __VA_ARGS__);    \
        __ELF_LOG_LEVEL__(prntBuf);                 \
    } while (0)                                     \

#endif

#define vpuxElfLog(LOG_TYPE, ...)                                               \
    do {                                                                        \
        constexpr bool elf_showlog___ = LOG_TYPE##_LEVEL <= VPUX_ELF_LOG_LEVEL; \
        if (elf_showlog___ && VPUX_ELF_ENABLE_LOGGING) {                        \
            vpuxElfLogFunc(LOG_TYPE, ##__VA_ARGS__);                            \
        }                                                                       \
    } while (0)

#endif // __VPUX_ELF_LOG_H__
