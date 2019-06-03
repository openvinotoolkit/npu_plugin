// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <thread>

namespace HDDLTestsUtils {

inline bool runHddlService(std::string hddlServiceConfigFilename) {
#if defined(_WIN32) || defined(WIN32)
    std::cout << "Unavailable on windows due to issues with hddl-service" << std::endl;
    return false;
#endif
    if (hddlServiceConfigFilename.empty()) {
        return false;
    }

    const char *hddlInstallDir = std::getenv("HDDL_INSTALL_DIR");
    if (hddlInstallDir == nullptr) {
        return false;
    }

    std::stringstream runHddlServiceCmdStream;
    runHddlServiceCmdStream << hddlInstallDir << "/bin/hddldaemon -c " << hddlServiceConfigFilename;

    std::string runHddlServiceCmd = runHddlServiceCmdStream.str();
    std::thread t([runHddlServiceCmd]{int result = std::system(runHddlServiceCmd.c_str());});
    t.detach();
    // Waiting for initialization of hddldaemon
    std::this_thread::sleep_for(std::chrono::seconds(10));

    return true;
}


inline bool killHddlService() {
#if defined(_WIN32) || defined(WIN32)
    std::cout << "Unavailable on windows due to issues with hddl-service" << std::endl;
    return true;
#endif
    std::string killHddlServiceCmd;
#if defined(_WIN32) || defined(_WIN64)
    killHddlServiceCmd = "taskkill /F /IM hddldaemon.exe";
#else
    killHddlServiceCmd = "pkill -f hddldaemon";
#endif

    int result = std::system(killHddlServiceCmd.c_str());

    std::this_thread::sleep_for(std::chrono::seconds(1));

    return true;
}

} // HDDLTestsUtils
