//
// Copyright 2019 Intel Corporation.
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

#include "tests_timeout.hpp"

#include <unistd.h>

namespace TestsTimeout {

int runWithTimeout(const std::function<void(int&)>& runFunc, std::string& statusMessage, double dSecRunTimeout) {
    std::ostringstream statusMessageStream;

    int runStatus;
    int retStatus = RunStatus::UNKNOWN;
    if (dSecRunTimeout == 0.) {  // without timeout
        runFunc(retStatus);
    } else {
        sigset_t sigset;
        sigemptyset(&sigset);
        // It is supposed that signals are not used by the testing (target) function
        sigaddset(&sigset, SIGCHLD);
        sigprocmask(SIG_BLOCK, &sigset, nullptr);

        int chPid = -1;
        chPid = fork();
        if (chPid == 0) {
            // Child process
            try {
                runFunc(runStatus);
            } catch (...) {
                runStatus = RunStatus::EXECUTION_FAILURE;
            }
            exit(runStatus);
        } else if (chPid < 0) {
            statusMessageStream << "CAN NOT CREATE CHILD PROCES: fork() returned: " << chPid;
            retStatus = RunStatus::FORK_FAILURE;
        } else {
            // Parent process
            int sec = static_cast<int>(dSecRunTimeout);
            timespec timeout = {sec, static_cast<int>((dSecRunTimeout - static_cast<double>(sec)) * 1e9)};
            siginfo_t info;
            int signo = -100;
            // Wait for the child process to terminate or timeout.
            do {
                // Supposes that tests are not executed in parallel
                // because is is undefined which thread catch the signal.
                // 1 test and 1 thread of parent process should be executed in each one moment
                signo = sigtimedwait(&sigset, &info, &timeout);
                if (-1 == signo) {
                    if (EAGAIN == errno) {  // Timed out.
                        kill(chPid, SIGKILL);
                        statusMessageStream << "TIMEOUT: " << dSecRunTimeout << " s";
                        retStatus = RunStatus::TIMEOUT;
                    } else {
                        statusMessageStream << "UNEXPECTED SIGNAL CAUGHT: sigtimedwait response: " << signo
                                            << " errno: " << errno;
                        retStatus = RunStatus::UNEXPECTED;
                    }
                } else {  // The child has terminated.
                    if (info.si_pid == chPid) {
                        statusMessageStream << "";
                        retStatus = info.si_status;
                    }
                }
            } while (retStatus == RunStatus::UNKNOWN);
        }
    }
    statusMessage = statusMessageStream.str();
    return retStatus;
}

}  // namespace TestsTimeout
