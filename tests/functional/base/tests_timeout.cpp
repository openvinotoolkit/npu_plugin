//
// Copyright 2019 Intel Corporation.
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

#include "tests_timeout.hpp"
#include <unistd.h>

namespace TestsTimeout
{

void cleanPendingSignals(sigset_t& sigset, siginfo_t& info, int millySecCleanPendingTimeout) {
    timespec timeout = {0, millySecCleanPendingTimeout * 1000000}; // milliseconds * 10^6 -> nanoseconds
    while (-1 != sigtimedwait(&sigset, &info, &timeout));
}

int runWithTimeout (
        const std::function<void(int&)>& runFunc,
        std::string& statusMessage,
        int secRunTimeout, int millySecKillSignalWait, int millySecCleanPendingTimeout) {
    std::ostringstream statusMessageStream;

    int runStatus;
    int retStatus = RunStatus::UNKNOWN;
    if (secRunTimeout == 0) { // without timeout
        runFunc(retStatus);
    } else {
        sigset_t sigset;
        sigemptyset(&sigset);
        sigaddset(&sigset, SIGCHLD);
        sigprocmask(SIG_BLOCK, &sigset, nullptr);

        int chPid = -1;
        chPid = fork();
        if (!chPid) {
            runFunc(runStatus);
            exit(runStatus);
        } else {
            timespec timeout = {secRunTimeout, 0};
            siginfo_t info;
            int signo = -100;
            // Wait for the child process to terminate or timeout.
            signo = sigtimedwait(&sigset, &info, &timeout);
            if(-1 == signo) {
                if(EAGAIN == errno) { // Timed out.
                    kill(chPid, SIGKILL);
                    usleep(millySecKillSignalWait * 1000); // waiting for signals to clean them before the next test started (millyseconds * 10^3 -> microseconds)
                    cleanPendingSignals(sigset, info, millySecCleanPendingTimeout);
                    statusMessageStream << "TIMEOUT: " << secRunTimeout << " s";
                    retStatus = RunStatus::TIMEOUT;
                } else {
                    cleanPendingSignals(sigset, info, millySecCleanPendingTimeout);
                    statusMessageStream << "UEXPECTED SIGNAL CATCHED: sigtimedwait responce: " << signo << " errno: " << errno;
                    retStatus = RunStatus::UEXPECTED;
                }
            } else { // The child has terminated.
                cleanPendingSignals(sigset, info, millySecCleanPendingTimeout);
                statusMessageStream << "";
                retStatus = info.si_status;
            }
        }
    }
    statusMessage = statusMessageStream.str();
    return retStatus;
}

} // namespace TestsTimeout
