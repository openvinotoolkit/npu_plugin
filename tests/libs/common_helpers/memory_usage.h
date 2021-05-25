//
// Copyright 2020 Intel Corporation.
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

#include <fstream>
#include <string>

namespace MemoryUsage {
inline void procMemUsage(double& vm_usage, double& res_usage) {
    vm_usage = 0.0;
    res_usage = 0.0;

#if !defined(_WIN32)

    std::ifstream stat_stream("/proc/self/stat", std::ios_base::in);

    // dummy vars
    std::string pid, comm, state, ppid, pgrp, session, tty_nr;
    std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    std::string utime, stime, cutime, cstime, priority, nice;
    std::string O, itrealvalue, starttime;

    // the two fields we want
    unsigned long vsize;
    long rss;

    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >> tpgid >> flags >> minflt >> cminflt >>
        majflt >> cmajflt >> utime >> stime >> cutime >> cstime >> priority >> nice >> O >> itrealvalue >> starttime >>
        vsize >> rss;

    stat_stream.close();

    double page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024.0;
    vm_usage = vsize / 1024.0;
    res_usage = rss * page_size_kb;
#endif
}
}
