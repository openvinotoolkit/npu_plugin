//
// Copyright 2020 Intel Corporation.
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

#include <fstream>
#include <string>

namespace MemoryUsage {
inline void procMemUsage(double& vm_usage, double& res_usage) {
    vm_usage = 0.0;
    res_usage = 0.0;

#if !defined(_WIN32) && !defined(__arm__) && !defined(__aarch64__)

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
