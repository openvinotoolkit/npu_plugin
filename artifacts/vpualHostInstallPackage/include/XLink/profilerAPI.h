///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
///
/// @file      profilerAPI.h
/// 

#ifndef __PROFILER_API_H_
#define __PROFILER_API_H_
#ifdef __cplusplus
extern "C" {
#endif

#if defined(TRACE_PROFILER_ENABLED) || \
    defined(LOS_PROFILE) || \
    defined(LRT_PROFILE) || \
    defined(SHAVE_PROFILE) || \
    defined(SAMPLE_PROFILING)

static inline void enableProfiler() {
    extern int __profileEnable; // do not use this directly, may change without notice
    __profileEnable = 1;
}

static inline void disableProfiler() {
    extern int __profileEnable; // do not use this directly, may change without notice
    __profileEnable = 0;
}

static inline int profilerEnabled() {
    extern int __profileEnable; // do not use this directly, may change without notice
    return __profileEnable;
}

#else

#define enableProfiler()
#define disableProfiler()
#define profilerEnabled() (0)

#endif // one of the profilers defined

#ifdef __cplusplus
}
#endif
#endif // __PROFILER_API_H_
