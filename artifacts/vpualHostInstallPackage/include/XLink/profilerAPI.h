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

__attribute__((no_instrument_function))
static inline void enableProfiler() {
    extern int __profileEnable; // do not use this directly, may change without notice
    __profileEnable = 1;
}
__attribute__((no_instrument_function))
static inline void disableProfiler() {
    extern int __profileEnable; // do not use this directly, may change without notice
    __profileEnable = 0;
}
__attribute__((no_instrument_function))
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
