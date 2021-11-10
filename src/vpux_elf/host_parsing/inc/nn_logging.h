/*
 * {% copyright %}
 */
#pragma once
#if __has_include(<nn_log.h>)
#include <nn_log.h>
#else
inline void nnLogParamSink(...) {}

#if NN_LOG_VERBOSITY >= 1
#define nnLog_MVLOG_FATAL(...) printf("FATAL: " __VA_ARGS__); printf("\n");
#define NNLOG_DEFAULT_LEVEL MVLOG_FATAL
#else
#define nnLog_MVLOG_FATAL(...) nnLogParamSink(__VA_ARGS__)
#define NNLOG_DEFAULT_LEVEL MVLOG_LAST
#endif

#if NN_LOG_VERBOSITY >= 2
#define nnLog_MVLOG_ERROR(...) printf("ERROR: " __VA_ARGS__); printf("\n");
#undef NNLOG_DEFAULT_LEVEL
#define NNLOG_DEFAULT_LEVEL MVLOG_ERROR
#else
#define nnLog_MVLOG_ERROR(...) nnLogParamSink(__VA_ARGS__)
#endif

#if NN_LOG_VERBOSITY >= 3
#define nnLog_MVLOG_WARN(...) printf("WARN: " __VA_ARGS__); printf("\n");
#undef NNLOG_DEFAULT_LEVEL
#define NNLOG_DEFAULT_LEVEL MVLOG_WARN
#else
#define nnLog_MVLOG_WARN(...) nnLogParamSink(__VA_ARGS__)
#endif

#if NN_LOG_VERBOSITY >= 4
#define nnLog_MVLOG_PERF(...) printf("PERF: "  __VA_ARGS__); printf("\n");
#undef NNLOG_DEFAULT_LEVEL
#define NNLOG_DEFAULT_LEVEL MVLOG_WARN
#else
#define nnLog_MVLOG_PERF(...) nnLogParamSink(__VA_ARGS__)
#endif

#if NN_LOG_VERBOSITY >= 5
#define nnLog_MVLOG_INFO(...) printf("INFO: "  __VA_ARGS__); printf("\n");
#undef NNLOG_DEFAULT_LEVEL
#define NNLOG_DEFAULT_LEVEL MVLOG_INFO
#else
#define nnLog_MVLOG_INFO(...) nnLogParamSink(__VA_ARGS__)
#endif

#if NN_LOG_VERBOSITY >= 6
#define nnLog_MVLOG_DEBUG(...) printf("DEBUG: " __VA_ARGS__); printf("\n");
#undef NNLOG_DEFAULT_LEVEL
#define NNLOG_DEFAULT_LEVEL MVLOG_DEBUG
#else
#define nnLog_MVLOG_DEBUG(...) nnLogParamSink(__VA_ARGS__)
#endif

#define nnLog(level, ...) nnLog_##level(__VA_ARGS__)
#endif
