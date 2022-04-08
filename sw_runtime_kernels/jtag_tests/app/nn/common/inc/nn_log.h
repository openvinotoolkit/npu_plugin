/*
* {% copyright %}
*/
#ifndef NN_LOG_H_
#define NN_LOG_H_

inline void nnLogParamSink(...)
{
}

#if NN_LOG_VERBOSITY >= 1
#include <mvLog.h>
#define nnLog_MVLOG_FATAL(...) mvLog(MVLOG_FATAL, __VA_ARGS__)
#define NNLOG_DEFAULT_LEVEL MVLOG_FATAL
#else
#define nnLog_MVLOG_FATAL(...) nnLogParamSink(__VA_ARGS__)
#define NNLOG_DEFAULT_LEVEL MVLOG_LAST
#endif

#if NN_LOG_VERBOSITY >= 2
#define nnLog_MVLOG_ERROR(...) mvLog(MVLOG_ERROR, __VA_ARGS__)
#undef  NNLOG_DEFAULT_LEVEL
#define NNLOG_DEFAULT_LEVEL MVLOG_ERROR
#else
#define nnLog_MVLOG_ERROR(...) nnLogParamSink(__VA_ARGS__)
#endif

#if NN_LOG_VERBOSITY >= 3
#define nnLog_MVLOG_WARN(...) mvLog(MVLOG_WARN, __VA_ARGS__)
#undef  NNLOG_DEFAULT_LEVEL
#define NNLOG_DEFAULT_LEVEL MVLOG_WARN
#else
#define nnLog_MVLOG_WARN(...) nnLogParamSink(__VA_ARGS__)
#endif

#if NN_LOG_VERBOSITY >= 4
#define nnLog_MVLOG_PERF(...) mvLog(MVLOG_WARN, __VA_ARGS__)
#undef  NNLOG_DEFAULT_LEVEL
#define NNLOG_DEFAULT_LEVEL MVLOG_WARN
#else
#define nnLog_MVLOG_PERF(...) nnLogParamSink(__VA_ARGS__)
#endif

#if NN_LOG_VERBOSITY >= 5
#define nnLog_MVLOG_INFO(...) mvLog(MVLOG_INFO, __VA_ARGS__)
#undef  NNLOG_DEFAULT_LEVEL
#define NNLOG_DEFAULT_LEVEL MVLOG_INFO
#else
#define nnLog_MVLOG_INFO(...) nnLogParamSink(__VA_ARGS__)
#endif

#if NN_LOG_VERBOSITY >= 6
#define nnLog_MVLOG_DEBUG(...) mvLog(MVLOG_DEBUG, __VA_ARGS__)
#undef  NNLOG_DEFAULT_LEVEL
#define NNLOG_DEFAULT_LEVEL MVLOG_DEBUG
#else
#define nnLog_MVLOG_DEBUG(...) nnLogParamSink(__VA_ARGS__)
#endif

#define nnLog(level, ...) nnLog_##level(__VA_ARGS__)

#endif // NN_LOG_H_
