/*
* {% copyright %}
*/
#pragma once

#if defined(__leon_rt__) || defined(__leon__)
#   include <nn_log.h>
#else
#   if defined(DEBUG_NN_SVU_RUNTIME)
#       include <stdio.h>
#       define dbgPrint(...) printf(__VA_ARGS__)
#   else
#       define dbgPrint(...)
#   endif
#endif

namespace nn {
namespace shave_lib {

#if defined(__leon_rt__) || defined(__leon__)
#   define logI(...) nnLog(MVLOG_INFO, __VA_ARGS__)
#   define logE(...) nnLog(MVLOG_ERROR, __VA_ARGS__)
#   define logW(...) nnLog(MVLOG_WARN, __VA_ARGS__)
#   define logD(...) nnLog(MVLOG_DEBUG, __VA_ARGS__)
#else
#   define logE(...) dbgPrint("%s:%d: [ERROR] ", __FILE__, __LINE__); dbgPrint(__VA_ARGS__); dbgPrint("\n")
#endif

#define RETURN_NULL_UNLESS(expr)                \
    if (!(expr)) {                              \
        logE(#expr" pointer is null");          \
        return nullptr;                         \
    }

#define RETURN_FALSE_UNLESS(expr, ...)          \
    if (!(expr)) {                              \
        logE(__VA_ARGS__);                      \
        return false;                           \
    }
}
}
