/*
* {% copyright %}
*/
#pragma once

#include <nn_log.h>
#include <elf.h>

#ifndef MODULE_NAME
#define MODULE_NAME "[CUSTOM] "
#endif

namespace nn {
namespace shave_lib {

#define logI(...) nnLog(MVLOG_INFO,  MODULE_NAME __VA_ARGS__)
#define logE(...) nnLog(MVLOG_ERROR, MODULE_NAME __VA_ARGS__)
#define logD(...) nnLog(MVLOG_DEBUG, MODULE_NAME __VA_ARGS__)

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

const Elf32_Shdr *get_elf_section_with_name(const uint8_t *elf_data, const char *section_name);
bool loadElf(const uint8_t *elfAddr, void *buffer);

} // namespace shave_lib
} // namespace nn
