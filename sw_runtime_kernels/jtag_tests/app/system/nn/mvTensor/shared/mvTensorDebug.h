// {% copyright %}
/*
 * MvAssert.h
 *
 *  Created on: Oct 20, 2016
 *      Author: ian-movidius
 */

#ifndef SHARED_MVASSERT_H_
#define SHARED_MVASSERT_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mv_types.h>
#include <cpuWhoAmI.h>
#include "mprintf.h"
#include "mvTensorConfig.h"

static inline void mvTensorAssertHelper(const u32 cond, const char* condText,
    const char* file, const u32 line, const char* func,
    const char* msg = "")
{
    if (cond)
        return;

    MPRINTF(
        "%s: assertion failed in\n"
        "\tfile: %s, line %u, \n"
        "\tfunction: %s\n"
        "\tcondition: %s\n"
        "\tmessage: %s\n",
        cpuGetProcessorName(cpuWhoAmI()),
        file, (unsigned)line,
        func,
        condText,
        msg);

#ifndef __MOVICOMPILE__
    // sleep(1); // can give UART a chance to be printed before device resetting
#ifndef SOFT_ASSERT
    exit(EXIT_FAILURE);
#endif
#endif
}

#define mvTensorAssert(cond, ...) \
    ::mvTensorAssertHelper(cond, #cond, __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__)


#endif /* SHARED_MVASSERT_H_ */
