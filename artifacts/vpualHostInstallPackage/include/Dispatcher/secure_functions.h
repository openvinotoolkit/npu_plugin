///
/// @file      secure_functions.h
///
/// @brief     Secure version of memcpy.
///

#include <errno.h>
#include <stdint.h>

#ifndef __SECURE_FUNCTIONS_H__
#define __SECURE_FUNCTIONS_H__

inline static int memcpy_s(void * dest, size_t destsz, const void * const src, size_t count)
{
    if (dest == NULL) return EINVAL; // dest should not be a NULL ptr
    if (destsz > SIZE_MAX) return ERANGE;
    if (count > SIZE_MAX) return ERANGE;
    if (destsz < count) { memset(dest, 0, destsz); return ERANGE; }
    if (src == NULL) { memset(dest, 0, destsz); return EINVAL; } // src should not be a NULL ptr

    // Copying shall not take place between regions that overlap.
    if( ((dest > src) && (dest < (src+count))) ||
        ((src > dest) && (src < (dest+destsz))) ) {
        memset(dest, 0, destsz);
        return ERANGE;
    }

    memcpy(dest, src, count);
    return 0;
}

#endif /* __SECURE_FUNCTIONS_H__ */
