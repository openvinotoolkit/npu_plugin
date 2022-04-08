//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#ifndef SHARED_MODULES_BLOBREADER_H_
#define SHARED_MODULES_BLOBREADER_H_


template <typename T>
T readBlob(const unsigned char *&src)
{
    T t;
    unsigned char *dst = reinterpret_cast<unsigned char *>(&t);

    for (unsigned int bytes = sizeof(T); bytes > 0; --bytes, ++dst, ++src)
        *dst = *src;

    return t;
}

template <typename T>
T readBlob(const unsigned char * const &origin)
{
    const unsigned char *src = origin;
    return readBlob<T>(src);
}

#endif /* SHARED_MODULES_BLOBREADER_H_ */
