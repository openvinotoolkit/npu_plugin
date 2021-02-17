#include "include/mcm/utils/helpers.hpp"

void mv::utils::releaseFile(FILE* ptr) {
    if(ptr) {
        fclose(ptr);
    }
}

// std::hash<std::string> STL implementation taken from C++7 gcc 7.5.0
// https://github.com/gcc-mirror/gcc/blob/releases/gcc-7.5.0/libstdc++-v3/libsupc++/hash_bytes.cc
// for blob consistnecy on different OS and systems

inline size_t mv::utils::unaligned_load(const char* p) {
    size_t result;
    __builtin_memcpy(&result, p, sizeof(result));
    return result;
}

size_t mv::utils::constatnt_string_hash(const std::string str) {
    const size_t m = 0x5bd1e995;
    const void* ptr = str.c_str();
    size_t len = str.length(); // * sizeof(wchar_t);
    const size_t seed = static_cast<size_t>(0xc70f6907UL);
    size_t hash = seed ^ len;
    const char* buf = static_cast<const char*>(ptr);

    // Mix 4 bytes at a time into the hash.
    while(len >= 4) {
        size_t k = mv::utils::unaligned_load(buf);
        k *= m;
        k ^= k >> 24;
        k *= m;
        hash *= m;
        hash ^= k;
        buf += 4;
        len -= 4;
    }

    // Handle the last few bytes of the input array.
    switch(len) {
        case 3:
            hash ^= static_cast<unsigned char>(buf[2]) << 16;
            [[gnu::fallthrough]];
        case 2:
            hash ^= static_cast<unsigned char>(buf[1]) << 8;
            [[gnu::fallthrough]];
        case 1:
            hash ^= static_cast<unsigned char>(buf[0]);
            hash *= m;
    };

    // Do a few final mixes of the hash.
    hash ^= hash >> 13;
    hash *= m;
    hash ^= hash >> 15;
    return hash;
}
