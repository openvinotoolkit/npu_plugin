//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifndef NN_MEMORY_MAP_H_
#define NN_MEMORY_MAP_H_

namespace nn {
static constexpr unsigned long long operator"" _KB(unsigned long long x) {
    return x << 10;
}

static constexpr unsigned long long operator"" _MB(unsigned long long x) {
    return x << 20;
}

namespace util {
struct MemoryMap {
    template <unsigned int Size, unsigned int Alignment = 1>
    class alignas(Alignment) Fragment {
        unsigned char data_[Size];
    };

    template <typename T, typename A>
    static inline T *project(A address) {
        return reinterpret_cast<T *>(address);
    }
};
} // namespace util
} // namespace nn

#endif // NN_MEMORY_MAP_H_
