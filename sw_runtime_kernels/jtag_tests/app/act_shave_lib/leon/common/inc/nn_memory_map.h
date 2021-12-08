/*
 * {% copyright %}
 */
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
