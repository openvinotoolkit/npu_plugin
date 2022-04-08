/*
* {% copyright %}
*/
#ifndef NN_MATH_H_
#define NN_MATH_H_

#include <type_traits>
#include <assert.h>

namespace nn
{
    namespace math
    {
        template <size_t N>
        struct is_power_of_2
        {
            enum { value = ((N & 1) == 0) && is_power_of_2<N / 2>::value };
        };

        template <>
        struct is_power_of_2<1>
        {
            enum { value = 1 };
        };

        template <>
        struct is_power_of_2<0>
        {
            enum { value = 0 };
        };

        template <size_t N, typename T>
        typename std::enable_if<is_power_of_2<N>::value, T>::type round_up(T t)
        {
            return static_cast<T>((t + N - 1) & ~(N - 1));
        }

        template <typename T>
        inline T round_up_power_of_2(T round_to, T t)
        {
            assert(round_to && ((round_to & (round_to - 1)) == 0));
            return (t + round_to - 1) & ~(round_to - 1);
        }

        template <size_t N, typename T>
        typename std::enable_if<is_power_of_2<N>::value, T>::type round_down(T t)
        {
            return t & ~(N - 1);
        }

        template <size_t N, typename T>
        inline typename std::enable_if<std::is_pointer<T>::value, T>::type ptr_align_up(T t)
        {
            return reinterpret_cast<T>(round_up<N>(reinterpret_cast<size_t>(t)));
        }

        template <size_t N, typename T>
        inline typename std::enable_if<std::is_pointer<T>::value, T>::type ptr_align_down(T t)
        {
            return reinterpret_cast<T>(round_down<N>(reinterpret_cast<size_t>(t)));
        }

        template <typename T>
        inline T mask(unsigned int bits)
        {
            T m = (bits < sizeof(T) * 8) ? (1 << bits) : 0;
            return m - 1;
        }

        template <typename T>
        unsigned int count(T value)
        {
            unsigned int bits = 0;

            for (; value > 0; value >>= 1)
                if (value & 1)
                    ++bits;

            return bits;
        }

        template <typename T>
        int firstBitIndex(T mask)
        {
            for (int i = 0; mask > 0; ++i, mask >>= 1)
                if (mask & 1)
                    return i;

            return -1;
        }

        template <typename T>
        int lastBitIndex(T mask)
        {
            for (int i = 0; mask > 0; ++i, mask >>= 1)
                if (mask == 1)
                    return i;

            return -1;
        }

        template <typename T>
        inline typename std::enable_if<std::is_integral<T>::value, T>::type divide_up(T dividend, T divisor) {
            return (dividend + divisor - 1) / divisor;
        }
    }
}

#endif // NN_MATH_H_
