// {% copyright %}
#ifndef MV_TENSOR_UTIL_H_
#define MV_TENSOR_UTIL_H_

#include <Op.h>
#include <algorithm>
#include <type_traits>

#include <assert.h>

namespace mv
{
    namespace tensor
    {
        namespace util
        {
            inline bool is_ddr(const void *p)
            {
                return reinterpret_cast<unsigned int>(p) & 0x80000000;
            }

            template <typename T>
            inline T *uncached_ddr(T *t)
            {
#ifndef DDR_UNCACHED_ACCESS
                UNUSED(t);
                assert(false);
                return reinterpret_cast<T *>(reinterpret_cast<unsigned int>(t));
#else
                return reinterpret_cast<T *>(reinterpret_cast<unsigned int>(t) | 0x40000000);
#endif
            }

            template <typename T>
            inline T *cached_ddr(T *t)
            {
                return reinterpret_cast<T *>(reinterpret_cast<unsigned int>(t) & ~0x40000000);
            }

            inline bool is_cmx(const void *p)
            {
                return (reinterpret_cast<unsigned int>(p) & 0x70000000) == 0x70000000;
            }

            template <typename T>
            inline T *uncached_cmx(T *t)
            {
                return reinterpret_cast<T *>(reinterpret_cast<unsigned int>(t) | 0x08000000);
            }

            template <typename T>
            inline T *cached_cmx(T *t)
            {
                return reinterpret_cast<T *>(reinterpret_cast<unsigned int>(t) & ~0x08000000);
            }

            template <typename T>
            inline T *uncached(T *t)
            {
                return
                    is_ddr(t) ? uncached_ddr(t) :
                    is_cmx(t) ? uncached_cmx(t) :
                    t;
            }

            template <typename T>
            inline T *cached(T *t)
            {
                return
                    is_ddr(t) ? cached_ddr(t) :
                    is_cmx(t) ? cached_cmx(t) :
                    t;
            }

            template <unsigned int N>
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

            template <unsigned int N, typename T>
            constexpr typename std::enable_if<is_power_of_2<N>::value, T>::type round_up(T t)
            {
                return (t + N - 1) & ~(N - 1);
            }

            template <unsigned int N, typename T>
            constexpr typename std::enable_if<is_power_of_2<N>::value, T>::type round_down(T t)
            {
                return t & ~(N - 1);
            }

            template <unsigned int N, typename T>
            inline constexpr typename std::enable_if<is_power_of_2<N>::value, T>::type align_up(T t)
            {
                return round_up<N>(t);
            }

            template <unsigned int N, typename T>
            inline constexpr typename std::enable_if<std::is_pointer<T>::value, T>::type ptr_align_up(T t)
            {
                return reinterpret_cast<T>(round_up<N>(reinterpret_cast<unsigned int>(t)));
            }

            template <unsigned int N, typename T>
            inline constexpr typename std::enable_if<std::is_pointer<T>::value, T>::type ptr_align_down(T t)
            {
                return reinterpret_cast<T>(round_down<N>(reinterpret_cast<unsigned int>(t)));
            }

            struct Slice
            {
                Slice(void *start, int size) :
                    start_(reinterpret_cast<unsigned char *>(start)),
                    size_(size)
                {
                }

                template <typename T, int Alignment>
                T *allocate(int size)
                {
                    unsigned char *aligned = reinterpret_cast<unsigned char *>(mv::tensor::util::align_up<Alignment>(reinterpret_cast<unsigned int>(start_)));
                    size *= sizeof(T);

                    if (aligned - start_ + size <= size_)
                    {
                        size_ -= aligned - start_ + size;
                        start_ = aligned + size;
                    }
                    else
                        aligned = nullptr;

                    return reinterpret_cast<T *>(aligned);
                }

                template <typename T, int Alignment>
                T *allocate_all(int *size = nullptr)
                {
                    unsigned char *aligned = reinterpret_cast<unsigned char *>(mv::tensor::util::align_up<Alignment>(reinterpret_cast<unsigned int>(start_)));

                    if (aligned - start_ <= size_)
                    {
                        const unsigned int fits = (size_ - (aligned - start_)) / sizeof(T);
                        const unsigned int bytes = fits * sizeof(T);

                        if (size != nullptr)
                            *size = fits;

                        size_ -= aligned - start_ + bytes;
                        start_ = aligned + bytes;
                    }
                    else
                        aligned = nullptr;

                    return reinterpret_cast<T *>(aligned);
                }

            private:
                unsigned char *start_;
                int size_;
            };

            u32 getBpp(t_MvTensorDataType type);
            /**
            * @brief uses os drv module to calculate ddrSize
            */
            u32 getRuntimeDDRSize();

            bool isContinuous(const nn::TensorRefNDData& buffer);

            bool convPoolSizesCheck(int sizeI, int sizeO, int kernel, int stride, int lPad, int rPad,
                    int dilation = 1, bool positivePad = true, bool shouldRealDataUsed = true);

            int convPoolSizesSizeOoutputByRPad(int sizeI, int kernel, int stride, int lPad, int rPad, int dilation = 1);

            int convPoolSizesRPadBySizeOutput(int sizeI, int sizeO, int kernel, int stride, int lPad, int dilation = 1);

        }  // namespace util
    }
}

#endif // MV_TENSOR_UTIL_H_
