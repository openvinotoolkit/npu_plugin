// {% copyright %}

#include <math.h>
#include <nn_log.h>

#include "mvTensorUtil.h"

namespace mv
{
    namespace tensor
    {
        namespace util
        {
            u32 getBpp(t_MvTensorDataType type)
            {
                u32 bpp = 0;

                switch (type)
                {
                    case t_fp16:
                        bpp = 2;
                        break;

                    case t_u8f:
                        bpp = 1;
                        break;

                    case t_int:
                    case t_fp32:
                        bpp = 4;
                        break;

                    default:
                        bpp= 2;
                        break;
                }

                return bpp;
            }

            static u32 DDR_SIZE = 0;
            u32 getRuntimeDDRSize() {
                if (DDR_SIZE) {
                    return DDR_SIZE;
                }

                DDR_SIZE = 512 * 1024 * 1024;

                return DDR_SIZE;
            }

            bool isContinuous(const nn::TensorRefNDData& buffer)
            {
                if (buffer.ndims <= 0)
                {
                    return true;
                }

                const auto elementByteSize = getBpp(static_cast<t_MvTensorDataType>(buffer.dType));
                if (static_cast<decltype(elementByteSize)>(buffer.strides[0]) != elementByteSize)
                {
                    return false;
                }

                for (decltype(buffer.ndims) dim = 1; dim < buffer.ndims; ++dim)
                {
                    if (buffer.strides[dim] != buffer.dims[dim - 1] * buffer.strides[dim - 1])
                    {
                        return false;
                    }
                }
                return true;
            }

            int convPoolSizesSizeOutputByRPad(int sizeI, int kernel, int stride, int lPad, int rPad, int dilation) {
                return std::min(1 + (sizeI + lPad -1) / stride, 1 + (sizeI + lPad - 1 - (kernel - 1) * dilation + rPad) / stride);
            }

            int convPoolSizesRPadBySizeOutput(int sizeI, int sizeO, int kernel, int stride, int lPad, int dilation) {
                return (sizeO - 1) * stride - sizeI - lPad + 1 + (kernel - 1) * dilation;
            }

            bool convPoolSizesCheck(int sizeI, int sizeO, int kernel, int stride, int lPad, int rPad,
                    int dilation, bool positivePad, bool shouldRealDataUsed) {
                if (sizeI <= 0 || sizeO <= 0 || kernel <= 0 || stride <= 0 || dilation <= 0) {
                    nnLog(MVLOG_INFO, "sizeI, sizeO kernel, stride or dilation values do not make sense\n");
                    return false;
                }
                if (positivePad) {
                    if (lPad < 0) {
                        nnLog(MVLOG_INFO, "Left/top side padding can not be negative %d\n", lPad);
                        return false;
                    }
                    if (rPad < 0) {
                        nnLog(MVLOG_INFO, "Right/bottom side padding can not be negative %d\n", rPad);
                        return false;
                    }
                }
                if (shouldRealDataUsed) {
                    if (lPad > (kernel - 1) * dilation) {
                        nnLog(MVLOG_INFO, "Left/top padding is too big %d\n", lPad);
                        return false;
                    }
                    if ((sizeO - 1) * stride - lPad > sizeI - 1) {
                        nnLog(MVLOG_INFO, "Output size is too big. The last kernel application is out of real data %d\n", sizeO);
                        return false;
                    }
                }
                if ((sizeO - 1) * stride - lPad + (kernel - 1) * dilation > sizeI - 1 + rPad) {
                    nnLog(MVLOG_INFO, "The last kernel application is out of input size + rPad range\n");
                    return false;
                }
                return true;
            }
        }
    }
}
