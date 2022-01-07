// {% copyright %}

#ifdef CONFIG_HAS_LRT_SRCS
#include <nn_log.h>
#else
#define nnLog(level, ...)
#endif
#include <mvSubspaces.h>
#include <moviVectorConvert.h>

#include <math.h>
#include <moviVectorTypes.h>
#include <param_mvn.h>

using namespace sw_params;
using namespace subspace;

namespace {

#define WORK_BUFFER_SIZE (((p->availableCmxBytes) / 4))

#define MAX_CHANNEL_SIZE 64
#define MAX_JOBS_NUM 1

struct t_MvMVNParamNClasses {
    half* input;
    Location inLocation;
    half* output;
    Location outLocation;
    u8* cmxslice;
    int32_t availableCmxBytes;

    s32 nShaves;
    s32 jobNum;

    NDOrder storageOrder;
    s32 channels;
    s32 height;
    s32 width;
    s32 stride;

    bool inputInCmx;
    bool outputInCmx;

    uint32_t acrossChannels;
    uint32_t normalize;
    float eps;

    float intermedia_mean[MAX_CHANNEL_SIZE * MAX_JOBS_NUM * 2];
};

static void calc_mean_NHWC_fp16(const half* line, int W, int C, int stride, float* intermedia_mean) {
    for (int c = 0; c < C; c += 8) {
        float8 m_sum = 0;

        for (int w = 0; w < W; w++) {
            half8 temp = *((half8*)(line + w * stride + c));

            m_sum += mvuConvert_float8(temp);
        }

        for (int i = 0; i < 8 && c + i < C; i++) {
            intermedia_mean[c + i] += m_sum[i];
        }
    }
}

static void calc_mean_var_NHWC_fp16(const half* line, int W, int C, int stride, float* intermedia_mean, int buf_size) {
    for (int c = 0; c < C; c += 8) {
        float8 m_sum = 0;
        float8 v_sum = 0;

        for (int w = 0; w < W; w++) {
            half8 temp = *((half8*)(line + w * stride + c));
            float8 ftemp = mvuConvert_float8(temp);

            m_sum += ftemp;
            v_sum += ftemp * ftemp;
        }

        for (int i = 0; i < 8 && c + i < C; i++) {
            intermedia_mean[c + i] += m_sum[i];
            intermedia_mean[buf_size + c + i] += v_sum[i];
        }
    }
}

static void calc_mean_CHW_fp16(const half* line, int W, float* intermedia_mean, int index) {
    float8 m_sum = 0;
    int w;

    for (w = 0; w < (W / 8) * 8; w += 8) {
        half8 temp = *((half8*)(line + w));
        m_sum += mvuConvert_float8(temp);
    }
    for (int i = 0; i < 8; i++) {
        intermedia_mean[index] += m_sum[i];
    }

    for (; w < W; w++) {
        intermedia_mean[index] += (float)*((half*)(line + w));
    }
}

static void calc_mean_var_CHW_fp16(const half* line, int W, float* intermedia_mean, int index, int buf_size) {
    float8 m_sum = 0;
    float8 v_sum = 0;
    int w;

    for (w = 0; w < (W / 8) * 8; w += 8) {
        half8 temp = *((half8*)(line + w));
        float8 ftemp = mvuConvert_float8(temp);

        m_sum += ftemp;
        v_sum += ftemp * ftemp;
    }
    for (int i = 0; i < 8; i++) {
        intermedia_mean[index] += m_sum[i];
        intermedia_mean[buf_size + index] += v_sum[i];
    }

    for (; w < W; w++) {
        float temp = (float)*((half*)(line + w));

        intermedia_mean[index] += temp;
        intermedia_mean[buf_size + index] += temp * temp;
    }
}

void mvMVN_1(t_MvMVNParamNClasses* p) {
    NDOrder order = p->storageOrder;

    int W = p->width;
    int H = p->height;
    int C = p->channels;
    int stride = p->stride;

    uint32_t normalize_variance = p->normalize;
    int nShaves = p->nShaves;  // Use only one shave for now
    int idy = p->jobNum;
    int buf_size = nShaves * C;

    half* input = (p->inputInCmx) ? p->input : reinterpret_cast<half*>(p->cmxslice + 0 * WORK_BUFFER_SIZE);
    float* intermedia_mean = p->intermedia_mean;

    for (int c = 0; c < C; ++c) {
        intermedia_mean[c * nShaves + idy] = 0;
        intermedia_mean[buf_size + c * nShaves + idy] = 0;
    }

    if (order == ND_HWC || order == ND_NHWC) {
        for (int h = 0; h < H; h++) {
            half* line = input + h * W * stride;

            if (!normalize_variance) {
                calc_mean_NHWC_fp16(line, W, C, stride, intermedia_mean);
            } else {
                calc_mean_var_NHWC_fp16(line, W, C, stride, intermedia_mean, buf_size);
            }
        }
    } else {
        for (int c = 0; c < C; c++) {
            int index = c * nShaves + idy;

            for (int h = 0; h < H; h++) {
                half* line = input + c * H * stride + h * stride;

                if (!normalize_variance) {
                    calc_mean_CHW_fp16(line, W, intermedia_mean, index);
                } else {
                    calc_mean_var_CHW_fp16(line, W, intermedia_mean, index, buf_size);
                }
            }
        }
    }
}

void mvMVN_23(t_MvMVNParamNClasses* p) {
    NDOrder order = p->storageOrder;

    int W = p->width;
    int H = p->height;
    int C = p->channels;
    int stride = p->stride;

    uint32_t normalize_variance = p->normalize;
    uint32_t acrossChannels = p->acrossChannels;
    float epsilon = p->eps;
    int nShaves = p->nShaves;  // Use only one shave for now

    const float* variance_part = p->intermedia_mean + nShaves * C;
    const float* mean_part = p->intermedia_mean;

    half* input = (p->inputInCmx) ? p->input : reinterpret_cast<half*>(p->cmxslice + 0 * WORK_BUFFER_SIZE);
    half* output = (p->outputInCmx) ? p->output : reinterpret_cast<half*>(p->cmxslice + 2 * WORK_BUFFER_SIZE);

    float mean;
    float variance;

    for (int c = 0; c < C; c++) {
        float m_acc;
        float v_acc;

        m_acc = mean_part[c];
        v_acc = variance_part[c];
        m_acc = m_acc / (H * W);
        v_acc = v_acc / (H * W);
        v_acc = v_acc - m_acc * m_acc;
        v_acc = v_acc < 0 ? 1.f : v_acc;
        v_acc = sqrtf(v_acc) + epsilon;

        mean = m_acc;
        variance = 1.f / v_acc;

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int offset;
                if (order == ND_HWC || order == ND_NHWC) {
                    offset = h * W * stride + w * stride + c;
                } else {
                    offset = c * H * stride + h * stride + w;
                }

                output[offset] = input[offset] - mean;
                if (normalize_variance) {
                    output[offset] = output[offset] * variance;
                }
            }
        }
    }
}

}  // namespace

using namespace subspace;

namespace nn {
namespace shave_lib {

extern "C" {
void singleShaveMVN(uint32_t lParams) {
    const MvnParams* layerParams = reinterpret_cast<const MvnParams*>(lParams);

    uint8_t* cmxData = nullptr;  // TODO: Restore the possibility of working with DDR tensors
    int32_t availableCmxBytes = 0;

    half* p_act_data = (half*)(layerParams->input.dataAddr);  // 0x1F000000
    half* p_act_out = (half*)(layerParams->output.dataAddr);  // 0x1F004000
    t_MvMVNParamNClasses mvnParamsCMX;
    t_MvMVNParamNClasses* sp = &mvnParamsCMX;

    sp->inputInCmx = true;
    sp->outputInCmx = true;

    int32_t* pDims = (int32_t*)(layerParams->input.dimsAddr);
    int64_t* iPStrides = (int64_t*)(layerParams->input.stridesAddr);
    int64_t* oPStrides = (int64_t*)(layerParams->output.stridesAddr);

    sp->acrossChannels = layerParams->acrossChannels;
    sp->normalize = layerParams->normalize;
    sp->eps = layerParams->eps;

    if (layerParams->acrossChannels != false) {
        nnLog(MVLOG_ERROR, "Unsuported case, expected across_channels = false");
        return;
    }

    sp->storageOrder = layerParams->input.dimsOrder;
    s32 ndims = layerParams->input.numDims;
    bool success;
    NDDims indices = orderNDToIndices(layerParams->input.dimsOrder, success);
    sp->channels = pDims[indices[ndims - 3]];
    sp->height = pDims[indices[ndims - 2]];
    sp->width = pDims[indices[ndims - 1]];
    sp->stride = iPStrides[1] / (CHAR_BIT * sizeof(half));

    unsigned int shaves_no = 1;
    int32_t firstShave = 0;
    int32_t jobNum = 0;
    int32_t lastShave = firstShave + static_cast<int>(shaves_no) - 1;
    nnLog(MVLOG_DEBUG, "singleShaveMVN: run on %d SHAVEs\n", shaves_no);
    nnLog(MVLOG_DEBUG, "MVNParamNClasses %d\n", __LINE__);

    t_MvMVNParamNClasses* mvnParamNClasses = &mvnParamsCMX;
    mvnParamNClasses->input = reinterpret_cast<half*>(p_act_data);
    mvnParamNClasses->inLocation = sw_params::Location::NN_CMX;  // layerParams->input.location;
    mvnParamNClasses->inputInCmx = true;                         // layerParams->input.location;
    mvnParamNClasses->output = reinterpret_cast<half*>(p_act_out);
    mvnParamNClasses->outLocation = sw_params::Location::NN_CMX;  // layerParams->output.location;
    mvnParamNClasses->outputInCmx = true;                         // layerParams->input.location;
    mvnParamNClasses->cmxslice = cmxData;
    mvnParamNClasses->availableCmxBytes = availableCmxBytes;
    mvnParamNClasses->nShaves = shaves_no;
    mvnParamNClasses->jobNum = jobNum;

    mvMVN_1(mvnParamNClasses);
    mvMVN_23(mvnParamNClasses);
}
}

}  // namespace shave_lib
}  // namespace nn
