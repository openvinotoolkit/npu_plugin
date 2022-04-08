//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0

#include <moviVectorConvert.h>
#include <mv_types.h>

#include <param_convert.h>

#define VECTOR_SIZE (8) /* Changes to this should be reflected in the code as well */

using namespace sw_params;

namespace nn {
namespace shave_lib {

template <class T_SRC, class T_DST, class T_SRC_V, class T_DST_V, class ConvertVectors, class ConvertScalars>
__attribute__((always_inline)) void calculate(T_SRC* __restrict p_act_data, T_DST* __restrict p_act_out,
                                              T_SRC_V* __restrict p_act_data_v, T_DST_V* __restrict p_act_out_v,
                                              int nElements, int numVectors, ConvertVectors convertVectors,
                                              ConvertScalars convertScalars) {
#pragma clang loop unroll_count(8)
    for (int i = 0; i < numVectors; i++) {
        p_act_out_v[i] = convertVectors(p_act_data_v[i]);
    }
    for (int i = numVectors * VECTOR_SIZE; i < nElements; i++) {
        p_act_out[i] = convertScalars(p_act_data[i]);
    }
}

extern "C" {

void single_shave_convert(const struct ConvertParams* lParams) {
    u8* p_act_data = (u8*)(lParams->input.dataAddr);  // 0x1F000000
    u8* p_act_out = (u8*)(lParams->output.dataAddr);  // 0x1F004000

    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);

    int32_t nElements = 1;
    int32_t i = 0;

    for (i = 0; i != lParams->input.numDims; i++) {
        nElements *= pDims[i];
    }
    const int numVectors = nElements / VECTOR_SIZE;

    auto in_type = lParams->input.dataType;
    auto out_type = lParams->output.dataType;

    if (in_type == NN_U8 && out_type == NN_I8) {
        calculate((u8*)p_act_data, (s8*)p_act_out, (uchar8*)p_act_data, (char8*)p_act_out, nElements, numVectors,
                  [](uchar8 vin) -> char8 {
                      return mvuConvert_char8(vin);
                  },
                  [](u8 vin) -> s8 {
                      return mvuConvert_char(vin);
                  });
    } else if (in_type == NN_U8 && out_type == NN_I32) {
        calculate((u8*)p_act_data, (s32*)p_act_out, (uchar8*)p_act_data, (int8*)p_act_out, nElements, numVectors,
                  [](uchar8 vin) -> int8 {
                      return mvuConvert_int8(vin);
                  },
                  [](u8 vin) -> s32 {
                      return mvuConvert_int(vin);
                  });
    } else if (in_type == NN_U8 && out_type == NN_FP16) {
        calculate((u8*)p_act_data, (fp16*)p_act_out, (uchar8*)p_act_data, (half8*)p_act_out, nElements, numVectors,
                  [](uchar8 vin) -> half8 {
                      return mvuConvert_half8(vin);
                  },
                  [](u8 vin) -> fp16 {
                      return mvuConvert_half(vin);
                  });
    } else if (in_type == NN_U8 && out_type == NN_FP32) {
        calculate((u8*)p_act_data, (fp32*)p_act_out, (uchar8*)p_act_data, (float8*)p_act_out, nElements, numVectors,
                  [](uchar8 vin) -> float8 {
                      return mvuConvert_float8(vin);
                  },
                  [](u8 vin) -> fp32 {
                      return mvuConvert_float(vin);
                  });
    } else if (in_type == NN_I8 && out_type == NN_U8) {
        calculate((s8*)p_act_data, (u8*)p_act_out, (char8*)p_act_data, (uchar8*)p_act_out, nElements, numVectors,
                  [](char8 vin) -> uchar8 {
                      return mvuConvert_uchar8(vin);
                  },
                  [](s8 vin) -> u8 {
                      return mvuConvert_uchar(vin);
                  });
    } else if (in_type == NN_I8 && out_type == NN_I32) {
        calculate((s8*)p_act_data, (s32*)p_act_out, (char8*)p_act_data, (int8*)p_act_out, nElements, numVectors,
                  [](char8 vin) -> int8 {
                      return mvuConvert_int8(vin);
                  },
                  [](s8 vin) -> s32 {
                      return mvuConvert_int(vin);
                  });
    } else if (in_type == NN_I8 && out_type == NN_FP16) {
        calculate((s8*)p_act_data, (fp16*)p_act_out, (char8*)p_act_data, (half8*)p_act_out, nElements, numVectors,
                  [](char8 vin) -> half8 {
                      return mvuConvert_half8(vin);
                  },
                  [](s8 vin) -> fp16 {
                      return mvuConvert_half(vin);
                  });
    } else if (in_type == NN_I8 && out_type == NN_FP32) {
        calculate((s8*)p_act_data, (fp32*)p_act_out, (char8*)p_act_data, (float8*)p_act_out, nElements, numVectors,
                  [](char8 vin) -> float8 {
                      return mvuConvert_float8(vin);
                  },
                  [](s8 vin) -> fp32 {
                      return mvuConvert_float(vin);
                  });
    } else if (in_type == NN_I32 && out_type == NN_U8) {
        calculate((s32*)p_act_data, (u8*)p_act_out, (int8*)p_act_data, (uchar8*)p_act_out, nElements, numVectors,
                  [](int8 vin) -> uchar8 {
                      return mvuConvert_uchar8(vin);
                  },
                  [](s32 vin) -> u8 {
                      return mvuConvert_uchar(vin);
                  });
    } else if (in_type == NN_I32 && out_type == NN_I8) {
        calculate((s32*)p_act_data, (s8*)p_act_out, (int8*)p_act_data, (char8*)p_act_out, nElements, numVectors,
                  [](int8 vin) -> char8 {
                      return mvuConvert_char8(vin);
                  },
                  [](s32 vin) -> s8 {
                      return mvuConvert_char(vin);
                  });
    } else if (in_type == NN_I32 && out_type == NN_FP16) {
        calculate((s32*)p_act_data, (fp16*)p_act_out, (int8*)p_act_data, (half8*)p_act_out, nElements, numVectors,
                  [](int8 vin) -> half8 {
                      return mvuConvert_half8(vin);
                  },
                  [](s32 vin) -> fp16 {
                      return mvuConvert_half(vin);
                  });
    } else if (in_type == NN_I32 && out_type == NN_FP32) {
        calculate((s32*)p_act_data, (fp32*)p_act_out, (int8*)p_act_data, (float8*)p_act_out, nElements, numVectors,
                  [](int8 vin) -> float8 {
                      return mvuConvert_float8(vin);
                  },
                  [](s32 vin) -> fp32 {
                      return mvuConvert_float(vin);
                  });
    } else if (in_type == NN_FP16 && out_type == NN_U8) {
        calculate((fp16*)p_act_data, (u8*)p_act_out, (half8*)p_act_data, (uchar8*)p_act_out, nElements, numVectors,
                  [](half8 vin) -> uchar8 {
                      return mvuConvert_uchar8(vin);
                  },
                  [](fp16 vin) -> u8 {
                      return mvuConvert_uchar(vin);
                  });
    } else if (in_type == NN_FP16 && out_type == NN_I8) {
        calculate((fp16*)p_act_data, (s8*)p_act_out, (half8*)p_act_data, (char8*)p_act_out, nElements, numVectors,
                  [](half8 vin) -> char8 {
                      return mvuConvert_char8(vin);
                  },
                  [](fp16 vin) -> s8 {
                      return mvuConvert_char(vin);
                  });
    } else if (in_type == NN_FP16 && out_type == NN_I32) {
        calculate((fp16*)p_act_data, (s32*)p_act_out, (half8*)p_act_data, (int8*)p_act_out, nElements, numVectors,
                  [](half8 vin) -> int8 {
                      return mvuConvert_int8(vin);
                  },
                  [](fp16 vin) -> s32 {
                      return mvuConvert_int(vin);
                  });
    } else if (in_type == NN_FP16 && out_type == NN_FP32) {
        calculate((fp16*)p_act_data, (fp32*)p_act_out, (half8*)p_act_data, (float8*)p_act_out, nElements, numVectors,
                  [](half8 vin) -> float8 {
                      return mvuConvert_float8(vin);
                  },
                  [](fp16 vin) -> fp32 {
                      return mvuConvert_float(vin);
                  });
    } else if (in_type == NN_FP32 && out_type == NN_U8) {
        calculate((fp32*)p_act_data, (u8*)p_act_out, (float8*)p_act_data, (uchar8*)p_act_out, nElements, numVectors,
                  [](float8 vin) -> uchar8 {
                      return mvuConvert_uchar8(vin);
                  },
                  [](fp32 vin) -> u8 {
                      return mvuConvert_uchar(vin);
                  });
    } else if (in_type == NN_FP32 && out_type == NN_I8) {
        calculate((fp32*)p_act_data, (s8*)p_act_out, (float8*)p_act_data, (char8*)p_act_out, nElements, numVectors,
                  [](float8 vin) -> char8 {
                      return mvuConvert_char8(vin);
                  },
                  [](fp32 vin) -> s8 {
                      return mvuConvert_char(vin);
                  });
    } else if (in_type == NN_FP32 && out_type == NN_I32) {
        calculate((fp32*)p_act_data, (s32*)p_act_out, (float8*)p_act_data, (int8*)p_act_out, nElements, numVectors,
                  [](float8 vin) -> int8 {
                      return mvuConvert_int8(vin);
                  },
                  [](fp32 vin) -> s32 {
                      return mvuConvert_int(vin);
                  });
    } else if (in_type == NN_I32 && out_type == NN_I64) {
        calculate((s32*)p_act_data, (s64*)p_act_out, (int8*)p_act_data, (longlong8*)p_act_out, nElements, numVectors,
                  [](int8 vin) -> longlong8 {
                      return mvuConvert_longlong8(vin);
                  },
                  [](s32 vin) -> s64 {
                      return mvuConvert_longlong(vin);
                  });
    } else if (in_type == NN_I8 && out_type == NN_I64) {
        calculate((s8*)p_act_data, (s64*)p_act_out, (char8*)p_act_data, (longlong8*)p_act_out, nElements, numVectors,
                  [](char8 vin) -> longlong8 {
                      return mvuConvert_longlong8(vin);
                  },
                  [](s8 vin) -> s64 {
                      return mvuConvert_longlong(vin);
                  });
    } else if (in_type == NN_FP16 && out_type == NN_I64) {
        calculate((fp16*)p_act_data, (s64*)p_act_out, (half8*)p_act_data, (longlong8*)p_act_out, nElements, numVectors,
                  [](half8 vin) -> longlong8 {
                      return mvuConvert_longlong8(vin);
                  },
                  [](fp16 vin) -> s64 {
                      return mvuConvert_longlong(vin);
                  });
    } else if (in_type == NN_FP32 && out_type == NN_I64) {
        calculate((fp32*)p_act_data, (s64*)p_act_out, (float8*)p_act_data, (longlong8*)p_act_out, nElements, numVectors,
                  [](float8 vin) -> longlong8 {
                      return mvuConvert_longlong8(vin);
                  },
                  [](fp32 vin) -> s64 {
                      return mvuConvert_longlong(vin);
                  });
    } else if (in_type == NN_U8 && out_type == NN_I64) {
        calculate((u8*)p_act_data, (s64*)p_act_out, (uchar8*)p_act_data, (longlong8*)p_act_out, nElements, numVectors,
                  [](uchar8 vin) -> longlong8 {
                      return mvuConvert_longlong8(vin);
                  },
                  [](u8 vin) -> s64 {
                      return mvuConvert_longlong(vin);
                  });
    } else if (in_type == NN_I64 && out_type == NN_I32) {
        calculate((s64*)p_act_data, (s32*)p_act_out, (longlong8*)p_act_data, (int8*)p_act_out, nElements, numVectors,
                  [](longlong8 vin) -> int8 {
                      return mvuConvert_int8(vin);
                  },
                  [](s64 vin) -> s32 {
                      return mvuConvert_int(vin);
                  });
    } else if (in_type == NN_I64 && out_type == NN_I8) {
        calculate((s64*)p_act_data, (s8*)p_act_out, (longlong8*)p_act_data, (char8*)p_act_out, nElements, numVectors,
                  [](longlong8 vin) -> char8 {
                      return mvuConvert_char8(vin);
                  },
                  [](s64 vin) -> s8 {
                      return mvuConvert_char(vin);
                  });
   } else if (in_type == NN_I64 && out_type == NN_FP16) {
        calculate((s64*)p_act_data, (fp16*)p_act_out, (longlong8*)p_act_data, (half8*)p_act_out, nElements, numVectors,
                  [](longlong8 vin) -> half8 {
                      return mvuConvert_half8(vin);
                  },
                  [](s64 vin) -> fp16 {
                      return mvuConvert_half(vin);
                  });
    } else if (in_type == NN_I64 && out_type == NN_FP32) {
        calculate((s64*)p_act_data, (fp32*)p_act_out, (longlong8*)p_act_data, (float8*)p_act_out, nElements, numVectors,
                  [](longlong8 vin) -> float8 {
                      return mvuConvert_float8(vin);
                  },
                  [](s64 vin) -> fp32 {
                      return mvuConvert_float(vin);
                  });
    } else if (in_type == NN_I64 && out_type == NN_U8) {
        calculate((s64*)p_act_data, (u8*)p_act_out, (longlong8*)p_act_data, (uchar8*)p_act_out, nElements, numVectors,
                  [](longlong8 vin) -> uchar8 {
                      return mvuConvert_uchar8(vin);
                  },
                  [](s64 vin) -> u8 {
                      return mvuConvert_uchar(vin);
                  });
    } else {
        calculate((fp32*)p_act_data, (fp16*)p_act_out, (float8*)p_act_data, (half8*)p_act_out, nElements, numVectors,
                  [](float8 vin) -> half8 {
                      return mvuConvert_half8(vin);
                  },
                  [](fp32 vin) -> fp16 {
                      return mvuConvert_half(vin);
                  });
    }
}
}
}  // namespace shave_lib
}  // namespace nn
