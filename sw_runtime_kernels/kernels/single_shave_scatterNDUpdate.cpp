//
// Copyright 2022 Intel Corporation.
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

#ifdef CONFIG_HAS_LRT_SRCS
#include <nn_log.h>
#else
#define nnLog(level, ...)
#endif
#include <moviVectorConvert.h>
#include <mvSubspaces.h>

#include <math.h>
#include <moviVectorTypes.h>
#include <moviVectorUtils.h>

#include <cmath>
#include <param_scatterNDUpdate.h>

using namespace sw_params;
using namespace subspace;

namespace {
// structs and functions

}  // namespace

namespace nn {
namespace shave_lib {

extern "C" {
void singleShaveScatterNDUpdate(uint32_t lParams) {

}

}  // namespace shave_lib
}  // namespace nn
}
