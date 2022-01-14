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

#include "Permute.h"

#include <stdio.h>
#include <stdlib.h>
#include <mv_types.h>
#include <nn_log.h>

#include "shave_task_runner.hpp"

using namespace mv::tensor;

Permute::~Permute()
{
}

namespace {
template <typename T>
void copyElement(void * in, void * out) {
    *(static_cast<T*>(out)) = *(static_cast<T*>(in));
}

void standartLayoutPermutationToStorageOrderPermutation(OpTensor &input, OpTensor &output,
        const int32_t standartLayoutPerm[], int32_t storageOrderPerm[]) {
    int32_t orderPerm[subspace::MAX_DIMS] = {0, };
    int32_t orderInd[subspace::MAX_DIMS] = {0, };
    /*
    orderPerm - (Op) permutation from logicalDims to memoryDims (logicalDims --> (orderPerm) --> memoryDims)
                or memoryDims = Op(logicalDims)
    orderInd  - (Oi) inverse permutation from logicalDims to memoryDims i.e.
                permutation from memoryDims to logicalDims (memoryDims --> (orderInd) --> logicalDims)
                or logicalDims = Oi(memoryDims)
    logicalPermutation - (P) permutation of logical dims (logOutDims = P(logInDims)):
    memory_order_permutation - (Pm) corresponding permutation of memory dims (memOutDims = Ps(memInDims)),
    then:
    logOutDims = P(logInDims)) ->
    Oi(memOutDims) = P(Oi(memInDims)) -> applying Op permutation (inverse for Oi) to both equality sides ->
    memOutDims = Op(P(Oi(memInDims)))
    then, memory_order_permutation can be found as:
    Pm = Op(P(Oi))
    */
    int32_t tmp[subspace::MAX_DIMS] = {0, };
    orderToPermutation(output.order, orderPerm);
    orderToIndices(input.order, orderInd);
    subspace::permuteArray(orderInd, standartLayoutPerm, tmp, input.ndims);
    subspace::permuteArray(tmp, orderPerm, storageOrderPerm, input.ndims);
}

using FpCopyElement = void (*)(void*, void*);
}

void Permute::run(mv::tensor::Processor& ,
                  t_MvTensorMyriadResources& myriadRes,
                  t_MvTensorDebugInfo&) {

    nnLog(MVLOG_DEBUG, "ndims  = %ld\n", input.ndims);
    nnLog(MVLOG_DEBUG, "order  = 0x%x\n", input.order);
    nnLog(MVLOG_DEBUG, "dims.0 = %ld(%ld)\n", input.dims[0], input.strides[0]);
    nnLog(MVLOG_DEBUG, "dims.1 = %ld(%ld)\n", input.dims[1], input.strides[1]);
    nnLog(MVLOG_DEBUG, "dims.2 = %ld(%ld)\n", input.dims[2], input.strides[2]);
    nnLog(MVLOG_DEBUG, "dims.3 = %ld(%ld)\n", input.dims[3], input.strides[3]);
    nnLog(MVLOG_DEBUG, "ops.order.0 = %ld\n", ops.order[0]);
    nnLog(MVLOG_DEBUG, "ops.order.1 = %ld\n", ops.order[1]);
    nnLog(MVLOG_DEBUG, "ops.order.2 = %ld\n", ops.order[2]);
    nnLog(MVLOG_DEBUG, "ops.order.3 = %ld\n", ops.order[3]);
    nnLog(MVLOG_DEBUG, "ops.allow_permute_nd = %ld\n", ops.allow_permute_nd);
    nnLog(MVLOG_DEBUG, "input.dType = %u bpp=%d\n", input.dType, nn::getBpp(input.dType));

    if (this->executeInTestingSystem) {
        FpCopyElement fpCopy;
        switch (nn::getBpp(input.dType)) {
        case 2:
            fpCopy = copyElement<uint16_t>;
            break;
        case 4:
            fpCopy = copyElement<uint32_t>;
            break;
        case 8:
            fpCopy = copyElement<uint64_t>;
            break;
        case 1:
        default:
            fpCopy = copyElement<uint8_t>;
            break;
        }
        int32_t inDims[subspace::MAX_DIMS] = {0, };
        int32_t outDims[subspace::MAX_DIMS] = {0, };
        int32_t storagePerm[subspace::MAX_DIMS] = {0, };
        standartLayoutPermutationToStorageOrderPermutation(input, output, this->ops.order, storagePerm);

        int totalElements = subspace::getTotal(input.dims, input.ndims);
        nnLog(MVLOG_DEBUG, "totalElements=%d\n", totalElements);
        nnLog(MVLOG_DEBUG, "inTensorDims: (%d(%d) %d(%d) %d(%d) %d(%d) %d(%d)\n",
                input.dims[0], input.strides[0],
                input.dims[1], input.strides[1],
                input.dims[2], input.strides[2],
                input.dims[3], input.strides[3],
                input.dims[4], input.strides[4]);
        nnLog(MVLOG_DEBUG, "outTensorDims: (%d(%d) %d(%d) %d(%d) %d(%d) %d(%d)\n",
                output.dims[0], output.strides[0],
                output.dims[1], output.strides[1],
                output.dims[2], output.strides[2],
                output.dims[3], output.strides[3],
                output.dims[4], output.strides[4]);
        for (int i = 0; i < totalElements; i++) {
            permuteArray(inDims, storagePerm, outDims, input.ndims);
            fpCopy(nn::element(input, inDims), nn::element(output, outDims));
            subspace::increment1Coord(inDims, input.dims, input.ndims);
        }
    } //else {
        // ---- KMB infra
//        std::unique_ptr<MVCNN::UPALayerTaskT> upaTask (new MVCNN::UPALayerTaskT());
//        if (ops.allow_permute_nd) {
//            upaTask->softLayerParams.type = MVCNN::SoftwareLayerParams_PermuteNDParams;
//            std::unique_ptr<MVCNN::PermuteNDParamsT> softLayerParamsValue(new MVCNN::PermuteNDParamsT);
//            for(int order_pos = 0; order_pos < input.ndims; order_pos++) {
//                auto reversedOrderIndex = input.ndims - order_pos - 1;
//                auto reversedOrderValue = input.ndims - ops.order[reversedOrderIndex] - 1;
//                softLayerParamsValue->permute_nd_order.push_back(reversedOrderValue);
//            }
//            upaTask->softLayerParams.value = softLayerParamsValue.release();
//        } else {
//            upaTask->softLayerParams.type = MVCNN::SoftwareLayerParams_PermuteParams;
//            std::unique_ptr<MVCNN::PermuteParamsT> softLayerParamsValue(new MVCNN::PermuteParamsT);
//
//            softLayerParamsValue->permute_order.reset(new MVCNN::order3(2 - ops.order[2], 2 - ops.order[1], 2 - ops.order[0]));
//            upaTask->softLayerParams.value = softLayerParamsValue.release();
//        }
//
//        UPATaskRunner runner;
//        mvTensorAssert(runner.enqueTask(std::move(upaTask), {input}, {output}, myriadRes.lastShave - myriadRes.firstShave + 1, &perfData), "permute layer run failed");
//        mvTensorAssert(runner.dequeResult(), "permute layer run failed");
//    }
}
