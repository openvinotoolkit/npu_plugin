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

#define MVLOG_UNIT_NAME MvOp
#include <nn_log.h>

#include "Op.h"

Op::Op(t_MvTensorOpType op_type) : opType(op_type)
{
    mvLogLevelSet(MVLOG_INFO);
    mvLog(MVLOG_DEBUG, "%lu", op_type);
    dataIO = nullptr;
    dataParams = nullptr;
    executor = nullptr;
}

Op::~Op()
{
}

void Op::run(mv::tensor::Processor& /*mvtp*/,
        t_MvTensorMyriadResources& /*myriadRes*/,
        t_MvTensorDebugInfo& /*debugInfo*/)
{
    mvLog(MVLOG_DEBUG, "Op::run");
}

t_MvTensorOpType Op::GetOpType() const
{
    return opType;
}

unsigned int Op::getByteSize() const
{
    mvLog(MVLOG_DEBUG, "Op::getByteSize");
    return 0;
}

unsigned int Op::getBytesRead() const
{
    return getByteSize() + sizeof(uint32_t);
}

void Op::setNumShaves(uint32_t nShaves)
{
    mvLog(MVLOG_DEBUG, "Op::setNumShaves %lu", nShaves);
    numShaves = nShaves;
}

uint32_t Op::getNumShaves() const
{
    return numShaves;
}

void OpTensor::set(void* addr, uint32_t dataType, subspace::t_D8StorageOrder oldOrder, const int32_t dims[], const int32_t strides[]) {
    this->order = oldOrder;
    NDOrder newOrder = subspace::orderToNDOrder(oldOrder);
    TensorRef::set(addr, dataType, newOrder, dims, strides);
}

void OpTensor::printDims(const char * prefix) {
    int32_t permutation[subspace::MAX_DIMS] = {};
    int nDims = subspace::orderToPermutation(order, permutation);

    const char *prefixVal = (prefix) ? prefix : "";
    (void)prefixVal; // supress 'unused variable' build waring
    nnLog(MVLOG_INFO, "%s(order:%08" PRIx32 " | inner - ", prefixVal, order);

    for (int i = 0; i < nDims; i++) {
        nnLog(MVLOG_INFO, "%c:%" PRId32 "(%" PRId32 ") ",
                nn::getDimName(static_cast<nn::Dim>(permutation[i])), dims[i], strides[i]);
    }
    nnLog(MVLOG_INFO, "- outer)\n");
}
