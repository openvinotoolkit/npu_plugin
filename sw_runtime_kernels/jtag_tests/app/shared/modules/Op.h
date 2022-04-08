//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef SHARED_MODULES_OP_H
#define SHARED_MODULES_OP_H

#include "nn_tensor_ref.h"
#include <mvTensor.h>
#include "mvTensor_cpp.h"
#include "mvTensorResources.h"
#include "mvTensorDebug.h"

#include <mvSubspaces8d.h>

#include <nn_perf_manager.h>

class OpTensor: public nn::TensorRef {
public:
    unsigned int getByteSize() const { return 0; };
    subspace::t_D8StorageOrder order;
    void set(void* addr, uint32_t dataType, subspace::t_D8StorageOrder oldOrder, const int32_t dims[], const int32_t strides[]);
    void set(void* addr, uint32_t dataType, subspace::NDDims order, const int32_t dims[], const int32_t strides[]) = delete;
    void printDims(const char * prefix);
};

// TODO: Redo performance counters for MTL perf
// struct PerformanceData {
//     ShavePerfCounters* perfCounters{};
//     int32_t elapsedTimeNs{};
// };

class Op {
public:
    Op() = default;
    Op(t_MvTensorOpType op_type);

    virtual ~Op();

    virtual unsigned int getByteSize() const;

    virtual uint32_t getNumIterations() const { return 1; }
    virtual uint32_t getNumStages() const { return 1; }

    virtual void run();

    virtual unsigned int getBytesRead() const;

    virtual void setNumShaves(uint32_t nShaves);

    virtual uint32_t getNumShaves() const;

    virtual t_MvTensorOpType GetOpType() const;

    t_MvTensorOpType opType = kNone0;

    uint32_t offsetIO = 0;

    // PerformanceData perfData;

protected:

    const unsigned char * getParamsPtr()
    {
        return dataParams;
    };

    const unsigned char * getIOPtr()
    {
        return dataIO;
    };

private:
    uint32_t numShaves = 0;
    Op(const Op &) = delete;
    Op &operator =(const Op &) = delete;
    const unsigned char * dataParams;
    const unsigned char * dataIO;
};
#endif
