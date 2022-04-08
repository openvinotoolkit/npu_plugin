//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#pragma once

#include "icv_test_suite.h"
#include <CustomCpp.h>
#include <nn_log.h>
#include "common_types.h"

using namespace icv_tests;

namespace
{

#define MAX_LOCAL_PARAMS    32

struct Dimensions {
    int width;
    int height;
    int channels;
};

template<int WD>
struct GroupParams {
    int32_t localWorkGroupSize[WD];
    int32_t numGroups[WD];
    int32_t offset[WD];
    uint32_t workDim;
    uint32_t kernelId;
};

struct CustomParams {
    int32_t layerParams[MAX_LOCAL_PARAMS];
};

typedef std::initializer_list<int32_t> Dims;

struct SingleTest {
    Dims inputDims;
    Dims outputDims;
    StorageOrder storageOrder;
    CustomParams customLayerParams;
};

// A base class for OpenCL tests.
// It is aware of test type (structure, that contains all the test input).
// Test input is used to load ELF and pass layer params into layer.
class CustomCppTestBase : public TestSuite {
public:
    explicit CustomCppTestBase() = default;
    virtual ~CustomCppTestBase() = default;
protected:
    // initTestCase must initialize m_currentTest variable;
    // may redefine threshold and debug-specific variables
    virtual void initTestCase() = 0;
    virtual void generateInputData() = 0;
    virtual void generateReferenceData() = 0;

    const CustomCppLayerParams* getParams() const { return &m_params; }

    t_MvTensorOpType opType() const override { return kCustomCpp; }

    void generateData() override
    {
        nnLog(MVLOG_DEBUG, "leonPreambleID %ld", m_params.leonPreambleID);
        nnLog(MVLOG_DEBUG, "ParamData  %p with length %ld.", m_params.paramData,  m_params.paramDataLen);

        generateInputData();
        generateReferenceData();
        rtems_cache_flush_entire_data();
    }

protected:
    CustomCppLayerParams m_params;
};

} // anonymous namespace

