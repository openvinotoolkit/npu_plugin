// {% copyright %}
#pragma once

#include "icv_test_suite.h"
#include <CustomCpp.h>
#include <nn_log.h>
#include "layers/param_custom_cpp.h"
#include "common_types.h"

using namespace icv_tests;

namespace
{

#define MAX_BINARY_SIZE     (1024 * 80)
#define MAX_LOCAL_PARAMS    32

// FPE = Full Path to ELF
#define FPE(filename) ((char *) ICV_TESTS_STRINGIFY(DATA_FOLDER) "custom/bin-r1/" filename)

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

//template<int WD>
struct CustomParams {
//    GroupParams<WD> groupParams;
    int32_t layerParams[MAX_LOCAL_PARAMS];
};

// A base class for OpenCL tests.
// It is aware of test type (structure, that contains all the test input).
// Test input is used to load ELF and pass layer params into layer.
template <class TestType>
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

    void initElfBuffer() {
        m_elfBuffer = allocData<u8>(MAX_BINARY_SIZE + 1024);
        m_elfBuffer = (u8 * )ICV_TESTS_MEMORY_ALIGN(1024, (unsigned) m_elfBuffer);
    }

    const CustomCppLayerParams* getParams() const { return &m_params; }

    t_MvTensorOpType opType() const override { return kCustomCpp; }

    void generateData() override
    {
        rtems_cache_invalidate_multiple_data_lines(m_elfBuffer, MAX_BINARY_SIZE);
//        loadMemFromFileSimple(const_cast<char*>(m_currentTest->kernelName), 0, m_elfBuffer);

        nnLog(MVLOG_DEBUG, "leonPreambleID %ld", m_params.leonPreambleID);
        nnLog(MVLOG_DEBUG, "KernelData %p with length %ld.", m_params.kernelData, m_params.kernelDataLen);
        nnLog(MVLOG_DEBUG, "ParamData  %p with length %ld.", m_params.paramData,  m_params.paramDataLen);

        generateInputData();
        generateReferenceData();
        rtems_cache_flush_entire_data();
    }

protected:
    // A pointer to current test case. It is set by initTestCase() in derived class.
    // Whole set of tests is stored in derived class.
    const TestType* m_currentTest;
    CustomCppLayerParams m_params;
    u8* m_elfBuffer;
};

} // anonymous namespace

