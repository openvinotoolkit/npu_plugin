// {% copyright %}

#ifndef SHARED_MODULES_OP_H
#define SHARED_MODULES_OP_H

#include <sw_tensor_ref.h>
#include <mvTensor.h>
#include "mvTensor_cpp.h"
#include "mvTensorResources.h"
#include "mvTensorDebug.h"

#include <mvSubspaces8d.h>
#include <sw_layer.h>

#include <nn_perf_measurement.h>

using namespace nn::shave_lib;

struct MvNCIExecutor;
typedef struct MvNCIExecutor *MvNCIExecutorHandle;

class OpTensor: public nn::TensorRef {
public:
    unsigned int getByteSize() const { return 0; };
    subspace::t_D8StorageOrder order;
    void set(void* addr, uint32_t dataType, subspace::t_D8StorageOrder oldOrder, const int32_t dims[], const int32_t strides[]);
    void set(void* addr, uint32_t dataType, subspace::NDDims order, const int32_t dims[], const int32_t strides[]) = delete;
    void printDims(const char * prefix);
};

struct PerformanceData {
    ShavePerfCounters* perfCounters{};
    int32_t elapsedTimeNs{};
};

class Op
{
public:
//    const uint32_t STAGE_HEADER_SIZE      = 3 * sizeof(uint32_t);
//    const uint32_t STAGE_FOOTER_SIZE      = 2 * sizeof(uint32_t);
//    const uint32_t STAGE_DATA_OFFSET_SIZE = 1 * sizeof(uint32_t);

    Op() = default;
    Op(t_MvTensorOpType op_type);

    virtual ~Op();

    virtual unsigned int getByteSize() const;

    virtual uint32_t getNumIterations() const { return 1; }
    virtual uint32_t getNumStages() const { return 1; }

protected:

    const unsigned char * getParamsPtr()
    {
        return dataParams;
    };

    const unsigned char * getIOPtr()
    {
        return dataIO;
    };

public:
    virtual void run(mv::tensor::Processor& mvtp,
                     t_MvTensorMyriadResources& myriadRes,
                     t_MvTensorDebugInfo& debugInfo);

    virtual bool parse(Layer */*layer*/) {return true;};

    virtual unsigned int getBytesRead() const;

    virtual void setNumShaves(uint32_t nShaves);

    virtual uint32_t getNumShaves() const;

    virtual t_MvTensorOpType GetOpType() const;

    t_MvTensorOpType opType = kNone0;

    uint32_t offsetIO = 0;

    /* It is necessary to use executor to nested MvNCIExecutor call.
     * This field is currently used only from custom_mtcnn
     */
    MvNCIExecutorHandle executor;

    PerformanceData perfData;

private:
    uint32_t numShaves = 0;
    Op(const Op &) = delete;
    Op &operator =(const Op &) = delete;
    const unsigned char * dataParams;
    const unsigned char * dataIO;
};
#endif
