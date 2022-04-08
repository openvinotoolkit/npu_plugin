//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include <cmath>
#include <random>
#include "mvSubspaces.h"
#include <algorithm>

#define F16_MAX 65504.0f
#define DEF_ABS_THRESHOLD 0.016f

enum ReduceOpType : int32_t
{
    op_L1,
    op_L2,
    op_LogicalAnd,
    op_LogicalOr,
    op_LogicalSum,
    op_LogicalSumExp,
    op_Max,
    op_Mean,
    op_Min,
    op_Prod,
    op_Sum,
    op_SumSquare

};

typedef struct {
   ReduceOpType type;
   void   *kernel;
} ReduceOpInfo;

struct ReduceTestParams {
    std::initializer_list<int32_t> data;
    std::initializer_list<int32_t> axes;
};

__attribute__((aligned(1024)))
#include "sk.reduce_l1.3720xx.text.xdat"
#include "sk.reduce_l2.3720xx.text.xdat"
#include "sk.reduce_sum.3720xx.text.xdat"
#include "sk.reduce_mean.3720xx.text.xdat"
#include "sk.reduce_prod.3720xx.text.xdat"
#include "sk.reduce_and.3720xx.text.xdat"
#include "sk.reduce_or.3720xx.text.xdat"
#include "sk.reduce_max.3720xx.text.xdat"
#include "sk.reduce_min.3720xx.text.xdat"

//Not supported
// #include "sk.reduce_logical_sum.3720xx.text.xdat"
// #include "sk.reduce_logical_sum_exp.3720xx.text.xdat"
// #include "sk.reduce_sum_square.3720xx.text.xdat"

#include "param_reduce.h"

namespace{

template <class DataType>
class LogicalAnd;

template <>
class LogicalAnd<fp16> {
    typedef fp16 DataType;
    const fp16 one = f32Tof16(1.0);
    const fp16 zero = f32Tof16(0.0);

public:
    void init() { m_val = true; }
    void accumulate(const DataType &val) { m_val &= bool(f16Tof32(val) != 0.0); }
    DataType result() const { return (m_val ? one : zero); }

private:
    bool m_val;
};

template <>
class LogicalAnd<int32_t> {
    typedef int32_t DataType;
    const int32_t one = 1;
    const int32_t zero = 0;

public:
    void init() { m_val = true; }
    void accumulate(const DataType &val) { m_val &= bool(val != zero); }
    DataType result() const { return (m_val ? one : zero); }

private:
    bool m_val;
};

template <class DataType>
class Max;

template <>
class Max<fp16> {
    typedef fp16 DataType;

public:
    void init() { m_val = INT16_MIN; }
    void accumulate(const DataType &val) {
        auto fval = f16Tof32(val);
        m_val = (m_val > fval ? m_val : fval);
    }
    DataType result() const { return f32Tof16(m_val); }

private:
    float m_val;
};

template <>
class Max<int32_t> {
    typedef int32_t DataType;

public:
    void init() { m_val = static_cast<int64_t>(INT32_MIN); }
    void accumulate(const DataType &val) {
        int64_t fval = val;
        m_val = (m_val > fval ? m_val : fval);
    }
    DataType result() const { return (DataType)(m_val); }

private:
    int64_t m_val;
};

template <class DataType>
class L1;

template <>
class L1<fp16> {
    typedef fp16 DataType;

public:
    void init() { m_val = 0; }
    void accumulate(const DataType &val) { m_val += std::abs(f16Tof32(val)); }
    DataType result() const { return m_val > F16_MAX ? f32Tof16(F16_MAX) : f32Tof16(m_val); }

private:
    float m_val;
};

template <>
class L1<int32_t> {
    typedef int32_t DataType;

public:
    void init() { m_val = 0; }
    void accumulate(const DataType &val) { m_val += std::abs(val); }
    DataType result() const {
        return m_val > INT32_MAX ? static_cast<DataType>(INT32_MAX) : static_cast<DataType>(m_val); }

private:
    int64_t m_val;
};

template <class DataType>
class Sum;

template <>
class Sum<fp16> {
    typedef fp16 DataType;

public:
    void init() { m_val = 0; }
    void accumulate(const DataType &val) { m_val += f16Tof32(val); }
    DataType result() const { return m_val > F16_MAX ? f32Tof16(F16_MAX) : f32Tof16(m_val); }

private:
    float m_val;
};

template <>
class Sum<int32_t> {
    typedef int32_t DataType;

public:
    void init() { m_val = 0; }
    void accumulate(const DataType &val) { m_val += val; }
    DataType result() const { return m_val > INT32_MAX ? static_cast<DataType>(INT32_MAX) : static_cast<DataType>(m_val); }

private:
    int64_t m_val;
};

template <class DataType>
class Mean;

template <>
class Mean<fp16> {
    typedef fp16 DataType;

public:
    void init() {
        m_val = 0;
        m_count = 0;
    }
    void accumulate(const DataType &val) {
        m_val += f16Tof32(val);
        m_count++;
    }
    DataType result() const {
        if (m_count == 0)
            return m_val > F16_MAX ? f32Tof16(F16_MAX) : f32Tof16(m_val);
        else{
            float result = m_val / m_count;
            return result > F16_MAX ? f32Tof16(F16_MAX) : f32Tof16(result);
        }
    }

private:
    float m_val;
    int m_count;
};

template <>
class Mean<int32_t> {
    typedef int32_t DataType;

public:
    void init() {
        m_val = 0;
        m_count = 0;
    }
    void accumulate(const DataType &val) {
        m_val += val;
        m_count++;
    }
    DataType result() const {
        if (m_count == 0)
            return m_val > INT32_MAX ? static_cast<DataType>(INT32_MAX) : static_cast<DataType>(m_val);
        else{
            int64_t result = m_val / m_count;
            return result > INT32_MAX ? static_cast<DataType>(INT32_MAX) : static_cast<DataType>(result);
        }
    }

private:
    int64_t m_val;
    int m_count;
};

template <class DataType>
class Min;

template <>
class Min<fp16> {
    typedef fp16 DataType;

public:
    void init() { m_val = INT16_MAX; }
    void accumulate(const DataType &val) {
        auto fval = f16Tof32(val);
        m_val = (m_val < fval ? m_val : fval);
    }
    DataType result() const { return f32Tof16(m_val); }

private:
    float m_val;
};

template <>
class Min<int32_t> {
    typedef int32_t DataType;

public:
    void init() { m_val = INT32_MAX; }
    void accumulate(const DataType &val) {
        int64_t fval = val;
        m_val = (m_val < fval ? m_val : fval);
    }
    DataType result() const { return (DataType)m_val; }

private:
    int64_t m_val;
};

template <class DataType>
class LogicalOr;

template <>
class LogicalOr<fp16> {
    typedef fp16 DataType;
    const fp16 one = f32Tof16(1.0);
    const fp16 zero = f32Tof16(0.0);

public:
    void init() { m_val = false; }
    void accumulate(const DataType &val) { m_val |= bool(f16Tof32(val) != 0.0); }
    DataType result() const { return (m_val ? one : zero); }

private:
    bool m_val;
};

template <>
class LogicalOr<int32_t> {
    typedef int32_t DataType;
    const int32_t one = 1;
    const int32_t zero = 0;

public:
    void init() { m_val = false; }
    void accumulate(const DataType &val) { m_val |= bool(val != zero); }
    DataType result() const { return (m_val ? one : zero); }

private:
    bool m_val;
};

template <class DataType>
class L2;

template <>
class L2<fp16> {
    typedef fp16 DataType;

public:
    void init() { m_val = 0; }
    void accumulate(const DataType &val) { m_val += std::pow(f16Tof32(val), 2); }
    DataType result() const {
        return m_val > F16_MAX ? f32Tof16(F16_MAX) : f32Tof16(std::sqrt(m_val)); }

private:
    float m_val;
};

template <>
class L2<int32_t> {
    typedef int32_t DataType;

public:
    void init() { m_val = 0; }
    void accumulate(const DataType &val) { m_val += std::pow(val, 2); }
    DataType result() const {
        return m_val > INT32_MAX ? static_cast<DataType>(INT32_MAX) : static_cast<DataType>(std::sqrt(m_val)); }

private:
    int64_t m_val;
};

template <class DataType>
class Prod;

template <>
class Prod<fp16> {
    typedef fp16 DataType;

public:
    void init() { m_val = 1.0f; }
    void accumulate(const DataType &val) { m_val *= f16Tof32(val); }
    DataType result() const { return m_val > F16_MAX ? f32Tof16(F16_MAX) : f32Tof16(m_val); }

private:
    float m_val;
};

template <>
class Prod<int32_t> {
    typedef int32_t DataType;

public:
    void init() { m_val = 1; }
    void accumulate(const DataType &val) { m_val *= val; }
    DataType result() const { return m_val > INT32_MAX ? INT32_MAX : (DataType)(m_val); }

private:
    int64_t m_val;
};

    static constexpr std::initializer_list<SingleTest> test_list {
          {{1, 1, 1},    {1, 1, 1},    orderXYZ, {false, sw_params::Location::NN_CMX}},
          {{1, 1, 1},    {1, 1, 1},    orderXYZ, {true,  sw_params::Location::NN_CMX}},
    };

    static constexpr std::initializer_list<ReduceTestParams> dims_list {
        {{2,  3,  4, 5}, {1}},
#ifdef CONFIG_RUN_LARGE_TESTS
        {{19, 23, 4, 2}, {3}},
#endif
    };

    static constexpr std::initializer_list<ReduceOpInfo> kernel_list {
        // Supported operations
        {ReduceOpType::op_L1,        sk_reduce_l1_3720xx_text},
        {ReduceOpType::op_L2,        sk_reduce_l2_3720xx_text},
        {ReduceOpType::op_Sum,       sk_reduce_sum_3720xx_text},
        {ReduceOpType::op_Mean,      sk_reduce_mean_3720xx_text},
        {ReduceOpType::op_Prod,      sk_reduce_prod_3720xx_text},
        {ReduceOpType::op_LogicalAnd,sk_reduce_and_3720xx_text},
        {ReduceOpType::op_LogicalOr, sk_reduce_or_3720xx_text},
        {ReduceOpType::op_Max,       sk_reduce_max_3720xx_text},
        {ReduceOpType::op_Min,       sk_reduce_min_3720xx_text},

        // Not supported
        // {ReduceOpType::op_LogicalSum,   sk_reduce_logical_sum_3720xx_text},
        // {ReduceOpType::op_LogicalSumExp,sk_reduce_logical_sum_exp_3720xx_text},
        // {ReduceOpType::op_SumSquare,    sk_reduce_sum_square_3720xx_text},
    };

    template <typename TensorType>
    class CustomCppReduceTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppReduceTest():
                    m_testsLoop(test_list, "keep_dims"),
                    m_opInfoLoop(kernel_list, "kernel"),
                    m_dimsLoop(dims_list, "dims")
                    {}

        virtual ~CustomCppReduceTest() { }

    protected:
        const char* suiteName() const override {
            if(std::is_same<TensorType, fp16>::value)
                return "CustomCppReduceTest_FP16";
            else
                return "CustomCppReduceTest_INT32";
            
        }
        void userLoops() override {
            addLoop(m_opInfoLoop);
            addLoop(m_testsLoop);
            addLoop(m_dimsLoop);
        }

        static inline t_MvTensorStorageOrder ieOrderFromNumDims(int numDims) {
            if (numDims == 0 || numDims == 1) {
                return orderC;
            } else if (numDims == 2) {
                return orderNC;
            } else if (numDims == 3) {
                return orderCHW;
            } else if (numDims == 4) {
                return orderNCHW;
            } else {
                return maskOrder(FULL_ORDER, numDims);
            }
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

            initTestCase();

            std::vector<int32_t> dataDims = m_dimsLoop.value().data;
            std::vector<int32_t> axesDims = m_dimsLoop.value().axes;

            const MemoryDims md_dataDims(dataDims.data(), dataDims.size());
            const MemoryDims md_axesDims(axesDims.data(), axesDims.size());

            const auto dataOrder = ieOrderFromNumDims(dataDims.size());
            const auto axesOrder = ieOrderFromNumDims(axesDims.size());

            m_inTensor.init(dataOrder, md_dataDims);
            m_axesTensor.init(axesOrder, md_axesDims);

            allocBuffer(m_inTensor);
            allocBuffer(m_axesTensor);

            const SingleTest* test = m_currentTest;
            m_reduceParams = reinterpret_cast<sw_params::ReduceParams *>(paramContainer);
            *m_reduceParams = sw_params::ReduceParams();
            m_reduceParams->keep_dims = test->customLayerParams.layerParams[0];

            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::ReduceParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[1]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_reduceParams);

            m_params.kernel = reinterpret_cast<uint32_t>(m_opInfoLoop.value().kernel);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = DEF_ABS_THRESHOLD;
        }

        void resetOutputData() override {
            resetTensorBuffer(m_outputTensor);
        }

        void generateInputData() override {
            rand_seed();
            m_inTensor.forEach(false, [&](const MemoryDims& indices)
            {   
                if(std::is_same<TensorType, fp16>::value)
                    m_inTensor.at(indices) = f32Tof16(float(rand()) / RAND_MAX * 256 - 128);
                else
                    m_inTensor.at(indices) = static_cast<TensorType>(float(rand()) / RAND_MAX * 256 - 128);
            });

            std::vector<int32_t> deleted_axes;
            if(m_inTensor.ndims() < m_axesTensor.memoryDims().dims[0])
                return;
            m_axesTensor.forEach(false, [&](const MemoryDims& indices)
            {
                bool is_not_unique = false;
                int val = 0;
                int pozitive_val = 0;
                do{
                    val = (static_cast<int>(rand()) % ( 2 * m_inTensor.ndims())) - m_inTensor.ndims();
                    pozitive_val = val < 0 ? val + m_inTensor.ndims() : val;
                    is_not_unique = std::find(deleted_axes.begin(), deleted_axes.end(), pozitive_val) != deleted_axes.end();
                }while(is_not_unique);
                m_axesTensor.at(indices) = val;
                deleted_axes.push_back(pozitive_val);
            });
            
            //Construct output
            std::vector<int32_t> outputDims;
            bool keepDims = (bool)m_currentTest->customLayerParams.layerParams[0];
            for (int32_t i = 0; i < m_inTensor.ndims(); i++) {
                if (std::find(deleted_axes.begin(), deleted_axes.end(), i) == deleted_axes.end()) {
                    outputDims.push_back(m_inTensor.memoryDims().dims[i]);
                } else {
                    if (keepDims){
                        outputDims.push_back(1);
                    }
                }
            }
            if(m_inTensor.ndims() == (int)deleted_axes.size())
                outputDims.push_back(1);

            const MemoryDims md_outputDims(outputDims.data(), outputDims.size());
            const auto outOrder  = ieOrderFromNumDims(outputDims.size());
            m_outputTensor.init(outOrder, md_outputDims);
            m_referenceOutputTensor.init(outOrder, md_outputDims);
            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceOutputTensor);
        }

        static uint32_t axes2mask(int N, int K, Tensor<int32_t> &axes) {
            uint32_t mask = 0;
            for (int j = 0; j < K; ++j) {
                int32_t i = axes.data()[j];
                if ((i >= 0) && (i < N))
                    mask |= (1 << i);
            }
            return mask;
        }

        static void split(int N, uint32_t mask, const MemoryDims &D, MemoryDims &DR, MemoryDims &DI) {
            int jr = 0, ji = 0;
            for (int i = 0; i < N; ++i) {
                if (mask & (1 << i)){
                    DR.dims[jr++] = D.dims[i];
                }
                else
                    DI.dims[ji++] = D.dims[i];
            }
        }

        static MemoryDims merge(int N, uint32_t mask, const MemoryDims &DR, const MemoryDims &DI) {
            MemoryDims D;
            int jr = 0, ji = 0;
            for (int i = 0; i < N; ++i) {
                if (mask & (1 << i))
                    D.dims[i] = DR.dims[jr++];
                else
                    D.dims[i] = DI.dims[ji++];
            }
            return D;
        }

        static MemoryDims fill(int K, int val) {
            MemoryDims Z;
            for (int j = 0; j < K; ++j)
                Z.dims[j] = val;
            return Z;
        }

        template <class DataType, class Op>
        static void fullReduce(Tensor<DataType> &in, Tensor<DataType> &out, Op op){
            int N = in.ndims();
            op.init();
            in.memoryDims().forEach(N, [&](const MemoryDims &i) { op.accumulate(in.at(i)); });
            out.data()[0] = op.result();
        }

        template <class DataType, class Op>
        static void partReduce(Tensor<DataType> &in, Tensor<int32_t> &axes, Tensor<DataType> &out, bool keep_dims, Op op){
            int N = in.ndims();
            int K = axes.memoryDims().dims[0];
            unsigned mask = axes2mask(N, K, axes);
            MemoryDims DR, DI;
            split(N, mask, in.memoryDims(), DR, DI);
            const MemoryDims ZR = fill(K, 0);

            DI.forEach(N - K, [&](const MemoryDims &di) {
                op.init();
                DR.forEach(K, [&](const MemoryDims &dr) {
                    MemoryDims id = merge(N, mask, dr, di);
                    op.accumulate(in.at(id));
                });
                if (keep_dims) {
                    MemoryDims od = merge(N, mask, ZR, di);
                    out.at(od) = op.result();
                } else {
                    out.at(di) = op.result();
                }
            });
        }

        template <class DataType, class Op>
        static void refReduce(Tensor<DataType> &in, Tensor<int32_t> &axes, Tensor<DataType> &out, bool keep_dims, Op op) {
            int N = in.ndims();
            int K = axes.memoryDims().dims[0];

            if ((K <= 0) || (K >= N)) {
                if (K >= N)
                    fullReduce(in, out, op);
            } else {
                partReduce(in, axes, out, keep_dims, op);
            }
        }

        template <class DataType, template <class DT> class Op>
        static void refReduceOp(Tensor<DataType> &in, Tensor<int32_t> &axes, Tensor<DataType> &out, bool keep_dims) {
            refReduce<DataType, Op<DataType>>(in, axes, out, keep_dims, Op<DataType>());
        }

        void generateReferenceData() override {

            //Make axes pozitive
            for(int i = 0; i < m_axesTensor.memoryDims().dims[0]; i++)
                m_axesTensor.data()[i] = m_axesTensor.data()[i] < 0 ? m_axesTensor.data()[i] + m_inTensor.ndims() : m_axesTensor.data()[i];

            switch(m_opInfoLoop.value().type){
              case ReduceOpType::op_L1:            refReduceOp<TensorType, L1>(m_inTensor, m_axesTensor, m_referenceOutputTensor, m_reduceParams->keep_dims); break;
              case ReduceOpType::op_L2:            refReduceOp<TensorType, L2>(m_inTensor, m_axesTensor, m_referenceOutputTensor, m_reduceParams->keep_dims); break;
              case ReduceOpType::op_LogicalAnd:    refReduceOp<TensorType, LogicalAnd>(m_inTensor, m_axesTensor, m_referenceOutputTensor, m_reduceParams->keep_dims); break;
              case ReduceOpType::op_LogicalOr:     refReduceOp<TensorType, LogicalOr>(m_inTensor, m_axesTensor, m_referenceOutputTensor, m_reduceParams->keep_dims); break;
              case ReduceOpType::op_LogicalSum:    nnLog(MVLOG_ERROR, "Reduce: Operation LogicalSum it's not supported."); break;
              case ReduceOpType::op_LogicalSumExp: nnLog(MVLOG_ERROR, "Reduce: Operation LogicalSumExp it's not supported."); break;
              case ReduceOpType::op_Max:           refReduceOp<TensorType, Max>(m_inTensor, m_axesTensor, m_referenceOutputTensor, m_reduceParams->keep_dims);break;
              case ReduceOpType::op_Mean:          refReduceOp<TensorType, Mean>(m_inTensor, m_axesTensor, m_referenceOutputTensor, m_reduceParams->keep_dims); break;
              case ReduceOpType::op_Min:           refReduceOp<TensorType, Min>(m_inTensor, m_axesTensor, m_referenceOutputTensor, m_reduceParams->keep_dims); break;
              case ReduceOpType::op_Prod:          refReduceOp<TensorType, Prod>(m_inTensor, m_axesTensor, m_referenceOutputTensor, m_reduceParams->keep_dims); break;
              case ReduceOpType::op_Sum:           refReduceOp<TensorType, Sum>(m_inTensor, m_axesTensor, m_referenceOutputTensor, m_reduceParams->keep_dims); break;
              case ReduceOpType::op_SumSquare:     nnLog(MVLOG_ERROR, "Reduce: Operation SumSquare it's not supported.");break;
              default: assert(0); //unimp
            }
        }

        void formatTestParams(char* str, int maxLength) const override
        {
            const auto& d = m_outputTensor.tensorDims();
            const auto& l = m_outputTensor.tensorLimits();

            const char* layout_text = layoutString(m_currentTest->storageOrder);

            snprintf_append(str, maxLength, "H W C = %u %u %u (%u %u %u), %s",
                            d.height, d.width, d.channels, l.height, l.width, l.channels, layout_text);
        }

        void initParserRunner() override
        {
            initMyriadResources();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);
            OpTensor inBuff;
            OpTensor axesBuff;
            OpTensor outBuff;
            m_inTensor.exportToBuffer(inBuff);
            m_axesTensor.exportToBuffer(axesBuff);
            m_outputTensor.exportToBuffer(outBuff);

            customCppOp->addInputBuffer(inBuff, m_requiredTensorLocation);
            customCppOp->addInputBuffer(axesBuff, m_requiredTensorLocation);
            customCppOp->addOutputBuffer(outBuff,  m_requiredTensorLocation);
            customCppOp->ops = *getParams();
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();
            m_referenceOutputTensor.confirmBufferData();

            // save output data
            if (m_save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_inTensor.buffer()), m_inTensor.bufferSize(),
                                 "inMyriad0.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_axesTensor.buffer()), m_axesTensor.bufferSize(),
                                 "inMyriad1.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                 "outMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_referenceOutputTensor.buffer()),
                                 m_referenceOutputTensor.bufferSize(), "refOutMyriad.bin");
            }

            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float value, gt_value;
                if(std::is_same<TensorType, fp16>::value){
                    value    = f16Tof32(m_outputTensor.at(indices));
                    gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                }
                else{
                    value    = (float)(m_outputTensor.at(indices));
                    gt_value = (float)(m_referenceOutputTensor.at(indices));
                }
                float abs_diff = fabsf(value - gt_value);

                if (isnanf(value) && isnan(gt_value)) {
                    abs_diff = 0.0f;
                }
                else if (isinf(value) && isinf(gt_value)) {
                   if(signbit(value) == signbit(gt_value)) {
                     abs_diff = 0.0f;
                   }
                }

                bool differ = !bool(abs_diff <= m_test_threshold);
                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs)
                {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d:%d] outputShv = %f outputRef = %f diff = %f\n", ti.batch, ti.channels, ti.height, ti.width,
                            value, gt_value, abs_diff);
                }
            });

            return !threshold_test_failed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;
        ListIterator<ReduceOpInfo>  m_opInfoLoop;
        ListIterator<ReduceTestParams> m_dimsLoop;
        
        Tensor<TensorType> m_inTensor;
        Tensor<TensorType> m_outputTensor;
        Tensor<TensorType> m_referenceOutputTensor;
        Tensor<int32_t> m_axesTensor;
        sw_params::ReduceParams * m_reduceParams;
    };
}

// Currently disable Reduce family on HW, need to investigate when more resources are available.
#ifdef CONFIG_MOVISIM_RUN
namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Reduce_INT32))
{
typedef CustomCppReduceTest<int32_t> reduce_tests;
ICV_TESTS_REGISTER_SUITE(reduce_tests)
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Reduce_FP16))
{
typedef CustomCppReduceTest<fp16> reduce_tests;
ICV_TESTS_REGISTER_SUITE(reduce_tests)
}
#endif
