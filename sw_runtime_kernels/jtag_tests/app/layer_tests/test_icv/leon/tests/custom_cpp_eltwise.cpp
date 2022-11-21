//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include <cmath>
#include <random>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

enum EltOpType : int32_t {
    POWER,
    ADD,
    SUB,
    MIN,
    MAX,
    MUL,
    DIV,
};

typedef struct {
    EltOpType type;
    void* kernel;
} EltOpInfo;

struct EltwiseTestParams {
    std::initializer_list<int32_t> input1;
    std::initializer_list<int32_t> input2;
};

__attribute__((aligned(1024)))
#include "sk.eltwise_add_fp16.3720xx.text.xdat"
#include "sk.eltwise_div_fp16.3720xx.text.xdat"
#include "sk.eltwise_max_fp16.3720xx.text.xdat"
#include "sk.eltwise_min_fp16.3720xx.text.xdat"
#include "sk.eltwise_mul_fp16.3720xx.text.xdat"
#include "sk.eltwise_power_fp16.3720xx.text.xdat"
#include "sk.eltwise_sub_fp16.3720xx.text.xdat"

#define KERNEL_SELECT(a, b) (a)

#include "param_eltwise.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, EltwiseBinaryMath)) {

    // ToDo: to be removed after this PR is merged, it does some infrastructure cleanup:
    // [E#32635][VPUX37XX] VPUX37XX_SW_Layers Gather test and kernel #830
    static constexpr std::initializer_list<SingleTest> test_list{
            {{1, 1, 1}, {1, 1, 1}, orderXYZ, FPE("eltwise_fp16.elf"), {sw_params::Location::NN_CMX}},
    };

    static constexpr std::initializer_list<EltwiseTestParams> dims_list{
            {{1, 2, 1, 1}, {10, 2, 1, 1}},
            {{10, 2, 1, 1}, {1, 2, 1, 1}},
            {{10, 2, 1, 1}, {10, 2, 1, 2}},
    };

    static constexpr std::initializer_list<EltOpInfo> kernel_list{
            {EltOpType::POWER, KERNEL_SELECT(sk_eltwise_power_fp16_3720xx_text, &SLK_eltwise_power_fp16)},
            {EltOpType::ADD, KERNEL_SELECT(sk_eltwise_add_fp16_3720xx_text, &SLK_eltwise_add_fp16)},
            {EltOpType::SUB, KERNEL_SELECT(sk_eltwise_sub_fp16_3720xx_text, &SLK_eltwise_sub_fp16)},
            {EltOpType::MIN, KERNEL_SELECT(sk_eltwise_min_fp16_3720xx_text, &SLK_eltwise_min_fp16)},
            {EltOpType::MAX, KERNEL_SELECT(sk_eltwise_max_fp16_3720xx_text, &SLK_eltwise_max_fp16)},
            {EltOpType::MUL, KERNEL_SELECT(sk_eltwise_mul_fp16_3720xx_text, &SLK_eltwise_mul_fp16)},
            {EltOpType::DIV, KERNEL_SELECT(sk_eltwise_div_fp16_3720xx_text, &SLK_eltwise_div_fp16)},
    };

    class CustomCppEltwiseBinaryMathTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppEltwiseBinaryMathTest()
                : m_testsLoop(test_list, "test"), m_opInfoLoop(kernel_list, "kernel"), m_dimsLoop(dims_list, "dims") {
        }

        virtual ~CustomCppEltwiseBinaryMathTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppEltwiseBinaryMathTest";
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

        void broadcastShapeTo4D(std::vector<int32_t>& shape) {
            mvTensorAssert((shape.size() <= 4), "Tensors with rank > 4 are not supported");

            for (int i = shape.size(); i < 4; i++) {
                shape.push_back(1);
            }
        }

        std::vector<int32_t> calcOutputDims(std::vector<int32_t>& shape1, std::vector<int32_t>& shape2) {
            std::vector<int32_t> outShape(std::max(shape1.size(), shape2.size()), 0);

            for (uint32_t i = 0; i < outShape.size(); i++) {
                mvTensorAssert((shape1[i] == 1 || shape2[i] == 1 || shape1[i] == shape2[i]),
                               "Got non broadcastable dimensions pair");

                outShape[i] = std::max(shape1[i], shape2[i]);
            }

            return outShape;
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, emptyParamData, MAX_LOCAL_PARAMS, 0};

            initElfBuffer();
            initTestCase();

            std::vector<int32_t> inputDims1 = m_dimsLoop.value().input1;
            std::vector<int32_t> inputDims2 = m_dimsLoop.value().input2;
            broadcastShapeTo4D(inputDims1);
            broadcastShapeTo4D(inputDims2);

            std::vector<int32_t> outputDims = calcOutputDims(inputDims1, inputDims2);

            const MemoryDims md_inputDims1(inputDims1.data(), inputDims1.size());
            const MemoryDims md_inputDims2(inputDims2.data(), inputDims2.size());
            const MemoryDims md_outputDims(outputDims.data(), outputDims.size());

            const auto inOrder1 = ieOrderFromNumDims(inputDims1.size());
            const auto inOrder2 = ieOrderFromNumDims(inputDims2.size());
            const auto outOrder = ieOrderFromNumDims(outputDims.size());

            m_inTensor[0].init(inOrder1, md_inputDims1);
            m_inTensor[1].init(inOrder2, md_inputDims2);
            m_outputTensor.init(outOrder, md_outputDims);
            m_referenceOutputTensor.init(outOrder, md_outputDims);

            allocBuffer(m_inTensor[0]);
            allocBuffer(m_inTensor[1]);
            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceOutputTensor);

            m_powParams = reinterpret_cast<sw_params::EltwiseParams*>(paramContainer);
            *m_powParams = sw_params::EltwiseParams();  // default ctor init obj

            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::EltwiseParams);
            m_requiredTensorLocation =
                    static_cast<sw_params::Location>(m_currentTest->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_powParams);

            m_params.kernel = reinterpret_cast<uint64_t>(m_opInfoLoop.value().kernel);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.05f;
        }

        void defaultTensorInitializer(Tensor<fp16>& tensor) {
            tensor.forEach(false, [&](const MemoryDims& indices) {
                tensor.at(indices) = f32Tof16(float(rand()) / RAND_MAX * 256 - 128);
            });
        }

        void generateInputData() override {
            rand_seed();

            if (m_opInfoLoop.value().type == EltOpType::POWER) {
                m_inTensor[0].forEach(false, [&](const MemoryDims& indices) {
                    const auto range = static_cast<int32_t>(m_inTensor[0].fullSize()) + 3;
                    const auto index1 = m_inTensor[0].index(indices);
                    const auto index2 = static_cast<int32_t>((static_cast<uint64_t>(index1) * index1 + 17) % range);

                    auto val = 2 * static_cast<float>(index2 * 127 + 13 - range / 2) / range;
                    val = (val < 2) ? 3 : val;

                    m_inTensor[0].at(indices) = f32Tof16(val);
                });

                const auto pow_range = 0.8f;
                const auto pow_steps = m_inTensor[1].fullSize();
                m_inTensor[1].forEach(false, [&](const MemoryDims& indices) {
                    const auto step = pow_range / pow_steps;
                    const auto index = m_inTensor[1].index(indices);
                    const auto sign = (index & 1) * 2 - 1;  // +1 or -1
                    const auto val = sign * (pow_range + index * step);

                    m_inTensor[1].at(indices) = f32Tof16(val);
                });
            } else if (m_opInfoLoop.value().type == EltOpType::DIV) {
                defaultTensorInitializer(m_inTensor[0]);

                m_inTensor[1].forEach(false, [&](const MemoryDims& indices) {
                    const float floatValue = float(rand()) / RAND_MAX * 256 - 128;
                    const fp16 tValue = f32Tof16(floatValue);
                    const fp16 tOne = f32Tof16(1.f);
                    const fp16 tZero = f32Tof16(0.f);
                    m_inTensor[1].at(indices) = (tValue == tZero) ? tOne : tValue;
                });
            } else {
                defaultTensorInitializer(m_inTensor[0]);
                defaultTensorInitializer(m_inTensor[1]);
            }
        }

        static float myPow(float base, float exp) {
            // CVadim:  in some cases powf(float, float) computes a wrong result, on leon
            //          ex. powf(0.290039, -1.889648) = 0.000000 (10.36 is the right answer)
            //          use `double` as a workaround
            // ToDo: to find out if it is a known problem
            float out = std::pow((double)base, exp);

            // Workaround for Leon-compiler for this case (which returns -Inf instead of +Inf)
            // "powf(+/-0, exponent) where exponent is negative, finite and is even-integer or non-integer, returns
            // +Inf"
            if ((base == 0) && ((exp < 0) && !isinf(exp))) {
                bool isInt = (ceilf(exp) == exp);
                bool isEvenInt = isInt && ((((int)exp) & 0x1) == 0);
                if ((!isInt) || isEvenInt) {
                    out = (+INFINITY);
                }
            }

            return out;
        }

        void broadcastIndices(Tensor<fp16>& tensor, MemoryDims& inIndices, const MemoryDims& outIndices) {
            int i;
            for (i = 0; i < tensor.ndims(); i++) {
                if (tensor.memoryDims().dims[i] == 1)
                    inIndices.dims[i] = 0;
                else
                    inIndices.dims[i] = outIndices.dims[i];
            }
        }

        void generateReferenceData() override {
            std::function<float(const float&, const float&)> reference;

            switch (m_opInfoLoop.value().type) {
            case EltOpType::POWER:
                reference = [](const float& a, const float& b) {
                    return myPow(a, b);
                };
                break;
            case EltOpType::ADD:
                reference = [](const float& a, const float& b) {
                    return a + b;
                };
                break;
            case EltOpType::SUB:
                reference = [](const float& a, const float& b) {
                    return a - b;
                };
                break;
            case EltOpType::MIN:
                reference = [](const float& a, const float& b) {
                    return a < b ? a : b;
                };
                break;
            case EltOpType::MAX:
                reference = [](const float& a, const float& b) {
                    return a > b ? a : b;
                };
                break;
            case EltOpType::MUL:
                reference = [](const float& a, const float& b) {
                    return a * b;
                };
                break;
            case EltOpType::DIV:
                reference = [](const float& a, const float& b) {
                    return a / b;
                };
                break;
            default:
                assert(0);  // unimp
            }

            m_referenceOutputTensor.forEach(false, [&](const MemoryDims& indices) {
                MemoryDims inIndices1, inIndices2;
                broadcastIndices(m_inTensor[0], inIndices1, indices);
                broadcastIndices(m_inTensor[1], inIndices2, indices);

                float a = f16Tof32(m_inTensor[0].at(inIndices1));
                float b = f16Tof32(m_inTensor[1].at(inIndices2));
                float ref = reference(a, b);
                m_referenceOutputTensor.at(indices) = f32Tof16(ref);
            });
        }

        void initParserRunner() override {
            initMyriadResources();
            initDebugInfo();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);
            OpTensor inBuff[2];
            OpTensor outBuff;
            m_inTensor[0].exportToBuffer(inBuff[0]);
            m_inTensor[1].exportToBuffer(inBuff[1]);
            m_outputTensor.exportToBuffer(outBuff);

            customCppOp->addInputBuffer(inBuff[0], m_requiredTensorLocation);
            customCppOp->addInputBuffer(inBuff[1], m_requiredTensorLocation);
            customCppOp->addOutputBuffer(outBuff, m_requiredTensorLocation);
            customCppOp->ops = *getParams();
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            // save output data
            if (m_save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_inTensor[0].buffer()), m_inTensor[0].bufferSize(),
                                 "inMyriad0.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_inTensor[1].buffer()), m_inTensor[1].bufferSize(),
                                 "inMyriad1.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                 "outMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_referenceOutputTensor.buffer()),
                                 m_referenceOutputTensor.bufferSize(), "refOutMyriad.bin");
            }

            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                float abs_diff = fabsf(value - gt_value);

                if (isnanf(value) && isnan(gt_value)) {
                    abs_diff = 0.0f;
                } else if (isinf(value) && isinf(gt_value)) {
                    if (signbit(value) == signbit(gt_value)) {
                        abs_diff = 0.0f;
                    }
                }

                bool differ = !bool(abs_diff <= m_test_threshold);
                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    MemoryDims inIndices1, inIndices2;
                    broadcastIndices(m_inTensor[0], inIndices1, indices);
                    broadcastIndices(m_inTensor[1], inIndices2, indices);

                    float inValue1 = f16Tof32(m_inTensor[0].at(inIndices1));
                    float inValue2 = f16Tof32(m_inTensor[1].at(inIndices2));

                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] input1 = %f input2 = %f outputShv = %f outputRef = %f diff = %f\n",
                           ti.height, ti.width, ti.channels, inValue1, inValue2, value, gt_value, abs_diff);
                }
            });
            return !threshold_test_failed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;
        ListIterator<EltOpInfo> m_opInfoLoop;
        ListIterator<EltwiseTestParams> m_dimsLoop;
        Tensor<fp16> m_inTensor[2];
        sw_params::EltwiseParams* m_powParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppEltwiseBinaryMathTest)
}  // namespace )