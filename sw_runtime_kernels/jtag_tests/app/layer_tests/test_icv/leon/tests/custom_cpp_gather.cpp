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

#include <custom_cpp_tests.h>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.singleShaveGather.3010xx.text.xdat"
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_gather.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Gather)) {
    const bool saveToFile = false;

    static constexpr std::initializer_list<SingleTest> gather_test_list {
        {{4, 3, 2}, {4, 3, 2}, orderCHW, FPE("gather.elf"), {sw_params::Location::NN_CMX}}
    };

    class CustomCppGatherTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppGatherTest(): m_testsLoop(gather_test_list, "test") {
        }
        virtual ~CustomCppGatherTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppGatherTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            printf("init data.\n");
            sw_params::BaseKernelParams emptyParamData;
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, emptyParamData, MAX_LOCAL_PARAMS, 0};

            initElfBuffer();
            initTestCase();
            const Dimensions& dimIn = m_currentTest->inDim;
            //const Dimensions& dimOut = m_currentTest->outDim;
            const StorageOrder& storageOrder = m_currentTest->storageOrder;

            const TensorDims dims3Input(dimIn.width,   dimIn.height,  dimIn.channels,  1);
            const TensorDims dims3Indices(2, 3, 1, 1);
            const TensorDims dims3Axis(1, 1, 1, 1);
            const TensorDims dims3Output(4, 2, 3, 2);

            m_inputTensor.init(storageOrder, dims3Input);
            m_indicesTensor.init(0x21, dims3Indices); // orderHW Unsupported?
            m_axisTensor.init(storageOrder, dims3Axis);
            m_outputValueTensor.init(orderNCHW, dims3Output);
            m_referenceValueTensor.init(orderNCHW, dims3Output);

            allocBuffer(m_inputTensor);
            allocBuffer(m_indicesTensor);
            allocBuffer(m_axisTensor);
            allocBuffer(m_outputValueTensor);
            allocBuffer(m_referenceValueTensor);

            // TODO: Check window in kernel
            m_windowfp16.init(storageOrder, TensorDims(50, 1, 1, 1));
            m_windowint32.init(storageOrder, TensorDims(50, 1, 1, 1));
            allocBuffer(m_windowfp16);
            allocBuffer(m_windowint32);

            CustomCppTests<fp16>::initData();
            const SingleTest* test = m_currentTest;
            int32_t ind[subspace::MAX_DIMS] = {0};
            subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
            m_gatherParams = reinterpret_cast<sw_params::GatherParams*>(paramContainer);
            *m_gatherParams = sw_params::GatherParams();
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::GatherParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_gatherParams);

#ifdef CONFIG_TARGET_SOC_3720
            m_params.kernel = reinterpret_cast<uint64_t>(sk_singleShaveGather_3010xx_text);
#else
            m_params.kernel = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(singleShaveGather));
#endif
        }

        void initTestCase() override {
            printf("init test case.\n");
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.001f;
        }

        void initParserRunner() override {
            printf("init parser runner.\n");
            initMyriadResources();
            initDebugInfo();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);

            OpTensor inputBuff;
            m_inputTensor.exportToBuffer(inputBuff);
            customCppOp->addInputBuffer(inputBuff, m_requiredTensorLocation);

            OpTensor indicesBuff;
            m_indicesTensor.exportToBuffer(indicesBuff);
            customCppOp->addInputBuffer(indicesBuff, m_requiredTensorLocation);

            OpTensor axisBuff;
            m_axisTensor.exportToBuffer(axisBuff);
            customCppOp->addInputBuffer(axisBuff, m_requiredTensorLocation);

            OpTensor outputBuff;
            m_outputValueTensor.exportToBuffer(outputBuff);
            customCppOp->addOutputBuffer(outputBuff, m_requiredTensorLocation);

            // TODO: Check window in kernel
            OpTensor windowfp16Buff;
            m_windowfp16.exportToBuffer(windowfp16Buff);
            customCppOp->addOutputBuffer(windowfp16Buff, m_requiredTensorLocation);
            OpTensor windowint32Buff;
            m_windowint32.exportToBuffer(windowint32Buff);
            customCppOp->addOutputBuffer(windowint32Buff, m_requiredTensorLocation);

            customCppOp->ops = *getParams();
        }

        void generateInputData() override {
            printf("generate input data.\n");

            // input value
            float fp32Val = 1.0;
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                m_inputTensor.at(indices) = f32Tof16(fp32Val);
                fp32Val++;
            });

            // indices value
            m_indicesTensor.forEach(false, [&](const MemoryDims& indices) {
                m_indicesTensor.at(indices) = indices.dims[1];
            });

            // axis
            int32_t axis = 1;
            m_axisTensor.forEach(false, [&](const MemoryDims& indices) {
                m_axisTensor.at(indices) = axis;
            });
        }

        void generateReferenceData() override {
            printf("generate reference data.\n");

            int numIndicesDims = m_indicesTensor.ndims();
            int numInputDims = m_inputTensor.ndims();
            int axis = m_axisTensor.at(MemoryDims(0, 0, 0, 0));
            m_referenceValueTensor.forEach(false, [&](const MemoryDims& outIdx) {
                int32_t indicesDims[MAX_DIMS];
                for (int i = 0; i < numIndicesDims; i++) {
                    indicesDims[i] = outIdx.dims[axis + i];
                }
                MemoryDims indicesIdx(indicesDims, numIndicesDims);
                int inputDimIdx = m_indicesTensor.at(indicesIdx);

                int32_t inputDims[MAX_DIMS];
                for (int i = 0; i < numInputDims; i++) {
                    if (i < axis) {
                        inputDims[i] = outIdx.dims[i];
                    } else if (i == axis) {
                        inputDims[i] = inputDimIdx;
                    }
                    else {
                        inputDims[i] = outIdx.dims[i + numIndicesDims - 1];
                    }
                }
                MemoryDims inputIdx(inputDims, numInputDims);

                m_referenceValueTensor.at(outIdx) = m_inputTensor.at(inputIdx);
            });
        }

        virtual bool checkResult() override {
            printf("check results.\n");
            m_outputValueTensor.confirmBufferData();

            if (saveToFile) {
                saveMemoryToFile(reinterpret_cast<u32>(m_inputTensor.buffer()), m_inputTensor.bufferSize(),
                        "inMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_outputValueTensor.buffer()), m_outputValueTensor.bufferSize(),
                        "outMyriad.bin");

                saveMemoryToFile(reinterpret_cast<u32>(m_referenceValueTensor.buffer()),
                        m_referenceValueTensor.bufferSize(), "refOutMyriad.bin");
            }

            bool thresholdTestFailed = false;

            m_outputValueTensor.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_outputValueTensor.at(indices));
                float gtValue = f16Tof32(m_referenceValueTensor.at(indices));
                float absDiff = fabs(value - gtValue);
                bool differ = !bool(absDiff <= m_test_threshold);

                thresholdTestFailed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputValueTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] %f %f %f\n", ti.height, ti.width, ti.channels, value, gtValue,
                           absDiff);
                }
            });

            // TODO: Check window in kernel
            float fp32Val;
            for (int i = 0; i < 24; i++) {
                fp32Val = f16Tof32(m_windowfp16.at(MemoryDims(i, 0, 0, 0, 0, 0, 0, 0)));
                printf("fp16 window [%d]: %f\n", i, fp32Val);
            }

            int32_t int32Val;
            for (int i = 0; i < 50; i++) {
                int32Val = m_windowint32.at(MemoryDims(i, 0, 0, 0, 0, 0, 0, 0));
                printf("int32 window [%d]: %ld\n", i, int32Val);
            }

            for (int i = 0; i < 48; i++) {
                fp32Val = f16Tof32(m_outputValueTensor.at(MemoryDims(i, 0, 0, 0, 0, 0, 0, 0)));
                printf("output value [%d]: %f\n", i, fp32Val);
            }

            return !thresholdTestFailed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;

        Tensor<int32_t> m_indicesTensor;
        Tensor<int32_t> m_axisTensor;
        // Tensor params will align with m_input if output is not redefined.
        Tensor<half> m_outputValueTensor;
        Tensor<half> m_referenceValueTensor;

        // TODO: Check window in kernel
        // Tensor class is in icv_test_suite.h
        Tensor<fp16> m_windowfp16;
        Tensor<int32_t> m_windowint32;

        sw_params::GatherParams* m_gatherParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppGatherTest)
} // namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Gather))
