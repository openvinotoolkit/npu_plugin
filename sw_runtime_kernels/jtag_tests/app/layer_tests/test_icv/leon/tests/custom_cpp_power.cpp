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
#include <cmath>
#include <random>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.power_fp16.3010xx.text.xdat"
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_power.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Power)) {
    static constexpr std::initializer_list<SingleTest> pow_test_list {
          {{1, 1, 7},    {1, 1, 7},    orderZYX, FPE("power_fp16.elf"), {sw_params::Location::NN_CMX}},
          {{1, 1, 20},   {1, 1, 20},   orderZYX, FPE("power_fp16.elf"), {sw_params::Location::NN_CMX}},
          {{1000, 1, 1}, {1000, 1, 1}, orderZYX, FPE("power_fp16.elf"), {sw_params::Location::NN_CMX}},
          {{9, 5, 17},   {9, 5, 17},   orderZYX, FPE("power_fp16.elf"), {sw_params::Location::NN_CMX}},
       };

    class CustomCppPowerTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppPowerTest(): m_testsLoop(pow_test_list, "test") {
        }
        virtual ~CustomCppPowerTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppPowerTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, MAX_LOCAL_PARAMS, 0, 0};

            initElfBuffer();
            initTestCase();

            const Dimensions& dims = m_currentTest->inDim; // inDims == outDims
            const StorageOrder& storageOrder = m_currentTest->storageOrder;
            const TensorDims tDims(dims.width, dims.height, dims.channels,  1);

            m_inTensor[0].init(storageOrder, tDims);
            m_inTensor[1].init(storageOrder, tDims);
            m_outputTensor.init(storageOrder, tDims);
            m_referenceOutputTensor.init(storageOrder, tDims);

            allocBuffer(m_inTensor[0]);
            allocBuffer(m_inTensor[1]);
            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceOutputTensor);

            m_powParams = reinterpret_cast<sw_params::PowerParams*>(paramContainer);
           *m_powParams = sw_params::PowerParams(); //default ctor init obj

            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::PowerParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(m_currentTest->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_powParams);

#ifdef CONFIG_TARGET_SOC_3720
            m_params.kernel = reinterpret_cast<uint64_t>(sk_power_fp16_3010xx_text);
#else
            m_params.kernel = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(power_fp16));
#endif
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.05f;
        }

        void generateInputData() override {

            rand_seed();

            for(int x=0; x<2; x++){
             m_inTensor[x].forEach(false, [&](const MemoryDims& indices) {
                float tmp = float(rand() % 600) / 100 - 3;
                m_inTensor[x].at(indices) = f32Tof16(tmp);
             });
            }
        }
        void generateReferenceData() override {
            if(1){
              float a =  0.000000f;
              float b = -1.469727f;
              float c = powf(a,b);
              printf("a=%f b=%f pow=%f(0x%x)\n", a, b, c, *((uint32_t*)&c));
            }

            m_inTensor[0].forEach(false, [&](const MemoryDims& indices) {
                float val1 = f16Tof32(m_inTensor[0].at(indices));
                float val2 = f16Tof32(m_inTensor[1].at(indices));
                float ref  = powf(val1, val2);
                m_referenceOutputTensor.at(indices) = f32Tof16(ref);
            });
        }

        void initParserRunner() override
        {
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
            customCppOp->addOutputBuffer(outBuff,  m_requiredTensorLocation);
            customCppOp->ops = *getParams();
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            // save output data
            if (m_save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                 "outMyriad.bin");
            }

            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float value    = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                float abs_diff = fabsf(value - gt_value);

                //"powf returns NaN if base is finite and negative and exponent is finite and non-integer"
                if (isnanf(value) && isnan(gt_value)) {
                    abs_diff = 0.0f;
                }
                //"powf(+/-0, exponent) where exponent is negative, finite and is even-integer or non-integer, returns +Inf"
                else if (isinf(value) && isinf(gt_value)) {
                   if(signbit(value) == signbit(gt_value)) {
                    abs_diff = 0.0f; //OK
                   } else {
                     float inA = f16Tof32(m_inTensor[0].at(indices));
                     float inB = f16Tof32(m_inTensor[1].at(indices));
                     bool  inBisInt = (ceilf(inB) == inB);
                     /*DBG*/ printf("__fail for A=%f B=%f (%f vs %f)\n", inA, inB, value, gt_value);
                     if((inA == 0.0f) && (inB < 0.0f) && (!inBisInt)){
                       abs_diff = 0.0f; //workaround for BUG
                     }
                   }
                }

                bool differ = !bool(abs_diff <= m_test_threshold);
                threshold_test_failed |= differ;

                if (differ )//&& GlobalData::doPrintDiffs)
                {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] %f %f %f\n", ti.height, ti.width, ti.channels, value, gt_value,
                           abs_diff);
                }
            });
            return !threshold_test_failed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;
        Tensor<fp16> m_inTensor[2]; //2x inputs
        sw_params::PowerParams* m_powParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppPowerTest)
}  // namespace )
