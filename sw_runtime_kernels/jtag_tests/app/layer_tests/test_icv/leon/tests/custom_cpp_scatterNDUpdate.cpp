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
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.singleShaveScatterNDUpdate.3010xx.text.xdat"
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_scatterNDUpdate.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, ScatterNDUpdate)) {

    const bool save_to_file = false;
// #define ALL_PARAMS_SET

    static constexpr std::initializer_list<SingleTest> scatterNDUpdate_test_list{

    };

    class CustomCppScatterNDUpdateTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppScatterNDUpdateTest(): m_testsLoop(scatterNDUpdate_test_list, "test") {
        }
        virtual ~CustomCppScatterNDUpdateTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppScatterNDUpdateTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {

        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
        }

        void generateInputData() override {

        }

        void generateReferenceData() override {

        }

        virtual bool checkResult() override {

        }

    private:
        ListIterator<SingleTest> m_testsLoop;

        sw_params::ScatterNDUpdateParams* m_scatterNDUpdateParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppScatterNDUpdateTest)
}  // namespace )
