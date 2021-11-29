#include <custom_cpp_tests.h>
#include <cmath>
#include <random>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
//kernel name 
extern void*(shvNN0_hswish_fp16);
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_topk.h"

#define USE_SEED_VALUE 0xbdd1cb13  // defined to use this value as random seed

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, TopK)) {
#define ALL_MODES_SET
//#define ALL_OUTPUTS_SET
#define ALL_SORTS_SET
    
    //#define USE_ARGMAX_TESTS /* use testing data from old ArgMax tests */

#define TOPKSORT_NONE_SUPPORTED /* uncomment when and only if TopKSort::none will be supported */

    const bool save_to_file = false;

#define WHOLE_SLICE (-1) /* special K value, means 'all' */
typedef int32_t Index;

typedef t_D8StorageOrder StorageOrder;
typedef std::initializer_list<int32_t> Dims;
typedef std::initializer_list<int32_t> Gaps;
enum class TopKOutput { value = 1 << 0, index = 1 << 1, valueAndIndex = value | index };

static constexpr std::initializer_list<SingleTest> topk_test_list{
    {{5, 6, 7}, {5, 6, 7}, orderZYX, FPE("topk.elf"), {{{0, 0, 0, 0, 0, 0, 0, 0}/*gaps_list*/,1 /*k_value*/,1 /*axes*/,TopKMode::max/*mode*/,TopKSort::index/*sort*/,TopKOutput::valueAndIndex/*outputs_list*/,sw_params::Location::UPA_CMX /*mem type*/,}}},
    {{5, 6, 7}, {5, 6, 7}, orderZYX, FPE("topk.elf"), {{0 /*axis*/,0 /*index_value*/,0 /*mode {max, min}*/,2 /*sort {none, value, index};*/,sw_params::Location::NN_CMX/*mem type*/,}}},
};
    
    class CustomCppTopKTest: public CustomCppTests<fp16> {
    public:
        
    };
}