#include <custom_cpp_tests.h>
#include <cmath>
#include <random>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
//kernel name 
extern void*(shvNN0_topk);
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

static constexpr std::initializer_list<SingleTest> topk_test_list{
    {{5, 6, 7}, {1, 6, 1}, orderZYX, FPE("topk.elf"), {{1 /*axes*/,0/*mode=0,1(max,min)*/,2/*sort=0,1,2(none,value,index)*/,sw_params::Location::NN_CMX /*mem type*/,}}}, //int only
    //{{5, 6, 7}, {1, 6, 1}, orderZYX, FPE("topk.elf"), {{0/*gap*/,1 /*k_value*/,1 /*axes*/,0/*mode=0,1(max,min)*/,2/*sort=0,1,2(none,value,index)*/,2/*outputs=0,1,2(value,index,valueandindex)*/,sw_params::Location::NN_CMX /*mem type*/,}}},
};

// pair of (value, index) used in sorting
typedef std::pair<fp16, Index> Pair;

// comparison function comp(a,b) should return T if a precedes b
typedef std::function<bool(const Pair&, const Pair&)> CompareFunction;

static bool compareIndices(const Pair& a, const Pair& b)
{
    const Index aIndex = a.second;
    const Index bIndex = b.second;

    if (aIndex < bIndex) return true;
    if (aIndex > bIndex) return false;

    return true;
}

static bool compareValuesMax(const Pair& a, const Pair& b)
{
    const float aValue = f16Tof32(a.first);
    const float bValue = f16Tof32(b.first);

    if (!(aValue <= bValue)) return true;
    if (!(aValue >= bValue)) return false;

    return compareIndices(a, b);
}

static bool compareValuesMin(const Pair& a, const Pair& b)
{
    const float aValue = f16Tof32(a.first);
    const float bValue = f16Tof32(b.first);

    if (!(aValue >= bValue)) return true;
    if (!(aValue <= bValue)) return false;

    return compareIndices(a, b);
}

class CustomCppTopKTest: public CustomCppTests<fp16> {
public:
    explicit CustomCppTopKTest(): m_testsLoop(topk_test_list){}
    virtual ~CustomCppTopKTest(){}
protected:
    const char* suiteName() const override {
        return "CustomCppTopKTest";
    }
    void userLoops() override {
        addLoop(m_testsLoop);
    }
    
    void reverse(MemoryDims& dims, int m_dims) {
        for (int l = 0,r = m_dims - 1; l < r; l++, r--) {
            int32_t dim = dims.dims[l];
            dims.dims[l] = dims.dims[r];
            dims.dims[r] = dim;
        }
    }
    
    void initData() override{
        
        initElfBuffer();
        initTestCase();
        
        m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, MAX_LOCAL_PARAMS, 0, 0};
            
        paramContainer.resize(((int)sizeof(sw_params::TopKParams) + 7) / 8);
        CustomCppTests<fp16>::initData();
        const SingleTest* test = m_currentTest;
        int32_t ind[subspace::MAX_DIMS] = {0,};
        subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
        
        m_axis = ind[test->customLayerParams.layerParams[0]];
        m_mode = ind[test->customLayerParams.layerParams[1]];
        m_sort = ind[test->customLayerParams.layerParams[2]];
        m_hasOutputValues = 1;
        m_hasOutputIndices = 1;
        
        m_topkParams = reinterpret_cast<sw_params::TopKParams*>(paramContainer.data());
        *m_topkParams = sw_params::TopKParams();
        
        m_topkParams->axis = m_axis;
        m_topkParams->mode = m_mode;
        m_topkParams->sort = m_sort;
        m_topkParams->hasValues = 1;
        m_topkParams->hasIndices = 1;
        
        m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer.data());
        m_params.paramDataLen = paramContainer.size() * sizeof(uint64_t);
        m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[6]);
        m_params.baseParamData = sw_params::TopKParamsToBaseKernelParams(m_topkParams);
        
        const Dimensions& dimIn = m_currentTest->inDim;
        const Dimensions& dimOut = m_currentTest->outDim;
        const StorageOrder& storageOrder = m_currentTest->storageOrder;
        
        m_ndims = m_topkParams->inputValues.numDims;
        
        const TensorDims dims3In(dimIn.width,dimIn.height,  dimIn.channels,  1);
        const TensorDims dims3InKvalue(1, 1,  1,  1);
        const TensorDims dims3Out(dimOut.width, dimOut.height, dimOut.channels, 1);

        m_inputTensor.init(storageOrder, dims3In);
        m_inputKTensor.init(storageOrder, dims3InKvalue);
        m_refIndicesTensor.init(storageOrder, dims3Out);
        m_refValuesTensor.init(storageOrder, dims3Out);
        
        if (m_hasOutputValues==1)
            m_outputValuesTensor.init(storageOrder, dims3Out);
        if (m_hasOutputIndices==1)
            m_outputIndicesTensor.init(storageOrder, dims3Out);
        
        allocBuffer(m_inputTensor);
        allocBuffer(m_inputKTensor);
        allocBuffer(m_refIndicesTensor);
        allocBuffer(m_refValuesTensor);
        
        if (m_hasOutputValues==1)
            allocBuffer(m_outputValuesTensor);
        if (m_hasOutputIndices==1)
            allocBuffer(m_outputIndicesTensor);
        
    }
    
    void initTestCase() override {
        m_currentTest = &m_testsLoop.value();
        m_test_threshold = 0.0005f;
    }
    
    void initParserRunner() override {
        printf("init parser runner.\n");
        initMyriadResources();
        initDebugInfo();
        
        static_assert(std::is_base_of<Op, CustomCpp>());
        CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);

        Buffer inBuff;
        m_inputTensor.exportToBuffer(inBuff);
        customCppOp->addInputBuffer(inBuff, m_requiredTensorLocation);

        Buffer kBuff;
        m_inputKTensor.exportToBuffer(kBuff);
        customCppOp->addInputBuffer(kBuff, m_requiredTensorLocation);

        Buffer valueBuff;
        m_outputValuesTensor.exportToBuffer(valueBuff);
        customCppOp->addOutputBuffer(valueBuff, m_requiredTensorLocation);

        Buffer indexBuff;
        m_outputIndicesTensor.exportToBuffer(indexBuff);
        customCppOp->addOutputBuffer(indexBuff, m_requiredTensorLocation);

        customCppOp->ops = *getParams();
    }
    
    void generateInputData() override {
#ifdef CONFIG_TARGET_SOC_3720
        m_params.kernel  = reinterpret_cast<uint64_t>(&shvNN0_topk);
#else
        m_params.kernel  = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(topk));
#endif
        
        rand_seed();
        
        // set random seed
        u64 ticks_for_seed = rtems_clock_get_uptime_nanoseconds();
        srand(ticks_for_seed);
        
        // input values
        m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
            float tmp = float(rand() % 600) / 100 - 3.0f;
            m_inputTensor.at(indices) = f32Tof16(tmp);
        });

        // input K
        int32_t k = 1;
        m_inputKTensor.forEach(false, [&](const MemoryDims& indices) {
            m_inputKTensor.at(indices) = k;
        });

        // reference
        generateReferenceData();
    }
    
    void generateReferenceData() override {
        const auto k = 1;
        const auto mode = m_mode;
        const auto sort = m_sort;
        const auto ndims = m_ndims;
        const auto axis = m_ndims - 1 - m_axis; // Change axis due to layout
        
        MemoryDims dims = m_inputValuesTensor.memoryDims();
        const int n = dims.dims[axis];
        dims.dims[axis] = 1;
        
        const int inputValuesAxisStep = m_inputTensor.memorySteps().dims[axis];
        const int refValuesAxisStep = m_refValuesTensor.memorySteps().dims[axis];
        const int refIndicesAxisStep = m_refIndicesTensor.memorySteps().dims[axis];

        std::vector<Pair> temp(n);
        CompareFunction compareValues = modeComparison(mode);

        dims.forEach(ndims, dims.dims, [&](const MemoryDims& id){
            const auto inputValuesData = &m_inputTensor.at(id);
            for (int i = 0; i < n; ++i)
                temp[i] = Pair(inputValuesData[i * inputValuesAxisStep], i);
                         
            std::partial_sort(temp.begin(), temp.begin() + k, temp.begin() + n, compareValues);
            if (sort == 2) {
                std::sort(temp.begin(), temp.begin() + k, compareIndices);
            }

            auto refValuesData = &m_refValuesTensor.at(id);
            auto refIndicesData = &m_refIndicesTensor.at(id);
            for (int i = 0; i < k; ++i){
                const auto& t = temp[i];
                refValuesData[i * refValuesAxisStep] = t.first;
                refIndicesData[i * refIndicesAxisStep] = t.second;
            }
        });
    }
    static CompareFunction modeComparison(int32_t mode)
    {
        switch (mode)
        {
        case 0/*max*/: return compareValuesMax;
        case 1/*min*/: return compareValuesMin;
        default: mvTensorAssert(false); return nullptr;
        }
    }
    
    virtual bool checkResult() override{
        bool test_failed = false;
        m_outputTensor.confirmBufferData();
        
        if (m_hasOutputValues==1)
            m_outputValuesTensor.confirmBufferData();
        if (m_hasOutputIndices==1)
            m_outputIndicesTensor.confirmBufferData();
        
        m_refValuesTensor.forEach(true, [this, &test_failed](const MemoryDims& indices){
            const float gt_value    = f16Tof32(m_refValuesTensor.at(indices));
            float out_value   = (m_hasOutputValues == 1) ? f16Tof32(m_outputValuesTensor.at(indices)) : 0.f;
            const bool value_differ = (m_hasOutputValues == 1) && (out_value != gt_value);
            
            const Index gt_index    = m_refIndicesTensor.at(indices);
            const Index out_index   = (m_hasOutputIndices == 1) ? m_outputIndicesTensor.at(indices) : 0;
            const bool index_differ = (m_hasOutputIndices == 1) && (out_index != gt_index);

            const bool differ = value_differ || index_differ;
            test_failed = test_failed || differ;

            if (m_hasOutputValues!=1)
            {
                int32_t setCoords[MAX_DIMS];
                subspace::getCoord(out_index, m_inputValuesTensor.memoryDims().dims, m_inputValuesTensor.ndims(), setCoords);
                MemoryDims indicesIn(setCoords, m_inputValuesTensor.ndims());
                out_value = m_inputValuesTensor.at(indicesIn);
            }

            if (differ && GlobalData::doPrintDiffs)
            {
                char indices_str[64];
                printf("DIFF [%s] %f %s %f, %ld %s %ld\n",
                       ((m_hasOutputValues==1) ? m_outputValuesTensor.indicesToString(indices, indices_str) : m_outputIndicesTensor.indicesToString(indices, indices_str)),
                       out_value, value_differ ? "!=" : "==", gt_value,
                       out_index, index_differ ? "!=" : "==", gt_index);
            }
        });
        return !test_failed;
    }
private:
    ListIterator<SingleTest> m_testsLoop;
    int m_ndims;
    int m_axis;
    int m_mode;
    int m_sort;
    int m_hasOutputValues;
    int m_hasOutputIndices;
    
    std::vector<uint64_t> paramContainer;
    sw_params::TopKParams * m_topkParams;
    
    Tensor<int32_t> m_inputValuesTensor;
    Tensor<fp16> m_outputValuesTensor;
    Tensor<int32_t> m_inputKTensor;
    Tensor<int32_t> m_outputIndicesTensor;
    Tensor<fp16> m_refValuesTensor;
    Tensor<int32_t> m_refIndicesTensor;
};

ICV_TESTS_REGISTER_SUITE(CustomCppTopKTest)
}