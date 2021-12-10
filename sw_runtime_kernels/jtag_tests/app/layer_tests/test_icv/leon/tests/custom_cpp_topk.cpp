#include <custom_cpp_tests.h>
#include <cmath>
#include <random>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
//kernel name 
extern void*(shvNN0_singleShaveTopK);
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_topk.h"

#define USE_SEED_VALUE 0xbdd1cb13  // defined to use this value as random seed

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, TopK)) {
typedef int32_t Index;
typedef t_D8StorageOrder StorageOrder;

static constexpr std::initializer_list<SingleTest> topk_test_list{
    {{4, 3, 3}, {4, 3, 1}, orderZYX, FPE("topk.elf"), {{1,2/*axes*/,0/*mode=0,1(max,min)*/,1/*sort=0,1(value,index)*/,sw_params::Location::NN_CMX /*mem type*/,}}}, //int only
//    {{1, 4, 5}, {1, 1, 5}, orderZYX, FPE("topk.elf"), {{1,1/*axes*/,0/*mode=0,1(max,min)*/,1/*sort=0,1(value,index)*/,sw_params::Location::NN_CMX /*mem type*/,}}}, //int only
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
    explicit CustomCppTopKTest(): m_testsLoop(topk_test_list,"test"){}
    virtual ~CustomCppTopKTest(){}
protected:
    const char* suiteName() const override {
        return "CustomCppTopKTest";
    }
    void userLoops() override {
        addLoop(m_testsLoop);
    }
    
    void initData() override{
        printf("init data.\n");
        m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, MAX_LOCAL_PARAMS, 0, 0};
        paramContainer.resize(((int)sizeof(sw_params::TopKParams) + 7) / 8);
        
        initElfBuffer();
        initTestCase();
        
        const Dimensions& dimIn = m_currentTest->inDim;
        const Dimensions& dimOut = m_currentTest->outDim;
        const StorageOrder& storageOrder = m_currentTest->storageOrder;
        
        const TensorDims dims3K(1, 1, 1, 1);
        const TensorDims dims3In(dimIn.width,   dimIn.height,  dimIn.channels,  1);
        const TensorDims dims3Out(dimOut.width, dimOut.height, dimOut.channels, 1);

        m_inputTensor.init(storageOrder, dims3In);
        m_kTensor.init(storageOrder, dims3K);
        m_valueTensor.init(storageOrder, dims3Out);
        m_indexTensor.init(storageOrder, dims3Out);
        m_referenceValueTensor.init(storageOrder, dims3Out);
        m_referenceIndexTensor.init(storageOrder, dims3Out);
        
        allocBuffer(m_inputTensor);
        allocBuffer(m_kTensor);
        allocBuffer(m_valueTensor);
        allocBuffer(m_indexTensor);
        allocBuffer(m_referenceValueTensor);
        allocBuffer(m_referenceIndexTensor);
        
        const SingleTest* test = m_currentTest;
        int32_t ind[subspace::MAX_DIMS] = {0};
        subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
        
        m_k = ind[test->customLayerParams.layerParams[0]];
        m_axis = ind[test->customLayerParams.layerParams[1]];
        m_mode = ind[test->customLayerParams.layerParams[2]];
        m_sort = ind[test->customLayerParams.layerParams[3]];
        m_hasOutputValues = 1;
        m_hasOutputIndices = 1;
        
        m_TopKParams = reinterpret_cast<sw_params::TopKParams*>(paramContainer.data());
        *m_TopKParams = sw_params::TopKParams();
        m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer.data());
        m_params.paramDataLen = paramContainer.size() * sizeof(uint64_t);
        m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
        m_params.baseParamData = sw_params::ToBaseKernelParams(m_TopKParams);
        
        m_TopKParams->axis = m_axis;
        m_TopKParams->mode = m_mode;
        m_TopKParams->sort = m_sort;
        m_TopKParams->hasValues = m_hasOutputValues;
        m_TopKParams->hasIndices = m_hasOutputIndices;
    }
    
    void initTestCase() override {
        printf("init test case.\n");
        m_currentTest = &m_testsLoop.value();
        m_test_threshold = 0.01f;
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
        m_kTensor.exportToBuffer(kBuff);
        customCppOp->addInputBuffer(kBuff, m_requiredTensorLocation);
        
        Buffer valueBuff;
        m_valueTensor.exportToBuffer(valueBuff);
        customCppOp->addOutputBuffer(valueBuff, m_requiredTensorLocation);

        Buffer indexBuff;
        m_indexTensor.exportToBuffer(indexBuff);
        customCppOp->addOutputBuffer(indexBuff, m_requiredTensorLocation);

        customCppOp->ops = *getParams();
    }
    
    void generateInputData() override {
        printf("generate input data.\n");
#ifdef CONFIG_TARGET_SOC_3720
        m_params.kernel  = reinterpret_cast<uint64_t>(&shvNN0_singleShaveTopK);
#else
        m_params.kernel  = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(singleShaveTopK));
#endif
        
        const u32 range = m_inputTensor.fullSize() + 3;

        // input values
//        const int total = m_inputTensor.dataSize();
//        float scale = std::min(total, 50000) / float(total); // prevent FP16 overflow
//        m_inputTensor.forEach(false, [&](const MemoryDims& indices)
//        {
//            int index = m_inputTensor.index(indices);
//            u32 index1 = u32( (u64(index) * index + 17) % range );
//            float val1 = scale * float(index1*1 + range/3) / float(range);
//            m_inputTensor.at(indices) = f32Tof16(val1);
//        });
        
//        float tmp_val = 1;
//        m_inputTensor.forEach(false, [&](const MemoryDims& indices)
//        {
//            m_inputTensor.at(indices) = f32Tof16(tmp_val);
//            tmp_val++;
//        });
        
        m_inputTensor.forEach(false, [&](const MemoryDims& indices)
                              {
                                  float tmp = float(rand() % 1000) / 100 - 5.0f;
                                  m_inputTensor.at(indices) = f32Tof16(tmp);
                              });
        
        int32_t k = m_k;
        m_kTensor.forEach(false, [&](const MemoryDims& indices) {
            m_kTensor.at(indices) = k;
        });
    }
    
    void generateReferenceData() override {
        printf("generate reference data.\n");
        const auto k = m_k;
        const auto mode = m_mode;
        const auto sort = m_sort;
        const auto ndims = m_inputTensor.ndims();
        const auto axis = m_axis;
        
        printf("k = %d\n", k);
        printf("axis = %d\n", axis);
        printf("mode = %d\n", mode);
        printf("sort = %d\n", sort);
        printf("ndims = %d\n", ndims);
        
        MemoryDims dims = m_inputTensor.memoryDims();
        const int n = dims.dims[axis];
        dims.dims[axis] = 1;
        
        mvTensorAssert(k <= n);
        mvTensorAssert(m_inputTensor.storageOrder() == m_referenceValueTensor.storageOrder());
        mvTensorAssert(m_inputTensor.storageOrder() == m_referenceIndexTensor.storageOrder());
        
        const int inputValuesAxisStep = m_inputTensor.memorySteps().dims[axis];
        for (int i = 0; i < 3; i++) {
            printf("m_inputTensor.memorySteps().dims[%d] = %d\n", i, m_inputTensor.memorySteps().dims[i]);
        }

        const int refValuesAxisStep = m_referenceValueTensor.memorySteps().dims[axis];
        const int refIndicesAxisStep = m_referenceIndexTensor.memorySteps().dims[axis];

        std::vector<Pair> temp(n);
        CompareFunction compareValues = modeComparison(mode);

        dims.forEach(ndims, dims.dims, [&](const MemoryDims& id){
            const auto inputValuesData = &m_inputTensor.at(id);
            
            for (int i = 0; i < n; ++i) {
                temp[i] = Pair(inputValuesData[i * inputValuesAxisStep], i);
                printf("inputValuesData[i * inputValuesAxisStep] = %f\n",
                       f16Tof32(inputValuesData[i * inputValuesAxisStep]));
            }
            
            std::partial_sort(temp.begin(), temp.begin() + k, temp.begin() + n, compareValues);
            if (sort == 0) {
                std::sort(temp.begin(), temp.begin() + k, compareIndices);
            }

            auto refValuesData = &m_referenceValueTensor.at(id);
            auto refIndicesData = &m_referenceIndexTensor.at(id);
            for (int i = 0; i < k; ++i){
                const auto& t = temp[i];
                refValuesData[i * refValuesAxisStep] = t.first;
                refIndicesData[i * refIndicesAxisStep] = t.second;
                printf("refValuesData[%d] = %f, refIndicesData[%d] = %d\n", i * refValuesAxisStep, f16Tof32(refValuesData[i * refValuesAxisStep]), i * refIndicesAxisStep, refIndicesData[i * refIndicesAxisStep]);
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
        printf("check result.\n");
        bool test_failed = false;
        
        m_inputTensor.confirmBufferData();
        m_valueTensor.confirmBufferData();
        m_indexTensor.confirmBufferData();
        m_referenceValueTensor.confirmBufferData();
        
        saveMemoryToFile(reinterpret_cast<u32>(m_inputTensor.buffer()), m_inputTensor.bufferSize(), "input.bin");
        saveMemoryToFile(reinterpret_cast<u32>(m_valueTensor.buffer()), m_valueTensor.bufferSize(), "outvalue.bin");
        saveMemoryToFile(reinterpret_cast<u32>(m_referenceValueTensor.buffer()), m_referenceValueTensor.bufferSize(), "outref.bin");
        saveMemoryToFile(reinterpret_cast<u32>(m_indexTensor.buffer()), m_indexTensor.bufferSize(), "outindex.bin");
        
        m_referenceValueTensor.forEach(true, [this, &test_failed](const MemoryDims& indices){
            const float gt_value    = f16Tof32(m_referenceValueTensor.at(indices));
            float value = f16Tof32(m_valueTensor.at(indices));
            float abs_diff = fabs(value - gt_value);
            bool value_differ = !bool(abs_diff <= m_test_threshold);
            const Index gt_index = m_referenceIndexTensor.at(indices);
            const Index out_index = m_indexTensor.at(indices);
            const bool index_differ = (out_index != gt_index);
            const bool differ = value_differ || index_differ;
            test_failed = test_failed || differ;
            printf("m_valueTensor value = %f, gt_value = %f\n", value, gt_value);
            printf("m_indexTensor out_index = %ld, gt_index = %ld\n", out_index, gt_index);
        });
        return !test_failed;
    }
private:
    ListIterator<SingleTest> m_testsLoop;
    int32_t m_k;
    int32_t m_axis;
    int32_t m_mode;
    int32_t m_sort;
    int m_hasOutputValues;
    int m_hasOutputIndices;
    
    std::vector<uint64_t> paramContainer;
    sw_params::TopKParams* m_TopKParams;
    
    Tensor<fp16> m_valueTensor;
    Tensor<int32_t> m_indexTensor;
    Tensor<int32_t> m_kTensor;
    
    Tensor<fp16> m_referenceValueTensor;
    Tensor<int32_t> m_referenceIndexTensor;
};

ICV_TESTS_REGISTER_SUITE(CustomCppTopKTest)
}