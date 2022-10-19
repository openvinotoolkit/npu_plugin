// {% copyright %}

#include "icv_test_suite.h"

#include "jtag_interactions.h" // get_from_debug()
#include "mvTensor_cpp.h" // Processor

#if defined(CONFIG_OS_DRV_SVUNN_USE_CMX)
# define NN_CMX_SECTION __attribute__((section(".cmx_direct.data")))
#else
# define NN_CMX_SECTION __attribute__((section(".nncmx0.shared.data")))
#endif

using namespace nn::shave_lib;

namespace icv_tests
{

u8 DDR_BSS GlobalData::memoryBuffer[ICV_TESTS_MEMORY_BUFFER_SIZE] __attribute__((aligned(ICV_TESTS_MEMORY_BUFFER_ALIGN)));

TensorBase::ErrorHandler* TensorBase::errorHandler = TensorBase::defaultHandler;

ShavePerfCounters NN_CMX_SECTION GlobalData::shavePerfCounters{};

u8 * GlobalData::getMemoryBuffer() {
    return GlobalData::memoryBuffer;
}

const int GlobalData::dataPartitionNo = 0;

const int GlobalData::instrPartitionNo = 1;

const int GlobalData::bypassPartitionNo = 2;

const int GlobalData::maxShaves = MVTENSOR_MAX_SHAVES;

int GlobalData::startShave = 0;

int GlobalData::numShavesToBegin = MVTENSOR_MAX_SHAVES;
int GlobalData::numShavesToEnd   = MVTENSOR_MAX_SHAVES;

int GlobalData::numRepeats = 1;

bool GlobalData::doPrintDiffs     = false;
bool GlobalData::doPrintDiffRange = false;
bool GlobalData::doPrintDiffMax   = false;

RunMode GlobalData::runMode     = RunMode::Run;
bool GlobalData::doPrintName    = false;
bool GlobalData::doPrintTime    = false;
bool GlobalData::doPrintParams  = false;
CheckMode GlobalData::checkMode = CheckMode::Continue;
bool GlobalData::doCallOnce     = true;

bool GlobalData::doPrintPerfCounters = false;

char GlobalData::testFilter[ICV_TEST_TEST_FILTER_STR_MAXSIZE] = "";

void GlobalData::init()
{
    startShave       = get_from_debug(startShave);
    numShavesToBegin = get_from_debug(numShavesToBegin);
    numShavesToEnd   = get_from_debug(numShavesToEnd);
    numRepeats       = get_from_debug(numRepeats);

    doPrintDiffs     = get_from_debug(doPrintDiffs);
    doPrintDiffRange = get_from_debug(doPrintDiffRange);
    doPrintDiffMax   = get_from_debug(doPrintDiffMax);

    runMode          = get_from_debug(runMode);
    doPrintName      = get_from_debug(doPrintName);
    doPrintTime      = get_from_debug(doPrintTime);
    doPrintParams    = get_from_debug(doPrintParams);
    checkMode        = get_from_debug(checkMode);
    doCallOnce       = get_from_debug(doCallOnce);

    doPrintPerfCounters = get_from_debug(doPrintPerfCounters);

    int testFilterLength = get_from_debug(0);
    get_from_debug(testFilterLength, testFilter);

    startShave = std::max(0, std::min(startShave, maxShaves - 1));

    const int limit = maxShaves - startShave;
    numShavesToBegin = std::max(0, std::min(numShavesToBegin, limit));
    numShavesToEnd   = std::max(0, std::min(numShavesToEnd, limit));

    numRepeats = std::max(1, numRepeats);
}

bool Logger::globalEnabled = true;
int Logger::level = 0;

SuiteRegistry& SuiteRegistry::getInstance()
{
    static SuiteRegistry registry;
    return registry;
}

void SuiteRegistry::registerSuite(const TestSuiteFactory& suite)
{
    auto& registry = SuiteRegistry::getInstance();
    auto& suites = registry.m_suiteBuilders;
    // Insert as a first element to have test suites sorted in ascending order
    suites.push_front(suite);
}

namespace ulp
{

//bool debug = true;
bool debug = false;

constexpr int fp16_sigbits = 10; // significand length, in bits (hidden leading '1' not counted)
constexpr int fp32_sigbits = 23;

constexpr float scale_fp16_ulp_to_fp32 = static_cast<float>(1 << (fp32_sigbits - fp16_sigbits));

constexpr float fp16_min = 1.0f / (1 << 14); // 2^(-14) = 0.00006103515625f == min non-zero FP16 (no denormals)

// fp16 ulp difference for two values both >= fp16_min (regular ulp definition)
// required: fmin <= fmax
float bigdiff_fp32(float fmin, float fmax)
{
    fp32_union umin = { .f = fmin };
    fp32_union umax = { .f = fmax };

    const float res = float(umax.u - umin.u) / scale_fp16_ulp_to_fp32;
    if (debug) printf("# -> bigdiff_fp32 : %f %f (0x%08x 0x%08x) %f\n", fmin, fmax, umin.u, umax.u, res);
    return res;
}

// fp16 ulp difference for two values both <= fp16_min (no denormals: [0..fp16_min] treated as 1 ulp)
// required: fmin <= fmax
float smalldiff_fp32(float fmin, float fmax)
{
    const float res = (fmax - fmin) / fp16_min;
    if (debug) printf("# -> smalldiff_fp32 : %f %f %f\n", fmin, fmax, res);
    return res;
}

float absdiff_fp32(float a, float b)
{
    const float fa = std::abs(a);
    const float fb = std::abs(b);

    if (debug) printf("# fa fb = %f %f\n", fa, fb);

    const float fmin = std::min(fa, fb);
    const float fmax = std::max(fa, fb);

    if (debug) printf("# fmin fmax = %f %f\n", fmin, fmax);

    if (fmin >= fp16_min)
    {
        if (debug) printf("# -> big case\n");
        return bigdiff_fp32(fmin, fmax);
    }
    if (fmax <= fp16_min)
    {
        if (debug) printf("# -> small case\n");
        return smalldiff_fp32(fmin, fmax);
    }
    if (debug) printf("# -> mixed case\n");
    return smalldiff_fp32(fmin, fp16_min) + bigdiff_fp32(fp16_min, fmax);
}

// fp16 ulp difference
// 'b' is the approximation for exact mathematical value, so it's provided in wide format
float absdiff_fp16(fp16 a16, float b)
{
    const float a = f16Tof32(a16);

    // handle NaNs

    const bool aIsNaN = (a != a);
    const bool bIsNaN = (b != b);
    if (aIsNaN || bIsNaN)
        return (aIsNaN && bIsNaN) ? 0 : (a - b);

    // handle overflow & underflow cases

    const int a16abs = (a16 & 0x7fff);
    if ((a16abs == 0x7c00) || (a16abs == 0x0000)) // INF | 0 | FTZ
    {
        const fp16 b16 = f32Tof16(b);
        if (debug) printf("# a16 b16 = 0x%04x 0x%04x\n", (uint16_t)a16, (uint16_t)b16);
        if ((uint16_t)a16 == (uint16_t)b16)
            return 0;
    }

    // main path: both are numbers, no overflow/underflow

    const fp32_union ua = { .f = a };
    const fp32_union ub = { .f = b };

    // check whether signs are different
    if ((ua.u ^ ub.u) & 0x80000000)
        return absdiff_fp32(a, 0) + absdiff_fp32(0, b);
    else
        return absdiff_fp32(a, b);
}

} // namespace ulp

//== UnitTest local replacement ================================================

uint32_t numTestsPassed = 0;
uint32_t numTestsRan = 0;

int unitTestInit()
{
    numTestsPassed = 0;
    numTestsRan = 0;
    return 0;
}

int unitTestLogFail()
{
    ++numTestsRan;
    return 0;
}

int unitTestLogPass()
{
    ++numTestsRan;
    ++numTestsPassed;
    return 0;
}

int unitTestFinalReport()
{
    const uint32_t numFailures = numTestsRan - numTestsPassed;

    if (numTestsPassed == numTestsRan)
        printf("\nmoviUnitTest:PASSED\n");
    else
        printf("\nmoviUnitTest:FAILED : %ld failures\n", (long int)numFailures);

    return numFailures;
}

}; // namespace icv_tests
