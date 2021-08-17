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
