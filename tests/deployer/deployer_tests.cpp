#include "gtest/gtest.h"
#include "include/mcm/utils/deployer/configuration.hpp"
#include "include/mcm/utils/deployer/executor.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"

#include <chrono>
#include <thread>
using namespace mv;
using namespace exe;
TEST(basic_test, goldAllZero)
{
    //Logger::setVerboseLevel(VerboseLevel::Info);
    //Create Configuration
    std::string graphFile = utils::projectRootPath() + std::string("/tests/data/gold_11.blob");
    Configuration config(graphFile);
    std::cout << "Configuration graph file " << config.getGraphFilePath() << std::endl;
    Executor exec(config);
    Tensor res = exec.execute();

    unsigned short max = 0;
    unsigned int max_idx = 0;
    for (unsigned int i=0; i < res.getShape().totalSize(); i++)
    {
        if (res(i) > max)
        {
            max = res(i);
            max_idx = i;
        }
        //if (res(i) != 0)
        //    std::cout << "res[" << i << "] = " << res(i) << std::endl;

    }
    std::cout << "res max idx " << max_idx << " val " << (float) max << std::endl;
    //mv_num_convert cvtr;

    //EXPECT_EQ(max, cvtr.fp32_to_fp16(0.27124));
    //EXPECT_EQ(max_idx, 885);
}

TEST(basic_test, gold01AllOnes)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(2));

    //Logger::setVerboseLevel(VerboseLevel::Info);
    //Create Configuration
    std::string graphFile = utils::projectRootPath() + std::string("/tests/data/gold_01.blob");
    Configuration config(graphFile);
    config.setInputMode(InputMode::ALL_ONE);
    std::cout << "Configuration graph file " << config.getGraphFilePath() << std::endl;
    Executor exec(config);
    Tensor res = exec.execute();

    unsigned short max = 0;
    unsigned int max_idx = 0;
    for (unsigned int i=0; i < res.getShape().totalSize(); i++)
    {
        if (res(i) > max)
        {
            max = res(i);
            max_idx = i;
        }
        //if (res(i) != 0)
        //    std::cout << "res[" << i << "] = " << res(i) << std::endl;

    }
    std::cout << "res max idx " << max_idx << " val " << (float) max << std::endl;
    //mv_num_convert cvtr;

    //EXPECT_EQ(max, cvtr.fp32_to_fp16(0.24194));
    //EXPECT_EQ(max_idx, 885);
}

TEST(basic_test, goldFromFile)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    //Logger::setVerboseLevel(VerboseLevel::Info);
    //Create Configuration
    std::string graphFile = utils::projectRootPath() + std::string("/tests/data/gold_01.blob");

    Configuration config(graphFile);
    config.setInputMode(InputMode::FILE);
    config.setInputFilePath(utils::projectRootPath() + std::string("/tests/data/nps_guitar.bin"));
    std::cout << "Configuration graph file " << config.getGraphFilePath() << std::endl;
    Executor exec(config);
    Tensor res = exec.execute();
    unsigned short max = 0;
    unsigned int max_idx = 0;
    for (unsigned int i=0; i < res.getShape().totalSize(); i++)
    {
        if (res(i) > max)
        {
            max = res(i);
            max_idx = i;
        }
    }
    //mv_num_convert cvtr;

    //std::cout << "expected max prob " << cvtr.fp32_to_fp16(0.99609) << std::endl;
    std::cout << "res max idx " << max_idx << " val " << max << std::endl;
    //EXPECT_EQ(max, cvtr.fp32_to_fp16(0.99609));
    //EXPECT_EQ(max_idx, 546);
}