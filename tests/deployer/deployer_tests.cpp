#include "gtest/gtest.h"
#include "include/mcm/utils/deployer/configuration.hpp"
#include "include/mcm/utils/deployer/executor.hpp"
#include "include/mcm/utils/deployer/deployer_utils.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"

#include <chrono>
#include <thread>
using namespace mv;
using namespace exe;
TEST(basic_test, goldAllZero)
{
    //Logger::setVerboseLevel(VerboseLevel::Info);
    //Create Configuration
    //std::string graphFile = utils::projectRootPath() + std::string("/tests/data/googlenet_graph.blob");
    std::string graphFile = utils::projectRootPath() + std::string("/tests/data/gold_11.blob");
    Configuration config(graphFile);
    std::cout << "Configuration graph file " << config.getGraphFilePath() << std::endl;
    Executor exec;
    Order order("NHWC");
    Shape shape({64, 64 ,3 ,1});
    //Shape shape({224, 224 ,3 ,1});

    Tensor inputTensor = mv::exe::dep_utils::getInputData(InputMode::ALL_ZERO, order, shape);
    Tensor res = exec.execute(config, inputTensor);

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
    //std::string graphFile = utils::projectRootPath() + std::string("/tests/data/googlenet_graph.blob");
    std::string graphFile = utils::projectRootPath() + std::string("/tests/data/gold_01.blob");
    Configuration config(graphFile);

    Shape shape({32, 32, 1, 1});
    //Shape shape({224, 224 ,3 ,1});

    Order order("NHWC");
    Executor exec;
    Tensor inputTensor = mv::exe::dep_utils::getInputData(InputMode::ALL_ONE, order, shape);
    Tensor res = exec.execute(config, inputTensor);

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
    //std::string graphFile = utils::projectRootPath() + std::string("/tests/data/googlenet_graph.blob");
    std::string graphFile = utils::projectRootPath() + std::string("/tests/data/gold_01.blob");
    Configuration config(graphFile);
    //Shape shape({224, 224 ,3 ,1});
    Shape shape({32, 32, 1, 1});

    Order order("NHWC");
    Executor exec;
    std::string inputFile = utils::projectRootPath() + std::string("/tests/data/nps_guitar.bin");
    Tensor inputTensor = mv::exe::dep_utils::getInputData(inputFile, order, shape);
    Tensor res = exec.execute(config, inputTensor);

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