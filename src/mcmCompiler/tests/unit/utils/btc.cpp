#include "mcm/tensor/shape.hpp"
#include "mcm/tensor/order/order.hpp"
#include "mcm/tensor/dtype/dtype.hpp"
#include "utils/compression/btc.hpp"
#include <stdlib.h>
#include "gtest/gtest.h"

using namespace testing;

TEST(btc, self_test)
{
    int randomSize;
    int randomNumber;
    std::vector<uint8_t> inputData;

    std::size_t numCompressionModules = 1;
    std::size_t bufferAlignment = 1;
    std::size_t bitmapPreprocEnable = 0;
    bool bypassMode = false;
    bool pStatsOnly = false;
    uint32_t verbosity = 0;

    std::unique_ptr<mv::BTC> codec_(new mv::BTC(bufferAlignment, bitmapPreprocEnable, pStatsOnly, bypassMode, verbosity));

    // initialize random seed:
    srand (time(NULL));

    // generate random number between 1000 and 10000 for array size:
    randomSize = rand() % 10000 + 1000;

    for(unsigned i = 0; i < randomSize; ++i)
    {
        randomNumber = rand() % 255 + 0; /* generate random uint8 number*/
        inputData.push_back(randomNumber);
    }

    // placeholder tensor
    mv::Shape tShape({1, randomSize});
    mv::Tensor t("t", tShape, mv::DType("UINT8"), mv::Order("HW"));

    // compress data
    auto compressedData = codec_->compress(inputData, t);

    // downcast from uint64_t to uint8_t
    std::vector<uint64_t> compressedIn = compressedData.first;
    std::vector<uint8_t> compressedOut;

    for (size_t i = 0; i < compressedIn.size(); i++)
    {
        compressedOut[i] = compressedIn[i];
    }

    // decompress data
    std::vector<uint8_t> outputData = codec_->decompress(compressedOut);

    // test that decompressed data equals original data
    for(unsigned i = 0; i < randomSize; ++i)
        ASSERT_EQ(inputData[i], outputData[i]);
}
