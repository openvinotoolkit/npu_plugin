
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/tensor/order/order.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"
#include "include/mcm/utils/compression/btc.hpp"

#include "gtest/gtest.h"

#include <stdlib.h>
#include <cstdio>


TEST(BitCompactor, self_test)
{
    size_t randomSize;
    uint8_t randomNumber;
    std::vector<uint8_t> inputData;

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

    for(size_t i = 0; i < randomSize; ++i)
    {
        randomNumber = static_cast<int64_t>(rand() % 256); /* generate random uint8 number*/
        inputData.push_back(randomNumber);
    }

    // compress data
    auto compressedData = codec_->compress(inputData);

    // decompress data
    std::vector<uint8_t> outputData = codec_->decompress(compressedData);

    // test that decompressed data equals original data
    for(unsigned i = 0; i < outputData.size(); ++i)
        ASSERT_EQ(inputData[i], outputData[i]);

}
