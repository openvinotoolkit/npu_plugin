#ifndef BTC_HPP
#define BTC_HPP

#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/compression/compressor.hpp"
#include "include/bitcompactor/bitCompactor.h"


namespace mv
{
class BTC : public Compressor
{
    public:
        struct BTC_Config
        {
                uint32_t bitsPerSym;
                uint32_t maxEncSyms;
                uint32_t verbosity;
                uint32_t blockSizeBytes;
                uint32_t superBlockSizeBytes;
                uint32_t minFixedBitLn;

                bool     compressFlag;
                bool     statsOnly;
                bool     bypass;
                bool     dualEncodeEn;
                bool     procBinningEn;
                bool     procBitmapEn;
                bool     mixedBlockSizeEn;
                bool     ratioEn;

                int      alignMode;
        };

        BTC(uint32_t align, uint32_t bitmapPreprocEnable, bool pStatsOnly, bool bypassMode, uint32_t verbosity);
        std::unique_ptr<BitCompactor> codec_ = nullptr;

        std::pair<std::vector<int64_t>, uint32_t> compress(std::vector<int64_t>& data, mv::Tensor& t);
        std::pair<std::vector<int64_t>, uint32_t> compress(std::vector<int64_t>& data, mv::Data::TensorIterator& t);
        std::vector<uint8_t> decompress(std::vector<uint8_t>& compressedData);

    private:
        void setDefaultConfig();
        void updateConfig(uint32_t align, uint32_t bitmapPreprocEnable, bool pStatsOnly, bool bypassMode, uint32_t verbosity);
};
}
#endif //BTC_HPP
