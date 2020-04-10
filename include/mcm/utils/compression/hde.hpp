#ifndef HDE_HPP_
#define HDE_HPP_

#include "include/mcm/tensor/tensor.hpp"
#include "contrib/huffman_encoding/include/Huffman.hpp"
#include "contrib/huffman_encoding/include/huffmanCodec.hpp"


namespace mv
{
class Hde
{
    public:
        Hde(uint32_t bitPerSymbol, uint32_t maxNumberEncodedSymbols, uint32_t verbosity, uint32_t blockSize, bool pStatsOnly, uint32_t bypassMode);
        std::unique_ptr<huffmanCodec> codec_ = nullptr;

        std::vector<int64_t> hdeCompress(std::vector<int64_t>& data, mv::Tensor& t);
        std::vector<uint8_t> hdeDecompress(std::vector<uint8_t>& compressedData);
        
};   
} 
#endif 