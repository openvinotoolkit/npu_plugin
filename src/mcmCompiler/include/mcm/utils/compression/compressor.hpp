#ifndef COMPRESSOR_HPP
#define COMPRESSOR_HPP

#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"


namespace mv
{
class Compressor
{
    public:
        Compressor() {}

        virtual std::pair<std::vector<int64_t>, uint32_t> compress(std::vector<int64_t>& data, mv::Tensor& t) = 0;
        virtual std::pair<std::vector<int64_t>, uint32_t> compress(std::vector<int64_t>& data, mv::Data::TensorIterator& t) = 0;
        virtual std::vector<uint8_t> decompress(std::vector<uint8_t>& compressedData) = 0;

        virtual ~Compressor() = default;
};
}
#endif //COMPRESSOR_HPP