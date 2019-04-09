#ifndef QUANTIZATION_PARAMS_HPP_
#define QUANTIZATION_PARAMS_HPP_

#include <vector>
#include "include/mcm/base/element.hpp"
#include "include/mcm/utils/data_generator.hpp"

namespace mv
{

    class QuantizationParams: public Element
    {
    private:

    template <class T>
    std::vector<T> extendToK_(size_t size, std::vector<T> value)
    {
        if (value.size() == 1)
            return mv::utils::generateSequence<T>(size, static_cast<T>(value[0]) , 0);

        //for Operations with no quant Params
        if (value.size() == 0)
            return mv::utils::generateSequence<T>(size, static_cast<T>(1) , 0);

        if (value.size() == size)
            return value;

        throw mv::ArgumentError("QuantizationPass", "extendToK", "parameters dimensions doesn't match size of output_channels or 1",
                    std::to_string(value.size()));
    }
    public:
        QuantizationParams(const json::Value& content);
        QuantizationParams(std::vector<unsigned> zp, std::vector<double> scale, std::vector<double> min, std::vector<double> max);

        inline std::vector<unsigned> getZeroPoint() const
        {
            return get<std::vector<unsigned>>("zeroPoint");
        }

        inline std::vector<double> getScale() const
        {
            return get<std::vector<double>>("scale");
        }

        inline std::vector<double> getMin() const
        {
            return get<std::vector<double>>("min");
        }

        inline std::vector<double> getMax() const
        {
            return get<std::vector<double>>("max");
        }

        void extendParamsToOutputChannelSize(const size_t outputChannelSize);

        unsigned getZeroPoint(const size_t channel) const;
        virtual std::string getLogID() const override;
        virtual std::string toString() const override;
    };

}

#endif // QUANTIZATION_PARAMS_HPP_
