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

        if (value.size() == size)
            return value;

        throw mv::ArgumentError("QuantizationPass", "extendToK", "parameters dimensions doesn't match size of output_channels or 1",
                    std::to_string(value.size()));
    }
    public:
        QuantizationParams(const json::Value& content);
        QuantizationParams(std::vector<int64_t> zp, std::vector<double> scale, std::vector<double> min, std::vector<double> max);
        QuantizationParams(std::vector<int64_t> zp, std::vector<double> scale, std::vector<double> min, std::vector<double> max, std::vector <uint8_t> shift, std::vector<uint16_t> mult);

        inline std::vector<int64_t> getZeroPoint() const
        {
            return get<std::vector<int64_t>>("zeroPoint");
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

        inline std::vector<uint8_t> getShift() const
        {
            return get<std::vector<uint8_t>>("shift");
        }

        inline std::vector<uint16_t> getMult() const
        {
            return get<std::vector<uint16_t>>("mult");
        }

        void quantize(std::vector<uint8_t> shift, std::vector<uint16_t> mult);
        void extendParamsToOutputChannelSize(const size_t outputChannelSize);
        void extendParamsPartialToOutputChannelSize(const size_t outputChannelSize);

        int64_t getZeroPoint(const size_t channel) const;
        virtual std::string getLogID() const override;
        virtual std::string toString() const override;
        virtual bool isEmpty() const;
    };

}

#endif // QUANTIZATION_PARAMS_HPP_
