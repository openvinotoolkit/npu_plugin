#ifndef QUANTIZATION_PARAMS_HPP_
#define QUANTIZATION_PARAMS_HPP_

#include <vector>
#include "include/mcm/base/element.hpp"
#include "include/mcm/utils/data_generator.hpp"

namespace mv
{

    class QuantizationParams: public Element
    {
    public:
        QuantizationParams(const json::Value& content);
        QuantizationParams(std::vector<int64_t> zp, std::vector<double> scale, std::vector<double> min, std::vector<double> max);
        QuantizationParams(std::vector<int64_t> zp, std::vector<double> scale, std::vector<double> min, std::vector<double> max, std::vector <unsigned> shift, std::vector<unsigned> mult);

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

        inline std::vector<unsigned> getShift() const
        {
            return get<std::vector<unsigned>>("shift");
        }

        inline std::vector<unsigned> getMult() const
        {
            return get<std::vector<unsigned>>("mult");
        }

        void quantize(std::vector<unsigned> shift, std::vector<unsigned> mult);

        int64_t getZeroPoint(const size_t channel) const;
        virtual std::string getLogID() const override;
        virtual std::string toString() const override;
        virtual bool isEmpty() const;
    };

}

#endif // QUANTIZATION_PARAMS_HPP_
