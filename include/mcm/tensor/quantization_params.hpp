#ifndef QUANTIZATION_PARAMS_HPP_
#define QUANTIZATION_PARAMS_HPP_

#include <vector>
#include "include/mcm/base/element.hpp"

namespace mv
{

    class QuantizationParams: public Element
    {
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

        virtual std::string getLogID() const override;
        virtual std::string toString() const override;
    };

}

#endif // QUANTIZATION_PARAMS_HPP_
