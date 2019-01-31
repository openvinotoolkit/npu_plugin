#ifndef QUANTIZATION_PARAMS_HPP_
#define QUANTIZATION_PARAMS_HPP_

#include <vector>
#include "include/mcm/base/element.hpp"

namespace mv
{

    class QuantizationParams: public Element
    {
    public:

        QuantizationParams(std::vector<size_t> zp, std::vector<double> scale, std::vector<double> min, std::vector<double> max);

        inline std::vector<int64_t> getZeroPoint() const
        {
            return get<std::vector<int64_t>>("zero_point");
        }

        inline std::vector<float> getScale() const
        {
            return get<std::vector<float>>("scale");
        }

        inline std::vector<float> getMin() const
        {
            return get<std::vector<float>>("min");
        }

        inline std::vector<float> getMax() const
        {
            return get<std::vector<float>>("max");
        }

        virtual std::string getLogID() const override;
        virtual std::string toString() const override;
    };

}

#endif // QUANTIZATION_PARAMS_HPP_
