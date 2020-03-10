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
        QuantizationParams(const std::vector<int64_t>& zp, const std::vector<double>& scale, const std::vector<double>& min, const std::vector<double>& max);
        QuantizationParams(const std::vector<int64_t>& zp, const std::vector<double>& scale, const std::vector<double>& min, const std::vector<double>& max, const std::vector <unsigned>& shift, const std::vector<unsigned>& mult);
//        QuantizationParams & operator=(const QuantizationParams& quantObject);

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
        void setScale(std::vector<double> scale_);

        int64_t getZeroPoint(const size_t channel) const;
        virtual std::string getLogID() const override;
        virtual std::string toString() const override;
        virtual bool isEmpty() const;
        virtual bool isNeutral() const;
        virtual bool infinitelimits() const;
    };

}

#endif // QUANTIZATION_PARAMS_HPP_
