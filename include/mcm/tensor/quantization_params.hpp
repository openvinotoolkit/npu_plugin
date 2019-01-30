#ifndef QUANTIZATION_PARAMS_HPP_
#define QUANTIZATION_PARAMS_HPP_

#include <vector>
#include "include/mcm/logger/log_sender.hpp"
#include "include/mcm/base/printable.hpp"

namespace mv
{

    class QuantizationParams: public Printable, public LogSender
    {
        int64_t zero_point_;
        float scale_;
        float min_; //TBD: min/max are they needed?
        float max_;
    public:

        QuantizationParams(int64_t zp, float scale, float min, float max);
        QuantizationParams(const QuantizationParams& other);

        inline int64_t getZeroPoint() const
        {
            return zero_point_;
        }

        inline float getScale() const
        {
            return scale_;
        }

        inline float getMin() const
        {
            return min_;
        }

        inline float getMax() const
        {
            return max_;
        }

        inline void setScale(float scale)
        {
            scale_ = scale;
        }

        inline void setZeroPoint(int64_t zp)
        {
            zero_point_ = zp;
        }

        void setMin(float min);
        void setMax(float max);

        bool operator==(const QuantizationParams& other) const;

        virtual std::string getLogID() const override;
        virtual std::string toString() const override;

    };

}

#endif // QUANTIZATION_PARAMS_HPP_
