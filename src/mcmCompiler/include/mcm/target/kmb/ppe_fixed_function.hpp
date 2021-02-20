#ifndef MV_PPEFIXEDFUNCTION
#define MV_PPEFIXEDFUNCTION

#include <string>
#include <array>
#include <climits>
#include "include/mcm/target/kmb/ppe_layer_type.hpp"

namespace mv
{
    class PPEFixedFunction : public LogSender
    {
        private:
            int lowClamp_;
            int highClamp_;
            int32_t reluMult_;
            uint8_t reluShift_;
            std::vector<PPELayerType> layers_;
        public:
            PPEFixedFunction(int32_t lRelumult = 1, uint8_t lRelushift = 0, int lowClamp = INT_MIN, int highClamp = INT_MAX);

            int getLowClamp() const;
            int getHighClamp() const ;
            int32_t getLReluMult() const ;
            uint8_t getLReluShift() const ;

            void setLowClamp(int lowClamp);
            void setHighClamp(int highClamp);
            void setLReluMult(int32_t lRelumult);
            void setLReluShift(uint8_t lRelushift);

            const std::vector<mv::PPELayerType>& getLayers() const;
            void addLayer(PPELayerType layer);

            std::string getLogID() const;
            std::string toString() const;
    };
}

#endif
