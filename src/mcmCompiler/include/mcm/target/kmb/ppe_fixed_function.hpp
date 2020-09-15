#ifndef MV_PPEFIXEDFUNCTION
#define MV_PPEFIXEDFUNCTION

#include <string>
#include <array>
#include "include/mcm/target/kmb/ppe_layer_type.hpp"

namespace mv
{
    class PPEFixedFunction : public LogSender
    {
        private:
            int lowClamp_;
            int highClamp_;
            int8_t reluMult_;
            uint8_t reluShift_;
            std::vector<PPELayerType> layers_;
        public:
            PPEFixedFunction(int8_t lRelumult = 1, uint8_t lRelushift = 0, int lowClamp = -2147483648, int highClamp = 2147483647);

            int getLowClamp() const;
            int getHighClamp() const ;
            int8_t getLReluMult() const ;
            uint8_t getLReluShift() const ;

            void setLowClamp(int lowClamp);
            void setHighClamp(int highClamp);
            void setLReluMult(int8_t lRelumult);
            void setLReluShift(uint8_t lRelushift);

            const std::vector<mv::PPELayerType>& getLayers() const;
            void addLayer(PPELayerType layer);

            std::string getLogID() const;
            std::string toString() const;
    };
}

#endif
