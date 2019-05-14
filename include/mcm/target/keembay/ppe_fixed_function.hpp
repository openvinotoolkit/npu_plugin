#ifndef MV_PPEFIXEDFUNCTION
#define MV_PPEFIXEDFUNCTION

#include <string>
#include <array>
#include "include/mcm/target/keembay/ppe_layer_type.hpp"

namespace mv
{
    class PPEFixedFunction : public LogSender
    {
        private:
            int lowClamp_;
            int highClamp_;
            std::vector<PPELayerType> layers_;
        public:
            PPEFixedFunction(int low_clamp = 0, int high_clamp = 0);

            int getLowClamp() const;
            int getHighClamp() const ;
            const std::vector<mv::PPELayerType>& getLayers() const;
            void addLayer(PPELayerType layer);

            std::string getLogID() const;
            std::string toString() const;
    };
}

#endif
