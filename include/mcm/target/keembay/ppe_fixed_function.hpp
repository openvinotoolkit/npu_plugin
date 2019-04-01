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
            std::vector<PpeLayerType> layers_;

        public:
            std::string toString() const;
            PPEFixedFunction(unsigned low_clamp = 0, unsigned high_clamp = 0);
            unsigned getLowClamp() const;
            unsigned getHighClamp() const ;
            const std::vector<PpeLayerType>& getLayers() const;
            void addLayer(PpeLayerType layer);
            std::string getLogID() const;
    };
}

#endif
