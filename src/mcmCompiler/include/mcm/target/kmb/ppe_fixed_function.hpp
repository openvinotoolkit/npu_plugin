#ifndef MV_PPEFIXEDFUNCTION
#define MV_PPEFIXEDFUNCTION

#include <string>
#include <array>
#include <climits>
#include <limits>
#include "include/mcm/target/kmb/ppe_layer_type.hpp"

namespace mv
{
    /**
     * @brief class of storing values for PPE fixed function mapping
     */
    class PPEFixedFunction : public LogSender
    {
        private:
            int lowClamp_;
            int highClamp_;
            /**
             * Note: the size of reluMult_ is always >= 1
             */
            std::vector<int32_t> reluMult_;
            uint8_t reluShift_;
            std::vector<PPELayerType> layers_;
        public:
            /**
             * @brief Construct a PPEFixedFunction with multiple multipliers
             * @param lRelumult a single multiplier (e.g., LeakyReLU)
             * @param lRelushift a common shift value for the multipliers
             * @param lowClamp specifies the lower boundary of clamp
             * @param hightClamp specifies the upper boundary of clamp
             */
            PPEFixedFunction(int32_t lRelumult = 1, uint8_t lRelushift = 0, int lowClamp = std::numeric_limits<int>::min(), int highClamp = std::numeric_limits<int>::max());
            
            /**
             * @brief Construct a PPEFixedFunction with multiple multipliers
             * @param lRelumult a vector of multipliers (e.g., PReLU)
             * @param lRelushift a common shift value for the multipliers
             * @param lowClamp specifies the lower boundary of clamp
             * @param hightClamp specifies the upper boundary of clamp
             */
            PPEFixedFunction(const std::vector<int32_t>& lRelumult, uint8_t lRelushift = 0, int lowClamp = std::numeric_limits<int>::min(), int highClamp = std::numeric_limits<int>::max() );

            int getLowClamp() const;
            int getHighClamp() const ;

            /**
             *@brief Returns a multiplier.
             *@note The function is to keep the compatibility with mcm emulator
             *@note LReluMult is an deprecated attribute as it is passed throught weight table
             */
            int32_t getLReluMult() const;
            
            /**
             *@brief Returns a vector of mulitpliers
             */
            const std::vector<int32_t>& getLReluMults() const;

            /**
             * @brief Returns the common shift for PPE
             */
            uint8_t getLReluShift() const ;

            void setLowClamp(int lowClamp);
            void setHighClamp(int highClamp);

            /**
             * @brief set one multipliers (e.g., LeakyReLU)
             */
            void setLReluMult(int32_t lRelumult);

            /**
             * @brief set a vector of multipliers (e.g., PReLU)
             * Note: the size of reluMult_ is always >= 1
             */
            void setLReluMults(const std::vector<int32_t>& lRelumult);

            /**
             * @brief set the common shift value
             */
            void setLReluShift(uint8_t lRelushift);

            const std::vector<mv::PPELayerType>& getLayers() const;
            void addLayer(PPELayerType layer);

            std::string getLogID() const;
            std::string toString() const;
    };
}

#endif
