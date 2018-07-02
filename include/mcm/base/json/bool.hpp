#ifndef MV_JSON_BOOL_HPP_
#define MV_JSON_BOOL_HPP_

#include "include/mcm/base/json/value.hpp"

namespace mv
{

    namespace json
    {

        class Bool : public Value
        {

            bool value_;

        public:

            Bool(Object& owner, const std::string& key, bool value);
            explicit operator bool&() override;
            std::string stringify() const override;

        }; 
        
    }

}

#endif // MV_JSON_BOOL_HPP_