#ifndef MV_JSON_ARRAY_HPP_
#define MV_JSON_ARRAY_HPP_

#include <vector>
#include <memory>
#include "include/mcm/base/json/value.hpp"
#include "include/mcm/base/json/exception/index_error.hpp"

namespace mv
{

    namespace json
    {

        class Array : public Value
        {

            friend class Value;
            std::vector<std::unique_ptr<Value>> elements_;

            void deepCopyMembers_(const std::vector<std::unique_ptr<Value>>& input);

        public:

            Array();
            Array(const Array& other);
            Array(Object& owner, const std::string& key);
            void erase(unsigned idx);
            unsigned size() const;
            Value& operator[](unsigned idx);
            std::string stringify() const override;
            Array& operator=(const Array& other);

        };  

    }

}

#endif // MV_JSON_ARRAY_HPP_