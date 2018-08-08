#ifndef MV_JSON_DETAIL_VALUE_CONTENT_HPP_
#define MV_JSON_DETAIL_VALUE_CONTENT_HPP_

#include "include/mcm/base/exception/value_error.hpp"

namespace mv
{

    namespace json
    {

        class Object;
        class Array;
    
        namespace detail
        {

            class ValueContent
            {

            public:

                virtual ~ValueContent() = 0;
                virtual std::string stringify() const = 0;
                virtual std::string stringifyPretty() const = 0;
                virtual explicit operator float&();
                virtual explicit operator long long&();
                virtual explicit operator std::string&();
                virtual explicit operator bool&();
                virtual explicit operator Object&();
                virtual explicit operator Array&();

            };  

        }

    }

}

#endif // MV_JSON_DETAIL_VALUE_CONTENT_HPP_