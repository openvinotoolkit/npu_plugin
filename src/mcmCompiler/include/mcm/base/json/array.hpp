#ifndef MV_JSON_ARRAY_HPP_
#define MV_JSON_ARRAY_HPP_

#include <vector>
#include <memory>
#include "include/mcm/base/json/value.hpp"
#include "include/mcm/base/exception/index_error.hpp"

namespace mv
{

    namespace json
    {

        class Array : public detail::ValueContent
        {

            std::vector<Value> elements_;

        public:

            Array();
            Array(const Array& other);
            Array(std::initializer_list<Value> l);
            Array(std::size_t s);
            void append(const Value& value);
            void erase(std::size_t idx);
            unsigned size() const;
            void clear();
            Value& operator[](std::size_t idx);
            const Value& operator[](std::size_t idx) const;
            Value& last();
            std::string stringify() const override;
            std::string stringifyPretty() const override;
            Array& operator=(const Array& other);
            bool operator==(const Array& other) const;
            bool operator!=(const Array& other) const;

            virtual std::string getLogID() const override;

        };  

    }

}

#endif // MV_JSON_ARRAY_HPP_