
#ifndef DATA_ELEMENT_HPP_
#define DATA_ELEMENT_HPP_
#include <cstdint>
#include <string>

namespace mv
{
    class DataElement
    {
        bool isDouble_;
        union {
            double fp64_;
            int64_t i64_;
        } data_;

    public:
        DataElement(bool isDouble) : isDouble_(isDouble), data_{0} {}
        DataElement(bool isDouble, double val);
        DataElement(bool isDouble, int64_t val);
        DataElement(const DataElement& other);
        ~DataElement();

        DataElement& operator=(int64_t val);
        DataElement& operator=(double val);
        DataElement& operator+=(int64_t val);
        DataElement& operator+=(double val);
        DataElement& operator-=(int64_t val);
        DataElement& operator-=(double val);
        DataElement& operator*=(int64_t val);
        DataElement& operator*=(double val);
        DataElement& operator/=(int64_t val);
        DataElement& operator/=(double val);

        DataElement& operator=(DataElement src);
        bool operator==(DataElement& rhs) const;
        bool operator==(const double& rhs) const;
        bool operator==(const int& rhs) const;
        bool operator==(const unsigned int& rhs) const;
        bool operator==(const long unsigned int& rhs) const;
        bool operator==(const int64_t& rhs) const;
        bool operator==(const float& rhs) const;
        operator int64_t() const;
        operator double() const;
        operator float() const;

        operator std::string() const;
        bool isDouble() const {return isDouble_;}
    };

}
#endif // DATA_ELEMENT_HPP_
