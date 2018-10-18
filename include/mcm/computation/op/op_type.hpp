#ifndef MV_OP_OP_TYPE_HPP_
#define MV_OP_OP_TYPE_HPP_

#include <string>
#include <unordered_map>
#include <functional>
#include "include/mcm/base/json/value.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/logger/log_sender.hpp"

namespace mv
{

    class OpType : public LogSender
    {

    public:

        static constexpr unsigned short Input = 0;
        static constexpr unsigned short Output = 1;
        static constexpr unsigned short Constant = 2;
        static constexpr unsigned short Conv2D = 3;
        static constexpr unsigned short Conversion = 4;
        static constexpr unsigned short MatMul = 5;
        static constexpr unsigned short MaxPool2D = 6;
        static constexpr unsigned short AvgPool2D = 7;
        static constexpr unsigned short Concat = 8;
        static constexpr unsigned short ReLU = 9;
        static constexpr unsigned short Softmax = 10;
        static constexpr unsigned short Scale = 11;
        static constexpr unsigned short BatchNorm = 12;
        static constexpr unsigned short Add = 13;
        static constexpr unsigned short Subtract = 14;
        static constexpr unsigned short Multiply = 15;
        static constexpr unsigned short Divide = 16;
        static constexpr unsigned short Reshape = 17;
        static constexpr unsigned short Bias = 18;
        static constexpr unsigned short FullyConnected = 19;
        static constexpr unsigned short PReLU = 20;
        static constexpr unsigned short DropOut = 21;
        static constexpr unsigned short DepthwiseConv2D = 22;

    private:

        static const std::unordered_map<unsigned short, std::string> opTypeStrings_;

        int opType_;

    public:

        OpType();
        OpType(unsigned short value);
        OpType(const std::string& value);

        std::string toString() const;

        bool operator==(const OpType &other) const;
        bool operator!=(const OpType &other) const;
        bool operator==(unsigned short value) const;
        bool operator!=(unsigned short value) const;
        bool operator<(const OpType &other) const;
        explicit operator unsigned short() const;

        std::string getLogID() const override;

    };

}

#endif // MV_OP_OP_TYPE_HPP_
