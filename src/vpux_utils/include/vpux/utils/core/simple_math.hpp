//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

//
// Simple integer arithmetics to be used for the work sizes calculation.
// Supported operations : +,-,*,/,%,(,)
// no unary -,+
// Variables defined as single chars and should not include one of the ops, whitespaces or 0-9
//

#pragma once

#include "vpux/utils/core/hash.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace vpux {

//
// IntOrFloat
//

class IntOrFloat final {
public:
    explicit IntOrFloat(int x);
    explicit IntOrFloat(float x);
    explicit IntOrFloat(StringRef str);

public:
    bool isInt() const {
        return _isInt;
    }

    int asInt() const {
        return _isInt ? _value.i : static_cast<int>(_value.f);
    }

    float asFloat() const {
        return _isInt ? static_cast<float>(_value.i) : _value.f;
    }

private:
    union {
        int i;
        float f;
    } _value = {};

    bool _isInt = true;
};

IntOrFloat operator+(const IntOrFloat& a, const IntOrFloat& b);
IntOrFloat operator-(const IntOrFloat& a, const IntOrFloat& b);
IntOrFloat operator*(const IntOrFloat& a, const IntOrFloat& b);
IntOrFloat operator/(const IntOrFloat& a, const IntOrFloat& b);
IntOrFloat operator%(const IntOrFloat& a, const IntOrFloat& b);

//
// MathExpression
//

class MathExpression final {
public:
    using VarMap = std::map<std::string, std::string>;

public:
    void setVariables(const VarMap& variables);
    void parse(StringRef expression);
    IntOrFloat evaluate() const;

private:
    enum class TokenType { Value, Operator, Function };

    struct Token final {
        TokenType type;
        IntOrFloat value;
        std::string opName;

        Token(TokenType type, IntOrFloat value, StringRef opName): type(type), value(value), opName(opName.str()) {
        }
    };

private:
    std::unordered_map<std::string, IntOrFloat> _vars;
    std::vector<Token> _tokens;
};

}  // namespace vpux
