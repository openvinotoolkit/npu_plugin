//
// Copyright 2016-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include <set>
#include <map>
#include <vector>
#include <functional>
#include <string>
#include <utility>

#include <vpu/utils/containers.hpp>

//
// Simple integer arithmetics to be used for the work sizes calculation.
// Supported operations : +,-,*,/,%,(,)
// no unary -,+
// Variables defined as single chars and should not include one of the ops, whitespaces or 0-9
//

namespace vpu {

class SimpleMathExpression final {
public:
    void setVariables(const std::map<char, int>& vars) { _vars = vars; }

    void parse(const std::string& expression);

    int evaluate() const;

private:
    struct Token final {
        enum TokenType {
            Value,
            Operator,
        };

        TokenType type;
        int value;
        char op;

        explicit Token(TokenType t = Value, int v = 0, char o = 0) : type(t), value(v), op(o) {}
    };

private:
    std::map<char, int> _vars;
    SmallVector<Token> _parsedTokens;
};

}  // namespace vpu
