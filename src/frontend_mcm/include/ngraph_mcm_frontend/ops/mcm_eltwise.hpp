//
// Copyright 2020 Intel Corporation.
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

// clang-format off
#ifdef ENABLE_MCM_COMPILER

#include <ngraph/op/op.hpp>
#include <memory>

class McmEltwise final : public ngraph::op::Op {
public:
    enum OperationType {
        SUM = 0
    };

    McmEltwise() = default;

    McmEltwise(
            const ngraph::Output<ngraph::Node>& input0,
            const ngraph::Output<ngraph::Node>& input1,
            const OperationType operation,
            const ngraph::element::Type& type);

    void setElemType(const ngraph::element::Type& type) {
        _type = type;
        validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    static const ngraph::NodeTypeInfo type_info;
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info; }
    OperationType getOperationType() const { return _operation; }

private:
    ngraph::element::Type _type;
    OperationType _operation;
};

#endif // ENABLE_MCM_COMPILER
// clang-format on
