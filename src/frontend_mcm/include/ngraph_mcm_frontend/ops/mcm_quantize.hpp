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

class McmQuantize final : public ngraph::op::Op {
public:
    McmQuantize() = default;

    McmQuantize(
            const ngraph::Output<ngraph::Node>& input,
            const ngraph::Output<ngraph::Node>& scale,
            const ngraph::Output<ngraph::Node>& zero_point,
            const ngraph::element::Type& type);

    void validate_and_infer_types() override;

    std::shared_ptr<ngraph::Node> copy_with_new_args(const ngraph::NodeVector& new_args) const override;

    static const ngraph::NodeTypeInfo type_info;
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info; }

private:
    ngraph::element::Type _type;
};

#endif
// clang-format on
