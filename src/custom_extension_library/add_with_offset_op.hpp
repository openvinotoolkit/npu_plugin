// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <ie_iextension.h>
#include <ie_api.h>
#include <ngraph/ngraph.hpp>

#if defined(_WIN32) && defined(IMPLEMENT_INFERENCE_EXTENSION_API)
#define INFERENCE_EXTENSION_API_CLASS(...) __declspec(dllexport) __VA_ARGS__
#else
#define INFERENCE_EXTENSION_API_CLASS(TYPE) INFERENCE_ENGINE_API_CLASS(TYPE)
#endif

namespace SampleExtension {

class INFERENCE_EXTENSION_API_CLASS(AddWOffsetOp) : public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"add_with_offset", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info; }
    AddWOffsetOp() = default;
    AddWOffsetOp(const ngraph::Output<ngraph::Node>& inp1, const ngraph::Output<ngraph::Node>& inp2, float offset);
    void validate_and_infer_types() override;
    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;
    void setOffsetAttr(float value) {offset = value;}
    float getOffsetAttr() {return offset;}
private:
    float offset;
};

}


