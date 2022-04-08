//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// clang-format off

#include <ie_precision.hpp>
#include <ngraph/type/element_type.hpp>

namespace ie = InferenceEngine;

ngraph::element::Type cvtPrecisionToElemType(const ie::Precision& precision);

// clang-format on
