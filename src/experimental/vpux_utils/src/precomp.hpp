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

#include <cpp/ie_cnn_network.h>
#include <ie_blob.h>
#include <ie_common.h>
#include <ie_data.h>
#include <ie_layouts.h>
#include <precision_utils.h>
#include <blob_factory.hpp>
#include <blob_transform.hpp>
#include <cpp_interfaces/exception2status.hpp>
#include <details/ie_exception.hpp>
#include <ie_core.hpp>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>
#include <ie_metric_helpers.hpp>
#include <ie_parallel.hpp>
#include <ie_parameter.hpp>
#include <ie_plugin_config.hpp>
#include <ie_precision.hpp>

#include <ngraph/function.hpp>
#include <ngraph/node_output.hpp>
#include <ngraph/ops.hpp>
#include <ngraph/partial_shape.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/bfloat16.hpp>
#include <ngraph/type/element_type.hpp>
#include <ngraph/type/float16.hpp>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/None.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/TypeName.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
