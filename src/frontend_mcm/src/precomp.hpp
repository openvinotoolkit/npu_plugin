// Copyright 2020-2021 Intel Corporation.
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

#include <schema/graphfile/graphfile_generated.h>

#include <blob_factory.hpp>
#include <blob_transform.hpp>
#include <caseless.hpp>
#include <description_buffer.hpp>
#include <legacy/graph_tools.hpp>
#include <ie_blob.h>
#include <ie_common.h>
#include <ie_compound_blob.h>
#include <ie_data.h>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>
#include <ie_itt.hpp>
#include <ie_layouts.h>
#include <ie_parallel.hpp>
#include <ie_precision.hpp>
#include <ie_utils.hpp>
#include <openvino/itt.hpp>
#include <parse_layers_helpers.hpp>
#include <precision_utils.h>
#include <quantization_helpers.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/ops.hpp>

#include <flatbuffers/flatbuffers.h>

#include <algorithm>
#include <functional>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
#include <unordered_map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cassert>
#include <cctype>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <cstring>
