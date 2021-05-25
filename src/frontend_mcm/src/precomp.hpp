// Copyright 2020-2021 Intel Corporation.
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
