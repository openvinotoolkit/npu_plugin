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

#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/logger.hpp>

#include <ie_common.h>
#include <ie_compound_blob.h>
#include <blob_factory.hpp>
#include <blob_transform.hpp>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <details/ie_exception.hpp>
#include <ie_blob.h>
#include <ie_common.h>
#include <ie_data.h>
#include <ie_layouts.h>
#include <ie_parallel.hpp>
#include <ie_utils.hpp>
#include <openvino/itt.hpp>
#include <precision_utils.h>

#include <algorithm>
#include <functional>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <ostream>
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
#include <cstdlib>
#include <cstring>
