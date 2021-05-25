// Copyright 2020 Intel Corporation.
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

#include <vpu/utils/logger.hpp>

#include <ie_common.h>
#include <ie_compound_blob.h>
#include <blob_factory.hpp>
#include <blob_transform.hpp>
#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
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
