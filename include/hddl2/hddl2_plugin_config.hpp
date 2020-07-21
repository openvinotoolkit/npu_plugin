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


/**
 * @brief A header that defines advanced related properties for HDDL2 plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file hddl2_plugin_config.hpp
 */

#pragma once

#include <vpu/vpu_plugin_config.hpp>

//
// HDDL2 plugin options
//

#define VPU_HDDL2_CONFIG_KEY(name) InferenceEngine::HDDL2ConfigParams::_CONFIG_KEY(VPU_HDDL2_##name)
#define VPU_HDDL2_CONFIG_VALUE(name) InferenceEngine::HDDL2ConfigParams::VPU_HDDL2_##name

#define DECLARE_VPU_HDDL2_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPU_HDDL2_##name)
#define DECLARE_VPU_HDDL2_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPU_HDDL2_##name)

namespace InferenceEngine {
namespace HDDL2ConfigParams {
//
// HDDL2 plugin options
//

/**
* @brief [Only for hddl2Plugin]
* Type: Arbitrary string.
* This option allows to specify color format.
*/
    DECLARE_VPU_HDDL2_CONFIG_KEY(GRAPH_COLOR_FORMAT);

    DECLARE_VPU_HDDL2_CONFIG_VALUE(BGR);
    DECLARE_VPU_HDDL2_CONFIG_VALUE(RGB);

}  // namespace HDDL2ConfigParams
}  // namespace InferenceEngine


