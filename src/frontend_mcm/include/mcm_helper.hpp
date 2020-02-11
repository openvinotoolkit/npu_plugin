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

#include <list>
#include <memory>
#include <queue>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vpu/utils/attributes_map.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/extra.hpp>
#include <vpu/utils/handle.hpp>
#include <vpu/utils/intrusive_handle_list.hpp>
#include <vpu/utils/range.hpp>
#include <vpu/utils/small_vector.hpp>

namespace vpu {

#define MCM_DEFINE_MODEL_TYPES(type, postfix)                                        \
    using type = Handle<VPU_COMBINE(type, postfix)>;                                 \
                                                                                     \
    using VPU_COMBINE(type, Vector) = SmallVector<type>;                             \
                                                                                     \
    using VPU_COMBINE(type, List) = IntrusiveHandleList<VPU_COMBINE(type, postfix)>; \
                                                                                     \
    using VPU_COMBINE(type, Set) = std::unordered_set<type, HandleHash>;             \
                                                                                     \
    template <typename Val>                                                          \
    using VPU_COMBINE(type, Map) = std::unordered_map<type, Val, HandleHash>;        \
                                                                                     \
    using VPU_COMBINE(type, Ptr) = std::shared_ptr<VPU_COMBINE(type, postfix)>;      \
                                                                                     \
    using VPU_COMBINE(type, PtrList) = std::list<VPU_COMBINE(type, Ptr)>;

#define MCM_MODEL_ATTRIBUTE(type, name, defVal) \
protected:                                      \
    type VPU_COMBINE(_, name) = defVal;         \
                                                \
public:                                         \
    inline const type& name() const { return VPU_COMBINE(_, name); }

}  // namespace vpu
