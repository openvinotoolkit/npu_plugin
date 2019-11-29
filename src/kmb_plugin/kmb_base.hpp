// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

namespace KmbPlugin {

//
// KMB_DEFINE_MODEL_TYPES
//

#define KMB_DEFINE_MODEL_TYPES(type, postfix)                                        \
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

//
// KMB_MODEL_ATTRIBUTE
//

#define KMB_MODEL_ATTRIBUTE(type, name, defVal) \
protected:                                      \
    type VPU_COMBINE(_, name) = defVal;         \
                                                \
public:                                         \
    inline const type& name() const { return VPU_COMBINE(_, name); }

#define KMB_MODEL_ATTRIBUTE_PTR_RANGE(type, name) \
protected:                                        \
    type VPU_COMBINE(_, name);                    \
                                                  \
public:                                           \
    inline auto name() const->decltype(contRange(VPU_COMBINE(_, name))) { return contRange(VPU_COMBINE(_, name)); }

}  // namespace KmbPlugin
}  // namespace vpu
