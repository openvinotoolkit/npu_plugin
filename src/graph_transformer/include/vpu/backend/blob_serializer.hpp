//
// Copyright 2018-2019 Intel Corporation.
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

#include <vector>

#include <vpu/model/base.hpp>

namespace vpu {

class BlobSerializer final {
public:
    template <typename T>
    int append(const T& val) {
        auto curPos = _data.size();

        _data.insert(
            _data.end(),
            reinterpret_cast<const char*>(&val),
            reinterpret_cast<const char*>(&val) + sizeof(val));

        return checked_cast<int>(curPos);
    }

    template <typename T>
    void overWrite(int pos, const T& val) {
        auto uPos = checked_cast<size_t>(pos);
        std::copy_n(reinterpret_cast<const char*>(&val), sizeof(val), _data.data() + uPos);
    }

    // Overwrites `uint32_t` value in `_data` at the position `pos`
    // to the size of the tail from `pos` to the end of `_data`.
    void overWriteTailSize(int pos) {
        auto uPos = checked_cast<size_t>(pos);
        IE_ASSERT(uPos < _data.size());
        auto size = checked_cast<uint32_t>(_data.size() - uPos);
        std::copy_n(reinterpret_cast<const char*>(&size), sizeof(uint32_t), _data.data() + uPos);
    }

    int size() const { return checked_cast<int>(_data.size()); }

    const char* data() const { return _data.data(); }

private:
    std::vector<char> _data;
};

}  // namespace vpu
