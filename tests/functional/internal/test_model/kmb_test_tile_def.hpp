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

#include "kmb_test_model.hpp"
#include "kmb_test_utils.hpp"

struct TileParams final {
    size_t _axis = 0;
    size_t _tiles = 0;

    TileParams& axis(const size_t& axis) {
        this->_axis = axis;
        return *this;
    }

    TileParams& tiles(const size_t& tiles) {
        this->_tiles = tiles;
        return *this;
    }
};

inline std::ostream& operator<<(std::ostream& os, const TileParams& p) {
    vpu::formatPrint(os, "[axis: %v, tiles: %v]",
        p._axis, p._tiles);
    return os;
}

struct TileLayerDef final {
    TestNetwork& testNet;

    std::string name;

    TileParams params;

    PortInfo inputPort;
    PortInfo repeatsPort;

    TileLayerDef(TestNetwork& testNet, std::string name, TileParams params)
        : testNet(testNet), name(std::move(name)), params(std::move(params)) {
    }

    TileLayerDef& input(const std::string& layerName, size_t index = 0) {
        inputPort = PortInfo(layerName, index);
        return *this;
    }

    TileLayerDef& repeats(const std::string& layer_name, size_t index = 0) {
        repeatsPort = PortInfo(layer_name, index);
        return *this;
    }

    TestNetwork& build();
};
