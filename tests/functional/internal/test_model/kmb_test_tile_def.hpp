//
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
