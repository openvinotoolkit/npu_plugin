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

#include <gtest/gtest.h>
#include <libgen.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <test_model_path.hpp>
#include <vector>

#ifndef KMB_ALPHA_TESTS_DATA_PATH
#define KMB_ALPHA_TESTS_DATA_PATH ""
#endif

#ifndef _WIN32
static std::string getDirname(std::string filePath) {
    std::vector<char> input(filePath.begin(), filePath.end());
    input.push_back(0);
    return dirname(&*input.begin());
}
#else
static std::string getDirname(std::string filePath) {
    char dirname[_MAX_DIR];
    _splitpath(filePath.c_str(), nullptr, dirname, nullptr, nullptr);
    return dirname;
}
#endif

static const char* getTestDataPathNonFatal() noexcept {
    const char* models_path = std::getenv("KMB_ALPHA_TESTS_DATA_PATH");

    if (models_path == NULL) {
        if (KMB_ALPHA_TESTS_DATA_PATH != NULL) {
            models_path = KMB_ALPHA_TESTS_DATA_PATH;
        } else {
            ::testing::AssertionFailure() << "KMB_ALPHA_TESTS_DATA_PATH not defined";
        }
    }
    return models_path;
}

static bool exist(const std::string& name) {
    std::ifstream file(name);
    return static_cast<bool>(file);
}

static std::vector<std::string> getModelsDirs() { return std::vector<std::string> {getTestDataPathNonFatal()}; }

class KmbModelsPath {
    std::stringstream _rel_path;
    mutable std::string _abs_path;

public:
    KmbModelsPath() = default;
    KmbModelsPath(const KmbModelsPath& that) { _rel_path << that._rel_path.str(); }

    template <class T>
    KmbModelsPath& operator+=(const T& relative_path) {
        _rel_path << relative_path;
        return *this;
    }

    template <class T>
    KmbModelsPath& operator<<(const T& serializable) {
        _rel_path << serializable;
        return *this;
    }

    std::string str() const { return this->operator std::string(); }

    const char* c_str() const {
        _abs_path = this->operator std::string();
        return _abs_path.c_str();
    }

    template <class T>
    KmbModelsPath operator+(const T& relative_path) const {
        KmbModelsPath newPath(*this);
        newPath += relative_path;
        return newPath;
    }

    operator std::string() const {
        std::vector<std::string> absModelsPath;
        for (auto& path : getModelsDirs()) {
            absModelsPath.push_back(path + _rel_path.str());
            if (exist(absModelsPath.back())) {
                return absModelsPath.back();
            }
            // checking models for precision encoded in folder name
            auto dirname = getDirname(absModelsPath.back());
            std::vector<std::pair<std::string, std::string>> stdprecisions = {
                {"_fp32", "FP32"}, {"_q78", "_Q78"}, {"_fp16", "FP16"}, {"_i16", "I16"}};

            auto filename = absModelsPath.back().substr(dirname.size() + 1);

            for (auto& precision : stdprecisions) {
                auto havePrecision = filename.find(precision.first);
                if (havePrecision == std::string::npos) continue;

                auto newName = filename.replace(havePrecision, precision.first.size(), "");
                newName = dirname + kPathSeparator + precision.second + kPathSeparator + newName;

                if (exist(newName)) {
                    return newName;
                }
            }
        }

        auto getModelsDirname = [](std::string path) -> std::string {
            std::string dir = getDirname(path);

            struct stat sb;
            if (stat(dir.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
                return "";
            }
            return dir;
        };

        for (auto& path : absModelsPath) {
            std::string publicDir = getModelsDirname(path);

            if (!publicDir.empty()) {
                return path;
            }
        }

        return "";
    }
};
