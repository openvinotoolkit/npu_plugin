//
// Copyright Intel Corporation.
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

#include "vpux/utils/core/small_vector.hpp"

#include <cassert>

namespace vpux {

template <typename T>
class RingBuffer final {
public:
    RingBuffer() = default;

    explicit RingBuffer(size_t capacity): _storage(capacity), _putInd(capacity), _getInd(0), _size(0) {
    }

public:
    template <typename T1>
    void push(T1&& val) {
        ++_size;

        if (++_putInd >= _storage.size()) {
            _putInd = 0;
        }

        _storage[_putInd] = std::forward<T1>(val);
    }

    void pop() {
        assert(_size > 0);
        --_size;

        if (++_getInd >= _storage.size()) {
            _getInd = 0;
        }
    }

    void clear() {
        _putInd = _storage.size();
        _getInd = 0;
        _size = 0;
    }

public:
    T& front() {
        return _storage[_getInd];
    }
    const T& front() const {
        return _storage[_getInd];
    }

    T& back() {
        return _storage[_putInd];
    }
    const T& back() const {
        return _storage[_putInd];
    }

    bool empty() const {
        return _size == 0;
    }
    bool full() const {
        return _size == _storage.size();
    }

    size_t size() const {
        return _size;
    }
    size_t capacity() const {
        return _storage.size();
    }

private:
    SmallVector<T> _storage;
    size_t _putInd = 0;
    size_t _getInd = 0;
    size_t _size = 0;
};

}  // namespace vpux
