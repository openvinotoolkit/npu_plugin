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

#include "vpux/compiler/core/attributes/dim.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <cassert>

namespace vpux {

//
// DimValues
//

namespace details {

template <typename T>
class DimValuesBase {
public:
    using ContainerType = SmallVector<T, MAX_NUM_DIMS>;
    using ValueType = T;

public:
    using value_type = typename ContainerType::value_type;

    using iterator = typename ContainerType::iterator;
    using reverse_iterator = typename ContainerType::reverse_iterator;

    using const_iterator = typename ContainerType::const_iterator;
    using const_reverse_iterator = typename ContainerType::const_reverse_iterator;

    using size_type = typename ContainerType::size_type;

public:
    DimValuesBase() = default;

    DimValuesBase(ArrayRef<ValueType> vals) {
        assign(vals);
    }

    DimValuesBase(std::initializer_list<ValueType> vals) {
        assign(vals);
    }

    template <class Iter>
    DimValuesBase(Iter&& b, Iter&& e) {
        assign(std::forward<Iter>(b), std::forward<Iter>(e));
    }

    explicit DimValuesBase(size_t size) {
        resize(size);
    }

    DimValuesBase(size_t size, const ValueType& initVal) {
        assign(size, initVal);
    }

    explicit DimValuesBase(const ContainerType& cont): _cont(cont) {
    }
    explicit DimValuesBase(ContainerType&& cont): _cont(std::move(cont)) {
    }

public:
    void assign(ArrayRef<ValueType> vals) {
        _cont.assign(vals.begin(), vals.end());
    }
    void assign(std::initializer_list<ValueType> vals) {
        _cont.assign(vals);
    }
    void assign(size_t size, const ValueType& initVal) {
        _cont.assign(size, initVal);
    }
    template <class Iter>
    void assign(Iter&& b, Iter&& e) {
        _cont.assign(std::forward<Iter>(b), std::forward<Iter>(e));
    }

    void append(ArrayRef<ValueType> vals) {
        _cont.append(vals.begin(), vals.end());
    }
    void append(std::initializer_list<ValueType> vals) {
        _cont.append(vals);
    }
    void append(size_t size, const ValueType& initVal) {
        _cont.append(size, initVal);
    }
    template <class Iter>
    void append(Iter&& b, Iter&& e) {
        _cont.append(std::forward<Iter>(b), std::forward<Iter>(e));
    }

    void push_back(const ValueType& val) {
        _cont.push_back(val);
    }
    void push_back(ValueType&& val) {
        _cont.push_back(std::move(val));
    }

    template <typename... Args>
    void emplace_back(Args&&... args) {
        _cont.emplace_back(std::forward<Args>(args)...);
    }

    void insert(iterator pos, const ValueType& val) {
        _cont.insert(pos, val);
    }
    void insert(iterator pos, ValueType&& val) {
        _cont.insert(pos, std::move(val));
    }

    void resize(size_t newSize) {
        _cont.resize(newSize);
    }
    void resize(size_t newSize, const ValueType& newValue) {
        _cont.resize(newSize, newValue);
    }

    void clear() {
        _cont.clear();
    }

public:
    size_t size() const {
        return _cont.size();
    }

    bool empty() const {
        return _cont.empty();
    }

public:
    auto begin() {
        return _cont.begin();
    }

    auto end() {
        return _cont.end();
    }

    auto begin() const {
        return _cont.begin();
    }

    auto end() const {
        return _cont.end();
    }

    auto rbegin() {
        return _cont.rbegin();
    }

    auto rend() {
        return _cont.rend();
    }

    auto rbegin() const {
        return _cont.rbegin();
    }

    auto rend() const {
        return _cont.rend();
    }

public:
    const auto& front() const {
        assert(!empty());
        return _cont.front();
    }
    auto& front() {
        assert(!empty());
        return _cont.front();
    }

    const auto& back() const {
        assert(!empty());
        return _cont.back();
    }
    auto& back() {
        assert(!empty());
        return _cont.back();
    }

public:
    auto& raw() {
        return _cont;
    }
    const auto& raw() const {
        return _cont;
    }

public:
    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "{0}", raw());
    }

private:
    ContainerType _cont;
};

template <typename D, typename T, template <class> class Tag>
class DimValues final : public Tag<DimValuesBase<T>> {
public:
    using DimType = D;

public:
    using Tag<DimValuesBase<T>>::Tag;

public:
    const auto& operator[](DimType d) const {
        assert(d.ind() >= 0 && static_cast<size_t>(d.ind()) < this->size());
        return this->raw()[static_cast<size_t>(d.ind())];
    }
    auto& operator[](DimType d) {
        assert(d.ind() >= 0 && static_cast<size_t>(d.ind()) < this->size());
        return this->raw()[static_cast<size_t>(d.ind())];
    }

public:
    bool operator==(const DimValues& other) const {
        return this->raw() == other.raw();
    }
    bool operator!=(const DimValues& other) const {
        return this->raw() != other.raw();
    }
};

}  // namespace details

//
// DimValuesRef
//

namespace details {

template <typename T>
class DimValuesRefBase {
public:
    using BaseRef = ArrayRef<T>;
    using ValueType = T;

public:
    using value_type = typename BaseRef::value_type;

    using iterator = typename BaseRef::iterator;
    using reverse_iterator = typename BaseRef::reverse_iterator;

    using const_iterator = typename BaseRef::const_iterator;
    using const_reverse_iterator = typename BaseRef::const_reverse_iterator;

    using size_type = typename BaseRef::size_type;

public:
    DimValuesRefBase() = default;

    DimValuesRefBase(const std::initializer_list<ValueType>& vals): _ref(vals) {
    }

    explicit DimValuesRefBase(BaseRef ref): _ref(std::move(ref)) {
    }

public:
    size_t size() const {
        return _ref.size();
    }

    bool empty() const {
        return _ref.empty();
    }

public:
    auto begin() const {
        return _ref.begin();
    }
    auto end() const {
        return _ref.end();
    }

    auto rbegin() const {
        return _ref.rbegin();
    }
    auto rend() const {
        return _ref.rend();
    }

public:
    const auto& front() const {
        assert(!empty());
        return _ref.front();
    }

    const auto& back() const {
        assert(!empty());
        return _ref.back();
    }

public:
    const auto& raw() const {
        return _ref;
    }

public:
    void printFormat(llvm::raw_ostream& stream) const {
        printTo(stream, "{0}", raw());
    }

private:
    BaseRef _ref;
};

template <typename D, typename T, template <class> class Tag>
class DimValuesRef final : public Tag<DimValuesRefBase<T>> {
public:
    using DimType = D;

public:
    using Tag<DimValuesRefBase<T>>::Tag;

    DimValuesRef(const DimValuesRef<D, T, Tag>& other) = default;
    DimValuesRef& operator=(const DimValuesRef<D, T, Tag>& other) = default;

    DimValuesRef(DimValuesRef<D, T, Tag>&& other) = default;
    DimValuesRef& operator=(DimValuesRef<D, T, Tag>&& other) = default;

    template <typename U, typename = require_t<std::is_convertible<U*, T*>>>
    DimValuesRef(const DimValuesRef<D, U, Tag>& values): Tag<DimValuesRefBase<T>>(values.raw()) {
    }

    template <typename U, typename = require_t<std::is_convertible<U*, T*>>>
    DimValuesRef(const DimValues<D, U, Tag>& values): Tag<DimValuesRefBase<T>>(values.raw()) {
    }

public:
    const auto& operator[](DimType d) const {
        assert(d.ind() >= 0 && static_cast<size_t>(d.ind()) < this->size());
        return this->raw()[static_cast<size_t>(d.ind())];
    }

public:
    bool operator==(const DimValuesRef& other) const {
        return this->raw() == other.raw();
    }
    bool operator!=(const DimValuesRef& other) const {
        return this->raw() != other.raw();
    }

public:
    auto toValues() const -> DimValues<D, typename std::remove_const<T>::type, Tag> {
        using ResultType = DimValues<D, typename std::remove_const<T>::type, Tag>;
        return ResultType(this->begin(), this->end());
    }
};

}  // namespace details

}  // namespace vpux
