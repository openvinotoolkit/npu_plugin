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

#include "vpux/compiler/core/attributes/dim_values.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/attributes/strides.hpp"

#include "vpux/utils/core/range.hpp"

namespace vpux {
namespace subspace {

///
/// \brief Uses number of section to calculate coordinates of section
/// \param [in] dims - dimensionality
/// \param [in] nSubspace - number of section
/// \return coordinates of section
///
MemShape getCoord(MemShapeRef dims, int64_t numSections);

///
/// \brief Uses coordinates of the section and strides to calculate offset
///        from beginning of original tensor to beginning of section
/// \param [in] subspaceCoord - coordinates of section
/// \param [in] strides - strides
/// \param [in] broadcast - broadcast flags, by dimensions (0=normal, 1=broadcasted)
/// \return offset in bits
///
Bit getOffset(MemShapeRef subspaceCoord, MemStridesRef strides, ArrayRef<bool> broadcast = {});

///
/// \brief Increments current subspaceCoord by 1 element
/// \param [in/out] subspaceCoord - coordinates of section
/// \param [in] dims - sizes of dimensions
///
void increment1Coord(MemShape& subspaceCoord, MemShapeRef dims);

///
/// \brief Increments current subspaceCoord by N elements
/// \param [in/out] subspaceCoord - coordinates of section
/// \param [in] dims - sizes of dimensions
/// \param [in] inc - value of the increment in elements
///
void incrementNCoord(MemShape& subspaceCoord, MemShapeRef dims, int64_t inc);

///
/// \brief Increments current coordinates of 1D section (line along axis coordinate) by 1
/// \param [in/out] lineCoord - full coordinate vector with line's coordinates
/// \param [in] dims - sizes of dimensions
/// \param [in] axis - coordinate along which the line goes
///
void incrementLine(MemShape& lineCoord, MemShapeRef dims, MemDim axis);

///
/// \brief Increments current coordinates of 2D section (plane on axis0, axis1 coordinates) by 1
/// \param [in/out] planeCoord - full coordinate vector with plane's coordinates
/// \param [in] dims - sizes of dimensions
/// \param [in] axis0, axis1 - coordinates on which the plane is built
///
void incrementPlane(MemShape& planeCoord, MemShapeRef dims, MemDim axis0, MemDim axis1);

///
/// \brief Calculates amount of different 1D sections in tensor
/// \param [in] dims - sizes of dimensions
/// \param [in] axis - coordinate along which the lines go
/// \return common amount of different 1D sections in tensor
///
int64_t getTotalLines(MemShapeRef dims, MemDim axis);

///
/// \brief Calculates amount of different 2D sections in tensor
/// \param [in] dims - sizes of dimensions
/// \param [in] axis0, axis1 - coordinates on which the plane is built
/// \return common amount of different 2D sections in tensor
///
int64_t getTotalPlanes(MemShapeRef dims, MemDim axis0, MemDim axis1);

///
/// \brief Calculates sizes (in elements) of included subtensors of smaller dimensionality,
/// \param [in] subspaceDims - sizes of dimensions
/// \return sizes of included subtensors
///
MemShape getSizes(MemShapeRef subspaceDims);

///
/// \brief Excludes 1 element from array
/// \param [in/out] arr - target arrays
/// \param [in] elPos - coordinate of element to be excluded
///
template <typename D, typename T, template <class> class Tag>
auto arrayElementExclude(details::DimValuesRef<D, T, Tag> arr, D elPos) {
    VPUX_THROW_UNLESS(checked_cast<size_t>(elPos.ind()) < arr.size(), "'{0}' index is out of '{1}' range", elPos, arr);

    auto out = arr.toValues();
    out.erase(out.begin() + elPos.ind());
    return out;
}
template <typename D, typename T, template <class> class Tag>
auto arrayElementExclude(const details::DimValues<D, T, Tag>& arr, D elPos) {
    return arrayElementExclude(details::DimValuesRef<D, T, Tag>(arr), elPos);
}

///
/// \brief Includes 1 element to array
/// \param [in/out] arr - target arrays
/// \param [in] elPos - coordinate of element to be included
/// \param [in] value - element value to be included
///
template <typename D, typename T, template <class> class Tag, typename V>
auto arrayElementInclude(details::DimValuesRef<D, T, Tag> arr, D elPos, V&& value) {
    VPUX_THROW_UNLESS(checked_cast<size_t>(elPos.ind()) <= arr.size(), "'{0}' index is out of '{1}' range", elPos, arr);

    auto out = arr.toValues();
    out.insert(out.begin() + elPos.ind(), std::forward<V>(value));
    return out;
}
template <typename D, typename T, template <class> class Tag, typename V>
auto arrayElementInclude(const details::DimValues<D, T, Tag>& arr, D elPos, V&& value) {
    return arrayElementInclude(details::DimValuesRef<D, T, Tag>(arr), elPos, std::forward<V>(value));
}

}  // namespace subspace
}  // namespace vpux
