//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/core/layers.hpp"

using namespace vpux;

//
// Dims4D
//

const Dim vpux::Dims4D::Act::N(0);
const Dim vpux::Dims4D::Act::C(1);
const Dim vpux::Dims4D::Act::H(2);
const Dim vpux::Dims4D::Act::W(3);

const Dim vpux::Dims4D::Filter::OC(0);
const Dim vpux::Dims4D::Filter::IC(1);
const Dim vpux::Dims4D::Filter::KY(2);
const Dim vpux::Dims4D::Filter::KX(3);

const Dim vpux::Dims4D::Kernel::Y(0);
const Dim vpux::Dims4D::Kernel::X(1);

const Dim vpux::Dims4D::Dilation::Y(0);
const Dim vpux::Dims4D::Dilation::X(1);

const Dim vpux::Dims4D::Strides::Y(0);
const Dim vpux::Dims4D::Strides::X(1);

const Dim vpux::Dims4D::PadsBegin::Top(0);
const Dim vpux::Dims4D::PadsBegin::Left(1);

const Dim vpux::Dims4D::PadsEnd::Bottom(0);
const Dim vpux::Dims4D::PadsEnd::Right(1);
