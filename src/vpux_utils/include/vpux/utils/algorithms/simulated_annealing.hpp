//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <random>

#include <vpux/utils/core/error.hpp>

namespace vpux {
namespace algorithm {

template <class T>
using StateGetter = std::function<const T(int, const T* const)>;
template <class T>
using CostGetter = std::function<double(const T&)>;
template <class T>
using Successor = std::function<void(const T&)>;

using StopCondition = std::function<bool(size_t)>;
using TemperatureCallback = std::function<void(size_t&)>;

bool defaultStopCondition(size_t temperature) {
    return temperature <= 0;
}

void defaultTemperatureCallback(size_t& temperature) {
    --temperature;
}

/*
 * Simulated Annealing Algorithm
 * Optimization algorithm for global minimum search
 * On each step it gets "neighbour" of current state and
 * tries to compare the cost of new state and current one.
 * If new solution is "better" it's accepted, if not,
 * it might also be accepted with a probability of e^(-delta/temperature).
 * Temperature decreases on each step, algorithm stops if it's zero.
 */
template <class State>
State simulatedAnnealing(size_t temperature, StateGetter<State> getState, CostGetter<State> getCost,
                         Successor<State> successCallback = nullptr, StopCondition stopCondition = defaultStopCondition,
                         TemperatureCallback changeTemperature = defaultTemperatureCallback) {
    VPUX_THROW_WHEN(getState == nullptr || getCost == nullptr, "Functions for getting state and cost are not set");

    auto currentState = getState(temperature, nullptr);

    const double lowerBound = 0;
    const double upperBound = 1;

    std::uniform_real_distribution<double> dist(lowerBound, upperBound);

    // same default seed is used to get deterministic sequence of numbers
    std::mt19937 randomEngine;

    double bestCost = getCost(currentState);
    auto bestState = currentState;

    VPUX_THROW_WHEN(stopCondition == nullptr || changeTemperature == nullptr,
                    "Functions for checking stop condition and changing temperature are not set");

    while (!stopCondition(temperature)) {
        auto neighbour = getState(temperature, &currentState);
        const auto newCost = getCost(neighbour);
        const auto currentCost = getCost(currentState);
        const auto delta = newCost - currentCost;

        const double acceptance = dist(randomEngine);

        // use here Metropolis Criterion to decide acceptance of worse states
        if (delta < 0 || exp(-(delta / temperature)) > acceptance) {
            currentState = neighbour;
            if (successCallback != nullptr) {
                successCallback(currentState);
            }
        }

        if (newCost < bestCost) {
            bestState = neighbour;
            bestCost = newCost;
        }

        changeTemperature(temperature);
    }

    return bestState;
}

}  // namespace algorithm
}  // namespace vpux
