//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/init.hpp"

#include "vpux/compiler/dialect/VPU/passes.hpp"

#include "vpux/compiler/dialect/VPUIPRegMapped/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/types.hpp"

#include "vpux/compiler/dialect/VPU37XX/ops.hpp"
#include "vpux/compiler/dialect/VPU37XX/types.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;

std::shared_ptr<mlir::MLIRContext> getGlobalContext() {
    static std::shared_ptr<mlir::MLIRContext> globalCtx = std::make_unique<mlir::MLIRContext>();
    return globalCtx;
}

class MLIR_VPUIPRegisterSerializationTest :
        public testing::TestWithParam<std::pair<VPUIPRegMapped::RegisterType, std::vector<uint8_t>>> {};

TEST_P(MLIR_VPUIPRegisterSerializationTest, Serialization) {
    const auto testedRegisterDesc = GetParam();

    auto res = testedRegisterDesc.first.serialize();
    EXPECT_EQ(res, testedRegisterDesc.second);
}

auto genRegister =
        [](uint32_t size, std::string name, uint32_t address,
           std::vector<std::tuple<uint8_t /*width*/, uint8_t /*pos*/, uint64_t /*value*/, std::string /*name*/>>
                   regFields) {
            mlir::DialectRegistry registry;
            registerDialects(registry);

            getGlobalContext()->loadDialect<VPUIPRegMapped::VPUIPRegMappedDialect>();
            mlir::OpBuilder globalBuilder(getGlobalContext().get());

            std::vector<VPUIPRegMapped::RegFieldType> fields;
            std::transform(regFields.cbegin(), regFields.cend(), std::back_inserter(fields),
                           [](std::tuple<uint8_t, uint8_t, uint64_t, std::string> fieldDesc) {
                               uint8_t width;
                               uint8_t pos;
                               uint64_t value;
                               std::string name;
                               std::tie(width, pos, value, name) = fieldDesc;
                               return vpux::VPUIPRegMapped::RegFieldType::get(getGlobalContext().get(), width, pos,
                                                                              value, name);
                           });
            mlir::ArrayRef<vpux::VPUIPRegMapped::RegFieldType> arrayRefFields(fields);
            mlir::ArrayAttr fieldsArrayAttr = getVPUIPRegMapped_RegisterFieldArrayAttr(globalBuilder, arrayRefFields);
            return VPUIPRegMapped::RegisterType::get(getGlobalContext().get(), size, name, address, fieldsArrayAttr);
        };

std::vector<std::pair<VPUIPRegMapped::RegisterType, std::vector<uint8_t>>> simpleRegistersSet = {
        {genRegister(64, "64_bit_reg", 0x0,
                     {std::tuple<uint8_t, uint8_t, uint64_t, std::string>(64, 0, 0xC4D5E5B8058C41E5, "field_0")}),
         std::vector<uint8_t>{0xE5, 0x41, 0x8C, 0x05, 0xB8, 0xE5, 0xD5, 0xC4}},
        {genRegister(32, "32_bit_reg", 0x0,
                     {std::tuple<uint8_t, uint8_t, uint64_t, std::string>(32, 0, 0xC4D5E5B8, "field_0")}),
         std::vector<uint8_t>{0xB8, 0xE5, 0xD5, 0xC4}},
        {genRegister(16, "16_bit_reg", 0x0,
                     {std::tuple<uint8_t, uint8_t, uint64_t, std::string>(16, 0, 0xC4D5, "field_0")}),
         std::vector<uint8_t>{0xD5, 0xC4}},
        {genRegister(8, "8_bit_reg", 0x0, {std::tuple<uint8_t, uint8_t, uint64_t, std::string>(8, 0, 0xC4, "field_0")}),
         std::vector<uint8_t>{0xC4}}};

INSTANTIATE_TEST_CASE_P(simpleRegisters, MLIR_VPUIPRegisterSerializationTest, testing::ValuesIn(simpleRegistersSet));

std::vector<std::pair<VPUIPRegMapped::RegisterType, std::vector<uint8_t>>> compoundRegistersSet = {
        {genRegister(64, "64_bit_reg", 0x0,
                     {std::tuple<uint8_t, uint8_t, uint64_t, std::string>(8, 0, 0xE5, "field_0"),
                      std::tuple<uint8_t, uint8_t, uint64_t, std::string>(48, 8, 0xFFFFFFFFFFFF, "field_1"),
                      std::tuple<uint8_t, uint8_t, uint64_t, std::string>(8, 56, 0xC4, "field_2")}),
         std::vector<uint8_t>{0xE5, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xC4}},
        {genRegister(32, "32_bit_reg", 0x0,
                     {
                             std::tuple<uint8_t, uint8_t, uint64_t, std::string>(4, 0, 0x8, "field_0"),
                             std::tuple<uint8_t, uint8_t, uint64_t, std::string>(4, 4, 0XB, "field_1"),
                             std::tuple<uint8_t, uint8_t, uint64_t, std::string>(8, 8, 0xE5, "field_2"),
                             std::tuple<uint8_t, uint8_t, uint64_t, std::string>(16, 16, 0xC4D5, "field_3"),
                     }),
         std::vector<uint8_t>{0xB8, 0xE5, 0xD5, 0xC4}},
        {genRegister(16, "16_bit_reg", 0x0,
                     {std::tuple<uint8_t, uint8_t, uint64_t, std::string>(2, 0, 0x3, "field_0"),
                      std::tuple<uint8_t, uint8_t, uint64_t, std::string>(4, 8, 0xA, "field_1")}),
         std::vector<uint8_t>{0x3, 0x0A}},
        {genRegister(8, "8_bit_reg", 0x0,
                     {std::tuple<uint8_t, uint8_t, uint64_t, std::string>(1, 0, 0x1, "field_0"),
                      std::tuple<uint8_t, uint8_t, uint64_t, std::string>(4, 2, 0xF, "field_1"),
                      std::tuple<uint8_t, uint8_t, uint64_t, std::string>(1, 7, 0x1, "field_2")}),
         std::vector<uint8_t>{0xBD}}};

INSTANTIATE_TEST_CASE_P(compoundRegisters, MLIR_VPUIPRegisterSerializationTest,
                        testing::ValuesIn(compoundRegistersSet));

class MLIR_VPUIPRegMappedSerializationTest :
        public testing::TestWithParam<std::pair<VPUIPRegMapped::RegMappedType, std::vector<uint8_t>>> {};

TEST_P(MLIR_VPUIPRegMappedSerializationTest, Serialization) {
    const auto testedRegisterDesc = GetParam();

    auto res = testedRegisterDesc.first.serialize();
    EXPECT_EQ(res, testedRegisterDesc.second);
}

auto genMappedRegister = [](std::string name, std::vector<VPUIPRegMapped::RegisterType> regs) {
    mlir::DialectRegistry registry;
    registerDialects(registry);

    getGlobalContext()->loadDialect<VPUIPRegMapped::VPUIPRegMappedDialect>();
    mlir::OpBuilder globalBuilder(getGlobalContext().get());

    mlir::ArrayRef<vpux::VPUIPRegMapped::RegisterType> arrayRefRegisters(regs);
    mlir::ArrayAttr regsArrayAttr = getVPUIPRegMapped_RegisterArrayAttr(globalBuilder, arrayRefRegisters);
    return VPUIPRegMapped::RegMappedType::get(getGlobalContext().get(), name, regsArrayAttr);
};
std::vector<std::pair<VPUIPRegMapped::RegMappedType, std::vector<uint8_t>>> simpleMappedRegistersSet = {
        // check serialization of dense MappedRegister.
        // Inner registers folow one by one without gaps.
        // Registers declared in order of increasing their adresses: 0x0, 0x8, 0xC, 0xE
        {genMappedRegister(
                 "MappedReg",
                 {
                         genRegister(64, "64_bit_reg", 0x0,
                                     {std::tuple<uint8_t, uint8_t, uint64_t, std::string>(64, 0, 0xC4D5E5B8058C41E5,
                                                                                          "field_0")}),
                         genRegister(
                                 32, "32_bit_reg", 0x8,
                                 {std::tuple<uint8_t, uint8_t, uint64_t, std::string>(32, 0, 0xC4D5E5B8, "field_0")}),
                         genRegister(16, "16_bit_reg", 0xC,
                                     {std::tuple<uint8_t, uint8_t, uint64_t, std::string>(16, 0, 0xC4D5, "field_0")}),
                         genRegister(8, "8_bit_reg", 0xE,
                                     {std::tuple<uint8_t, uint8_t, uint64_t, std::string>(8, 0, 0xC4, "field_0")}),
                 }),
         std::vector<uint8_t>{0xE5, 0x41, 0x8C, 0x05, 0xB8, 0xE5, 0xD5, 0xC4, 0xB8, 0xE5, 0xD5, 0xC4, 0xD5, 0xC4,
                              0xC4}},
        // check serialization of sparse MappedRegister.
        // Inner registers don't follow one after another. There are some gaps: [0x0-0x8), [0xC-0xE).
        // Registers declared in order of increasing their adresses: 0x8, 0xE
        {genMappedRegister(
                 "MappedReg",
                 {
                         genRegister(32, "32_bit_reg", 0x8,
                                     {std::tuple<uint8_t, uint8_t, uint64_t, std::string>(32, 0, 0xC4D5E5B8,
                                                                                          "field_0")}),
                         genRegister(8, "8_bit_reg", 0xE,
                                     {std::tuple<uint8_t, uint8_t, uint64_t, std::string>(8, 0, 0xC4, "field_0")}),
                 }),
         std::vector<uint8_t>{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xB8, 0xE5, 0xD5, 0xC4, 0x0, 0x0, 0xC4}},
        // check serialization of sparse MappedRegister.
        // Inner registers don't follow one after another. There are some gaps: [0x0-0x8), [0xC-0xE).
        // Registers are not declared in random order of their adresses: 0xE, 0x8
        {genMappedRegister(
                 "MappedReg",
                 {genRegister(8, "8_bit_reg", 0xE,
                              {std::tuple<uint8_t, uint8_t, uint64_t, std::string>(8, 0, 0xC4, "field_0")}),
                  genRegister(32, "32_bit_reg", 0x8,
                              {std::tuple<uint8_t, uint8_t, uint64_t, std::string>(32, 0, 0xC4D5E5B8, "field_0")})}),
         std::vector<uint8_t>{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0xB8, 0xE5, 0xD5, 0xC4, 0x0, 0x0, 0xC4}}};

INSTANTIATE_TEST_CASE_P(simpleMappedRegisters, MLIR_VPUIPRegMappedSerializationTest,
                        testing::ValuesIn(simpleMappedRegistersSet));
