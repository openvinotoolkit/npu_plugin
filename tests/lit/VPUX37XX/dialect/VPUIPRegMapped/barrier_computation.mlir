//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --barrier-computation %s | FileCheck %s
module @Test {

IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x1x2x1000xf16>
} outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x1x2x1000xf16>
}

func @main(%arg0: memref<1x1x2x1000xf16>, %arg1: memref<1x1x2x1000xf16>) -> memref<1x1x2x1000xf16> {
    // this is the most simple lit test that could be constructed (VPUX30XX arch to be used in order to only have 32 barriers)

    %buffer = VPURT.DeclareBuffer "DDR" <0> -> memref<1x1x2x1000xf16, @DDR>
    %buffer1 = VPURT.DeclareBuffer "DDR" <4000> -> memref<1x1x2x1000xf16, @DDR>

    %barrier0 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=0:ui8 } <0,-1> -> !VPUIPRegMapped.Index<0>
    // CHECK:       %[[VAL0:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 0 : ui8}<0, 32> -> !VPUIPRegMapped.Index<0>

    %barrier1 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <1,-1> -> !VPUIPRegMapped.Index<1>

    // CHECK:       %[[VAL1:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, 33> -> !VPUIPRegMapped.Index<1>

    %barrier2 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <2,-1> -> !VPUIPRegMapped.Index<2>

    // CHECK:       %[[VAL2:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<2, 34> -> !VPUIPRegMapped.Index<2>

    %barrier3 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <3,-1> -> !VPUIPRegMapped.Index<3>

    // CHECK:       %[[VAL3:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<3, 35> -> !VPUIPRegMapped.Index<3>

    %barrier4 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <4,-1> -> !VPUIPRegMapped.Index<4>

    // CHECK:       %[[VAL4:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<4, 36> -> !VPUIPRegMapped.Index<4>

    %barrier5 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <5,-1> -> !VPUIPRegMapped.Index<5>

    // CHECK:       %[[VAL5:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<5, 37> -> !VPUIPRegMapped.Index<5>

    %barrier6 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <6,-1> -> !VPUIPRegMapped.Index<6>

    // CHECK:       %[[VAL6:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<6, 38> -> !VPUIPRegMapped.Index<6>

    %barrier7 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <7,-1> -> !VPUIPRegMapped.Index<7>

    // CHECK:       %[[VAL7:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<7, 39> -> !VPUIPRegMapped.Index<7>

    %barrier8 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <8,-1> -> !VPUIPRegMapped.Index<8>

    // CHECK:       %[[VAL8:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<8, 40> -> !VPUIPRegMapped.Index<8>

    %barrier9 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <9,-1> -> !VPUIPRegMapped.Index<9>

    // CHECK:       %[[VAL9:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<9, 41> -> !VPUIPRegMapped.Index<9>

    %barrier10 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <10,-1> -> !VPUIPRegMapped.Index<10>

    // CHECK:       %[[VAL10:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<10, 42> -> !VPUIPRegMapped.Index<10>

    %barrier11 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <11,-1> -> !VPUIPRegMapped.Index<11>

    // CHECK:       %[[VAL11:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<11, 43> -> !VPUIPRegMapped.Index<11>

    %barrier12 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <12,-1> -> !VPUIPRegMapped.Index<12>

    // CHECK:       %[[VAL12:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<12, 44> -> !VPUIPRegMapped.Index<12>

    %barrier13 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <13,-1> -> !VPUIPRegMapped.Index<13>

    // CHECK:       %[[VAL13:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<13, 45> -> !VPUIPRegMapped.Index<13>

    %barrier14 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <14,-1> -> !VPUIPRegMapped.Index<14>

    // CHECK:       %[[VAL14:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<14, 46> -> !VPUIPRegMapped.Index<14>

    %barrier15 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <15,-1> -> !VPUIPRegMapped.Index<15>

    // CHECK:       %[[VAL15:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<15, 47> -> !VPUIPRegMapped.Index<15>

    %barrier16 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <16,-1> -> !VPUIPRegMapped.Index<16>

    // CHECK:       %[[VAL16:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<16, 48> -> !VPUIPRegMapped.Index<16>

    %barrier17 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <17,-1> -> !VPUIPRegMapped.Index<17>

    // CHECK:       %[[VAL17:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<17, 49> -> !VPUIPRegMapped.Index<17>

    %barrier18 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <18,-1> -> !VPUIPRegMapped.Index<18>

    // CHECK:       %[[VAL18:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<18, 50> -> !VPUIPRegMapped.Index<18>

    %barrier19 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <19,-1> -> !VPUIPRegMapped.Index<19>

    // CHECK:       %[[VAL19:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<19, 51> -> !VPUIPRegMapped.Index<19>

    %barrier20 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <20,-1> -> !VPUIPRegMapped.Index<20>

    // CHECK:       %[[VAL20:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<20, 52> -> !VPUIPRegMapped.Index<20>

    %barrier21 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <21,-1> -> !VPUIPRegMapped.Index<21>

    // CHECK:       %[[VAL21:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<21, 53> -> !VPUIPRegMapped.Index<21>

    %barrier22 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <22,-1> -> !VPUIPRegMapped.Index<22>

    // CHECK:       %[[VAL22:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<22, 54> -> !VPUIPRegMapped.Index<22>

    %barrier23 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <23,-1> -> !VPUIPRegMapped.Index<23>

    // CHECK:       %[[VAL23:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<23, 55> -> !VPUIPRegMapped.Index<23>

    %barrier24 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <24,-1> -> !VPUIPRegMapped.Index<24>

    // CHECK:       %[[VAL24:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<24, 56> -> !VPUIPRegMapped.Index<24>

    %barrier25 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <25,-1> -> !VPUIPRegMapped.Index<25>

    // CHECK:       %[[VAL25:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<25, 57> -> !VPUIPRegMapped.Index<25>

    %barrier26 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <26,-1> -> !VPUIPRegMapped.Index<26>

    // CHECK:       %[[VAL26:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<26, 58> -> !VPUIPRegMapped.Index<26>

    %barrier27 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <27,-1> -> !VPUIPRegMapped.Index<27>

    // CHECK:       %[[VAL27:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<27, 59> -> !VPUIPRegMapped.Index<27>

    %barrier28 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <28,-1> -> !VPUIPRegMapped.Index<28>

    // CHECK:       %[[VAL28:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<28, 60> -> !VPUIPRegMapped.Index<28>

    %barrier29 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <29,-1> -> !VPUIPRegMapped.Index<29>

    // CHECK:       %[[VAL29:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<29, 61> -> !VPUIPRegMapped.Index<29>

    %barrier30 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <30,-1> -> !VPUIPRegMapped.Index<30>

    // CHECK:       %[[VAL30:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<30, 62> -> !VPUIPRegMapped.Index<30>

    %barrier31 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <31,-1> -> !VPUIPRegMapped.Index<31>

    // CHECK:       %[[VAL31:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<31, 63> -> !VPUIPRegMapped.Index<31>

    %barrier32 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <0,-1> -> !VPUIPRegMapped.Index<32>

    // CHECK:       %[[VAL32:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, 64> -> !VPUIPRegMapped.Index<32>

    %barrier33 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <1,-1> -> !VPUIPRegMapped.Index<33>

    // CHECK:       %[[VAL33:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, 65> -> !VPUIPRegMapped.Index<33>

    %barrier34 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <2,-1> -> !VPUIPRegMapped.Index<34>

    // CHECK:       %[[VAL34:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<2, -1> -> !VPUIPRegMapped.Index<34>

    %barrier35 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <3,-1> -> !VPUIPRegMapped.Index<35>

    // CHECK:       %[[VAL35:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<3, -1> -> !VPUIPRegMapped.Index<35>

    %barrier36 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <4,-1> -> !VPUIPRegMapped.Index<36>

    // CHECK:       %[[VAL36:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<4, -1> -> !VPUIPRegMapped.Index<36>

    %barrier37 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <5,-1> -> !VPUIPRegMapped.Index<37>

    // CHECK:       %[[VAL37:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<5, -1> -> !VPUIPRegMapped.Index<37>

    %barrier38 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <6,-1> -> !VPUIPRegMapped.Index<38>

    // CHECK:       %[[VAL38:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<6, -1> -> !VPUIPRegMapped.Index<38>

    %barrier39 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <7,-1> -> !VPUIPRegMapped.Index<39>

    // CHECK:       %[[VAL39:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<7, -1> -> !VPUIPRegMapped.Index<39>

    %barrier40 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <8,-1> -> !VPUIPRegMapped.Index<40>

    // CHECK:       %[[VAL40:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<8, -1> -> !VPUIPRegMapped.Index<40>

    %barrier41 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <9,-1> -> !VPUIPRegMapped.Index<41>

    // CHECK:       %[[VAL41:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<9, -1> -> !VPUIPRegMapped.Index<41>

    %barrier42 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <10,-1> -> !VPUIPRegMapped.Index<42>

    // CHECK:       %[[VAL42:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<10, -1> -> !VPUIPRegMapped.Index<42>

    %barrier43 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <11,-1> -> !VPUIPRegMapped.Index<43>

    // CHECK:       %[[VAL43:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<11, -1> -> !VPUIPRegMapped.Index<43>

    %barrier44 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <12,-1> -> !VPUIPRegMapped.Index<44>

    // CHECK:       %[[VAL44:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<12, -1> -> !VPUIPRegMapped.Index<44>

    %barrier45 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <13,-1> -> !VPUIPRegMapped.Index<45>

    // CHECK:       %[[VAL45:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<13, -1> -> !VPUIPRegMapped.Index<45>

    %barrier46 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <14,-1> -> !VPUIPRegMapped.Index<46>

    // CHECK:       %[[VAL46:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<14, -1> -> !VPUIPRegMapped.Index<46>

    %barrier47 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <15,-1> -> !VPUIPRegMapped.Index<47>

    // CHECK:       %[[VAL47:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<15, -1> -> !VPUIPRegMapped.Index<47>

    %barrier48 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <16,-1> -> !VPUIPRegMapped.Index<48>

    // CHECK:       %[[VAL48:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<16, -1> -> !VPUIPRegMapped.Index<48>

    %barrier49 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <17,-1> -> !VPUIPRegMapped.Index<49>

    // CHECK:       %[[VAL49:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<17, -1> -> !VPUIPRegMapped.Index<49>

    %barrier50 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <18,-1> -> !VPUIPRegMapped.Index<50>

    // CHECK:       %[[VAL50:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<18, -1> -> !VPUIPRegMapped.Index<50>

    %barrier51 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <19,-1> -> !VPUIPRegMapped.Index<51>

    // CHECK:       %[[VAL51:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<19, -1> -> !VPUIPRegMapped.Index<51>

    %barrier52 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <20,-1> -> !VPUIPRegMapped.Index<52>

    // CHECK:       %[[VAL52:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<20, -1> -> !VPUIPRegMapped.Index<52>

    %barrier53 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <21,-1> -> !VPUIPRegMapped.Index<53>

    // CHECK:       %[[VAL53:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<21, -1> -> !VPUIPRegMapped.Index<53>

    %barrier54 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <22,-1> -> !VPUIPRegMapped.Index<54>

    // CHECK:       %[[VAL54:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<22, -1> -> !VPUIPRegMapped.Index<54>

    %barrier55 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <23,-1> -> !VPUIPRegMapped.Index<55>

    // CHECK:       %[[VAL55:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<23, -1> -> !VPUIPRegMapped.Index<55>

    %barrier56 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <24,-1> -> !VPUIPRegMapped.Index<56>

    // CHECK:       %[[VAL56:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<24, -1> -> !VPUIPRegMapped.Index<56>

    %barrier57 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <25,-1> -> !VPUIPRegMapped.Index<57>

    // CHECK:       %[[VAL57:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<25, -1> -> !VPUIPRegMapped.Index<57>

    %barrier58 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <26,-1> -> !VPUIPRegMapped.Index<58>

    // CHECK:       %[[VAL58:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<26, -1> -> !VPUIPRegMapped.Index<58>

    %barrier59 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <27,-1> -> !VPUIPRegMapped.Index<59>

    // CHECK:       %[[VAL59:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<27, -1> -> !VPUIPRegMapped.Index<59>

    %barrier60 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <28,-1> -> !VPUIPRegMapped.Index<60>

    // CHECK:       %[[VAL60:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<28, -1> -> !VPUIPRegMapped.Index<60>

    %barrier61 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <29,-1> -> !VPUIPRegMapped.Index<61>

    // CHECK:       %[[VAL61:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<29, -1> -> !VPUIPRegMapped.Index<61>

    %barrier62 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <30,-1> -> !VPUIPRegMapped.Index<62>

    // CHECK:       %[[VAL62:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<30, -1> -> !VPUIPRegMapped.Index<62>

    %barrier63 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <31,-1> -> !VPUIPRegMapped.Index<63>

    // CHECK:       %[[VAL63:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<31, -1> -> !VPUIPRegMapped.Index<63>

    %barrier64 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <0,-1> -> !VPUIPRegMapped.Index<64>

    // CHECK:       %[[VAL64:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<64>

    %barrier65 = VPUIPRegMapped.ConfigureBarrier {consumer_count=1:ui8, producer_count=1:ui8 } <1,-1> -> !VPUIPRegMapped.Index<65>

    // CHECK:       %[[VAL65:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<65>


    %d0 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier0 : !VPUIPRegMapped.Index<0>) updates(%barrier1 : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>


    %d1 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier1 : !VPUIPRegMapped.Index<1>) updates(%barrier2 : !VPUIPRegMapped.Index<2>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>


    %d2 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier2 : !VPUIPRegMapped.Index<2>) updates(%barrier3 : !VPUIPRegMapped.Index<3>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>


    %d3 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier3 : !VPUIPRegMapped.Index<3>) updates(%barrier4 : !VPUIPRegMapped.Index<4>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<3>


    %d4 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier4 : !VPUIPRegMapped.Index<4>) updates(%barrier5 : !VPUIPRegMapped.Index<5>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<4>


    %d5 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier5 : !VPUIPRegMapped.Index<5>) updates(%barrier6 : !VPUIPRegMapped.Index<6>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<5>


    %d6 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier6 : !VPUIPRegMapped.Index<6>) updates(%barrier7 : !VPUIPRegMapped.Index<7>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<6>


    %d7 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier7 : !VPUIPRegMapped.Index<7>) updates(%barrier8 : !VPUIPRegMapped.Index<8>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<7>


    %d8 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier8 : !VPUIPRegMapped.Index<8>) updates(%barrier9 : !VPUIPRegMapped.Index<9>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<8>


    %d9 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier9 : !VPUIPRegMapped.Index<9>) updates(%barrier10 : !VPUIPRegMapped.Index<10>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<9>


    %d10 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier10 : !VPUIPRegMapped.Index<10>) updates(%barrier11 : !VPUIPRegMapped.Index<11>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<10>


    %d11 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier11 : !VPUIPRegMapped.Index<11>) updates(%barrier12 : !VPUIPRegMapped.Index<12>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<11>


    %d12 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier12 : !VPUIPRegMapped.Index<12>) updates(%barrier13 : !VPUIPRegMapped.Index<13>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<12>


    %d13 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier13 : !VPUIPRegMapped.Index<13>) updates(%barrier14 : !VPUIPRegMapped.Index<14>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<13>


    %d14 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier14 : !VPUIPRegMapped.Index<14>) updates(%barrier15 : !VPUIPRegMapped.Index<15>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<14>


    %d15 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier15 : !VPUIPRegMapped.Index<15>) updates(%barrier16 : !VPUIPRegMapped.Index<16>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<15>


    %d16 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier16 : !VPUIPRegMapped.Index<16>) updates(%barrier17 : !VPUIPRegMapped.Index<17>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<16>


    %d17 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier17 : !VPUIPRegMapped.Index<17>) updates(%barrier18 : !VPUIPRegMapped.Index<18>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<17>


    %d18 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier18 : !VPUIPRegMapped.Index<18>) updates(%barrier19 : !VPUIPRegMapped.Index<19>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<18>


    %d19 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier19 : !VPUIPRegMapped.Index<19>) updates(%barrier20 : !VPUIPRegMapped.Index<20>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<19>


    %d20 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier20 : !VPUIPRegMapped.Index<20>) updates(%barrier21 : !VPUIPRegMapped.Index<21>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<20>


    %d21 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier21 : !VPUIPRegMapped.Index<21>) updates(%barrier22 : !VPUIPRegMapped.Index<22>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<21>


    %d22 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier22 : !VPUIPRegMapped.Index<22>) updates(%barrier23 : !VPUIPRegMapped.Index<23>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<22>


    %d23 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier23 : !VPUIPRegMapped.Index<23>) updates(%barrier24 : !VPUIPRegMapped.Index<24>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<23>


    %d24 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier24 : !VPUIPRegMapped.Index<24>) updates(%barrier25 : !VPUIPRegMapped.Index<25>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<24>


    %d25 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier25 : !VPUIPRegMapped.Index<25>) updates(%barrier26 : !VPUIPRegMapped.Index<26>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<25>


    %d26 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier26 : !VPUIPRegMapped.Index<26>) updates(%barrier27 : !VPUIPRegMapped.Index<27>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<26>


    %d27 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier27 : !VPUIPRegMapped.Index<27>) updates(%barrier28 : !VPUIPRegMapped.Index<28>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<27>


    %d28 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier28 : !VPUIPRegMapped.Index<28>) updates(%barrier29 : !VPUIPRegMapped.Index<29>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<28>


    %d29 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier29 : !VPUIPRegMapped.Index<29>) updates(%barrier30 : !VPUIPRegMapped.Index<30>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<29>


    %d30 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier30 : !VPUIPRegMapped.Index<30>) updates(%barrier31 : !VPUIPRegMapped.Index<31>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<30>


    %d31 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier31 : !VPUIPRegMapped.Index<31>) updates(%barrier32 : !VPUIPRegMapped.Index<32>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<31>


    %d32 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier32 : !VPUIPRegMapped.Index<32>) updates(%barrier33 : !VPUIPRegMapped.Index<33>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<32>


    %d33 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier33 : !VPUIPRegMapped.Index<33>) updates(%barrier34 : !VPUIPRegMapped.Index<34>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<33>


    %d34 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier34 : !VPUIPRegMapped.Index<34>) updates(%barrier35 : !VPUIPRegMapped.Index<35>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<34>


    %d35 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier35 : !VPUIPRegMapped.Index<35>) updates(%barrier36 : !VPUIPRegMapped.Index<36>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<35>


    %d36 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier36 : !VPUIPRegMapped.Index<36>) updates(%barrier37 : !VPUIPRegMapped.Index<37>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<36>


    %d37 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier37 : !VPUIPRegMapped.Index<37>) updates(%barrier38 : !VPUIPRegMapped.Index<38>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<37>


    %d38 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier38 : !VPUIPRegMapped.Index<38>) updates(%barrier39 : !VPUIPRegMapped.Index<39>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<38>


    %d39 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier39 : !VPUIPRegMapped.Index<39>) updates(%barrier40 : !VPUIPRegMapped.Index<40>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<39>


    %d40 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier40 : !VPUIPRegMapped.Index<40>) updates(%barrier41 : !VPUIPRegMapped.Index<41>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<40>


    %d41 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier41 : !VPUIPRegMapped.Index<41>) updates(%barrier42 : !VPUIPRegMapped.Index<42>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<41>


    %d42 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier42 : !VPUIPRegMapped.Index<42>) updates(%barrier43 : !VPUIPRegMapped.Index<43>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<42>


    %d43 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier43 : !VPUIPRegMapped.Index<43>) updates(%barrier44 : !VPUIPRegMapped.Index<44>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<43>


    %d44 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier44 : !VPUIPRegMapped.Index<44>) updates(%barrier45 : !VPUIPRegMapped.Index<45>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<44>


    %d45 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier45 : !VPUIPRegMapped.Index<45>) updates(%barrier46 : !VPUIPRegMapped.Index<46>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<45>


    %d46 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier46 : !VPUIPRegMapped.Index<46>) updates(%barrier47 : !VPUIPRegMapped.Index<47>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<46>


    %d47 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier47 : !VPUIPRegMapped.Index<47>) updates(%barrier48 : !VPUIPRegMapped.Index<48>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<47>


    %d48 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier48 : !VPUIPRegMapped.Index<48>) updates(%barrier49 : !VPUIPRegMapped.Index<49>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<48>


    %d49 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier49 : !VPUIPRegMapped.Index<49>) updates(%barrier50 : !VPUIPRegMapped.Index<50>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<49>


    %d50 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier50 : !VPUIPRegMapped.Index<50>) updates(%barrier51 : !VPUIPRegMapped.Index<51>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<50>


    %d51 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier51 : !VPUIPRegMapped.Index<51>) updates(%barrier52 : !VPUIPRegMapped.Index<52>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<51>


    %d52 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier52 : !VPUIPRegMapped.Index<52>) updates(%barrier53 : !VPUIPRegMapped.Index<53>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<52>


    %d53 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier53 : !VPUIPRegMapped.Index<53>) updates(%barrier54 : !VPUIPRegMapped.Index<54>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<53>


    %d54 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier54 : !VPUIPRegMapped.Index<54>) updates(%barrier55 : !VPUIPRegMapped.Index<55>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<54>


    %d55 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier55 : !VPUIPRegMapped.Index<55>) updates(%barrier56 : !VPUIPRegMapped.Index<56>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<55>


    %d56 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier56 : !VPUIPRegMapped.Index<56>) updates(%barrier57 : !VPUIPRegMapped.Index<57>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<56>


    %d57 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier57 : !VPUIPRegMapped.Index<57>) updates(%barrier58 : !VPUIPRegMapped.Index<58>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<57>


    %d58 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier58 : !VPUIPRegMapped.Index<58>) updates(%barrier59 : !VPUIPRegMapped.Index<59>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<58>


    %d59 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier59 : !VPUIPRegMapped.Index<59>) updates(%barrier60 : !VPUIPRegMapped.Index<60>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<59>


    %d60 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier60 : !VPUIPRegMapped.Index<60>) updates(%barrier61 : !VPUIPRegMapped.Index<61>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<60>


    %d61 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier61 : !VPUIPRegMapped.Index<61>) updates(%barrier62 : !VPUIPRegMapped.Index<62>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<61>


    %d62 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier62 : !VPUIPRegMapped.Index<62>) updates(%barrier63 : !VPUIPRegMapped.Index<63>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<62>


    %d63 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier63 : !VPUIPRegMapped.Index<63>) updates(%barrier64 : !VPUIPRegMapped.Index<64>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<63>


    %d64 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x2x1000xf16>) outputs(%arg1 : memref<1x1x2x1000xf16>) waits(%barrier64 : !VPUIPRegMapped.Index<64>) updates(%barrier65 : !VPUIPRegMapped.Index<65>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<64>


    return %arg1 : memref<1x1x2x1000xf16>
}
}
