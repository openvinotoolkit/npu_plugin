// {% copyright %}
//
// @file
// Code for matmul kernel (C += A * B)
// Number of instruction cycles: 22 + m*( 12 + n/8* ( 8 + 9*k/8 ) )
// Number of stall cycles:       0
// Error from reference output: RMS = , MAX =
// Constraints:
//         n%8 = 0, n>=8
//         k = 16
//

.version 00.70.00

.code .text
.salign 16
//------------------------------------------------------------------------------------------------------------
//void gemm_hhhh_nnn_k16(const half *A, const half *B, half *C, int m, int k, int n, int wA, int wB, int wC)//
//                        (i18)       (i17)          (i16)    (i15) (i14)  (i13)   (i12)   (i11)   (stack)
//------------------------------------------------------------------------------------------------------------
gemm_hhhh_nnn_k16:
.nowarn
  // ------ CNST usage ------
  .set PLUS_INF_FP16  0x7C00
  .set MINUS_INF_FP16 0xFC00

  // ------ IRF usage ------
  .set A            i18
  .set B            i17
  .set C            i16
  .set m            i15
  .set k            i14
  .set n            i13
  .set wA           i12
  .set wB           i11
  .set wC           i10

  .set pA0          i0
  .set pB0          i1
  .set pC           i2  // points to the next line of C
  .set pB1          i3
  .set incr         i4
  .set temp         i5
  .set temp.0         i5.0

  .set n_x_2        n
  .set row_out      i6
  .set col_out      i7
  .set min_fp16     temp
  .set min_fp16.0     temp.0
  .set max_fp16     i8
  .set max_fp16.0     i8.0
  .set wB_goback    i9

  // ------ VRF usage ------
  .set c_half8      v0
  .set half8_out    v1
  .set zero         v2
  .set a0           v3
  .set a1           v4
  .set a2           v5
  .set a3           v6
  .set a4           v7
  .set a5           v8
  .set a6           v9
  .set a7           v10
  .set b0           v11
  .set b1           v12
  .set b2           v13
  .set b3           v14
  .set b4           v15
  .set b5           v16
  .set b6           v17
  .set b7           v18
  .set v_min_fp16   v19
  .set v_max_fp16   v20
  .set aux1   v22
  .set a_rd   v21
  .set a_rd.0   v21.0
  .set a_rd.1   v21.1

  .set c_half8.0      v0.0
  .set half8_out.0    v1.0
  .set zero.0         v2.0
  .set a0.0           v3.0
  .set a1.0           v4.0
  .set a2.0           v5.0
  .set a3.0           v6.0
  .set a4.0           v7.0
  .set a5.0           v8.0
  .set a6.0           v9.0
  .set a7.0           v10.0
  .set b0.0           v11.0
  .set b1.0           v12.0
  .set b2.0           v13.0
  .set b3.0           v14.0
  .set b4.0           v15.0
  .set b5.0           v16.0
  .set b6.0           v17.0
  .set b7.0           v18.0
  .set v_min_fp16.0   v19.0
  .set v_max_fp16.0   v20.0

  .set c_half8.1      v0.1
  .set half8_out.1    v1.1
  .set zero.1         v2.1
  .set a0.1           v3.1
  .set a1.1           v4.1
  .set a2.1           v5.1
  .set a3.1           v6.1
  .set a4.1           v7.1
  .set a5.1           v8.1
  .set a6.1           v9.1
  .set a7.1           v10.1
  .set b0.1           v11.1
  .set b1.1           v12.1
  .set b2.1           v13.1
  .set b3.1           v14.1
  .set b4.1           v15.1
  .set b5.1           v16.1
  .set b6.1           v17.1
  .set b7.1           v18.1
  .set v_min_fp16.1   v19.1
  .set v_max_fp16.1   v20.1

  lsu0.ldil i0 0x3c00 || lsu1.ldih i0 0x3c00 || vau.xor v21 v21 v21
  cmu.cp.128.32.r v23 i0
  // wB_goback = 16 - 15*wB
  LSU0.LDIL wB_goback, 16
  LSU0.LDIL max_fp16, PLUS_INF_FP16 || LSU1.LDIL min_fp16, MINUS_INF_FP16 || CMU.CP.32 pC, C
  LSU0.LDIL incr, 16            || LSU1.LD.32 wC, i19           || IAU.XOR row_out, row_out, row_out || CMU.CPZV zero, 0
  LSU0.LDI.64 c_half8.0, C, incr       || LSU1.LDO.64 c_half8.1, C, 8  || IAU.SHL wB, wB, 1          || CMU.CP.128.16.r v_min_fp16, min_fp16.0
  LSU0.LD.64 a_rd.0, A          || LSU1.LDO.64 a_rd.1, A, 8     || IAU.ADDSU pB1, B, 8               || CMU.CP.32 pB0, B
  LSU0.LDI.64 b0.0, pB0, wB     || LSU1.LDI.64 b0.1, pB1, wB    || IAU.ADDSU pA0, A, incr            || SAU.ADD.I32 wB_goback, wB_goback, wB
  LSU0.LDI.64 b1.0, pB0, wB     || LSU1.LDI.64 b1.1, pB1, wB    || IAU.ADD col_out, incr, 0          || SAU.SHL.X32 temp, wB, 4
  LSU0.LDI.64 b2.0, pB0, wB     || LSU1.LDI.64 b2.1, pB1, wB    || IAU.SHL wA, wA, 1
  LSU0.LDI.64 b3.0, pB0, wB     || LSU1.LDI.64 b3.1, pB1, wB    || IAU.ADD row_out, row_out, 1
  LSU0.LDI.64 b4.0, pB0, wB     || LSU1.LDI.64 b4.1, pB1, wB    || IAU.SHL wC, wC, 1
  LSU0.LDI.64 b5.0, pB0, wB     || LSU1.LDI.64 b5.1, pB1, wB    || IAU.SUB wB_goback, wB_goback, temp
  LSU0.LDI.64 b6.0, pB0, wB     || LSU1.LDI.64 b6.1, pB1, wB    || CMU.VILV.X16 a0, a4, a_rd, a_rd || IAU.ADD pC, pC, wC
  LSU0.LDI.64 b7.0, pB0, wB     || LSU1.LDI.64 b7.1, pB1, wB    || CMU.VILV.X32 a0, a2, a0, a0 || IAU.SHL n_x_2, n, 1
  LSU0.LDO.64 a_rd.0, A, 16     || LSU1.LDO.64 a_rd.1, A, 24    || CMU.VILV.X32 a0, a1, a0, a0                 || VAU.MACPZ.F16 zero  v23 V_ACC1
  LSU1.LDIH wB_goback, 0                                        || CMU.CP.128.16.r v_max_fp16, max_fp16.0      || VAU.MACPZ.F16 c_half8  v23 V_ACC0


// _matmul_hhhh_k16_asm_loop labels (1) _matmul_hhhh_k16_asm_loop_col_out, (2) _matmul_hhhh_k16_asm_loop_common_dim and (3)_matmul_hhhh_k16_asm_loop_row_out
.lalign
_matmul_hhhh_k16_asm_loop:
  LSU0.LDI.64 b0.0, pB0, wB    || LSU1.LDI.64 b0.1, pB1, wB || CMU.VILV.X32 a2, a3, a2, a2                        || VAU.MACP.F16 a0, b0 V_ACC1
  LSU0.LDI.64 b1.0, pB0, wB    || LSU1.LDI.64 b1.1, pB1, wB || CMU.VILV.X32 a4, a6, a4, a4                        || VAU.MACP.F16 a1, b1 V_ACC0
  LSU0.LDI.64 b2.0, pB0, wB    || LSU1.LDI.64 b2.1, pB1, wB || CMU.VILV.X32 a4, a5, a4, a4                        || VAU.MACP.F16 a2, b2 V_ACC1
  LSU0.LDI.64 b3.0, pB0, wB    || LSU1.LDI.64 b3.1, pB1, wB || CMU.VILV.X32 a6, a7, a6, a6                        || VAU.MACP.F16 a3, b3 V_ACC0
  LSU0.LDI.64 b4.0, pB0, wB    || LSU1.LDI.64 b4.1, pB1, wB                                                       || VAU.MACP.F16 a4, b4 V_ACC1
  LSU0.LDI.64 b5.0, pB0, wB    || LSU1.LDI.64 b5.1, pB1, wB                                                       || VAU.MACP.F16 a5, b5 V_ACC0
  LSU0.LDI.64 b6.0, pB0, wB    || LSU1.LDI.64 b6.1, pB1, wB || CMU.VILV.X16 a0, a4, a_rd, a_rd                    || VAU.MACP.F16 a6, b6 V_ACC1
  LSU0.LDI.64 b7.0, pB0, wB_goback || LSU1.LDI.64 b7.1, pB1, wB_goback || CMU.VILV.X32 a0, a2, a0, a0             || VAU.MACP.F16 a7, b7 V_ACC0
  LSU0.LD.64 a_rd.0, A         || LSU1.LDO.64 a_rd.1, A, 8  || CMU.VILV.X32 a0, a1, a0, a0                        || VAU.MACP.F16 zero, zero V_ACC1

  LSU0.LDI.64 b0.0, pB0, wB    || LSU1.LDI.64 b0.1, pB1, wB || CMU.VILV.X32 a2, a3, a2, a2                        || VAU.MACP.F16 a0, b0 V_ACC0
  LSU0.LDI.64 b1.0, pB0, wB    || LSU1.LDI.64 b1.1, pB1, wB || CMU.VILV.X32 a4, a6, a4, a4                        || VAU.MACP.F16 a1, b1 V_ACC1
  LSU0.LDI.64 b2.0, pB0, wB    || LSU1.LDI.64 b2.1, pB1, wB || CMU.VILV.X32 a4, a5, a4, a4                        || VAU.MACP.F16 a2, b2 V_ACC0
  LSU0.LDI.64 b3.0, pB0, wB    || LSU1.LDI.64 b3.1, pB1, wB || CMU.VILV.X32 a6, a7, a6, a6                        || VAU.MACP.F16 a3, b3 V_ACC1
  LSU0.LDI.64 b4.0, pB0, wB    || LSU1.LDI.64 b4.1, pB1, wB                                                       || VAU.MACP.F16 a4, b4 V_ACC0
  LSU0.LDI.64 b5.0, pB0, wB    || LSU1.LDI.64 b5.1, pB1, wB                                                       || VAU.MACP.F16 a5, b5 V_ACC1
  LSU0.LDI.64 b6.0, pB0, wB    || LSU1.LDI.64 b6.1, pB1, wB || CMU.VILV.X16 a0, a4, a_rd, a_rd                    || VAU.MACP.F16 a6, b6 V_ACC0
  LSU0.LDI.64 b7.0, pB0, wB    || LSU1.LDI.64 b7.1, pB1, wB || CMU.VILV.X32 a0, a2, a0, a0                        || VAU.MACPw.F16 aux1 a7, b7 V_ACC1
  LSU0.LDO.64 a_rd.0, A, 16    || LSU1.LDO.64 a_rd.1, A, 24 || CMU.VILV.X32 a0, a1, a0, a0                        || VAU.MACPW.F16 half8_out, zero, zero V_ACC0

// ~_matmul_hhhh_k16_asm_loop_common_dim

  LSU0.LDI.64 c_half8.0, C, incr  || LSU1.LDO.64 c_half8.1, C, 8   || IAU.SUBSU  temp, n_x_2, col_out || CMU.CLAMPAB.F16 half8_out, v_min_fp16, v_max_fp16
                                                                      IAU.SUB temp, col_out, incr     || PEU.PCIX NEQ 0x00 || BRU.BRA _matmul_hhhh_k16_asm_loop
  LSU0.STO.64 half8_out.0, C, -48 || LSU1.STO.64 half8_out.1, C, -40 || IAU.ADD col_out, col_out, 16  || PEU.PCIX NEQ 0x30
  NOP 2
  vau.add.f16 half8_out half8_out aux1
  VAU.MACPZ.F16 zero  v23 V_ACC1
  VAU.MACPZ.F16 c_half8  v23 V_ACC0


// ~_matmul_hhhh_k16_asm_loop_col_out

  LSU0.LDI.64 c_half8.0, pC, wC || LSU1.LDO.64 c_half8.1, pC, 8      || IAU.ADDSU A, A, wA
  LSU0.LD.64 a_rd.0, A          || LSU1.LDO.64 a_rd.1, A, 8          || IAU.ADDSU pB1, B, 8 || CMU.CP.32 pB0, B
  LSU0.LDI.64 b0.0, pB0, wB     || LSU1.LDI.64 b0.1, pB1, wB         || IAU.ADD col_out, incr, 0
  LSU0.LDI.64 b1.0, pB0, wB     || LSU1.LDI.64 b1.1, pB1, wB         || CMU.CLAMPAB.F16 half8_out, v_min_fp16, v_max_fp16
  LSU0.LDI.64 b2.0, pB0, wB     || LSU1.LDI.64 b2.1, pB1, wB         || IAU.SUBSU temp, m, row_out
  LSU0.LDI.64 b3.0, pB0, wB     || LSU1.LDI.64 b3.1, pB1, wB         || IAU.ADDSU pA0, A, incr  || BRU.BRA _matmul_hhhh_k16_asm_loop || PEU.PCIX NEQ 0x0
  LSU0.LDI.64 b4.0, pB0, wB     || LSU1.LDI.64 b4.1, pB1, wB
  LSU0.LDI.64 b5.0, pB0, wB     || LSU1.LDI.64 b5.1, pB1, wB
  LSU0.LDI.64 b6.0, pB0, wB     || LSU1.LDI.64 b6.1, pB1, wB         || CMU.VILV.X16 a0, a4, a_rd, a_rd
  LSU0.LDI.64 b7.0, pB0, wB     || LSU1.LDI.64 b7.1, pB1, wB         || CMU.VILV.X32 a0, a2, a0, a0 || IAU.ADD row_out, row_out, 1
  LSU0.STO.64 half8_out.0, C, -32 || LSU1.STO.64 half8_out.1, C, -24 || CMU.VILV.X32 a0, a1, a0, a0 || IAU.SUBSU C, pC, wC || VAU.MACPZ.F16 zero  v23 V_ACC1
  LSU0.LDO.64 a_rd.0, A, 16     || LSU1.LDO.64 a_rd.1, A, 24         || IAU.ADD C, C, incr || VAU.MACPZ.F16 c_half8  v23 V_ACC0
// _matmul_hhhh_k16_asm_loop_row_out
.nowarnend

  BRU.JMP i30
  NOP 6
