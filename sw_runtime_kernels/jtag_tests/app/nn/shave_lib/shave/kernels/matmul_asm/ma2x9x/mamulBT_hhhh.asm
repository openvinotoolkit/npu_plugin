// {% copyright %}
//
// @file
// Code for matmul_BT_hhhh kernel -> C += A * B'
// Number of instruction cycles: 43 + m/2*( 15 + n/8*( 15 + 18*k/8 ) )
// Number of stall cycles:       0
// Error from reference output: RMS = , MAX =
// Constraints:       m%2 = 0
//                    k%8 = 0
//                    n%8 = 0

.version 00.70.00

.code .text.matmulBT_hhhh_asm
.code .text
.salign 16
//------------------------------------------------------------------------------------------------------------
//void matmulBT_hhhh_asm(const half *A, const half *B, half *C, int m, int k, int n, int wA, int wB, int wC);
//                            (i18)       (i17)        (i16)   (i15)  (i14)   (i13)  (i12)   (i11)   (stack)
//------------------------------------------------------------------------------------------------------------
matmulBT_hhhh_asm:
.nowarn
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
  .set pA1          i1
  .set pB           i2
  .set pC           i3
  .set pB0          i4
  .set incr         i5
  .set row_out      i6
  .set col_out      i7
  .set loop_label   i8
  .set temp         i9

  .set temp.0       i9.0


  .set k_x_2        temp
  .set off8         temp
  .set k_by_8       k

  .set iC0          i20
  .set iC0.0        i20.0
  .set iC1          i21
  .set iC1.0        i21.0
  .set iC2          i22
  .set iC2.0        i22.0
  .set iC3          i23
  .set iC3.0        i23.0
  .set iC4          i24
  .set iC4.0        i24.0
  .set iC5          i25
  .set iC5.0        i25.0
  .set iC6          i26
  .set iC6.0        i26.0
  .set iC7          i27
  .set iC7.0        i27.0
  .set incr_pB0     i28
  .set pC0          i29
  .set pC1          i30


  // ------ VRF usage ------
  .set vB0           v0
  .set vB1           v1
  .set vB2           v2
  .set vB3           v3
  .set vB4           v4
  .set vB5           v5
  .set vB6           v6
  .set vB7           v7
  .set vC0           v8
  .set vC1           v9

  .set vB0.0           v0.0
  .set vB1.0           v1.0
  .set vB2.0           v2.0
  .set vB3.0           v3.0
  .set vB4.0           v4.0
  .set vB5.0           v5.0
  .set vB6.0           v6.0
  .set vB7.0           v7.0
  .set vC0.0           v8.0
  .set vC1.0           v9.0

  .set vB0.1           v0.1
  .set vB1.1           v1.1
  .set vB2.1           v2.1
  .set vB3.1           v3.1
  .set vB4.1           v4.1
  .set vB5.1           v5.1
  .set vB6.1           v6.1
  .set vB7.1           v7.1
  .set vC0.1           v8.1
  .set vC1.1           v9.1


  .set vC0_temp      v10
  .set vC0_temp.0    v10.0
  .set vC0_temp.1    v10.1
  .set vC0_temp.2    v10.2
  .set vC0_temp.3    v10.3
  .set vC0_temp.4    v10.4
  .set vC0_temp.5    v10.5
  .set vC0_temp.6    v10.6
  .set vC0_temp.7    v10.7
  .set vC1_temp      v11
  .set vC1_temp.0    v11.0
  .set vC1_temp.1    v11.1
  .set vC1_temp.2    v11.2
  .set vC1_temp.3    v11.3
  .set vC1_temp.4    v11.4
  .set vC1_temp.5    v11.5
  .set vC1_temp.6    v11.6
  .set vC1_temp.7    v11.7
  .set vB0res        v12
  .set vB1res        v13
  .set vB2res        v14
  .set vB3res        v15
  .set vB4res        v16
  .set vB5res        v17
  .set vB6res        v18
  .set vB7res        v19
  .set vA0           v20
  .set vA1           v21
  .set vMIN          v22
  .set vMAX          v23

  .set vA0.0           v20.0
  .set vA1.0           v21.0

  .set vA0.1           v20.1
  .set vA1.1           v21.1


  .set PLUS_INF      0x7C00
  .set MINUS_INF     0xFC00

  // save IRF registers to stack
  LSU0.LD.32  wC, i19  || IAU.SUB i19 i19 4
  LSU0.ST.32  i20  i19 || IAU.SUB i19 i19 4
  LSU0.ST.32  i21  i19 || IAU.SUB i19 i19 4
  LSU0.ST.32  i22  i19 || IAU.SUB i19 i19 4
  LSU0.ST.32  i23  i19 || IAU.SUB i19 i19 4
  LSU0.ST.32  i24  i19 || IAU.SUB i19 i19 4
  LSU0.ST.32  i25  i19 || IAU.SUB i19 i19 4
  LSU0.ST.32  i26  i19 || IAU.SUB i19 i19 4
  LSU0.ST.32  i27  i19 || IAU.SUB i19 i19 4
  LSU0.ST.32  i28  i19 || IAU.SUB i19 i19 4
  LSU0.ST.32  i29  i19 || IAU.SUB i19 i19 4
  LSU0.ST.32  i30  i19 || IAU.SUB i19 i19 4
  LSU0.ST.32  i31  i19 || IAU.SHL wA, wA, 1

  IAU.SHL wB, wB, 1    || LSU0.LDIL temp, MINUS_INF
  IAU.SHL wC, wC, 1    || LSU0.LDIL temp, PLUS_INF || CMU.CP.128.16.r vMIN, temp.0
  IAU.SHL k_x_2, k, 1  || CMU.CP.128.16.r vMAX, temp.0
  IAU.SHR.u32 k_by_8, k, 3

  // pB0 increment = 8*2*wB - 2*k - 16
  IAU.SHL incr_pB0, wB, 3
  IAU.SUB incr_pB0, incr_pB0, k_x_2
  IAU.SUB incr_pB0, incr_pB0, 16 || LSU0.LDIL off8, 8 || LSU1.LDIH off8, 0

  IAU.XOR row_out, row_out, row_out                    || LSU0.LDIL incr, 16 || LSU1.LDIH incr, 0
  CMU.CP.32 pB0, B || IAU.XOR col_out, col_out, col_out || LSU0.LDIL loop_label, __matmulBT_hhhh_loop || LSU1.LDIH loop_label, __matmulBT_hhhh_loop
  CMU.CP.32 pA0, A || IAU.ADD pA1, A, wA                || LSU0.LD.64 vC0.0,  C    || LSU1.LDO.64 vC0.1,  C , 8
  CMU.CP.32 pC0, C || IAU.ADD pC1, C, wC                || LSU0.LD.64 vC1.0, pC1   || LSU1.LDO.64 vC1.1, pC1, 8
  LSU0.LDI.64 vA0.0, pA0, incr || LSU1.LDO.64 vA0.1, pA0, 8 || CMU.CP.32 pB, pB0
  LSU0.LD.64 vB0.0, pB         || LSU1.LDO.64 vB0.1, pB, 8 || IAU.ADD pB, pB, wB || SAU.XOR iC3, temp, temp
  LSU0.LD.64 vB1.0, pB         || LSU1.LDO.64 vB1.1, pB, 8 || IAU.ADD pB, pB, wB || SAU.XOR iC4, temp, temp
  LSU0.LD.64 vB2.0, pB         || LSU1.LDO.64 vB2.1, pB, 8 || IAU.ADD pB, pB, wB || SAU.XOR iC5, temp, temp
  LSU0.LD.64 vB3.0, pB         || LSU1.LDO.64 vB3.1, pB, 8 || IAU.ADD pB, pB, wB || SAU.XOR iC6, temp, temp
  LSU0.LD.64 vB4.0, pB         || LSU1.LDO.64 vB4.1, pB, 8 || IAU.ADD pB, pB, wB || SAU.XOR iC7, temp, temp
  LSU0.LD.64 vB5.0, pB         || LSU1.LDO.64 vB5.1, pB, 8 || IAU.ADD pB, pB, wB || CMU.CPZV vC1_temp, 0
  LSU0.LD.64 vB6.0, pB         || LSU1.LDO.64 vB6.1, pB, 8 || IAU.ADD pB, pB, wB
  LSU0.LD.64 vB7.0, pB         || LSU1.LDO.64 vB7.1, pB, 8 || IAU.ADD pB, pB, wB
  VAU.MUL.f16 vB0res, vB0, vA0 || LSU0.LDI.64 vA1.0, pA1, incr || LSU1.LDO.64 vA1.1, pA1, 8
  VAU.MUL.f16 vB1res, vB1, vA0 || LSU0.LD.64 vC1.0, pC1    || LSU1.LDO.64 vC1.1, pC1, 8 || IAU.ADD pB, pB0, 16
  VAU.MUL.f16 vB2res, vB2, vA0 || LSU0.LD.64 vC0.0, pC0    || LSU1.LDO.64 vC0.1, pC0, 8 || IAU.ADD pB0, pB0, 16


.lalign
_matmulBT_hhhh_row_out:
//    for (row_out = 0; row_out < m; row_out+=2)

.lalign
_matmulBT_hhhh_col_out:
//for (col_out = 0; col_out < n; col_out +=8)

// for (common_dim = 0 .. k/8)

//__matmulBT_hhhh_loop:
  VAU.MUL.f16 vB3res, vB3, vA0  || SAU.SUMX.f16 iC0, vB0res || CMU.CP.16 vC1_temp.3, iC3.0 || BRU.RPL loop_label, k_by_8
  VAU.MUL.f16 vB4res, vB4, vA0  || SAU.SUMX.f16 iC1, vB1res || CMU.CP.16 vC1_temp.4, iC4.0
  VAU.MUL.f16 vB5res, vB5, vA0  || SAU.SUMX.f16 iC2, vB2res || CMU.CP.16 vC1_temp.5, iC5.0
  VAU.MUL.f16 vB6res, vB6, vA0  || SAU.SUMX.f16 iC3, vB3res || CMU.CP.16 vC1_temp.6, iC6.0
  VAU.MUL.f16 vB7res, vB7, vA0  || SAU.SUMX.f16 iC4, vB4res || CMU.CP.16 vC1_temp.7, iC7.0
  VAU.ADD.F16 vC1, vC1, vC1_temp
  VAU.MUL.f16 vB0res, vB0, vA1  || SAU.SUMX.f16 iC5, vB5res || CMU.CP.16 vC0_temp.0, iC0.0   || LSU0.LDI.64 vA0.0, pA0, incr || LSU1.LDO.64 vA0.1, pA0, 8
  VAU.MUL.f16 vB1res, vB1, vA1  || SAU.SUMX.f16 iC6, vB6res || CMU.CP.16 vC0_temp.1, iC1.0   || LSU0.LD.64 vB0.0, pB         || LSU1.LDO.64 vB0.1, pB, 8 || IAU.ADD pB, pB, wB
  VAU.MUL.f16 vB2res, vB2, vA1  || SAU.SUMX.f16 iC7, vB7res || CMU.CP.16 vC0_temp.2, iC2.0   || LSU0.LD.64 vB1.0, pB         || LSU1.LDO.64 vB1.1, pB, 8 || IAU.ADD pB, pB, wB
  VAU.MUL.f16 vB3res, vB3, vA1  || SAU.SUMX.f16 iC0, vB0res || CMU.CP.16 vC0_temp.3, iC3.0   || LSU0.LD.64 vB2.0, pB         || LSU1.LDO.64 vB2.1, pB, 8 || IAU.ADD pB, pB, wB
  VAU.MUL.f16 vB4res, vB4, vA1  || SAU.SUMX.f16 iC1, vB1res || CMU.CP.16 vC0_temp.4, iC4.0   || LSU0.LD.64 vB3.0, pB         || LSU1.LDO.64 vB3.1, pB, 8 || IAU.ADD pB, pB, wB
__matmulBT_hhhh_loop:
  VAU.MUL.f16 vB5res, vB5, vA1  || SAU.SUMX.f16 iC2, vB2res || CMU.CP.16 vC0_temp.5, iC5.0   || LSU0.LD.64 vB4.0, pB         || LSU1.LDO.64 vB4.1, pB, 8 || IAU.ADD pB, pB, wB
  VAU.MUL.f16 vB6res, vB6, vA1  || SAU.SUMX.f16 iC3, vB3res || CMU.CP.16 vC0_temp.6, iC6.0   || LSU0.LD.64 vB5.0, pB         || LSU1.LDO.64 vB5.1, pB, 8 || IAU.ADD pB, pB, wB
  VAU.MUL.f16 vB7res, vB7, vA1  || SAU.SUMX.f16 iC4, vB4res || CMU.CP.16 vC0_temp.7, iC7.0   || LSU0.LD.64 vB6.0, pB         || LSU1.LDO.64 vB6.1, pB, 8 || IAU.ADD pB, pB, wB
  VAU.ADD.F16 vC0, vC0, vC0_temp                                                                || LSU0.LD.64 vB7.0, pB         || LSU1.LDO.64 vB7.1, pB, 8 || IAU.ADD pB, pB, wB
  VAU.MUL.f16 vB0res, vB0, vA0  || SAU.SUMX.f16 iC5, vB5res || CMU.CP.16 vC1_temp.0, iC0.0   || LSU0.LDI.64 vA1.0, pA1, incr || LSU1.LDO.64 vA1.1, pA1, 8
  VAU.MUL.f16 vB1res, vB1, vA0  || SAU.SUMX.f16 iC6, vB6res || CMU.CP.16 vC1_temp.1, iC1.0   || IAU.ADD pB, pB0, 16
  VAU.MUL.f16 vB2res, vB2, vA0  || SAU.SUMX.f16 iC7, vB7res || CMU.CP.16 vC1_temp.2, iC2.0   || IAU.ADD pB0, pB0, 16
//~__matmulBT_hhhh_loop

  SAU.XOR iC3, temp, temp        || CMU.CP.16 vC1_temp.3, iC3.0  || LSU0.LD.64 vA0.0, A          || LSU1.LDO.64 vA0.1, A, 8   || IAU.ADD col_out, col_out, 8
                                    CMU.CMII.I32  col_out, n                                                                     || IAU.ADD pB, pB0, incr_pB0
                                    CMU.CP.16 vC1_temp.4, iC4.0  || LSU0.LD.64 vB0.0, pB         || LSU1.LDO.64 vB0.1, pB, 8  || IAU.ADD pB, pB, wB
                                    CMU.CP.16 vC1_temp.5, iC5.0  || LSU0.LD.64 vB1.0, pB         || LSU1.LDO.64 vB1.1, pB, 8  || IAU.ADD pB, pB, wB
  SAU.XOR iC4, temp, temp        || CMU.CP.16 vC1_temp.6, iC6.0  || LSU0.LD.64 vB2.0, pB         || LSU1.LDO.64 vB2.1, pB, 8  || IAU.ADD pB, pB, wB
  SAU.XOR iC5, temp, temp        || CMU.CP.16 vC1_temp.7, iC7.0  || LSU0.LD.64 vB3.0, pB         || LSU1.LDO.64 vB3.1, pB, 8  || IAU.ADD pB, pB, wB
  VAU.ADD.F16 vC1, vC1, vC1_temp || CMU.CPZV vC1_temp, 0            || LSU0.LD.64 vB4.0, pB         || LSU1.LDO.64 vB4.1, pB, 8  || IAU.ADD pB, pB, wB
  SAU.XOR iC6, temp, temp        || CMU.CLAMPAB.F16 vC0, vMIN, vMAX || LSU0.LD.64 vB5.0, pB         || LSU1.LDO.64 vB5.1, pB, 8  || IAU.ADD pB, pB, wB
                                                                       LSU0.LD.64 vB6.0, pB         || LSU1.LDO.64 vB6.1, pB, 8  || IAU.ADD pB, pB, wB || PEU.PCCX NEQ 0x00 || BRU.BRA _matmulBT_hhhh_col_out
  SAU.XOR iC7, temp, temp        || CMU.CLAMPAB.F16 vC1, vMIN, vMAX || LSU0.LD.64 vB7.0, pB         || LSU1.LDO.64 vB7.1, pB, 8  || IAU.ADD pA1, A, wA
  VAU.MUL.f16 vB0res, vB0, vA0                                      || LSU0.STI.64 vC0.0, pC0, incr || LSU1.STO.64 vC0.1, pC0, 8 || IAU.ADD pA0, A, incr
  VAU.MUL.f16 vB1res, vB1, vA0                                      || LSU0.STI.64 vC1.0, pC1, incr || LSU1.STO.64 vC1.1, pC1, 8
  VAU.MUL.f16 vB2res, vB2, vA0                                      || LSU0.LD.64 vC0.0, pC0        || LSU1.LDO.64 vC0.1, pC0, 8 || IAU.ADD pB0, pB0, incr_pB0
                                                                       LSU0.LD.64 vC1.0, pC1        || LSU1.LDO.64 vC1.1, pC1, 8 || IAU.ADD pB, pB0, 16
                                                                       LSU0.LDI.64 vA1.0, pA1, incr || LSU1.LDO.64 vA1.1, pA1, 8 || IAU.ADD pB0, pB0, 16

//_matmulBT_hhhh_col_out

  IAU.ADD row_out, row_out, 2
  IAU.ADD A, A, wA         || SAU.ADD.I32 C, C, wC || CMU.CMII.I32 row_out, m
  IAU.ADD pA0, A, wA
  LSU0.LDI.64 vA0.0, pA0, incr || LSU1.LDO.64 vA0.1, pA0, 8|| IAU.ADD C, C, wC   || SAU.XOR iC3, temp, temp || CMU.CP.32 pB, B
  LSU0.LD.64 vB0.0, pB         || LSU1.LDO.64 vB0.1, pB, 8 || IAU.ADD pB, pB, wB || SAU.XOR iC4, temp, temp || CMU.CP.32 A, pA0
  LSU0.LD.64 vB1.0, pB         || LSU1.LDO.64 vB1.1, pB, 8 || IAU.ADD pB, pB, wB || SAU.XOR iC5, temp, temp || CMU.CP.32 pC0, C
  LSU0.LD.64 vB2.0, pB         || LSU1.LDO.64 vB2.1, pB, 8 || IAU.ADD pB, pB, wB || SAU.XOR iC6, temp, temp
  LSU0.LD.64 vB3.0, pB         || LSU1.LDO.64 vB3.1, pB, 8 || IAU.ADD pB, pB, wB || SAU.XOR iC7, temp, temp
  LSU0.LD.64 vB4.0, pB         || LSU1.LDO.64 vB4.1, pB, 8 || IAU.ADD pB, pB, wB || SAU.ADD.I32 pA1, A, wA  || PEU.PC1C LT  || BRU.BRA _matmulBT_hhhh_row_out
  LSU0.LD.64 vB5.0, pB         || LSU1.LDO.64 vB5.1, pB, 8 || IAU.ADD pB, pB, wB
  LSU0.LD.64 vB6.0, pB         || LSU1.LDO.64 vB6.1, pB, 8 || IAU.ADD pB, pB, wB                            || CMU.CPZV vC1_temp, 0
  LSU0.LD.64 vB7.0, pB         || LSU1.LDO.64 vB7.1, pB, 8 || IAU.XOR col_out, col_out, col_out || SAU.ADD.I32 pC1, C, wC
  LSU0.LDI.64 vA1.0, pA1, incr || LSU1.LDO.64 vA1.1, pA1, 8                      || VAU.MUL.f16 vB0res, vB0, vA0
  LSU0.LD.64 vC1.0, pC1    || LSU1.LDO.64 vC1.1, pC1, 8    || IAU.ADD pB, B, 16  || VAU.MUL.f16 vB1res, vB1, vA0
  LSU0.LD.64 vC0.0, pC0    || LSU1.LDO.64 vC0.1, pC0, 8    || IAU.ADD pB0, B, 16 || VAU.MUL.f16 vB2res, vB2, vA0

// _matmulBT_hhhh_row_out
.nowarnend

// restore used IRF registers from stack
  LSU0.LD.32  i31  i19 || IAU.ADD i19 i19 4
  LSU0.LD.32  i30  i19 || IAU.ADD i19 i19 4
  LSU0.LD.32  i29  i19 || IAU.ADD i19 i19 4
  LSU0.LD.32  i28  i19 || IAU.ADD i19 i19 4
  LSU0.LD.32  i27  i19 || IAU.ADD i19 i19 4
  LSU0.LD.32  i26  i19 || IAU.ADD i19 i19 4
  LSU0.LD.32  i25  i19 || IAU.ADD i19 i19 4
  LSU0.LD.32  i24  i19 || IAU.ADD i19 i19 4
  LSU0.LD.32  i23  i19 || IAU.ADD i19 i19 4
  LSU0.LD.32  i22  i19 || IAU.ADD i19 i19 4
  LSU0.LD.32  i21  i19 || IAU.ADD i19 i19 4
  LSU0.LD.32  i20  i19 || IAU.ADD i19 i19 4
  BRU.JMP i30
  NOP 6

