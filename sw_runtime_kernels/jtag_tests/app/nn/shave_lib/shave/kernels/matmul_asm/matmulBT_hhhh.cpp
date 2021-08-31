// {% copyright %}
/// @file
/// @copyright All code copyright Movidius Ltd 2016, all rights reserved
///            For License Warranty see: common/license.txt
///
#include "matmul_kernel.h"
#include "matmulBT.h"

//#define ADD_C_AT_THE_END

extern "C"
{

/*
    *  C += A * B'
    *
    *      A is m * k
    *      B is n * k
    *      C is m * n
    *
    *      Matrixes are row-major order
    *
    *      acc: accumulator, can it be float32? what is the impact in runtime?
    *
    *      dims m, k, n: any runtime improvement if they are multiple of 4, 8, etc?
    *
    */
void matmulBT_hhhh_c(const half *A, const half *B, half *C, int m, int k, int n,
    int wA, int wB, int wC)
{
    int row_out;
    int col_out;
    int common_dim;
    half acc;

    for (row_out = 0; row_out < m; row_out++)
    {
        for (col_out = 0; col_out < n; col_out++)
        {
#ifdef ADD_C_AT_THE_END
            acc = 0;
#else
            acc = C[row_out * wC + col_out];
#endif
            for (common_dim = 0; common_dim < k; common_dim++)
            {
                acc += A[row_out * wA + common_dim] * B[col_out * wB + common_dim];
            }
#ifdef ADD_C_AT_THE_END
            C[row_out * wC + col_out] += acc;
#else
            C[row_out * wC + col_out] = acc;
#endif
        }
    }
}





void matmulBT_hhhh_c_opt(const half *A, const half *B, half *C, int m, int k, int n,
    int wA, int wB, int wC)
{
    int row_out;
    int col_out;
    int common_dim;
    half8 a_half8;
    half8 b_half8;
    half8 *pv_A, *pv_B;
    half8 acc_half8;
    const half  *pA, *pB;
    half  *pC;


    for (row_out = 0; row_out < m; row_out++)
    {
        pC = C + row_out * wC;
        for (col_out = 0; col_out < n; col_out ++)
        {
#ifdef __PC__
            acc_half8[0] = 0;
            acc_half8[1] = 0;
            acc_half8[2] = 0;
            acc_half8[3] = 0;
            acc_half8[4] = 0;
            acc_half8[5] = 0;
            acc_half8[6] = 0;
            acc_half8[7] = 0;
#else
            acc_half8 = half8(0);
#endif // __PC__

            pA = &A[row_out * wA];
            pB = &B[col_out * wB];
            pv_A = (half8*)(pA);
            pv_B = (half8*)(pB);
            for (common_dim = 0; common_dim < k; common_dim+=8)
            {
                a_half8 = *pv_A;
                pv_A++;
                b_half8 = *pv_B;
                pv_B++;

                acc_half8 = acc_half8 + a_half8 * b_half8;
            }

#ifdef ADD_C_AT_THE_END
            *pC += __builtin_shave_sau_sumx_f16_r(acc_half8);
#else
            *pC += __builtin_shave_sau_sumx_f16_r(acc_half8);
#endif
            pC++;
        }
    }
}


void matmulBT_hhhh_c_ref(const float *A, const float *B, float *C, int m, int k, int n,
    int wA, int wB, int wC)
{
    int row_out;
    int col_out;
    int common_dim;
    float acc;

    for (row_out = 0; row_out < m; row_out++)
    {
        for (col_out = 0; col_out < n; col_out++)
        {
#ifdef ADD_C_AT_THE_END
            acc = 0;
#else
            acc = C[row_out * wC + col_out];
#endif
            for (common_dim = 0; common_dim < k; common_dim++)
            {
                acc += A[row_out * wA + common_dim] * B[col_out * wB + common_dim];
            }
#ifdef ADD_C_AT_THE_END
            C[row_out * wC + col_out] += acc;
#else
            C[row_out * wC + col_out] = acc;
#endif
        }
    }
}

} // extern "C"

