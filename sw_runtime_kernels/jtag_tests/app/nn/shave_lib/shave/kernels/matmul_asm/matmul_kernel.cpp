// {% copyright %}
/// @file
/// @copyright All code copyright Movidius Ltd 2016, all rights reserved
///            For License Warranty see: common/license.txt
///

#include "matmul_kernel.h"

extern "C"
{

/*
    *  C += A * B
    *
    *      A is m * k
    *      B is k * n
    *      C is m * n
    *
    *      Matrixes are row-major order
    *
    *      acc: accumulator, can it be float32? what is the impact in runtime?
    *
    *      dims m, k, n: any runtime improvement if they are multiple of 4, 8, etc?
    *
    */
void matmul_c(const half *A, const half *B, half *C, int m, int k, int n,
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
                acc += A[row_out * wA + common_dim] * B[common_dim * wB + col_out];
            }
#ifdef ADD_C_AT_THE_END
            C[row_out * wC + col_out] += acc;
#else
            C[row_out * wC + col_out] = acc;
#endif
        }
    }
}


void matmul_c_opt(const half *A, const half *B, half *C, int m, int k, int n,
    int wA, int wB, int wC)
{
    int row_out;
    int col_out;
    int common_dim;
    half8 a_half8;
    half8 b_half8;
    half8 *pv_C;
    half8 acc_half8;
    const half  *pA, *pB;

    for (row_out = 0; row_out < m; row_out++)
    {
        pv_C = reinterpret_cast<half8*>(C + row_out * wC);
        for (col_out = 0; col_out < n; col_out += 8)
        {
#ifdef ADD_C_AT_THE_END
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
#else
            acc_half8 = *pv_C;
#endif

            pA = &A[row_out * wA];
            pB = &B[col_out];
            for (common_dim = 0; common_dim < k; common_dim++)
            {
                half a_tmp = *pA;
                pA++;
#ifdef __PC__
                half8 a_tmp_half8;
                a_tmp_half8[0] = a_tmp;
                a_tmp_half8[1] = a_tmp;
                a_tmp_half8[2] = a_tmp;
                a_tmp_half8[3] = a_tmp;
                a_tmp_half8[4] = a_tmp;
                a_tmp_half8[5] = a_tmp;
                a_tmp_half8[6] = a_tmp;
                a_tmp_half8[7] = a_tmp;
                a_half8 = a_tmp_half8;
#else
                a_half8 = half8(a_tmp);
#endif
                b_half8 = *reinterpret_cast<half8*>((u32)pB);
                pB += wB;
                acc_half8 = acc_half8 + a_half8 * b_half8;
            }

#ifdef ADD_C_AT_THE_END
            *pv_C = *pv_C + acc_half8;
#else
            *pv_C = acc_half8;
#endif
            pv_C++;
        }
    }
}


void gemm_ssss_nnn_c(const float *A, const float *B, float *C, int m, int k, int n,
                int wA, int wB, int wC)
{
    int row_out;
    int col_out;
    int common_dim;
    float acc;
    // unused parameters
    (void)wA;
    (void)wB;
    (void)wC;

    for (row_out = 0; row_out < m; row_out++)
    {
        for (col_out = 0; col_out < n; col_out++)
        {
            acc = C[row_out * n + col_out];
            for (common_dim = 0; common_dim < k; common_dim++)
            {
                acc += A[row_out * k + common_dim] * B[common_dim * n + col_out];
            }
            C[row_out * n + col_out] = acc;
        }
    }
}

void gemm_hhhh_nnn_c(const half *A, const half *B, half *C, int m, int k, int n,
                int wA, int wB, int wC)
{
    int row_out;
    int col_out;
    int common_dim;
    half acc;
    // unused parameters
    (void)wA;
    (void)wB;
    (void)wC;

    for (row_out = 0; row_out < m; row_out++)
    {
        for (col_out = 0; col_out < n; col_out++)
        {
            acc = C[row_out * n + col_out];
            for (common_dim = 0; common_dim < k; common_dim++)
            {
                acc += A[row_out * k + common_dim] * B[common_dim * n + col_out];
            }
            C[row_out * n + col_out] = acc;
        }
    }
}

void gevm_ssss_nnn_c( const float *A, const float *B, float *C,int K, int N){
	int vecrow = 0;
	int matcol = 0;
	float acc = 0;
	for (vecrow = 0; vecrow < N; ++vecrow) {
		acc = 0;//C[vecrow];
		for (matcol = 0; matcol < K; ++matcol) {	//K
			acc += A[matcol] * B[matcol*N + vecrow];
		}
		C[vecrow] = acc;
	}
}

void gevm_ssss_ntn_c( const float *A, const float *B, float *C,int K, int N){
	int vecrow = 0;
	int matcol = 0;
	float acc = 0;
	for (vecrow = 0; vecrow < N; ++vecrow) {
		acc = C[vecrow];
		for (matcol = 0; matcol < K; ++matcol) {	//K
			acc += A[matcol] * B[vecrow*K + matcol];
		}
		C[vecrow] = acc;
	}
}

} // extern "C"
