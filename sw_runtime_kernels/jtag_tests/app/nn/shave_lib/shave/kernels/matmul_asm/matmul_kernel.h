/*
* {% copyright %}
*/
#ifndef MATMUL_KERNEL_H
#define MATMUL_KERNEL_H
#include "mv_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void gemm_ssss_nnn(const float *A, const float *B, float *C, int m, int k, int n,
                int wA, int wB, int wC );

void gemm_hhhh_nnn(const half *A, const half *B, half *C, int m, int k, int n,
               int wA, int wB, int wC);

void gemm_ssss_nnn_c(const float *A, const float *B, float *C, int m, int k, int n,
                int wA, int wB, int wC);

void gemm_hhhh_nnn_c(const half *A, const half *B, half *C, int m, int k, int n,
                int wA, int wB, int wC);

/*
 *  C = A * B
 */
void gemm_hhhh_nnn_nac(const half *A, const half *B, half *C, int m, int k, int n,
    int wA, int wB, int wC);

/*
    * Calculates C = C + (A*B)
    *   Where A is 1*K, B is K*N and C is 1*N
    */
void gevm_ssss_nnn_c(const float *A, const float *B, float *C, int K, int N);

void gevm_ssss_ntn_c(const float *A, const float *B, float *C, int K, int N);

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
    int wA, int wB, int wC);
void matmul_c_opt(const half *A, const half *B, half *C, int m, int k, int n,
    int wA, int wB, int wC);
void gemm_hhhh_nnn_k8(const half *A, const half *B, half *C, int m, int k, int n,
    int wA, int wB, int wC);
void gemm_hhhh_nnn_k16(const half *A, const half *B, half *C, int m, int k, int n,
    int wA, int wB, int wC);
void matmul_c_ref(const float *A, const float *B, float *C, int m, int k, int n,
    int wA, int wB, int wC);


#ifdef __cplusplus
}
#endif

#endif // MATMUL_KERNEL_H
