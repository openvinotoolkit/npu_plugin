/*
* {% copyright %}
*/
#ifndef MATMULBT_H
#define MATMULBT_H
#include "mv_types.h"

#ifdef __PC__
#include "half.h"
#include "VectorTypes.h"
#include "builtinFunctions.h"
#else
#include <moviVectorUtils.h>
#endif // __PC__

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
 */
void matmulBT_hhhh_c(const half *A, const half *B, half *C, int m, int k, int n,
    int wA, int wB, int wC);
void matmulBT_hhhh_c_opt(const half *A, const half *B, half *C, int m, int k, int n,
    int wA, int wB, int wC);
void matmulBT_hhhh_asm(const half *A, const half *B, half *C, int m, int k, int n,
    int wA, int wB, int wC);
void matmulBT_hhhh_c_ref(const float *A, const float *B, float *C, int m, int k, int n,
    int wA, int wB, int wC);

} // extern "C"


#endif // MATMULBT_H
