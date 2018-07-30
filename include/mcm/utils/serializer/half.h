// -----------------------------------------------------------------------------
// Copyright (C) 2012 Movidius Ltd. All rights reserved
//
// Company          : Movidius
// Description      : half implementation
//
// -----------------------------------------------------------------------------

#ifndef __SIPP_HALF__
#define __SIPP_HALF__

#include "moviFloat32.h"

#define HALF_MAX           65504.0f

#define HALF_SIGN          (1<<(sizeof(half)*8-1))
#ifndef HANDLE_NEGZERO
#define HANDLE_NEGZERO(x)  if(EXTRACT_F16_EXP(x.getPackedValue()) == 0 && EXTRACT_F16_FRAC(x.getPackedValue()) == 0) \
                             x.setPackedValue(x.getPackedValue() ^ (x.getPackedValue() & HALF_SIGN));
#endif

// Conversion macros
#define U8F_TO_HALF(x)     half((x)*(1.0f/255))
#define HALF_TO_U8F(x)     (F16_IS_NAN((x).getPackedValue()) ? (EXTRACT_F16_SIGN((x).getPackedValue()) ? 0x0 : 0xFF) : uint8_t(floor(((x)*(255)) + 0.5f)))
#define U12F_TO_HALF(x)    half((x)*(1.0f/4095))
#define HALF_TO_U12F(x)    uint16_t(floor(((x)*(4095)) + 0.5f))
#define U8F_TO_U12F(x)     (x << 4) | (x >> 4)
#define U12F_TO_U8F(x)     (x >> 4)
#define U16F_TO_HALF(x)    half((x)/65535.f)
#define HALF_TO_U16F(x)    uint16_t(floor(((x)*(65535.)) + 0.5))

#ifdef __cplusplus
// FIXME this class is named the same as the simple typedef half
class half{
private:
    unsigned short _h;
public:
    /* Constructors */
    half();
    half(unsigned char v);
    half(char v);
    half(int v);
    half(float v);
    half(double v);

    /* Destructor */
    ~half();

    /* Auxiliary swap function for assignments */
    friend void swap(half&, half&);

    /* General purpose set and get functions */
    void  setUnpackedValue(float);
    void  setPackedValue(unsigned short);
    float getUnpackedValue(void);
    unsigned short getPackedValue(void);

    /* Binary Comparison Operators */
    bool operator == (const half);
    bool operator != (const half);
    bool operator >  (const half);
    bool operator <  (const half);
    bool operator >= (const half);
    bool operator <= (const half);
    bool operator == (const float);
    bool operator != (const float);
    bool operator >  (const float);
    bool operator <  (const float);
    bool operator >= (const float);
    bool operator <= (const float);

    /* Assignment Operators return a reference to a half object in order to allow chained assignments, such as: f1 = f2 = f3 (supported with primitive types) */
    half & operator = (half);
    half & operator = (float);
    //half & operator = (double);

    /* Compound Assignment Operators */
    half & operator += (const  half);
    half & operator -= (const  half);
    half & operator *= (const  half);
    half & operator /= (const  half);
    half & operator += (const float);
    half & operator -= (const float);
    half & operator *= (const float);
    half & operator /= (const float);

    /* Binary Arithmetic Operators
     * -There's quite a major pitfall if these operators are overloaded; namely, the built-in
     * operator for each of these arithmetic operations is overloaded and the compiler will
     * constrain the end-user to use a trailing f if the second operand is an immediate value,
     * so that it is explicitly specified it as being a float value, otherwise it will fail to
     * find the proper overloaded operator and will generate an error (unless the operator
     * is overloaded with every single primitive type). Leaving it like that just makes the
     * compiler use the built-in operator+(float, float) which will get the right parameters
     * due to the conversion operator.
     */
    //half operator + (const  half);
    //half operator - (const  half);
    //half operator * (const  half);
    //half operator / (const  half);
    //half operator + (const float);
    //half operator - (const float);
    //half operator * (const float);
    //half operator / (const float);

    /* Conversion Operator */
    operator float () const;

    /* Unary Minus Operator */
    half operator - () const;

    /* Absolute Value Operator
     * -This is redundant since fabs can be called directly with a half instance (let me know if I should remove it)
     */
    friend float halfabs(half);
};
#else
typedef int16_t half;
#endif // __cplusplus

#endif // __SIPP_HALF__
