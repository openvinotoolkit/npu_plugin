#pragma once
#include <cmath>

/// reference functions

inline float reLU(float x) {
    if(x>0) {
        return x;
    }
    return 0;
}

inline float reLU6(float x) {
    if(x>6)
        return 6;
    else if(x>0 && x<=6)
        return x;
    else
        return 0;
}

inline double leakyReLU(double x) {
    if(x>0) {
        return x;
    }
    return 0.1*x;
}

inline float sigmoid(float x) {
    return 1/(1+exp(-x));
}

inline float elu(float x) {
    if(x>0) {
        return x;
    }
    return exp(x)-1;
}

inline float swish(float x) {
    return x/(1+exp(-x));
}

inline float hswish(float x) {
    return x * (reLU6(x+3)/6);
}

inline double mish(double x) {
    return x * tanh(log(1+exp(x)));
}
