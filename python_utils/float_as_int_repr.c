#include "stdint.h"

uint32_t float_as_int_repr(float x) {
    uint32_t y;
    y = *((int32_t*)(&x));
    return y;
}

float float_from_int_repr(uint32_t x) {
    float y;
    y = *((float*)(&x));
    return y;
}
