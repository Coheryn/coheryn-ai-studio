/* Coheryn-authored libm bridge for independent array arithmetic. No model graph. */
#include "scalar_math.c"
float coheryn_assessment_pow(float base, float exponent) { return powf(base, exponent); }
void coheryn_assessment_sincos(size_t count, const float * input, float * cosine, float * sine) {
    for (size_t i=0;i<count;i++) {
        const float c=cosf(input[i]);
        const float s=sinf(input[i]);
        cosine[i]=c;sine[i]=s;
    }
}
