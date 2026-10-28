/* Coheryn-authored standard-libm bridge for independent numerical assessment.
 * Contains no graph, weights, inference runtime or copied reference implementation. */
#include <math.h>
#include <stddef.h>
void coheryn_assessment_exp(size_t count, const float * input, float * output) {
    for (size_t i=0;i<count;i++) output[i]=expf(input[i]);
}
void coheryn_assessment_tanh(size_t count, const float * input, float * output) {
    for (size_t i=0;i<count;i++) output[i]=tanhf(input[i]);
}
