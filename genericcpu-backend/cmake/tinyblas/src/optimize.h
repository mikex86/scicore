#pragma once

#define unary_op_hook_optimizations(op_name, type, optimizations, fallback_impl) \
template<>                                                                       \
void tblas_##op_name(const type *in, type *out, size_t nElements) {              \
    optimizations;                                                              \
    fallback_impl;                                                              \
}

#define binary_op_hook_optimizations(op_name, type, optimizations, fallback_impl)\
template<>\
void tblas_##op_name(const type *a, const type *b, type *c,\
                    const size_t *shapeA, const size_t *stridesA, size_t nDimsA,\
                    const size_t *shapeB, const size_t *stridesB, size_t nDimsB,\
                    const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {\
\
    /* check for possible optimizations */ \
    optimizations\
    fallback_impl\
}

