#pragma once

#include <cstdint>
#include <cstddef>

// Unary operations
#define UNARY_OP_FOR_ALL_INPLACE_TYPES_PROTO(op_name) \
void tblas_tensor_b##op_name##_inplace(int8_t *A, size_t n); \
void tblas_tensor_s##op_name##_inplace(int16_t *A, size_t n);\
void tblas_tensor_i##op_name##_inplace(int32_t *A, size_t n);\
void tblas_tensor_l##op_name##_inplace(int64_t *A, size_t n);\
void tblas_tensor_f##op_name##_inplace(float *A, size_t n);  \
void tblas_tensor_d##op_name##_inplace(double *A, size_t n); \

#define UNARY_OP_FOR_ALL_TYPES_PROTO(op_name) \
void tblas_tensor_b##op_name(const int8_t *A, int8_t *B, size_t n); \
void tblas_tensor_s##op_name(const int16_t *A, int16_t *B, size_t n); \
void tblas_tensor_i##op_name(const int32_t *A, int32_t *B, size_t n); \
void tblas_tensor_l##op_name(const int64_t *A, int64_t *B, size_t n); \
void tblas_tensor_f##op_name(const float *A, float *B, size_t n);   \
void tblas_tensor_d##op_name(const double *A, double *B, size_t n); \

#define UNARY_OP_FOR_ALL_TYPES_ALL_VARIANTS_PROTO(op_name)\
UNARY_OP_FOR_ALL_INPLACE_TYPES_PROTO(op_name)\
UNARY_OP_FOR_ALL_TYPES_PROTO(op_name)

#define UNARY_OP_FOR_ALL_INPLACE_TYPES_IMPL(op_name)\
void tblas_tensor_b##op_name##_inplace(int8_t *A, size_t n) {\
    tblas_tensor_ge##op_name##_inplace(A, n);\
}\
void tblas_tensor_s##op_name##_inplace(int16_t *A, size_t n) {\
    tblas_tensor_ge##op_name##_inplace(A, n);\
}\
void tblas_tensor_i##op_name##_inplace(int32_t *A, size_t n) {\
    tblas_tensor_ge##op_name##_inplace(A, n);\
}\
void tblas_tensor_l##op_name##_inplace(int64_t *A, size_t n) {\
    tblas_tensor_ge##op_name##_inplace(A, n);\
}\
void tblas_tensor_f##op_name##_inplace(float *A, size_t n) {\
    tblas_tensor_ge##op_name##_inplace(A, n);\
}\
void tblas_tensor_d##op_name##_inplace(double *A, size_t n) {\
    tblas_tensor_ge##op_name##_inplace(A, n);\
}

#define UNARY_OP_FOR_ALL_TYPES_IMPL(op_name)\
void tblas_tensor_b##op_name(const int8_t *A, int8_t *B, size_t n) {\
    tblas_tensor_ge##op_name(A, B, n);\
}\
void tblas_tensor_s##op_name(const int16_t *A, int16_t *B, size_t n) {\
    tblas_tensor_ge##op_name(A, B, n);\
}\
void tblas_tensor_i##op_name(const int32_t *A, int32_t *B, size_t n) {\
    tblas_tensor_ge##op_name(A, B, n);\
}\
void tblas_tensor_l##op_name(const int64_t *A, int64_t *B, size_t n) {\
    tblas_tensor_ge##op_name(A, B, n);\
}\
void tblas_tensor_f##op_name(const float *A, float *B, size_t n) {\
    tblas_tensor_ge##op_name(A, B, n);\
}\
void tblas_tensor_d##op_name(const double *A, double *B, size_t n) {\
    tblas_tensor_ge##op_name(A, B, n);\
}

#define UNARY_OP_FOR_ALL_TYPES_ALL_VARIANTS_IMPL(op_name)\
UNARY_OP_FOR_ALL_INPLACE_TYPES_IMPL(op_name)\
UNARY_OP_FOR_ALL_TYPES_IMPL(op_name)


// Binary operations
#define BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_SCALAR_INPLACE_PROTO(op_name) \
/* Tensor by scalar (inplace, src=dst=A) */ \
void tblas_tensor_bb##op_name##_inplace(int8_t *A, int8_t b, size_t n);\
void tblas_tensor_bs##op_name##_inplace(int8_t *A, int16_t b, size_t n);\
void tblas_tensor_bi##op_name##_inplace(int8_t *A, int32_t b, size_t n);\
void tblas_tensor_bl##op_name##_inplace(int8_t *A, int64_t b, size_t n);\
void tblas_tensor_bf##op_name##_inplace(int8_t *A, float b, size_t n);\
void tblas_tensor_bd##op_name##_inplace(int8_t *A, double b, size_t n);\
void tblas_tensor_sb##op_name##_inplace(int16_t *A, int8_t b, size_t n);\
void tblas_tensor_ss##op_name##_inplace(int16_t *A, int16_t b, size_t n);\
void tblas_tensor_si##op_name##_inplace(int16_t *A, int32_t b, size_t n);\
void tblas_tensor_sl##op_name##_inplace(int16_t *A, int64_t b, size_t n);\
void tblas_tensor_sf##op_name##_inplace(int16_t *A, float b, size_t n);\
void tblas_tensor_sd##op_name##_inplace(int16_t *A, double b, size_t n);\
void tblas_tensor_ib##op_name##_inplace(int32_t *A, int8_t b, size_t n);\
void tblas_tensor_is##op_name##_inplace(int32_t *A, int16_t b, size_t n);\
void tblas_tensor_ii##op_name##_inplace(int32_t *A, int32_t b, size_t n);\
void tblas_tensor_il##op_name##_inplace(int32_t *A, int64_t b, size_t n);\
void tblas_tensor_if##op_name##_inplace(int32_t *A, float b, size_t n);\
void tblas_tensor_id##op_name##_inplace(int32_t *A, double b, size_t n);\
void tblas_tensor_lb##op_name##_inplace(int64_t *A, int8_t b, size_t n);\
void tblas_tensor_ls##op_name##_inplace(int64_t *A, int16_t b, size_t n);\
void tblas_tensor_li##op_name##_inplace(int64_t *A, int32_t b, size_t n);\
void tblas_tensor_ll##op_name##_inplace(int64_t *A, int64_t b, size_t n);\
void tblas_tensor_lf##op_name##_inplace(int64_t *A, float b, size_t n);\
void tblas_tensor_ld##op_name##_inplace(int64_t *A, double b, size_t n);\
void tblas_tensor_fb##op_name##_inplace(float *A, int8_t b, size_t n);\
void tblas_tensor_fs##op_name##_inplace(float *A, int16_t b, size_t n);\
void tblas_tensor_fi##op_name##_inplace(float *A, int32_t b, size_t n);\
void tblas_tensor_fl##op_name##_inplace(float *A, int64_t b, size_t n);\
void tblas_tensor_ff##op_name##_inplace(float *A, float b, size_t n);\
void tblas_tensor_fd##op_name##_inplace(float *A, double b, size_t n);\
void tblas_tensor_db##op_name##_inplace(double *A, int8_t b, size_t n);\
void tblas_tensor_ds##op_name##_inplace(double *A, int16_t b, size_t n);\
void tblas_tensor_di##op_name##_inplace(double *A, int32_t b, size_t n);\
void tblas_tensor_dl##op_name##_inplace(double *A, int64_t b, size_t n);\
void tblas_tensor_df##op_name##_inplace(double *A, float b, size_t n);\
void tblas_tensor_dd##op_name##_inplace(double *A, double b, size_t n);

#define BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_SCALAR_PROTO(op_name) \
/* Tensor by scalar (not in-place, dst = C) */ \
void tblas_tensor_bb##op_name(const int8_t *A, int8_t b, int8_t *C, size_t n);\
void tblas_tensor_bs##op_name(const int8_t *A, int16_t b, int16_t *C, size_t n);\
void tblas_tensor_bi##op_name(const int8_t *A, int32_t b, int32_t *C, size_t n);\
void tblas_tensor_bl##op_name(const int8_t *A, int64_t b, int64_t *C, size_t n);\
void tblas_tensor_bf##op_name(const int8_t *A, float b, float *C, size_t n);\
void tblas_tensor_bd##op_name(const int8_t *A, double b, double *C, size_t n);\
void tblas_tensor_sb##op_name(const int16_t *A, int8_t b, int16_t *C, size_t n);\
void tblas_tensor_ss##op_name(const int16_t *A, int16_t b, int16_t *C, size_t n);\
void tblas_tensor_si##op_name(const int16_t *A, int32_t b, int32_t *C, size_t n);\
void tblas_tensor_sl##op_name(const int16_t *A, int64_t b, int64_t *C, size_t n);\
void tblas_tensor_sf##op_name(const int16_t *A, float b, float *C, size_t n);\
void tblas_tensor_sd##op_name(const int16_t *A, double b, double *C, size_t n);\
void tblas_tensor_ib##op_name(const int32_t *A, int8_t b, int32_t *C, size_t n);\
void tblas_tensor_is##op_name(const int32_t *A, int16_t b, int32_t *C, size_t n);\
void tblas_tensor_ii##op_name(const int32_t *A, int32_t b, int32_t *C, size_t n);\
void tblas_tensor_il##op_name(const int32_t *A, int64_t b, int64_t *C, size_t n);\
void tblas_tensor_if##op_name(const int32_t *A, float b, int32_t *C, size_t n);\
void tblas_tensor_id##op_name(const int32_t *A, double b, double *C, size_t n);\
void tblas_tensor_lb##op_name(const int64_t *A, int8_t b, int64_t *C, size_t n);\
void tblas_tensor_ls##op_name(const int64_t *A, int16_t b, int64_t *C, size_t n);\
void tblas_tensor_li##op_name(const int64_t *A, int32_t b, int64_t *C, size_t n);\
void tblas_tensor_ll##op_name(const int64_t *A, int64_t b, int64_t *C, size_t n);\
void tblas_tensor_lf##op_name(const int64_t *A, float b, float *C, size_t n);\
void tblas_tensor_ld##op_name(const int64_t *A, double b, double *C, size_t n);\
void tblas_tensor_fb##op_name(const float *A, int8_t b, float *C, size_t n);\
void tblas_tensor_fs##op_name(const float *A, int16_t b, float *C, size_t n);\
void tblas_tensor_fi##op_name(const float *A, int32_t b, float *C, size_t n);\
void tblas_tensor_fl##op_name(const float *A, int64_t b, float *C, size_t n);\
void tblas_tensor_ff##op_name(const float *A, float b, float *C, size_t n);\
void tblas_tensor_fd##op_name(const float *A, double b, double *C, size_t n);\
void tblas_tensor_db##op_name(const double *A, int8_t b, double *C, size_t n);\
void tblas_tensor_ds##op_name(const double *A, int16_t b, double *C, size_t n);\
void tblas_tensor_di##op_name(const double *A, int32_t b, double *C, size_t n);\
void tblas_tensor_dl##op_name(const double *A, int64_t b, double *C, size_t n);\
void tblas_tensor_df##op_name(const double *A, float b, double *C, size_t n);\
void tblas_tensor_dd##op_name(const double *A, double b, double *C, size_t n);

#define BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_PROTO(op_name)\
void tblas_tensor_bb##op_name(const int8_t *A, const int8_t *B, int8_t *C, size_t n);\
void tblas_tensor_bs##op_name(const int8_t *A, const int16_t *B, int16_t *C, size_t n);\
void tblas_tensor_bi##op_name(const int8_t *A, const int32_t *B, int32_t *C, size_t n);\
void tblas_tensor_bl##op_name(const int8_t *A, const int64_t *B, int64_t *C, size_t n);\
void tblas_tensor_bf##op_name(const int8_t *A, const float *B, float *C, size_t n);\
void tblas_tensor_bd##op_name(const int8_t *A, const double *B, double *C, size_t n);\
void tblas_tensor_sb##op_name(const int16_t *A, const int8_t *B, int16_t *C, size_t n);\
void tblas_tensor_ss##op_name(const int16_t *A, const int16_t *B, int16_t *C, size_t n);\
void tblas_tensor_si##op_name(const int16_t *A, const int32_t *B, int32_t *C, size_t n);\
void tblas_tensor_sl##op_name(const int16_t *A, const int64_t *B, int64_t *C, size_t n);\
void tblas_tensor_sf##op_name(const int16_t *A, const float *B, float *C, size_t n);\
void tblas_tensor_sd##op_name(const int16_t *A, const double *B, double *C, size_t n);\
void tblas_tensor_ib##op_name(const int32_t *A, const int8_t *B, int32_t *C, size_t n);\
void tblas_tensor_is##op_name(const int32_t *A, const int16_t *B, int32_t *C, size_t n);\
void tblas_tensor_ii##op_name(const int32_t *A, const int32_t *B, int32_t *C, size_t n);\
void tblas_tensor_il##op_name(const int32_t *A, const int64_t *B, int64_t *C, size_t n);\
void tblas_tensor_if##op_name(const int32_t *A, const float *B, float *C, size_t n); \
void tblas_tensor_id##op_name(const int32_t *A, const double *B, double *C, size_t n);\
void tblas_tensor_lb##op_name(const int64_t *A, const int8_t *B, int64_t *C, size_t n);\
void tblas_tensor_ls##op_name(const int64_t *A, const int16_t *B, int64_t *C, size_t n);\
void tblas_tensor_li##op_name(const int64_t *A, const int32_t *B, int64_t *C, size_t n);\
void tblas_tensor_ll##op_name(const int64_t *A, const int64_t *B, int64_t *C, size_t n);\
void tblas_tensor_lf##op_name(const int64_t *A, const float *B, float *C, size_t n);\
void tblas_tensor_ld##op_name(const int64_t *A, const double *B, double *C, size_t n);\
void tblas_tensor_fb##op_name(const float *A, const int8_t *B, float *C, size_t n);\
void tblas_tensor_fs##op_name(const float *A, const int16_t *B, float *C, size_t n);\
void tblas_tensor_fi##op_name(const float *A, const int32_t *B, float *C, size_t n);\
void tblas_tensor_fl##op_name(const float *A, const int64_t *B, float *C, size_t n);\
void tblas_tensor_ff##op_name(const float *A, const float *B, float *C, size_t n);\
void tblas_tensor_fd##op_name(const float *A, const double *B, double *C, size_t n);\
void tblas_tensor_db##op_name(const double *A, const int8_t *B, double *C, size_t n);\
void tblas_tensor_ds##op_name(const double *A, const int16_t *B, double *C, size_t n);\
void tblas_tensor_di##op_name(const double *A, const int32_t *B, double *C, size_t n);\
void tblas_tensor_dl##op_name(const double *A, const int64_t *B, double *C, size_t n);\
void tblas_tensor_df##op_name(const double *A, const float *B, double *C, size_t n);\
void tblas_tensor_dd##op_name(const double *A, const double *B, double *C, size_t n);

#define BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_INPLACE_PROTO(op_name) \
/* Tensor by tensor of same length (element-wise, no broadcasting, in-place, src=dst=A) */ \
void tblas_tensor_bb##op_name##_inplace(int8_t *A, const int8_t *B, size_t n);\
void tblas_tensor_bs##op_name##_inplace(int8_t *A, const int16_t *B, size_t n);\
void tblas_tensor_bi##op_name##_inplace(int8_t *A, const int32_t *B, size_t n);\
void tblas_tensor_bl##op_name##_inplace(int8_t *A, const int64_t *B, size_t n);\
void tblas_tensor_bf##op_name##_inplace(int8_t *A, const float *B, size_t n);\
void tblas_tensor_bd##op_name##_inplace(int8_t *A, const double *B, size_t n);\
void tblas_tensor_sb##op_name##_inplace(int16_t *A, const int8_t *B, size_t n);\
void tblas_tensor_ss##op_name##_inplace(int16_t *A, const int16_t *B, size_t n);\
void tblas_tensor_si##op_name##_inplace(int16_t *A, const int32_t *B, size_t n);\
void tblas_tensor_sl##op_name##_inplace(int16_t *A, const int64_t *B, size_t n);\
void tblas_tensor_sf##op_name##_inplace(int16_t *A, const float *B, size_t n);\
void tblas_tensor_sd##op_name##_inplace(int16_t *A, const double *B, size_t n);\
void tblas_tensor_ib##op_name##_inplace(int32_t *A, const int8_t *B, size_t n);\
void tblas_tensor_is##op_name##_inplace(int32_t *A, const int16_t *B, size_t n);\
void tblas_tensor_ii##op_name##_inplace(int32_t *A, const int32_t *B, size_t n);\
void tblas_tensor_il##op_name##_inplace(int32_t *A, const int64_t *B, size_t n);\
void tblas_tensor_if##op_name##_inplace(int32_t *A, const float *B, size_t n);\
void tblas_tensor_id##op_name##_inplace(int32_t *A, const double *B, size_t n);\
void tblas_tensor_lb##op_name##_inplace(int64_t *A, const int8_t *B, size_t n);\
void tblas_tensor_ls##op_name##_inplace(int64_t *A, const int16_t *B, size_t n);\
void tblas_tensor_li##op_name##_inplace(int64_t *A, const int32_t *B, size_t n);\
void tblas_tensor_ll##op_name##_inplace(int64_t *A, const int64_t *B, size_t n);\
void tblas_tensor_lf##op_name##_inplace(int64_t *A, const float *B, size_t n);\
void tblas_tensor_ld##op_name##_inplace(int64_t *A, const double *B, size_t n);\
void tblas_tensor_fb##op_name##_inplace(float *A, const int8_t *B, size_t n);\
void tblas_tensor_fs##op_name##_inplace(float *A, const int16_t *B, size_t n);\
void tblas_tensor_fi##op_name##_inplace(float *A, const int32_t *B, size_t n);\
void tblas_tensor_fl##op_name##_inplace(float *A, const int64_t *B, size_t n);\
void tblas_tensor_ff##op_name##_inplace(float *A, const float *B, size_t n);\
void tblas_tensor_fd##op_name##_inplace(float *A, const double *B, size_t n);\
void tblas_tensor_db##op_name##_inplace(double *A, const int8_t *B, size_t n);\
void tblas_tensor_ds##op_name##_inplace(double *A, const int16_t *B, size_t n);\
void tblas_tensor_di##op_name##_inplace(double *A, const int32_t *B, size_t n);\
void tblas_tensor_dl##op_name##_inplace(double *A, const int64_t *B, size_t n);\
void tblas_tensor_df##op_name##_inplace(double *A, const float *B, size_t n);\
void tblas_tensor_dd##op_name##_inplace(double *A, const double *B, size_t n);

#define BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_INPLACE_BROADCASTING_PROTO(op_name) \
/* Tensor by tensor (element-wise, with broadcasting, in-place, src=dst=A) */ \
void tblas_tensor_bb##op_name##_broadcast_inplace(int8_t *A, const int8_t *B, size_t n, size_t r);\
void tblas_tensor_bs##op_name##_broadcast_inplace(int8_t *A, const int16_t *B, size_t n, size_t r);\
void tblas_tensor_bi##op_name##_broadcast_inplace(int8_t *A, const int32_t *B, size_t n, size_t r);\
void tblas_tensor_bl##op_name##_broadcast_inplace(int8_t *A, const int64_t *B, size_t n, size_t r);\
void tblas_tensor_bf##op_name##_broadcast_inplace(int8_t *A, const float *B, size_t n, size_t r);\
void tblas_tensor_bd##op_name##_broadcast_inplace(int8_t *A, const double *B, size_t n, size_t r);\
void tblas_tensor_sb##op_name##_broadcast_inplace(int16_t *A, const int8_t *B, size_t n, size_t r);\
void tblas_tensor_ss##op_name##_broadcast_inplace(int16_t *A, const int16_t *B, size_t n, size_t r);\
void tblas_tensor_si##op_name##_broadcast_inplace(int16_t *A, const int32_t *B, size_t n, size_t r);\
void tblas_tensor_sl##op_name##_broadcast_inplace(int16_t *A, const int64_t *B, size_t n, size_t r);\
void tblas_tensor_sf##op_name##_broadcast_inplace(int16_t *A, const float *B, size_t n, size_t r);\
void tblas_tensor_sd##op_name##_broadcast_inplace(int16_t *A, const double *B, size_t n, size_t r);\
void tblas_tensor_ib##op_name##_broadcast_inplace(int32_t *A, const int8_t *B, size_t n, size_t r);\
void tblas_tensor_is##op_name##_broadcast_inplace(int32_t *A, const int16_t *B, size_t n, size_t r);\
void tblas_tensor_ii##op_name##_broadcast_inplace(int32_t *A, const int32_t *B, size_t n, size_t r);\
void tblas_tensor_il##op_name##_broadcast_inplace(int32_t *A, const int64_t *B, size_t n, size_t r);\
void tblas_tensor_if##op_name##_broadcast_inplace(int32_t *A, const float *B, size_t n, size_t r);\
void tblas_tensor_id##op_name##_broadcast_inplace(int32_t *A, const double *B, size_t n, size_t r);\
void tblas_tensor_lb##op_name##_broadcast_inplace(int64_t *A, const int8_t *B, size_t n, size_t r);\
void tblas_tensor_ls##op_name##_broadcast_inplace(int64_t *A, const int16_t *B, size_t n, size_t r);\
void tblas_tensor_li##op_name##_broadcast_inplace(int64_t *A, const int32_t *B, size_t n, size_t r);\
void tblas_tensor_ll##op_name##_broadcast_inplace(int64_t *A, const int64_t *B, size_t n, size_t r);\
void tblas_tensor_lf##op_name##_broadcast_inplace(int64_t *A, const float *B, size_t n, size_t r);\
void tblas_tensor_ld##op_name##_broadcast_inplace(int64_t *A, const double *B, size_t n, size_t r);\
void tblas_tensor_fb##op_name##_broadcast_inplace(float *A, const int8_t *B, size_t n, size_t r);\
void tblas_tensor_fs##op_name##_broadcast_inplace(float *A, const int16_t *B, size_t n, size_t r);\
void tblas_tensor_fi##op_name##_broadcast_inplace(float *A, const int32_t *B, size_t n, size_t r);\
void tblas_tensor_fl##op_name##_broadcast_inplace(float *A, const int64_t *B, size_t n, size_t r);\
void tblas_tensor_ff##op_name##_broadcast_inplace(float *A, const float *B, size_t n, size_t r);\
void tblas_tensor_fd##op_name##_broadcast_inplace(float *A, const double *B, size_t n, size_t r);\
void tblas_tensor_db##op_name##_broadcast_inplace(double *A, const int8_t *B, size_t n, size_t r);\
void tblas_tensor_ds##op_name##_broadcast_inplace(double *A, const int16_t *B, size_t n, size_t r);\
void tblas_tensor_di##op_name##_broadcast_inplace(double *A, const int32_t *B, size_t n, size_t r);\
void tblas_tensor_dl##op_name##_broadcast_inplace(double *A, const int64_t *B, size_t n, size_t r);\
void tblas_tensor_df##op_name##_broadcast_inplace(double *A, const float *B, size_t n, size_t r);\
void tblas_tensor_dd##op_name##_broadcast_inplace(double *A, const double *B, size_t n, size_t r);

#define BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_BROADCAST_PROTO(op_name) \
/* Tensor by tensor (element-wise, with broadcasting, not in-place, src=A, dst=C) */ \
void tblas_tensor_bb##op_name##_broadcast(const int8_t *A, const int8_t *B, int8_t *C, size_t n, size_t r);\
void tblas_tensor_bs##op_name##_broadcast(const int8_t *A, const int16_t *B, int16_t *C, size_t n, size_t r);\
void tblas_tensor_bi##op_name##_broadcast(const int8_t *A, const int32_t *B, int32_t *C, size_t n, size_t r);\
void tblas_tensor_bl##op_name##_broadcast(const int8_t *A, const int64_t *B, int64_t *C, size_t n, size_t r);\
void tblas_tensor_bf##op_name##_broadcast(const int8_t *A, const float *B, float *C, size_t n, size_t r);\
void tblas_tensor_bd##op_name##_broadcast(const int8_t *A, const double *B, double *C, size_t n, size_t r);\
void tblas_tensor_sb##op_name##_broadcast(const int16_t *A, const int8_t *B, int16_t *C, size_t n, size_t r);\
void tblas_tensor_ss##op_name##_broadcast(const int16_t *A, const int16_t *B, int16_t *C, size_t n, size_t r);\
void tblas_tensor_si##op_name##_broadcast(const int16_t *A, const int32_t *B, int32_t *C, size_t n, size_t r);\
void tblas_tensor_sl##op_name##_broadcast(const int16_t *A, const int64_t *B, int64_t *C, size_t n, size_t r);\
void tblas_tensor_sf##op_name##_broadcast(const int16_t *A, const float *B, float *C, size_t n, size_t r);\
void tblas_tensor_sd##op_name##_broadcast(const int16_t *A, const double *B, double *C, size_t n, size_t r);\
void tblas_tensor_ib##op_name##_broadcast(const int32_t *A, const int8_t *B, int32_t *C, size_t n, size_t r);\
void tblas_tensor_is##op_name##_broadcast(const int32_t *A, const int16_t *B, int32_t *C, size_t n, size_t r);\
void tblas_tensor_ii##op_name##_broadcast(const int32_t *A, const int32_t *B, int32_t *C, size_t n, size_t r);\
void tblas_tensor_il##op_name##_broadcast(const int32_t *A, const int64_t *B, int64_t *C, size_t n, size_t r);\
void tblas_tensor_if##op_name##_broadcast(const int32_t *A, const float *B, float *C, size_t n, size_t r);\
void tblas_tensor_id##op_name##_broadcast(const int32_t *A, const double *B, double *C, size_t n, size_t r);\
void tblas_tensor_lb##op_name##_broadcast(const int64_t *A, const int8_t *B, int64_t *C, size_t n, size_t r);\
void tblas_tensor_ls##op_name##_broadcast(const int64_t *A, const int16_t *B, int64_t *C, size_t n, size_t r);\
void tblas_tensor_li##op_name##_broadcast(const int64_t *A, const int32_t *B, int64_t *C, size_t n, size_t r);\
void tblas_tensor_ll##op_name##_broadcast(const int64_t *A, const int64_t *B, int64_t *C, size_t n, size_t r);\
void tblas_tensor_lf##op_name##_broadcast(const int64_t *A, const float *B, float *C, size_t n, size_t r);\
void tblas_tensor_ld##op_name##_broadcast(const int64_t *A, const double *B, double *C, size_t n, size_t r);\
void tblas_tensor_fb##op_name##_broadcast(const float *A, const int8_t *B, float *C, size_t n, size_t r);\
void tblas_tensor_fs##op_name##_broadcast(const float *A, const int16_t *B, float *C, size_t n, size_t r);\
void tblas_tensor_fi##op_name##_broadcast(const float *A, const int32_t *B, float *C, size_t n, size_t r);\
void tblas_tensor_fl##op_name##_broadcast(const float *A, const int64_t *B, float *C, size_t n, size_t r);\
void tblas_tensor_ff##op_name##_broadcast(const float *A, const float *B, float *C, size_t n, size_t r);\
void tblas_tensor_fd##op_name##_broadcast(const float *A, const double *B, double *C, size_t n, size_t r);\
void tblas_tensor_db##op_name##_broadcast(const double *A, const int8_t *B, double *C, size_t n, size_t r);\
void tblas_tensor_ds##op_name##_broadcast(const double *A, const int16_t *B, double *C, size_t n , size_t  r);\
void tblas_tensor_di##op_name##_broadcast(const double *A, const int32_t *B, double *C, size_t n, size_t r);\
void tblas_tensor_dl##op_name##_broadcast(const double *A, const int64_t *B, double *C, size_t n, size_t r);\
void tblas_tensor_df##op_name##_broadcast(const double *A, const float *B, double *C, size_t n, size_t r);\
void tblas_tensor_dd##op_name##_broadcast(const double *A, const double *B, double *C, size_t n, size_t r);


#define BINARY_OP_FOR_ALL_TYPES_ALL_VARIANTS_PROTO(op_name)\
BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_SCALAR_INPLACE_PROTO(op_name)\
BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_SCALAR_PROTO(op_name)\
BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_PROTO(op_name)\
BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_INPLACE_PROTO(op_name)\
BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_INPLACE_BROADCASTING_PROTO(op_name)\
BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_BROADCAST_PROTO(op_name)


#define BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_SCALAR_INPLACE_IMPL(op_name) \
/* Tensor by scalar (inplace, src=dst=A) */ \
void tblas_tensor_bb##op_name##_inplace(int8_t *A, int8_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_bs##op_name##_inplace(int8_t *A, int16_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_bi##op_name##_inplace(int8_t *A, int32_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_bl##op_name##_inplace(int8_t *A, int64_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_bf##op_name##_inplace(int8_t *A, float b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_bd##op_name##_inplace(int8_t *A, double b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_sb##op_name##_inplace(int16_t *A, int8_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_ss##op_name##_inplace(int16_t *A, int16_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_si##op_name##_inplace(int16_t *A, int32_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_sl##op_name##_inplace(int16_t *A, int64_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_sf##op_name##_inplace(int16_t *A, float b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_sd##op_name##_inplace(int16_t *A, double b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_ib##op_name##_inplace(int32_t *A, int8_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_is##op_name##_inplace(int32_t *A, int16_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_ii##op_name##_inplace(int32_t *A, int32_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_il##op_name##_inplace(int32_t *A, int64_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_if##op_name##_inplace(int32_t *A, float b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_id##op_name##_inplace(int32_t *A, double b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_lb##op_name##_inplace(int64_t *A, int8_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_ls##op_name##_inplace(int64_t *A, int16_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_li##op_name##_inplace(int64_t *A, int32_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_ll##op_name##_inplace(int64_t *A, int64_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_lf##op_name##_inplace(int64_t *A, float b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_ld##op_name##_inplace(int64_t *A, double b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_fb##op_name##_inplace(float *A, int8_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_fs##op_name##_inplace(float *A, int16_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_fi##op_name##_inplace(float *A, int32_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_fl##op_name##_inplace(float *A, int64_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_ff##op_name##_inplace(float *A, float b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_fd##op_name##_inplace(float *A, double b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_db##op_name##_inplace(double *A, int8_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_ds##op_name##_inplace(double *A, int16_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_di##op_name##_inplace(double *A, int32_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_dl##op_name##_inplace(double *A, int64_t b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_df##op_name##_inplace(double *A, float b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}\
void tblas_tensor_dd##op_name##_inplace(double *A, double b, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, b, n);\
}

#define BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_SCALAR_IMPL(op_name) \
/* Tensor by scalar (not in-place, dst = C) */ \
void tblas_tensor_bb##op_name(const int8_t *A, int8_t b, int8_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_bs##op_name(const int8_t *A, int16_t b, int16_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_bi##op_name(const int8_t *A, int32_t b, int32_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_bl##op_name(const int8_t *A, int64_t b, int64_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_bf##op_name(const int8_t *A, float b, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_bd##op_name(const int8_t *A, double b, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_sb##op_name(const int16_t *A, int8_t b, int16_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_ss##op_name(const int16_t *A, int16_t b, int16_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_si##op_name(const int16_t *A, int32_t b, int16_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_sl##op_name(const int16_t *A, int64_t b, int64_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_sf##op_name(const int16_t *A, float b, int16_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_sd##op_name(const int16_t *A, double b, int16_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_ib##op_name(const int32_t *A, int8_t b, int32_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_is##op_name(const int32_t *A, int16_t b, int32_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_ii##op_name(const int32_t *A, int32_t b, int32_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_il##op_name(const int32_t *A, int64_t b, int64_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_if##op_name(const int32_t *A, float b, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_id##op_name(const int32_t *A, double b, int32_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_lb##op_name(const int64_t *A, int8_t b, int64_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_ls##op_name(const int64_t *A, int16_t b, int64_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_li##op_name(const int64_t *A, int32_t b, int64_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_ll##op_name(const int64_t *A, int64_t b, int64_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_lf##op_name(const int64_t *A, float b, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_ld##op_name(const int64_t *A, double b, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_fb##op_name(const float *A, int8_t b, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_fs##op_name(const float *A, int16_t b, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_fi##op_name(const float *A, int32_t b, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_fl##op_name(const float *A, int64_t b, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_ff##op_name(const float *A, float b, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_fd##op_name(const float *A, double b, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_db##op_name(const double *A, int8_t b, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_ds##op_name(const double *A, int16_t b, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_di##op_name(const double *A, int32_t b, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_dl##op_name(const double *A, int64_t b, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_df##op_name(const double *A, float b, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}\
void tblas_tensor_dd##op_name(const double *A, double b, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, b, C, n);\
}

#define BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_IMPL(op_name) \
/* Tensor by tensor of same length (element-wise, no broadcasting, not in-place, dst=C) */ \
void tblas_tensor_bb##op_name(const int8_t *A, const int8_t *B, int8_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_bs##op_name(const int8_t *A, const int16_t *B, int16_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_bi##op_name(const int8_t *A, const int32_t *B, int32_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_bl##op_name(const int8_t *A, const int64_t *B, int64_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_bf##op_name(const int8_t *A, const float *B, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_bd##op_name(const int8_t *A, const double *B, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_sb##op_name(const int16_t *A, const int8_t *B, int16_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_ss##op_name(const int16_t *A, const int16_t *B, int16_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_si##op_name(const int16_t *A, const int32_t *B, int32_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_sl##op_name(const int16_t *A, const int64_t *B, int64_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_sf##op_name(const int16_t *A, const float *B, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_sd##op_name(const int16_t *A, const double *B, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_ib##op_name(const int32_t *A, const int8_t *B, int32_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_is##op_name(const int32_t *A, const int16_t *B, int32_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_ii##op_name(const int32_t *A, const int32_t *B, int32_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_il##op_name(const int32_t *A, const int64_t *B, int64_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_if##op_name(const int32_t *A, const float *B, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_id##op_name(const int32_t *A, const double *B, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_lb##op_name(const int64_t *A, const int8_t *B, int64_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_ls##op_name(const int64_t *A, const int16_t *B, int64_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_li##op_name(const int64_t *A, const int32_t *B, int64_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_ll##op_name(const int64_t *A, const int64_t *B, int64_t *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_lf##op_name(const int64_t *A, const float *B, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_ld##op_name(const int64_t *A, const double *B, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_fb##op_name(const float *A, const int8_t *B, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_fs##op_name(const float *A, const int16_t *B, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_fi##op_name(const float *A, const int32_t *B, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_fl##op_name(const float *A, const int64_t *B, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_ff##op_name(const float *A, const float *B, float *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_fd##op_name(const float *A, const double *B, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_db##op_name(const double *A, const int8_t *B, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_ds##op_name(const double *A, const int16_t *B, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_di##op_name(const double *A, const int32_t *B, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_dl##op_name(const double *A, const int64_t *B, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_df##op_name(const double *A, const float *B, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}\
void tblas_tensor_dd##op_name(const double *A, const double *B, double *C, size_t n){\
    tblas_tensor_ge##op_name(A, B, C, n);\
}


#define BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_INPLACE_IMPL(op_name)\
/* Tensor by tensor of same length (element-wise, no broadcasting, in-place, src=dst=A) */ \
void tblas_tensor_bb##op_name##_inplace(int8_t *A, const int8_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_bs##op_name##_inplace(int8_t *A, const int16_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_bi##op_name##_inplace(int8_t *A, const int32_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_bl##op_name##_inplace(int8_t *A, const int64_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_bf##op_name##_inplace(int8_t *A, const float *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_bd##op_name##_inplace(int8_t *A, const double *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_sb##op_name##_inplace(int16_t *A, const int8_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_ss##op_name##_inplace(int16_t *A, const int16_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_si##op_name##_inplace(int16_t *A, const int32_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_sl##op_name##_inplace(int16_t *A, const int64_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_sf##op_name##_inplace(int16_t *A, const float *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_sd##op_name##_inplace(int16_t *A, const double *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_ib##op_name##_inplace(int32_t *A, const int8_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_is##op_name##_inplace(int32_t *A, const int16_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_ii##op_name##_inplace(int32_t *A, const int32_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_il##op_name##_inplace(int32_t *A, const int64_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_if##op_name##_inplace(int32_t *A, const float *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_id##op_name##_inplace(int32_t *A, const double *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_lb##op_name##_inplace(int64_t *A, const int8_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_ls##op_name##_inplace(int64_t *A, const int16_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_li##op_name##_inplace(int64_t *A, const int32_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_ll##op_name##_inplace(int64_t *A, const int64_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_lf##op_name##_inplace(int64_t *A, const float *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_ld##op_name##_inplace(int64_t *A, const double *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_fb##op_name##_inplace(float *A, const int8_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_fs##op_name##_inplace(float *A, const int16_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_fi##op_name##_inplace(float *A, const int32_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_fl##op_name##_inplace(float *A, const int64_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_ff##op_name##_inplace(float *A, const float *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_fd##op_name##_inplace(float *A, const double *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_db##op_name##_inplace(double *A, const int8_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_ds##op_name##_inplace(double *A, const int16_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_di##op_name##_inplace(double *A, const int32_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_dl##op_name##_inplace(double *A, const int64_t *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_df##op_name##_inplace(double *A, const float *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}\
void tblas_tensor_dd##op_name##_inplace(double *A, const double *B, size_t n){\
    tblas_tensor_ge##op_name##_inplace(A, B, n);\
}

#define BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_INPLACE_BROADCASTING_IMPL(op_name) \
/* Tensor by tensor (element-wise, with broadcasting, in-place, src=dst=A) */ \
void tblas_tensor_bb##op_name##_broadcast_inplace(int8_t *A, const int8_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_bs##op_name##_broadcast_inplace(int8_t *A, const int16_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_bi##op_name##_broadcast_inplace(int8_t *A, const int32_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_bl##op_name##_broadcast_inplace(int8_t *A, const int64_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_bf##op_name##_broadcast_inplace(int8_t *A, const float *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_bd##op_name##_broadcast_inplace(int8_t *A, const double *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_sb##op_name##_broadcast_inplace(int16_t *A, const int8_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_ss##op_name##_broadcast_inplace(int16_t *A, const int16_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_si##op_name##_broadcast_inplace(int16_t *A, const int32_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_sl##op_name##_broadcast_inplace(int16_t *A, const int64_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_sf##op_name##_broadcast_inplace(int16_t *A, const float *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_sd##op_name##_broadcast_inplace(int16_t *A, const double *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_ib##op_name##_broadcast_inplace(int32_t *A, const int8_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_is##op_name##_broadcast_inplace(int32_t *A, const int16_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_ii##op_name##_broadcast_inplace(int32_t *A, const int32_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_il##op_name##_broadcast_inplace(int32_t *A, const int64_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_if##op_name##_broadcast_inplace(int32_t *A, const float *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_id##op_name##_broadcast_inplace(int32_t *A, const double *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_lb##op_name##_broadcast_inplace(int64_t *A, const int8_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_ls##op_name##_broadcast_inplace(int64_t *A, const int16_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_li##op_name##_broadcast_inplace(int64_t *A, const int32_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_ll##op_name##_broadcast_inplace(int64_t *A, const int64_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_lf##op_name##_broadcast_inplace(int64_t *A, const float *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_ld##op_name##_broadcast_inplace(int64_t *A, const double *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_fb##op_name##_broadcast_inplace(float *A, const int8_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_fs##op_name##_broadcast_inplace(float *A, const int16_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_fi##op_name##_broadcast_inplace(float *A, const int32_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_fl##op_name##_broadcast_inplace(float *A, const int64_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_ff##op_name##_broadcast_inplace(float *A, const float *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_fd##op_name##_broadcast_inplace(float *A, const double *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_db##op_name##_broadcast_inplace(double *A, const int8_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_ds##op_name##_broadcast_inplace(double *A, const int16_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_di##op_name##_broadcast_inplace(double *A, const int32_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_dl##op_name##_broadcast_inplace(double *A, const int64_t *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_df##op_name##_broadcast_inplace(double *A, const float *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}\
void tblas_tensor_dd##op_name##_broadcast_inplace(double *A, const double *B, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast_inplace(A, B, n, r);\
}

#define BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_BROADCAST_IMPL(op_name)\
/* Multiply tensor by tensor (element-wise, with broadcasting, not in-place, src=A, dst=C) */ \
void tblas_tensor_bb##op_name##_broadcast(const int8_t *A, const int8_t *B, int8_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_bs##op_name##_broadcast(const int8_t *A, const int16_t *B, int16_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_bi##op_name##_broadcast(const int8_t *A, const int32_t *B, int32_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_bl##op_name##_broadcast(const int8_t *A, const int64_t *B, int64_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_bf##op_name##_broadcast(const int8_t *A, const float *B, float *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_bd##op_name##_broadcast(const int8_t *A, const double *B, double *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_sb##op_name##_broadcast(const int16_t *A, const int8_t *B, int16_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_ss##op_name##_broadcast(const int16_t *A, const int16_t *B, int16_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_si##op_name##_broadcast(const int16_t *A, const int32_t *B, int32_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_sl##op_name##_broadcast(const int16_t *A, const int64_t *B, int64_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_sf##op_name##_broadcast(const int16_t *A, const float *B, float *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_sd##op_name##_broadcast(const int16_t *A, const double *B, double *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_ib##op_name##_broadcast(const int32_t *A, const int8_t *B, int32_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_is##op_name##_broadcast(const int32_t *A, const int16_t *B, int32_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_ii##op_name##_broadcast(const int32_t *A, const int32_t *B, int32_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_il##op_name##_broadcast(const int32_t *A, const int64_t *B, int64_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_if##op_name##_broadcast(const int32_t *A, const float *B, float *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_id##op_name##_broadcast(const int32_t *A, const double *B, double *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_lb##op_name##_broadcast(const int64_t *A, const int8_t *B, int64_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_ls##op_name##_broadcast(const int64_t *A, const int16_t *B, int64_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_li##op_name##_broadcast(const int64_t *A, const int32_t *B, int64_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_ll##op_name##_broadcast(const int64_t *A, const int64_t *B, int64_t *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_lf##op_name##_broadcast(const int64_t *A, const float *B, float *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_ld##op_name##_broadcast(const int64_t *A, const double *B, double *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_fb##op_name##_broadcast(const float *A, const int8_t *B, float *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_fs##op_name##_broadcast(const float *A, const int16_t *B, float *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_fi##op_name##_broadcast(const float *A, const int32_t *B, float *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_fl##op_name##_broadcast(const float *A, const int64_t *B, float *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_ff##op_name##_broadcast(const float *A, const float *B, float *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_fd##op_name##_broadcast(const float *A, const double *B, double *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_db##op_name##_broadcast(const double *A, const int8_t *B, double *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_ds##op_name##_broadcast(const double *A, const int16_t *B, double *C, size_t n , size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_di##op_name##_broadcast(const double *A, const int32_t *B, double *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_dl##op_name##_broadcast(const double *A, const int64_t *B, double *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_df##op_name##_broadcast(const double *A, const float *B, double *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}\
void tblas_tensor_dd##op_name##_broadcast(const double *A, const double *B, double *C, size_t n, size_t r){\
    tblas_tensor_ge##op_name##_broadcast(A, B, C, n, r);\
}

#define BINARY_OP_FOR_ALL_TYPES_ALL_VARIANTS_IMPL(op_name)\
BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_SCALAR_INPLACE_IMPL(op_name)\
BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_SCALAR_IMPL(op_name)\
BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_IMPL(op_name)\
BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_INPLACE_IMPL(op_name)\
BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_INPLACE_BROADCASTING_IMPL(op_name)\
BINARY_OP_FOR_ALL_TYPES_TENSOR_BY_TENSOR_BROADCAST_IMPL(op_name)