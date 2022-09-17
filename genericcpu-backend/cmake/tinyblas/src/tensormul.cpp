#include "tensormul.h"

#define FORCE_INLINE __attribute__((always_inline)) inline

template<typename A, typename B>
FORCE_INLINE void tblas_tensor_gescl_inplace(A *a, B b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] *= b;
    }
}

void tblas_tensor_bbscl_inplace(uint8_t *A, int8_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_bsscl_inplace(uint8_t *A, int16_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_biscl_inplace(uint8_t *A, int32_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_blscl_inplace(uint8_t *A, int64_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_bfscl_inplace(uint8_t *A, float b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_bdscl_inplace(uint8_t *A, double b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_sbscl_inplace(uint16_t *A, int8_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_ssscl_inplace(uint16_t *A, int16_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_siscl_inplace(uint16_t *A, int32_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_slscl_inplace(uint16_t *A, int64_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_sfscl_inplace(uint16_t *A, float b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_sdscl_inplace(uint16_t *A, double b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_ibscl_inplace(uint32_t *A, int8_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_isscl_inplace(uint32_t *A, int16_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_iiscl_inplace(uint32_t *A, int32_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_ilscl_inplace(uint32_t *A, int64_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_ifscl_inplace(uint32_t *A, float b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_idscl_inplace(uint32_t *A, double b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_lbscl_inplace(uint64_t *A, int8_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_lsscl_inplace(uint64_t *A, int16_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_liscl_inplace(uint64_t *A, int32_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_llscl_inplace(uint64_t *A, int64_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_lfscl_inplace(uint64_t *A, float b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_ldscl_inplace(uint64_t *A, double b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_fbscl_inplace(float *A, int8_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_fsscl_inplace(float *A, int16_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_fiscl_inplace(float *A, int32_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_fliscl_inplace(float *A, int64_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_ffscl_inplace(float *A, float b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_fdscl_inplace(float *A, double b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_dbscl_inplace(double *A, int8_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_dsscl_inplace(double *A, int16_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_discl_inplace(double *A, int32_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_dlscl_inplace(double *A, int64_t b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_dfscl_inplace(double *A, float b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

void tblas_tensor_ddscl_inplace(double *A, double b, size_t n) {
    tblas_tensor_gescl_inplace(A, b, n);
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_gescl(A *a, B b, C *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] * b;
    }
}

void tblas_tensor_bbscl(uint8_t *A, int8_t b, uint8_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_bsscl(uint8_t *A, int16_t b, uint8_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_biscl(uint8_t *A, int32_t b, uint8_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_blscl(uint8_t *A, int64_t b, uint8_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_bfscl(uint8_t *A, float b, uint8_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_bdscl(uint8_t *A, double b, uint8_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_sbscl(uint16_t *A, int8_t b, uint16_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_ssscl(uint16_t *A, int16_t b, uint16_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_siscl(uint16_t *A, int32_t b, uint16_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_slscl(uint16_t *A, int64_t b, uint16_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_sfscl(uint16_t *A, float b, uint16_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_sdscl(uint16_t *A, double b, uint16_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_ibscl(uint32_t *A, int8_t b, uint32_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_isscl(uint32_t *A, int16_t b, uint32_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_iiscl(uint32_t *A, int32_t b, uint32_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_ilscl(uint32_t *A, int64_t b, uint32_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_ifscl(uint32_t *A, float b, uint32_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_idscl(uint32_t *A, double b, uint32_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_lbscl(uint64_t *A, int8_t b, uint64_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_lsscl(uint64_t *A, int16_t b, uint64_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_liscl(uint64_t *A, int32_t b, uint64_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_llscl(uint64_t *A, int64_t b, uint64_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_lfscl(uint64_t *A, float b, uint64_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_ldscl(uint64_t *A, double b, uint64_t *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_fbscl(float *A, int8_t b, float *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_fsscl(float *A, int16_t b, float *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_fiscl(float *A, int32_t b, float *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_fliscl(float *A, int64_t b, float *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_ffscl(float *A, float b, float *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_fdscl(float *A, double b, float *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_dbscl(double *A, int8_t b, double *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_dsscl(double *A, int16_t b, double *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_discl(double *A, int32_t b, double *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_dlscl(double *A, int64_t b, double *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_dfscl(double *A, float b, double *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

void tblas_tensor_ddscl(double *A, double b, double *C, size_t n) {
    tblas_tensor_gescl(A, b, C, n);
}

template<typename A, typename B>
FORCE_INLINE void tblas_tensor_gemul_inplace(A *a, B *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        a[i] *= b[i];
    }
}

void tblas_tensor_bbmul_inplace(uint8_t *A, uint8_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_bsmul_inplace(uint8_t *A, uint16_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_bimul_inplace(uint8_t *A, uint32_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_blmul_inplace(uint8_t *A, uint64_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_bfmul_inplace(uint8_t *A, float *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_bdmul_inplace(uint8_t *A, double *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_sbmul_inplace(uint16_t *A, uint8_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_ssmul_inplace(uint16_t *A, uint16_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_simul_inplace(uint16_t *A, uint32_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_slmul_inplace(uint16_t *A, uint64_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_sfmul_inplace(uint16_t *A, float *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_sdmul_inplace(uint16_t *A, double *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_ibmul_inplace(uint32_t *A, uint8_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_ismul_inplace(uint32_t *A, uint16_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_iimul_inplace(uint32_t *A, uint32_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_ilmul_inplace(uint32_t *A, uint64_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_ifmul_inplace(uint32_t *A, float *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_idmul_inplace(uint32_t *A, double *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_lbmul_inplace(uint64_t *A, uint8_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_lsmul_inplace(uint64_t *A, uint16_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_limul_inplace(uint64_t *A, uint32_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_llmul_inplace(uint64_t *A, uint64_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_lfmul_inplace(uint64_t *A, float *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_ldmul_inplace(uint64_t *A, double *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_fbmul_inplace(float *A, uint8_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_fsmul_inplace(float *A, uint16_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_fimul_inplace(float *A, uint32_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_flmul_inplace(float *A, uint64_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_ffmul_inplace(float *A, float *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_fdmul_inplace(float *A, double *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_dbmul_inplace(double *A, uint8_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_dsmul_inplace(double *A, uint16_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_dimul_inplace(double *A, uint32_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_dlmul_inplace(double *A, uint64_t *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_dfmul_inplace(double *A, float *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

void tblas_tensor_ddmul_inplace(double *A, double *B, size_t n) {
    tblas_tensor_gemul_inplace(A, B, n);
}

template<typename A, typename B, typename C>
FORCE_INLINE void tblas_tensor_gemul(A *a, B *b, C *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

void tblas_tensor_bbmul(uint8_t *A, uint8_t *B, uint8_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_bsmul(uint8_t *A, uint16_t *B, uint16_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_bimul(uint8_t *A, uint32_t *B, uint32_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_blmul(uint8_t *A, uint64_t *B, uint64_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_bfmul(uint8_t *A, float *B, float *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_bdmul(uint8_t *A, double *B, double *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_sbmul(uint16_t *A, uint8_t *B, uint16_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_ssmul(uint16_t *A, uint16_t *B, uint16_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_simul(uint16_t *A, uint32_t *B, uint32_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_slmul(uint16_t *A, uint64_t *B, uint64_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_sfmul(uint16_t *A, float *B, float *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_sdmul(uint16_t *A, double *B, double *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_ibmul(uint32_t *A, uint8_t *B, uint32_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_ismul(uint32_t *A, uint16_t *B, uint32_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_iimul(uint32_t *A, uint32_t *B, uint32_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_ilmul(uint32_t *A, uint64_t *B, uint64_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_ifmul(uint32_t *A, float *B, float *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_idmul(uint32_t *A, double *B, double *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_lbmul(uint64_t *A, uint8_t *B, uint64_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_lsmul(uint64_t *A, uint16_t *B, uint64_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_limul(uint64_t *A, uint32_t *B, uint64_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_llmul(uint64_t *A, uint64_t *B, uint64_t *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_lfmul(uint64_t *A, float *B, float *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_ldmul(uint64_t *A, double *B, double *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_fbmul(float *A, uint8_t *B, float *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_fsmul(float *A, uint16_t *B, float *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_fimul(float *A, uint32_t *B, float *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_flmul(float *A, uint64_t *B, float *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_ffmul(float *A, float *B, float *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_fdmul(float *A, double *B, double *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_dbmul(double *A, uint8_t *B, double *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_dsmul(double *A, uint16_t *B, double *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_dimul(double *A, uint32_t *B, double *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_dlmul(double *A, uint64_t *B, double *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_dfmul(double *A, float *B, double *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

void tblas_tensor_ddmul(double *A, double *B, double *C, size_t n) {
    tblas_tensor_gemul(A, B, C, n);
}

template<typename A, typename B>
FORCE_INLINE void tblas_tensor_gemul_broadcast_inplace(A *a, B *b, size_t n, size_t r) {
    for (size_t i = 0; i < n; i++) {
        size_t j = i % r;
        a[i] *= b[j];
    }
}

void tblas_tensor_bbmul_broadcast_inplace(uint8_t *A, uint8_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_bsmul_broadcast_inplace(uint8_t *A, uint16_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_bimul_broadcast_inplace(uint8_t *A, uint32_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_blmul_broadcast_inplace(uint8_t *A, uint64_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_bfmul_broadcast_inplace(uint8_t *A, float *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_bdmul_broadcast_inplace(uint8_t *A, double *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_sbmul_broadcast_inplace(uint16_t *A, uint8_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_ssmul_broadcast_inplace(uint16_t *A, uint16_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_simul_broadcast_inplace(uint16_t *A, uint32_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_slmul_broadcast_inplace(uint16_t *A, uint64_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_sfmul_broadcast_inplace(uint16_t *A, float *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_sdmul_broadcast_inplace(uint16_t *A, double *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_ibmul_broadcast_inplace(uint32_t *A, uint8_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_ismul_broadcast_inplace(uint32_t *A, uint16_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_iimul_broadcast_inplace(uint32_t *A, uint32_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_ilmul_broadcast_inplace(uint32_t *A, uint64_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_ifmul_broadcast_inplace(uint32_t *A, float *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_idmul_broadcast_inplace(uint32_t *A, double *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_lbmul_broadcast_inplace(uint64_t *A, uint8_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_lsmul_broadcast_inplace(uint64_t *A, uint16_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_limul_broadcast_inplace(uint64_t *A, uint32_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_llmul_broadcast_inplace(uint64_t *A, uint64_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_lfmul_broadcast_inplace(uint64_t *A, float *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_ldmul_broadcast_inplace(uint64_t *A, double *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_fbmul_broadcast_inplace(float *A, uint8_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_fsmul_broadcast_inplace(float *A, uint16_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_fimul_broadcast_inplace(float *A, uint32_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_flmul_broadcast_inplace(float *A, uint64_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_ffmul_broadcast_inplace(float *A, float *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_fdmul_broadcast_inplace(float *A, double *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_dbmul_broadcast_inplace(double *A, uint8_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_dsmul_broadcast_inplace(double *A, uint16_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_dimul_broadcast_inplace(double *A, uint32_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_dlmul_broadcast_inplace(double *A, uint64_t *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_dfmul_broadcast_inplace(double *A, float *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

void tblas_tensor_ddmul_broadcast_inplace(double *A, double *B, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast_inplace(A, B, n, r);
}

template<typename A, typename B, typename C>
void tblas_tensor_gemul_broadcast(A *a, B *b, C *c, size_t n, size_t r) {
    for (size_t i = 0; i < n; i++) {
        size_t j = i * r;
        c[i] = a[i] * b[j];
    }
}

void tblas_tensor_bbmul_broadcast(uint8_t *A, uint8_t *B, uint8_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_bsmul_broadcast(uint8_t *A, uint16_t *B, uint16_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_bimul_broadcast(uint8_t *A, uint32_t *B, uint32_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_blmul_broadcast(uint8_t *A, uint64_t *B, uint64_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_bfmul_broadcast(uint8_t *A, float *B, float *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_bdmul_broadcast(uint8_t *A, double *B, double *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_sbmul_broadcast(uint16_t *A, uint8_t *B, uint16_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_ssmul_broadcast(uint16_t *A, uint16_t *B, uint16_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_simul_broadcast(uint16_t *A, uint32_t *B, uint32_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_slmul_broadcast(uint16_t *A, uint64_t *B, uint64_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_sfmul_broadcast(uint16_t *A, float *B, float *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_sdmul_broadcast(uint16_t *A, double *B, double *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_ibmul_broadcast(uint32_t *A, uint8_t *B, uint32_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_ismul_broadcast(uint32_t *A, uint16_t *B, uint32_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_iimul_broadcast(uint32_t *A, uint32_t *B, uint32_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_ilmul_broadcast(uint32_t *A, uint64_t *B, uint64_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_ifmul_broadcast(uint32_t *A, float *B, float *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_idmul_broadcast(uint32_t *A, double *B, double *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_lbmul_broadcast(uint64_t *A, uint8_t *B, uint64_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_lsmul_broadcast(uint64_t *A, uint16_t *B, uint64_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_limul_broadcast(uint64_t *A, uint32_t *B, uint64_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

void tblas_tensor_llmul_broadcast(uint64_t *A, uint64_t *B, uint64_t *C, size_t n, size_t r) {
    tblas_tensor_gemul_broadcast(A, B, C, n, r);
}

#undef FORCE_INLINE