#include "pow.h"
#include "multiply.h"
#include "shapeutils.h"
#include "optimize.h"
#include <cstring>
#include <cmath>
#include <iostream>

// TODO: VECTORIZE POW

template<typename A, typename B, typename C>
void tblas_pow(const A *a, const B *b, C *c,
               const size_t *shapeA, const size_t *stridesA, size_t nDimsA,
               const size_t *shapeB, const size_t *stridesB, size_t nDimsB,
               const size_t *shapeC, const size_t *stridesC, size_t nDimsC) {
    auto *outputIndex = new size_t[nDimsC];
    memset(outputIndex, 0, sizeof(size_t) * nDimsC);

    size_t cIndexFlat = 0;
    do {
        size_t aIndexFlat = getFlatIndexConstrained(outputIndex, shapeA, stridesA, nDimsA, nDimsC);
        size_t bIndexFlat = getFlatIndexConstrained(outputIndex, shapeB, stridesB, nDimsB, nDimsC);
        c[cIndexFlat] = std::pow(a[aIndexFlat], b[bIndexFlat]);
        cIndexFlat++;
    } while (incrementIndex(outputIndex, shapeC, nDimsC));
    delete[] outputIndex;
}

binary_op_hook_optimizations(pow, float,
                             {
                          // check if B is scalar and is 2, then use tblas_multiply
                          if ((nDimsB == 0 || (nDimsB == 1 && shapeB[0] == 1)) && b[0] == 2) {
                              tblas_multiply(
                                      a, a, c,
                                      shapeA, stridesA, nDimsA,
                                      shapeA, stridesA, nDimsA,
                                      shapeC, stridesC, nDimsC
                              );
                              return;
                          }
                      },
                             {
                          auto *outputIndex = new size_t[nDimsC];
                          memset(outputIndex, 0, sizeof(size_t) * nDimsC);

                          size_t cIndexFlat = 0;
                          do {
                              size_t aIndexFlat = getFlatIndexConstrained(outputIndex, shapeA, stridesA, nDimsA,
                                                                          nDimsC);
                              size_t bIndexFlat = getFlatIndexConstrained(outputIndex, shapeB, stridesB, nDimsB,
                                                                          nDimsC);
                              c[cIndexFlat] = std::pow(a[aIndexFlat], b[bIndexFlat]);
                              cIndexFlat++;
                          } while (incrementIndex(outputIndex, shapeC, nDimsC));
                          delete[] outputIndex;
                      }
);


BINARY_OPERATION_FOR_ALL_DATA_TYPES_IMPL(tblas_pow)
