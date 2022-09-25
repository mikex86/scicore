#include "reducesum.h"
#include "forceinline.h"

template<typename T>
FORCE_INLINE void
tinyblas_tensor_reduce_gesum(T *a, dim_t *shape, size_t nDims, dim_t sumDim, T **sumOut, dim_t **shapeOut,
                             size_t *nDimsOutShape) {
    if (sumDim == -1) {
        // sum over every dimension
        *shapeOut = nullptr;
        *nDimsOutShape = 0;

        T sum = 0;
        for (size_t i = 0; i < nDims; i++) {
            sum += a[i];
        }
        *sumOut = sum;
    } else {
        // sum over a single dimension
        *nDimsOutShape = nDims - 1;
        *shapeOut = new dim_t[*nDimsOutShape];
        for (size_t i = 0; i < sumDim; i++) {
            (*shapeOut)[i] = shape[i];
        }
        for (size_t i = sumDim + 1; i < nDims; i++) {
            (*shapeOut)[i - 1] = shape[i];
        }

        dim_t sumSize = shape[sumDim];
        dim_t nElements = 1;
        for (size_t i = 0; i < nDims; i++) {
            nElements *= shape[i];
        }
        dim_t nSums = nElements / sumSize;

        T *result = new T[nSums];

        for (dim_t i = 0; i < nSums; i++) {
            T sum = 0;
            for (dim_t j = 0; j < sumSize; j++) {
                sum += a[i * sumSize + j];
            }
            result[i] = sum;
        }
        *sumOut = result;
    }
}
