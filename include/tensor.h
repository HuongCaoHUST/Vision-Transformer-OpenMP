#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TENSOR_MAX_DIMS 8

/* Opaque tensor handle: callers cannot depend on the internal representation. */
typedef struct Tensor Tensor;

typedef enum {
    TENSOR_OK = 0,
    TENSOR_ERROR_INVALID_ARGUMENT,
    TENSOR_ERROR_ALLOCATION,
    TENSOR_ERROR_SHAPE_MISMATCH,
    TENSOR_ERROR_OUT_OF_BOUNDS
} TensorStatus;

Tensor *tensor_create(const size_t *shape, size_t ndim);
Tensor *tensor_zeros(const size_t *shape, size_t ndim);
Tensor *tensor_ones(const size_t *shape, size_t ndim);
Tensor *tensor_clone(const Tensor *tensor);
void tensor_free(Tensor *tensor);

size_t tensor_numel(const Tensor *tensor);
size_t tensor_ndim(const Tensor *tensor);
const size_t *tensor_shape(const Tensor *tensor);
const float *tensor_data(const Tensor *tensor);
float *tensor_data_mut(Tensor *tensor);

TensorStatus tensor_fill(Tensor *tensor, float value);
TensorStatus tensor_get(const Tensor *tensor, const size_t *indices, float *value);
TensorStatus tensor_set(Tensor *tensor, const size_t *indices, float value);

/* Element-wise addition. The inputs and output must have the same shape. */
TensorStatus tensor_add(const Tensor *left, const Tensor *right, Tensor *out);
TensorStatus tensor_scale(const Tensor *tensor, float scalar, Tensor *out);

/* Matrix operations for two-dimensional tensors. */
Tensor *tensor_matmul(const Tensor *left, const Tensor *right);
Tensor *tensor_transpose2d(const Tensor *tensor);

const char *tensor_status_string(TensorStatus status);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_H */
