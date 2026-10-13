#include "tensor.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct Tensor {
    size_t ndim;
    size_t shape[TENSOR_MAX_DIMS];
    size_t numel;
    float *data;
};

static int size_multiply_overflows(size_t left, size_t right)
{
    return right != 0 && left > SIZE_MAX / right;
}

static int shape_is_valid(const size_t *shape, size_t ndim, size_t *numel)
{
    size_t i;
    size_t elements = 1;

    if (shape == NULL || numel == NULL || ndim == 0 || ndim > TENSOR_MAX_DIMS) {
        return 0;
    }

    for (i = 0; i < ndim; ++i) {
        if (shape[i] == 0 || size_multiply_overflows(elements, shape[i])) {
            return 0;
        }
        elements *= shape[i];
    }

    if (size_multiply_overflows(elements, sizeof(float))) {
        return 0;
    }

    *numel = elements;
    return 1;
}

static int tensors_have_same_shape(const Tensor *left, const Tensor *right)
{
    size_t i;

    if (left == NULL || right == NULL || left->ndim != right->ndim) {
        return 0;
    }

    for (i = 0; i < left->ndim; ++i) {
        if (left->shape[i] != right->shape[i]) {
            return 0;
        }
    }

    return 1;
}

static size_t flat_index(const Tensor *tensor, const size_t *indices)
{
    size_t i;
    size_t index = 0;

    for (i = 0; i < tensor->ndim; ++i) {
        index = index * tensor->shape[i] + indices[i];
    }

    return index;
}

Tensor *tensor_create(const size_t *shape, size_t ndim)
{
    Tensor *tensor;
    size_t numel;

    if (!shape_is_valid(shape, ndim, &numel)) {
        return NULL;
    }

    tensor = (Tensor *)calloc(1, sizeof(*tensor));
    if (tensor == NULL) {
        return NULL;
    }

    tensor->data = (float *)calloc(numel, sizeof(*tensor->data));
    if (tensor->data == NULL) {
        free(tensor);
        return NULL;
    }

    tensor->ndim = ndim;
    tensor->numel = numel;
    memcpy(tensor->shape, shape, ndim * sizeof(*shape));
    return tensor;
}

Tensor *tensor_zeros(const size_t *shape, size_t ndim)
{
    return tensor_create(shape, ndim);
}

Tensor *tensor_ones(const size_t *shape, size_t ndim)
{
    Tensor *tensor = tensor_create(shape, ndim);

    if (tensor != NULL) {
        (void)tensor_fill(tensor, 1.0f);
    }

    return tensor;
}

Tensor *tensor_clone(const Tensor *tensor)
{
    Tensor *copy;

    if (tensor == NULL) {
        return NULL;
    }

    copy = tensor_create(tensor->shape, tensor->ndim);
    if (copy != NULL) {
        memcpy(copy->data, tensor->data, tensor->numel * sizeof(*tensor->data));
    }

    return copy;
}

void tensor_free(Tensor *tensor)
{
    if (tensor != NULL) {
        free(tensor->data);
        free(tensor);
    }
}

size_t tensor_numel(const Tensor *tensor)
{
    return tensor == NULL ? 0 : tensor->numel;
}

size_t tensor_ndim(const Tensor *tensor)
{
    return tensor == NULL ? 0 : tensor->ndim;
}

const size_t *tensor_shape(const Tensor *tensor)
{
    return tensor == NULL ? NULL : tensor->shape;
}

const float *tensor_data(const Tensor *tensor)
{
    return tensor == NULL ? NULL : tensor->data;
}

float *tensor_data_mut(Tensor *tensor)
{
    return tensor == NULL ? NULL : tensor->data;
}

TensorStatus tensor_fill(Tensor *tensor, float value)
{
    size_t i;

    if (tensor == NULL) {
        return TENSOR_ERROR_INVALID_ARGUMENT;
    }

    for (i = 0; i < tensor->numel; ++i) {
        tensor->data[i] = value;
    }

    return TENSOR_OK;
}

TensorStatus tensor_get(const Tensor *tensor, const size_t *indices, float *value)
{
    size_t i;

    if (tensor == NULL || indices == NULL || value == NULL) {
        return TENSOR_ERROR_INVALID_ARGUMENT;
    }

    for (i = 0; i < tensor->ndim; ++i) {
        if (indices[i] >= tensor->shape[i]) {
            return TENSOR_ERROR_OUT_OF_BOUNDS;
        }
    }

    *value = tensor->data[flat_index(tensor, indices)];
    return TENSOR_OK;
}

TensorStatus tensor_set(Tensor *tensor, const size_t *indices, float value)
{
    size_t i;

    if (tensor == NULL || indices == NULL) {
        return TENSOR_ERROR_INVALID_ARGUMENT;
    }

    for (i = 0; i < tensor->ndim; ++i) {
        if (indices[i] >= tensor->shape[i]) {
            return TENSOR_ERROR_OUT_OF_BOUNDS;
        }
    }

    tensor->data[flat_index(tensor, indices)] = value;
    return TENSOR_OK;
}

TensorStatus tensor_add(const Tensor *left, const Tensor *right, Tensor *out)
{
    size_t i;

    if (left == NULL || right == NULL || out == NULL) {
        return TENSOR_ERROR_INVALID_ARGUMENT;
    }
    if (!tensors_have_same_shape(left, right) || !tensors_have_same_shape(left, out)) {
        return TENSOR_ERROR_SHAPE_MISMATCH;
    }

    for (i = 0; i < left->numel; ++i) {
        out->data[i] = left->data[i] + right->data[i];
    }

    return TENSOR_OK;
}

TensorStatus tensor_scale(const Tensor *tensor, float scalar, Tensor *out)
{
    size_t i;

    if (tensor == NULL || out == NULL) {
        return TENSOR_ERROR_INVALID_ARGUMENT;
    }
    if (!tensors_have_same_shape(tensor, out)) {
        return TENSOR_ERROR_SHAPE_MISMATCH;
    }

    for (i = 0; i < tensor->numel; ++i) {
        out->data[i] = tensor->data[i] * scalar;
    }

    return TENSOR_OK;
}

Tensor *tensor_matmul(const Tensor *left, const Tensor *right)
{
    size_t i;
    size_t j;
    size_t k;
    size_t result_shape[2];
    Tensor *result;

    if (left == NULL || right == NULL || left->ndim != 2 || right->ndim != 2) {
        return NULL;
    }
    if (left->shape[1] != right->shape[0]) {
        return NULL;
    }

    result_shape[0] = left->shape[0];
    result_shape[1] = right->shape[1];
    result = tensor_create(result_shape, 2);
    if (result == NULL) {
        return NULL;
    }

    for (i = 0; i < left->shape[0]; ++i) {
        for (j = 0; j < right->shape[1]; ++j) {
            float sum = 0.0f;
            for (k = 0; k < left->shape[1]; ++k) {
                sum += left->data[i * left->shape[1] + k]
                    * right->data[k * right->shape[1] + j];
            }
            result->data[i * result->shape[1] + j] = sum;
        }
    }

    return result;
}

Tensor *tensor_transpose2d(const Tensor *tensor)
{
    size_t i;
    size_t j;
    size_t result_shape[2];
    Tensor *result;

    if (tensor == NULL || tensor->ndim != 2) {
        return NULL;
    }

    result_shape[0] = tensor->shape[1];
    result_shape[1] = tensor->shape[0];
    result = tensor_create(result_shape, 2);
    if (result == NULL) {
        return NULL;
    }

    for (i = 0; i < tensor->shape[0]; ++i) {
        for (j = 0; j < tensor->shape[1]; ++j) {
            result->data[j * result->shape[1] + i] = tensor->data[i * tensor->shape[1] + j];
        }
    }

    return result;
}

const char *tensor_status_string(TensorStatus status)
{
    switch (status) {
    case TENSOR_OK:
        return "Success";
    case TENSOR_ERROR_INVALID_ARGUMENT:
        return "Invalid argument";
    case TENSOR_ERROR_ALLOCATION:
        return "Memory allocation failed";
    case TENSOR_ERROR_SHAPE_MISMATCH:
        return "Tensor shape mismatch";
    case TENSOR_ERROR_OUT_OF_BOUNDS:
        return "Tensor index out of bounds";
    default:
        return "Unknown tensor error";
    }
}
