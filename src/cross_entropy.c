#include "cross_entropy.h"

#include <float.h>
#include <math.h>

TensorStatus cross_entropy_loss(
    const Tensor *logits,
    size_t target_class,
    float *loss,
    Tensor **gradient)
{
    const size_t *shape;
    const float *logit_data;
    float *gradient_data;
    Tensor *gradient_tensor;
    float maximum;
    float sum_exp = 0.0f;
    size_t class_index;

    if (logits == NULL || loss == NULL || gradient == NULL
        || tensor_ndim(logits) != 2) {
        return TENSOR_ERROR_INVALID_ARGUMENT;
    }

    shape = tensor_shape(logits);
    if (shape[0] != 1 || target_class >= shape[1]) {
        return TENSOR_ERROR_INVALID_ARGUMENT;
    }

    gradient_tensor = tensor_zeros(shape, 2);
    if (gradient_tensor == NULL) {
        return TENSOR_ERROR_ALLOCATION;
    }

    logit_data = tensor_data(logits);
    gradient_data = tensor_data_mut(gradient_tensor);
    maximum = logit_data[0];

    for (class_index = 1; class_index < shape[1]; ++class_index) {
        if (logit_data[class_index] > maximum) {
            maximum = logit_data[class_index];
        }
    }

    for (class_index = 0; class_index < shape[1]; ++class_index) {
        sum_exp += expf(logit_data[class_index] - maximum);
    }

    if (sum_exp <= FLT_MIN) {
        tensor_free(gradient_tensor);
        return TENSOR_ERROR_INVALID_ARGUMENT;
    }

    *loss = logf(sum_exp) + maximum - logit_data[target_class];

    for (class_index = 0; class_index < shape[1]; ++class_index) {
        gradient_data[class_index] =
            expf(logit_data[class_index] - maximum) / sum_exp;
    }
    gradient_data[target_class] -= 1.0f;

    *gradient = gradient_tensor;
    return TENSOR_OK;
}
