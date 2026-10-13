#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include <stddef.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Compute cross-entropy for logits with shape [1, number_of_classes].
 * The returned gradient has the same shape as logits.
 */
TensorStatus cross_entropy_loss(
    const Tensor *logits,
    size_t target_class,
    float *loss,
    Tensor **gradient);

#ifdef __cplusplus
}
#endif

#endif /* CROSS_ENTROPY_H */
