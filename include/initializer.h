#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <stddef.h>
#include <stdint.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fill a tensor with deterministic Xavier/Glorot uniform values. */
TensorStatus tensor_fill_xavier(
    Tensor *tensor,
    size_t fan_in,
    size_t fan_out,
    uint32_t seed);

#ifdef __cplusplus
}
#endif

#endif /* INITIALIZER_H */
