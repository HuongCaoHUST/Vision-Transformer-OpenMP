#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Apply numerically stable softmax to the last dimension of a 2D tensor. */
Tensor *softmax_forward(const Tensor *input);

#ifdef __cplusplus
}
#endif

#endif /* SOFTMAX_H */
