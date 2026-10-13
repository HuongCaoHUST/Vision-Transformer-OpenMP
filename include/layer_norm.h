#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include <stddef.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LayerNorm LayerNorm;

LayerNorm *layer_norm_create(size_t normalized_dimension, float epsilon);
void layer_norm_free(LayerNorm *layer);

/* Normalize the last dimension of a 2D sequence tensor. */
Tensor *layer_norm_forward(const LayerNorm *layer, const Tensor *input);

float *layer_norm_gamma_mut(LayerNorm *layer);
float *layer_norm_beta_mut(LayerNorm *layer);
size_t layer_norm_dimension(const LayerNorm *layer);

#ifdef __cplusplus
}
#endif

#endif /* LAYER_NORM_H */
