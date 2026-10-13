#ifndef MLP_H
#define MLP_H

#include <stddef.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MlpBlock MlpBlock;

MlpBlock *mlp_create(size_t input_dimension, size_t hidden_dimension);
void mlp_free(MlpBlock *layer);

/* Apply Linear -> GELU -> Linear to a 2D sequence tensor. */
Tensor *mlp_forward(const MlpBlock *layer, const Tensor *input);

size_t mlp_input_dimension(const MlpBlock *layer);
size_t mlp_hidden_dimension(const MlpBlock *layer);

/* Parameter accessors for a future optimizer/training loop. */
float *mlp_input_weights_mut(MlpBlock *layer);
float *mlp_input_bias_mut(MlpBlock *layer);
float *mlp_output_weights_mut(MlpBlock *layer);
float *mlp_output_bias_mut(MlpBlock *layer);

#ifdef __cplusplus
}
#endif

#endif /* MLP_H */
