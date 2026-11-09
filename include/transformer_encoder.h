#ifndef TRANSFORMER_ENCODER_H
#define TRANSFORMER_ENCODER_H

#include <stddef.h>
#include <stdint.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TransformerEncoder TransformerEncoder;

TransformerEncoder *transformer_encoder_create(
    size_t embedding_dimension,
    size_t number_of_heads,
    size_t mlp_hidden_dimension);
TransformerEncoder *transformer_encoder_create_seeded(
    size_t embedding_dimension,
    size_t number_of_heads,
    size_t mlp_hidden_dimension,
    uint32_t seed);
void transformer_encoder_free(TransformerEncoder *block);

/* Apply a pre-normalized Transformer encoder block to [tokens, embedding]. */
Tensor *transformer_encoder_forward(
    const TransformerEncoder *block,
    const Tensor *input);

/* Backpropagate through the complete pre-norm block and update parameters. */
Tensor *transformer_encoder_backward(
    TransformerEncoder *block,
    const Tensor *input,
    const Tensor *output_gradient,
    float learning_rate);

size_t transformer_encoder_embedding_dimension(const TransformerEncoder *block);

#ifdef __cplusplus
}
#endif

#endif /* TRANSFORMER_ENCODER_H */
