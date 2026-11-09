#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include <stddef.h>
#include <stdint.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MultiHeadAttention MultiHeadAttention;

MultiHeadAttention *multi_head_attention_create(
    size_t embedding_dimension,
    size_t number_of_heads);
MultiHeadAttention *multi_head_attention_create_seeded(
    size_t embedding_dimension,
    size_t number_of_heads,
    uint32_t seed);
void multi_head_attention_free(MultiHeadAttention *layer);

/* Apply self-attention to a sequence with shape [tokens, embedding_dimension]. */
Tensor *multi_head_attention_forward(
    const MultiHeadAttention *layer,
    const Tensor *input);

size_t multi_head_attention_embedding_dimension(const MultiHeadAttention *layer);
size_t multi_head_attention_number_of_heads(const MultiHeadAttention *layer);
size_t multi_head_attention_head_dimension(const MultiHeadAttention *layer);

float *multi_head_attention_query_weights_mut(MultiHeadAttention *layer);
float *multi_head_attention_key_weights_mut(MultiHeadAttention *layer);
float *multi_head_attention_value_weights_mut(MultiHeadAttention *layer);
float *multi_head_attention_output_weights_mut(MultiHeadAttention *layer);
float *multi_head_attention_query_bias_mut(MultiHeadAttention *layer);
float *multi_head_attention_key_bias_mut(MultiHeadAttention *layer);
float *multi_head_attention_value_bias_mut(MultiHeadAttention *layer);
float *multi_head_attention_output_bias_mut(MultiHeadAttention *layer);

/* Backpropagate through self-attention and update all projection parameters. */
Tensor *multi_head_attention_backward(
    MultiHeadAttention *layer,
    const Tensor *input,
    const Tensor *output_gradient,
    float learning_rate);

#ifdef __cplusplus
}
#endif

#endif /* MULTI_HEAD_ATTENTION_H */
