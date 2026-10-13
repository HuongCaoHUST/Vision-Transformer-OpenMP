#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include <stddef.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MultiHeadAttention MultiHeadAttention;

MultiHeadAttention *multi_head_attention_create(
    size_t embedding_dimension,
    size_t number_of_heads);
void multi_head_attention_free(MultiHeadAttention *layer);

/* Apply self-attention to a sequence with shape [tokens, embedding_dimension]. */
Tensor *multi_head_attention_forward(
    const MultiHeadAttention *layer,
    const Tensor *input);

size_t multi_head_attention_embedding_dimension(const MultiHeadAttention *layer);
size_t multi_head_attention_number_of_heads(const MultiHeadAttention *layer);
size_t multi_head_attention_head_dimension(const MultiHeadAttention *layer);

#ifdef __cplusplus
}
#endif

#endif /* MULTI_HEAD_ATTENTION_H */
