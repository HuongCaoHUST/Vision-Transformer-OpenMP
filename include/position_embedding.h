#ifndef POSITION_EMBEDDING_H
#define POSITION_EMBEDDING_H

#include <stddef.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PositionEmbedding PositionEmbedding;

PositionEmbedding *position_embedding_create(
    size_t sequence_length,
    size_t embedding_dimension);
void position_embedding_free(PositionEmbedding *layer);

/* Add a learnable position vector to each sequence element. */
Tensor *position_embedding_forward(
    const PositionEmbedding *layer,
    const Tensor *sequence);

float *position_embedding_data_mut(PositionEmbedding *layer);
size_t position_embedding_sequence_length(const PositionEmbedding *layer);
size_t position_embedding_embedding_dimension(const PositionEmbedding *layer);

#ifdef __cplusplus
}
#endif

#endif /* POSITION_EMBEDDING_H */
