#include "position_embedding.h"

#include <stdlib.h>

struct PositionEmbedding {
    size_t sequence_length;
    size_t embedding_dimension;
    Tensor *value;
};

PositionEmbedding *position_embedding_create(
    size_t sequence_length,
    size_t embedding_dimension)
{
    PositionEmbedding *layer;
    size_t shape[2];

    if (sequence_length == 0 || embedding_dimension == 0) {
        return NULL;
    }

    layer = (PositionEmbedding *)calloc(1, sizeof(*layer));
    if (layer == NULL) {
        return NULL;
    }

    shape[0] = sequence_length;
    shape[1] = embedding_dimension;
    layer->value = tensor_zeros(shape, 2);
    if (layer->value == NULL) {
        free(layer);
        return NULL;
    }

    layer->sequence_length = sequence_length;
    layer->embedding_dimension = embedding_dimension;
    return layer;
}

void position_embedding_free(PositionEmbedding *layer)
{
    if (layer != NULL) {
        tensor_free(layer->value);
        free(layer);
    }
}

Tensor *position_embedding_forward(
    const PositionEmbedding *layer,
    const Tensor *sequence)
{
    const size_t *sequence_shape;
    const float *sequence_data;
    const float *position_data;
    float *output_data;
    Tensor *output;
    size_t i;
    size_t j;

    if (layer == NULL || sequence == NULL || tensor_ndim(sequence) != 2) {
        return NULL;
    }

    sequence_shape = tensor_shape(sequence);
    if (sequence_shape[0] != layer->sequence_length
        || sequence_shape[1] != layer->embedding_dimension) {
        return NULL;
    }

    output = tensor_clone(sequence);
    if (output == NULL) {
        return NULL;
    }

    sequence_data = tensor_data(sequence);
    position_data = tensor_data(layer->value);
    output_data = tensor_data_mut(output);

    for (i = 0; i < layer->sequence_length; ++i) {
        for (j = 0; j < layer->embedding_dimension; ++j) {
            output_data[i * layer->embedding_dimension + j] =
                sequence_data[i * layer->embedding_dimension + j]
                + position_data[i * layer->embedding_dimension + j];
        }
    }

    return output;
}

float *position_embedding_data_mut(PositionEmbedding *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->value);
}

size_t position_embedding_sequence_length(const PositionEmbedding *layer)
{
    return layer == NULL ? 0 : layer->sequence_length;
}

size_t position_embedding_embedding_dimension(const PositionEmbedding *layer)
{
    return layer == NULL ? 0 : layer->embedding_dimension;
}
