#include "cls_token.h"

#include <stdlib.h>

struct ClsToken {
    size_t embedding_dimension;
    Tensor *value;
};

ClsToken *cls_token_create(size_t embedding_dimension)
{
    ClsToken *layer;
    size_t shape[1];

    if (embedding_dimension == 0) {
        return NULL;
    }

    layer = (ClsToken *)calloc(1, sizeof(*layer));
    if (layer == NULL) {
        return NULL;
    }

    shape[0] = embedding_dimension;
    layer->value = tensor_zeros(shape, 1);
    if (layer->value == NULL) {
        free(layer);
        return NULL;
    }

    layer->embedding_dimension = embedding_dimension;
    return layer;
}

void cls_token_free(ClsToken *layer)
{
    if (layer != NULL) {
        tensor_free(layer->value);
        free(layer);
    }
}

Tensor *cls_token_forward(const ClsToken *layer, const Tensor *patch_embeddings)
{
    const size_t *input_shape;
    size_t output_shape[2];
    const float *input_data;
    const float *cls_data;
    float *output_data;
    Tensor *output;
    size_t sequence_length;
    size_t i;
    size_t j;

    if (layer == NULL || patch_embeddings == NULL
        || tensor_ndim(patch_embeddings) != 2) {
        return NULL;
    }

    input_shape = tensor_shape(patch_embeddings);
    if (input_shape[1] != layer->embedding_dimension
        || input_shape[0] == (size_t)-1) {
        return NULL;
    }

    sequence_length = input_shape[0] + 1;
    output_shape[0] = sequence_length;
    output_shape[1] = layer->embedding_dimension;
    output = tensor_zeros(output_shape, 2);
    if (output == NULL) {
        return NULL;
    }

    input_data = tensor_data(patch_embeddings);
    cls_data = tensor_data(layer->value);
    output_data = tensor_data_mut(output);

    for (j = 0; j < layer->embedding_dimension; ++j) {
        output_data[j] = cls_data[j];
    }

    for (i = 0; i < input_shape[0]; ++i) {
        for (j = 0; j < layer->embedding_dimension; ++j) {
            output_data[(i + 1) * layer->embedding_dimension + j] =
                input_data[i * layer->embedding_dimension + j];
        }
    }

    return output;
}

float *cls_token_data_mut(ClsToken *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->value);
}

size_t cls_token_embedding_dimension(const ClsToken *layer)
{
    return layer == NULL ? 0 : layer->embedding_dimension;
}
