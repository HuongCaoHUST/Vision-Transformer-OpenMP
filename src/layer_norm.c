#include "layer_norm.h"

#include <math.h>
#include <stdlib.h>

struct LayerNorm {
    size_t normalized_dimension;
    float epsilon;
    Tensor *gamma;
    Tensor *beta;
};

LayerNorm *layer_norm_create(size_t normalized_dimension, float epsilon)
{
    LayerNorm *layer;
    size_t shape[1];
    size_t i;

    if (normalized_dimension == 0 || epsilon <= 0.0f) {
        return NULL;
    }

    layer = (LayerNorm *)calloc(1, sizeof(*layer));
    if (layer == NULL) {
        return NULL;
    }

    shape[0] = normalized_dimension;
    layer->gamma = tensor_ones(shape, 1);
    layer->beta = tensor_zeros(shape, 1);
    if (layer->gamma == NULL || layer->beta == NULL) {
        layer_norm_free(layer);
        return NULL;
    }

    /* Gamma is initialized to one and beta to zero. */
    for (i = 0; i < normalized_dimension; ++i) {
        tensor_data_mut(layer->gamma)[i] = 1.0f;
        tensor_data_mut(layer->beta)[i] = 0.0f;
    }

    layer->normalized_dimension = normalized_dimension;
    layer->epsilon = epsilon;
    return layer;
}

void layer_norm_free(LayerNorm *layer)
{
    if (layer != NULL) {
        tensor_free(layer->gamma);
        tensor_free(layer->beta);
        free(layer);
    }
}

Tensor *layer_norm_forward(const LayerNorm *layer, const Tensor *input)
{
    const size_t *shape;
    const float *input_data;
    const float *gamma;
    const float *beta;
    float *output_data;
    Tensor *output;
    size_t row;
    size_t column;

    if (layer == NULL || input == NULL || tensor_ndim(input) != 2) {
        return NULL;
    }

    shape = tensor_shape(input);
    if (shape[1] != layer->normalized_dimension) {
        return NULL;
    }

    output = tensor_create(shape, 2);
    if (output == NULL) {
        return NULL;
    }

    input_data = tensor_data(input);
    gamma = tensor_data(layer->gamma);
    beta = tensor_data(layer->beta);
    output_data = tensor_data_mut(output);

    for (row = 0; row < shape[0]; ++row) {
        double mean = 0.0;
        double variance = 0.0;
        const size_t row_offset = row * layer->normalized_dimension;

        for (column = 0; column < layer->normalized_dimension; ++column) {
            mean += input_data[row_offset + column];
        }
        mean /= (double)layer->normalized_dimension;

        for (column = 0; column < layer->normalized_dimension; ++column) {
            const double difference = input_data[row_offset + column] - mean;
            variance += difference * difference;
        }
        variance /= (double)layer->normalized_dimension;

        for (column = 0; column < layer->normalized_dimension; ++column) {
            const float normalized = (float)(
                (input_data[row_offset + column] - mean)
                / sqrt(variance + layer->epsilon));
            output_data[row_offset + column] = gamma[column] * normalized + beta[column];
        }
    }

    return output;
}

float *layer_norm_gamma_mut(LayerNorm *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->gamma);
}

float *layer_norm_beta_mut(LayerNorm *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->beta);
}

size_t layer_norm_dimension(const LayerNorm *layer)
{
    return layer == NULL ? 0 : layer->normalized_dimension;
}
