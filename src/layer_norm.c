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

Tensor *layer_norm_backward(
    LayerNorm *layer,
    const Tensor *input,
    const Tensor *output_gradient,
    float learning_rate)
{
    const size_t *input_shape;
    const float *input_data;
    const float *output_gradient_data;
    float *input_gradient_data;
    float *gamma_data;
    float *beta_data;
    Tensor *input_gradient;
    float *gamma_gradient;
    float *beta_gradient;
    size_t row;
    size_t column;

    if (layer == NULL || input == NULL || output_gradient == NULL
        || learning_rate <= 0.0f || tensor_ndim(input) != 2
        || tensor_ndim(output_gradient) != 2) {
        return NULL;
    }

    input_shape = tensor_shape(input);
    if (tensor_shape(output_gradient)[0] != input_shape[0]
        || tensor_shape(output_gradient)[1] != input_shape[1]
        || input_shape[1] != layer->normalized_dimension) {
        return NULL;
    }

    input_gradient = tensor_zeros(input_shape, 2);
    gamma_gradient = (float *)calloc(layer->normalized_dimension, sizeof(*gamma_gradient));
    beta_gradient = (float *)calloc(layer->normalized_dimension, sizeof(*beta_gradient));
    if (input_gradient == NULL || gamma_gradient == NULL || beta_gradient == NULL) {
        tensor_free(input_gradient);
        free(gamma_gradient);
        free(beta_gradient);
        return NULL;
    }

    input_data = tensor_data(input);
    output_gradient_data = tensor_data(output_gradient);
    input_gradient_data = tensor_data_mut(input_gradient);
    gamma_data = tensor_data_mut(layer->gamma);
    beta_data = tensor_data_mut(layer->beta);

    for (row = 0; row < input_shape[0]; ++row) {
        const size_t row_offset = row * layer->normalized_dimension;
        double mean = 0.0;
        double variance = 0.0;
        double sum_scaled_gradient = 0.0;
        double sum_scaled_gradient_normalized = 0.0;
        double inverse_standard_deviation;

        for (column = 0; column < layer->normalized_dimension; ++column) {
            mean += input_data[row_offset + column];
        }
        mean /= (double)layer->normalized_dimension;

        for (column = 0; column < layer->normalized_dimension; ++column) {
            const double difference = input_data[row_offset + column] - mean;
            variance += difference * difference;
        }
        variance /= (double)layer->normalized_dimension;
        inverse_standard_deviation = 1.0 / sqrt(variance + layer->epsilon);

        for (column = 0; column < layer->normalized_dimension; ++column) {
            const double normalized =
                (input_data[row_offset + column] - mean) * inverse_standard_deviation;
            const double gradient = output_gradient_data[row_offset + column];
            const double scaled_gradient = (double)gamma_data[column] * gradient;
            sum_scaled_gradient += scaled_gradient;
            sum_scaled_gradient_normalized += scaled_gradient * normalized;
            gamma_gradient[column] += (float)(gradient * normalized);
            beta_gradient[column] += (float)gradient;
        }

        for (column = 0; column < layer->normalized_dimension; ++column) {
            const double normalized =
                (input_data[row_offset + column] - mean) * inverse_standard_deviation;
            const double gradient = output_gradient_data[row_offset + column];
            const double scaled_gradient = (double)gamma_data[column] * gradient;
            input_gradient_data[row_offset + column] = (float)(
                inverse_standard_deviation
                * (scaled_gradient * layer->normalized_dimension
                   - sum_scaled_gradient
                   - normalized * sum_scaled_gradient_normalized)
                / (double)layer->normalized_dimension);
        }
    }

    for (column = 0; column < layer->normalized_dimension; ++column) {
        gamma_data[column] -= learning_rate * gamma_gradient[column];
        beta_data[column] -= learning_rate * beta_gradient[column];
    }

    free(gamma_gradient);
    free(beta_gradient);

    return input_gradient;
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
