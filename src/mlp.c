#include "mlp.h"

#include <math.h>
#include <stdlib.h>

#include "initializer.h"

struct MlpBlock {
    size_t input_dimension;
    size_t hidden_dimension;
    Tensor *input_weight;
    Tensor *input_bias;
    Tensor *output_weight;
    Tensor *output_bias;
};

static Tensor *create_weight(
    size_t input_dimension,
    size_t output_dimension,
    uint32_t seed)
{
    size_t shape[2] = {input_dimension, output_dimension};
    Tensor *weight = tensor_zeros(shape, 2);

    if (weight == NULL) {
        return NULL;
    }

    (void)tensor_fill_xavier(weight, input_dimension, output_dimension, seed);

    return weight;
}

static Tensor *create_bias(size_t dimension)
{
    size_t shape[1] = {dimension};
    return tensor_zeros(shape, 1);
}

MlpBlock *mlp_create_seeded(
    size_t input_dimension,
    size_t hidden_dimension,
    uint32_t seed)
{
    MlpBlock *layer;

    if (input_dimension == 0 || hidden_dimension == 0) {
        return NULL;
    }

    layer = (MlpBlock *)calloc(1, sizeof(*layer));
    if (layer == NULL) {
        return NULL;
    }

    layer->input_dimension = input_dimension;
    layer->hidden_dimension = hidden_dimension;
    layer->input_weight = create_weight(input_dimension, hidden_dimension, seed + 1u);
    layer->input_bias = create_bias(hidden_dimension);
    layer->output_weight = create_weight(hidden_dimension, input_dimension, seed + 2u);
    layer->output_bias = create_bias(input_dimension);

    if (layer->input_weight == NULL || layer->input_bias == NULL
        || layer->output_weight == NULL || layer->output_bias == NULL) {
        mlp_free(layer);
        return NULL;
    }

    return layer;
}

MlpBlock *mlp_create(size_t input_dimension, size_t hidden_dimension)
{
    return mlp_create_seeded(input_dimension, hidden_dimension, 0x3000u);
}

void mlp_free(MlpBlock *layer)
{
    if (layer != NULL) {
        tensor_free(layer->input_weight);
        tensor_free(layer->input_bias);
        tensor_free(layer->output_weight);
        tensor_free(layer->output_bias);
        free(layer);
    }
}

static void linear_forward(
    const float *input,
    float *output,
    const float *weight,
    const float *bias,
    size_t rows,
    size_t input_dimension,
    size_t output_dimension)
{
    size_t row;
    size_t output_column;
    size_t input_column;

    for (row = 0; row < rows; ++row) {
        for (output_column = 0; output_column < output_dimension; ++output_column) {
            float value = bias[output_column];
            for (input_column = 0; input_column < input_dimension; ++input_column) {
                value += input[row * input_dimension + input_column]
                    * weight[input_column * output_dimension + output_column];
            }
            output[row * output_dimension + output_column] = value;
        }
    }
}

static float gelu(float value)
{
    const float coefficient = 0.7978845608f; /* sqrt(2 / pi) */
    const float cubic_coefficient = 0.044715f;
    const float inner = coefficient * (value + cubic_coefficient * value * value * value);
    return 0.5f * value * (1.0f + tanhf(inner));
}

static float gelu_derivative(float value)
{
    const float coefficient = 0.7978845608f; /* sqrt(2 / pi) */
    const float cubic_coefficient = 0.044715f;
    const float inner = coefficient * (value + cubic_coefficient * value * value * value);
    const float tanh_inner = tanhf(inner);
    const float inner_derivative =
        coefficient * (1.0f + 3.0f * cubic_coefficient * value * value);

    return 0.5f * (1.0f + tanh_inner)
        + 0.5f * value * (1.0f - tanh_inner * tanh_inner) * inner_derivative;
}

Tensor *mlp_forward(const MlpBlock *layer, const Tensor *input)
{
    const size_t *input_shape;
    size_t hidden_shape[2];
    Tensor *hidden;
    Tensor *output;
    const float *input_data;
    float *hidden_data;
    float *output_data;
    size_t i;

    if (layer == NULL || input == NULL || tensor_ndim(input) != 2) {
        return NULL;
    }

    input_shape = tensor_shape(input);
    if (input_shape[1] != layer->input_dimension) {
        return NULL;
    }

    hidden_shape[0] = input_shape[0];
    hidden_shape[1] = layer->hidden_dimension;
    hidden = tensor_zeros(hidden_shape, 2);
    output = tensor_zeros(input_shape, 2);
    if (hidden == NULL || output == NULL) {
        tensor_free(hidden);
        tensor_free(output);
        return NULL;
    }

    input_data = tensor_data(input);
    hidden_data = tensor_data_mut(hidden);
    output_data = tensor_data_mut(output);

    linear_forward(
        input_data,
        hidden_data,
        tensor_data(layer->input_weight),
        tensor_data(layer->input_bias),
        input_shape[0],
        layer->input_dimension,
        layer->hidden_dimension);

    for (i = 0; i < tensor_numel(hidden); ++i) {
        hidden_data[i] = gelu(hidden_data[i]);
    }

    linear_forward(
        hidden_data,
        output_data,
        tensor_data(layer->output_weight),
        tensor_data(layer->output_bias),
        input_shape[0],
        layer->hidden_dimension,
        layer->input_dimension);

    tensor_free(hidden);
    return output;
}

size_t mlp_input_dimension(const MlpBlock *layer)
{
    return layer == NULL ? 0 : layer->input_dimension;
}

size_t mlp_hidden_dimension(const MlpBlock *layer)
{
    return layer == NULL ? 0 : layer->hidden_dimension;
}

float *mlp_input_weights_mut(MlpBlock *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->input_weight);
}

float *mlp_input_bias_mut(MlpBlock *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->input_bias);
}

float *mlp_output_weights_mut(MlpBlock *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->output_weight);
}

float *mlp_output_bias_mut(MlpBlock *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->output_bias);
}

Tensor *mlp_backward(
    MlpBlock *layer,
    const Tensor *input,
    const Tensor *output_gradient,
    float learning_rate)
{
    const size_t *input_shape;
    const float *input_data;
    const float *output_gradient_data;
    float *input_gradient_data;
    float *input_weight_data;
    float *input_bias_data;
    float *output_weight_data;
    float *output_bias_data;
    Tensor *input_gradient;
    float *hidden_linear;
    float *hidden_gradient;
    size_t rows;
    size_t row;
    size_t input_column;
    size_t hidden_column;
    size_t output_column;

    if (layer == NULL || input == NULL || output_gradient == NULL
        || learning_rate <= 0.0f || tensor_ndim(input) != 2
        || tensor_ndim(output_gradient) != 2) {
        return NULL;
    }

    input_shape = tensor_shape(input);
    if (input_shape[1] != layer->input_dimension
        || tensor_shape(output_gradient)[0] != input_shape[0]
        || tensor_shape(output_gradient)[1] != layer->input_dimension) {
        return NULL;
    }

    rows = input_shape[0];
    input_gradient = tensor_zeros(input_shape, 2);
    hidden_linear = (float *)calloc(rows * layer->hidden_dimension, sizeof(*hidden_linear));
    hidden_gradient = (float *)calloc(rows * layer->hidden_dimension, sizeof(*hidden_gradient));
    if (input_gradient == NULL || hidden_linear == NULL || hidden_gradient == NULL) {
        tensor_free(input_gradient);
        free(hidden_linear);
        free(hidden_gradient);
        return NULL;
    }

    input_data = tensor_data(input);
    output_gradient_data = tensor_data(output_gradient);
    input_gradient_data = tensor_data_mut(input_gradient);
    input_weight_data = tensor_data_mut(layer->input_weight);
    input_bias_data = tensor_data_mut(layer->input_bias);
    output_weight_data = tensor_data_mut(layer->output_weight);
    output_bias_data = tensor_data_mut(layer->output_bias);

    for (row = 0; row < rows; ++row) {
        for (hidden_column = 0; hidden_column < layer->hidden_dimension; ++hidden_column) {
            float value = input_bias_data[hidden_column];
            for (input_column = 0; input_column < layer->input_dimension; ++input_column) {
                value += input_data[row * layer->input_dimension + input_column]
                    * input_weight_data[input_column * layer->hidden_dimension + hidden_column];
            }
            hidden_linear[row * layer->hidden_dimension + hidden_column] = value;
        }
    }

    /* Compute all input gradients before changing any MLP parameter. */
    for (row = 0; row < rows; ++row) {
        for (hidden_column = 0; hidden_column < layer->hidden_dimension; ++hidden_column) {
            float gradient = 0.0f;
            for (output_column = 0; output_column < layer->input_dimension; ++output_column) {
                gradient += output_gradient_data[row * layer->input_dimension + output_column]
                    * output_weight_data[hidden_column * layer->input_dimension + output_column];
            }
            hidden_gradient[row * layer->hidden_dimension + hidden_column] =
                gradient * gelu_derivative(
                    hidden_linear[row * layer->hidden_dimension + hidden_column]);
        }

        for (input_column = 0; input_column < layer->input_dimension; ++input_column) {
            float gradient = 0.0f;
            for (hidden_column = 0; hidden_column < layer->hidden_dimension; ++hidden_column) {
                gradient += hidden_gradient[row * layer->hidden_dimension + hidden_column]
                    * input_weight_data[input_column * layer->hidden_dimension + hidden_column];
            }
            input_gradient_data[row * layer->input_dimension + input_column] = gradient;
        }
    }

    for (hidden_column = 0; hidden_column < layer->hidden_dimension; ++hidden_column) {
        float bias_gradient = 0.0f;
        for (row = 0; row < rows; ++row) {
            bias_gradient += hidden_gradient[row * layer->hidden_dimension + hidden_column];
            for (input_column = 0; input_column < layer->input_dimension; ++input_column) {
                input_weight_data[input_column * layer->hidden_dimension + hidden_column] -=
                    learning_rate
                    * input_data[row * layer->input_dimension + input_column]
                    * hidden_gradient[row * layer->hidden_dimension + hidden_column];
            }
        }
        input_bias_data[hidden_column] -= learning_rate * bias_gradient;
    }

    for (hidden_column = 0; hidden_column < layer->hidden_dimension; ++hidden_column) {
        for (output_column = 0; output_column < layer->input_dimension; ++output_column) {
            float weight_gradient = 0.0f;
            for (row = 0; row < rows; ++row) {
                const float activated = gelu(
                    hidden_linear[row * layer->hidden_dimension + hidden_column]);
                weight_gradient += activated
                    * output_gradient_data[row * layer->input_dimension + output_column];
            }
            output_weight_data[hidden_column * layer->input_dimension + output_column] -=
                learning_rate * weight_gradient;
        }
    }

    for (output_column = 0; output_column < layer->input_dimension; ++output_column) {
        float bias_gradient = 0.0f;
        for (row = 0; row < rows; ++row) {
            bias_gradient += output_gradient_data[row * layer->input_dimension + output_column];
        }
        output_bias_data[output_column] -= learning_rate * bias_gradient;
    }

    free(hidden_linear);
    free(hidden_gradient);
    return input_gradient;
}
