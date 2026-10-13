#include "multi_head_attention.h"

#include <math.h>
#include <stdlib.h>

#include "initializer.h"

struct MultiHeadAttention {
    size_t embedding_dimension;
    size_t number_of_heads;
    size_t head_dimension;
    Tensor *query_weight;
    Tensor *key_weight;
    Tensor *value_weight;
    Tensor *output_weight;
    Tensor *query_bias;
    Tensor *key_bias;
    Tensor *value_bias;
    Tensor *output_bias;
};

static void initialize_projection(Tensor *matrix, size_t dimension, uint32_t seed)
{
    (void)tensor_fill_xavier(matrix, dimension, dimension, seed);
}

static Tensor *create_projection(size_t embedding_dimension, uint32_t seed)
{
    size_t shape[2] = {embedding_dimension, embedding_dimension};
    Tensor *projection = tensor_zeros(shape, 2);

    if (projection != NULL) {
        initialize_projection(projection, embedding_dimension, seed);
    }

    return projection;
}

static Tensor *create_bias(size_t embedding_dimension)
{
    size_t shape[1] = {embedding_dimension};
    return tensor_zeros(shape, 1);
}

MultiHeadAttention *multi_head_attention_create(
    size_t embedding_dimension,
    size_t number_of_heads)
{
    MultiHeadAttention *layer;

    if (embedding_dimension == 0 || number_of_heads == 0
        || embedding_dimension % number_of_heads != 0) {
        return NULL;
    }

    layer = (MultiHeadAttention *)calloc(1, sizeof(*layer));
    if (layer == NULL) {
        return NULL;
    }

    layer->embedding_dimension = embedding_dimension;
    layer->number_of_heads = number_of_heads;
    layer->head_dimension = embedding_dimension / number_of_heads;
    layer->query_weight = create_projection(embedding_dimension, 0x2001u);
    layer->key_weight = create_projection(embedding_dimension, 0x2002u);
    layer->value_weight = create_projection(embedding_dimension, 0x2003u);
    layer->output_weight = create_projection(embedding_dimension, 0x2004u);
    layer->query_bias = create_bias(embedding_dimension);
    layer->key_bias = create_bias(embedding_dimension);
    layer->value_bias = create_bias(embedding_dimension);
    layer->output_bias = create_bias(embedding_dimension);

    if (layer->query_weight == NULL || layer->key_weight == NULL
        || layer->value_weight == NULL || layer->output_weight == NULL
        || layer->query_bias == NULL || layer->key_bias == NULL
        || layer->value_bias == NULL || layer->output_bias == NULL) {
        multi_head_attention_free(layer);
        return NULL;
    }

    return layer;
}

void multi_head_attention_free(MultiHeadAttention *layer)
{
    if (layer != NULL) {
        tensor_free(layer->query_weight);
        tensor_free(layer->key_weight);
        tensor_free(layer->value_weight);
        tensor_free(layer->output_weight);
        tensor_free(layer->query_bias);
        tensor_free(layer->key_bias);
        tensor_free(layer->value_bias);
        tensor_free(layer->output_bias);
        free(layer);
    }
}

static void project_sequence(
    const float *input,
    float *output,
    const float *weight,
    const float *bias,
    size_t tokens,
    size_t dimension)
{
    size_t token;
    size_t output_column;
    size_t input_column;

    for (token = 0; token < tokens; ++token) {
        for (output_column = 0; output_column < dimension; ++output_column) {
            float value = bias[output_column];
            for (input_column = 0; input_column < dimension; ++input_column) {
                value += input[token * dimension + input_column]
                    * weight[input_column * dimension + output_column];
            }
            output[token * dimension + output_column] = value;
        }
    }
}

Tensor *multi_head_attention_forward(
    const MultiHeadAttention *layer,
    const Tensor *input)
{
    const size_t *input_shape;
    size_t output_shape[2];
    size_t tokens;
    size_t dimension;
    size_t head;
    size_t query_position;
    size_t key_position;
    size_t head_column;
    Tensor *query;
    Tensor *key;
    Tensor *value;
    Tensor *context;
    Tensor *output;
    const float *input_data;
    float *query_data;
    float *key_data;
    float *value_data;
    float *context_data;
    float *output_data;
    float *scores;

    if (layer == NULL || input == NULL || tensor_ndim(input) != 2) {
        return NULL;
    }

    input_shape = tensor_shape(input);
    tokens = input_shape[0];
    dimension = input_shape[1];
    if (tokens == 0 || dimension != layer->embedding_dimension) {
        return NULL;
    }

    output_shape[0] = tokens;
    output_shape[1] = dimension;
    query = tensor_zeros(output_shape, 2);
    key = tensor_zeros(output_shape, 2);
    value = tensor_zeros(output_shape, 2);
    context = tensor_zeros(output_shape, 2);
    output = tensor_zeros(output_shape, 2);
    scores = (float *)malloc(tokens * sizeof(*scores));

    if (query == NULL || key == NULL || value == NULL || context == NULL
        || output == NULL || scores == NULL) {
        tensor_free(query);
        tensor_free(key);
        tensor_free(value);
        tensor_free(context);
        tensor_free(output);
        free(scores);
        return NULL;
    }

    input_data = tensor_data(input);
    query_data = tensor_data_mut(query);
    key_data = tensor_data_mut(key);
    value_data = tensor_data_mut(value);
    context_data = tensor_data_mut(context);
    output_data = tensor_data_mut(output);

    project_sequence(input_data, query_data, tensor_data(layer->query_weight),
                     tensor_data(layer->query_bias), tokens, dimension);
    project_sequence(input_data, key_data, tensor_data(layer->key_weight),
                     tensor_data(layer->key_bias), tokens, dimension);
    project_sequence(input_data, value_data, tensor_data(layer->value_weight),
                     tensor_data(layer->value_bias), tokens, dimension);

    for (head = 0; head < layer->number_of_heads; ++head) {
        const size_t head_offset = head * layer->head_dimension;
        const float scale = 1.0f / sqrtf((float)layer->head_dimension);

        for (query_position = 0; query_position < tokens; ++query_position) {
            float maximum = -INFINITY;
            float sum = 0.0f;

            for (key_position = 0; key_position < tokens; ++key_position) {
                float score = 0.0f;
                for (head_column = 0; head_column < layer->head_dimension; ++head_column) {
                    score += query_data[query_position * dimension + head_offset + head_column]
                        * key_data[key_position * dimension + head_offset + head_column];
                }
                scores[key_position] = score * scale;
                if (scores[key_position] > maximum) {
                    maximum = scores[key_position];
                }
            }

            for (key_position = 0; key_position < tokens; ++key_position) {
                scores[key_position] = expf(scores[key_position] - maximum);
                sum += scores[key_position];
            }

            for (head_column = 0; head_column < layer->head_dimension; ++head_column) {
                float weighted_value = 0.0f;
                for (key_position = 0; key_position < tokens; ++key_position) {
                    weighted_value += (scores[key_position] / sum)
                        * value_data[key_position * dimension + head_offset + head_column];
                }
                context_data[query_position * dimension + head_offset + head_column] =
                    weighted_value;
            }
        }
    }

    project_sequence(context_data, output_data, tensor_data(layer->output_weight),
                     tensor_data(layer->output_bias), tokens, dimension);

    tensor_free(query);
    tensor_free(key);
    tensor_free(value);
    tensor_free(context);
    free(scores);
    return output;
}

size_t multi_head_attention_embedding_dimension(const MultiHeadAttention *layer)
{
    return layer == NULL ? 0 : layer->embedding_dimension;
}

size_t multi_head_attention_number_of_heads(const MultiHeadAttention *layer)
{
    return layer == NULL ? 0 : layer->number_of_heads;
}

size_t multi_head_attention_head_dimension(const MultiHeadAttention *layer)
{
    return layer == NULL ? 0 : layer->head_dimension;
}
