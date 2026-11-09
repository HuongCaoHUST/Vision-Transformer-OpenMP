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

MultiHeadAttention *multi_head_attention_create_seeded(
    size_t embedding_dimension,
    size_t number_of_heads,
    uint32_t seed)
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
    layer->query_weight = create_projection(embedding_dimension, seed + 1u);
    layer->key_weight = create_projection(embedding_dimension, seed + 2u);
    layer->value_weight = create_projection(embedding_dimension, seed + 3u);
    layer->output_weight = create_projection(embedding_dimension, seed + 4u);
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

MultiHeadAttention *multi_head_attention_create(
    size_t embedding_dimension,
    size_t number_of_heads)
{
    return multi_head_attention_create_seeded(embedding_dimension, number_of_heads, 0x2000u);
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

float *multi_head_attention_query_weights_mut(MultiHeadAttention *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->query_weight);
}

float *multi_head_attention_key_weights_mut(MultiHeadAttention *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->key_weight);
}

float *multi_head_attention_value_weights_mut(MultiHeadAttention *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->value_weight);
}

float *multi_head_attention_output_weights_mut(MultiHeadAttention *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->output_weight);
}

float *multi_head_attention_query_bias_mut(MultiHeadAttention *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->query_bias);
}

float *multi_head_attention_key_bias_mut(MultiHeadAttention *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->key_bias);
}

float *multi_head_attention_value_bias_mut(MultiHeadAttention *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->value_bias);
}

float *multi_head_attention_output_bias_mut(MultiHeadAttention *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->output_bias);
}

Tensor *multi_head_attention_backward(
    MultiHeadAttention *layer,
    const Tensor *input,
    const Tensor *output_gradient,
    float learning_rate)
{
    const size_t *input_shape;
    const float *input_data;
    const float *output_gradient_data;
    float *input_gradient_data;
    float *query_weight;
    float *key_weight;
    float *value_weight;
    float *output_weight;
    float *query_bias;
    float *key_bias;
    float *value_bias;
    float *output_bias;
    float *query;
    float *key;
    float *value;
    float *context;
    float *attention;
    float *query_gradient;
    float *key_gradient;
    float *value_gradient;
    float *context_gradient;
    float *query_weight_gradient;
    float *key_weight_gradient;
    float *value_weight_gradient;
    float *output_weight_gradient;
    float *query_bias_gradient;
    float *key_bias_gradient;
    float *value_bias_gradient;
    float *output_bias_gradient;
    Tensor *input_gradient;
    size_t tokens;
    size_t dimension;
    size_t head_dimension;
    size_t head;
    size_t query_position;
    size_t key_position;
    size_t input_column;
    size_t output_column;
    size_t head_column;
    size_t attention_stride;
    float *scores;

    if (layer == NULL || input == NULL || output_gradient == NULL
        || learning_rate <= 0.0f || tensor_ndim(input) != 2
        || tensor_ndim(output_gradient) != 2) {
        return NULL;
    }

    input_shape = tensor_shape(input);
    tokens = input_shape[0];
    dimension = input_shape[1];
    if (tokens == 0 || dimension != layer->embedding_dimension
        || tensor_shape(output_gradient)[0] != tokens
        || tensor_shape(output_gradient)[1] != dimension) {
        return NULL;
    }
    head_dimension = layer->head_dimension;
    attention_stride = tokens * tokens;

    input_gradient = tensor_zeros(input_shape, 2);
    query = (float *)calloc(tokens * dimension, sizeof(*query));
    key = (float *)calloc(tokens * dimension, sizeof(*key));
    value = (float *)calloc(tokens * dimension, sizeof(*value));
    context = (float *)calloc(tokens * dimension, sizeof(*context));
    attention = (float *)calloc(layer->number_of_heads * attention_stride, sizeof(*attention));
    query_gradient = (float *)calloc(tokens * dimension, sizeof(*query_gradient));
    key_gradient = (float *)calloc(tokens * dimension, sizeof(*key_gradient));
    value_gradient = (float *)calloc(tokens * dimension, sizeof(*value_gradient));
    context_gradient = (float *)calloc(tokens * dimension, sizeof(*context_gradient));
    query_weight_gradient = (float *)calloc(dimension * dimension, sizeof(*query_weight_gradient));
    key_weight_gradient = (float *)calloc(dimension * dimension, sizeof(*key_weight_gradient));
    value_weight_gradient = (float *)calloc(dimension * dimension, sizeof(*value_weight_gradient));
    output_weight_gradient = (float *)calloc(dimension * dimension, sizeof(*output_weight_gradient));
    query_bias_gradient = (float *)calloc(dimension, sizeof(*query_bias_gradient));
    key_bias_gradient = (float *)calloc(dimension, sizeof(*key_bias_gradient));
    value_bias_gradient = (float *)calloc(dimension, sizeof(*value_bias_gradient));
    output_bias_gradient = (float *)calloc(dimension, sizeof(*output_bias_gradient));
    scores = (float *)malloc(tokens * sizeof(*scores));

    if (input_gradient == NULL || query == NULL || key == NULL || value == NULL
        || context == NULL || attention == NULL || query_gradient == NULL
        || key_gradient == NULL || value_gradient == NULL || context_gradient == NULL
        || query_weight_gradient == NULL || key_weight_gradient == NULL
        || value_weight_gradient == NULL || output_weight_gradient == NULL
        || query_bias_gradient == NULL || key_bias_gradient == NULL
        || value_bias_gradient == NULL || output_bias_gradient == NULL
        || scores == NULL) {
        tensor_free(input_gradient);
        free(query);
        free(key);
        free(value);
        free(context);
        free(attention);
        free(query_gradient);
        free(key_gradient);
        free(value_gradient);
        free(context_gradient);
        free(query_weight_gradient);
        free(key_weight_gradient);
        free(value_weight_gradient);
        free(output_weight_gradient);
        free(query_bias_gradient);
        free(key_bias_gradient);
        free(value_bias_gradient);
        free(output_bias_gradient);
        free(scores);
        return NULL;
    }

    input_data = tensor_data(input);
    output_gradient_data = tensor_data(output_gradient);
    input_gradient_data = tensor_data_mut(input_gradient);
    query_weight = tensor_data_mut(layer->query_weight);
    key_weight = tensor_data_mut(layer->key_weight);
    value_weight = tensor_data_mut(layer->value_weight);
    output_weight = tensor_data_mut(layer->output_weight);
    query_bias = tensor_data_mut(layer->query_bias);
    key_bias = tensor_data_mut(layer->key_bias);
    value_bias = tensor_data_mut(layer->value_bias);
    output_bias = tensor_data_mut(layer->output_bias);

    for (query_position = 0; query_position < tokens; ++query_position) {
        for (output_column = 0; output_column < dimension; ++output_column) {
            float query_value = query_bias[output_column];
            float key_value = key_bias[output_column];
            float value_value = value_bias[output_column];
            for (input_column = 0; input_column < dimension; ++input_column) {
                const float input_value = input_data[query_position * dimension + input_column];
                query_value += input_value
                    * query_weight[input_column * dimension + output_column];
                key_value += input_value
                    * key_weight[input_column * dimension + output_column];
                value_value += input_value
                    * value_weight[input_column * dimension + output_column];
            }
            query[query_position * dimension + output_column] = query_value;
            key[query_position * dimension + output_column] = key_value;
            value[query_position * dimension + output_column] = value_value;
        }
    }

    for (head = 0; head < layer->number_of_heads; ++head) {
        const size_t head_offset = head * head_dimension;
        const float scale = 1.0f / sqrtf((float)head_dimension);

        for (query_position = 0; query_position < tokens; ++query_position) {
            float maximum = -INFINITY;
            float sum = 0.0f;

            for (key_position = 0; key_position < tokens; ++key_position) {
                float score = 0.0f;
                for (head_column = 0; head_column < head_dimension; ++head_column) {
                    score += query[query_position * dimension + head_offset + head_column]
                        * key[key_position * dimension + head_offset + head_column];
                }
                scores[key_position] = score * scale;
                if (scores[key_position] > maximum) {
                    maximum = scores[key_position];
                }
            }

            for (key_position = 0; key_position < tokens; ++key_position) {
                const float probability = expf(scores[key_position] - maximum);
                attention[head * attention_stride + query_position * tokens + key_position] =
                    probability;
                sum += probability;
            }

            for (key_position = 0; key_position < tokens; ++key_position) {
                attention[head * attention_stride + query_position * tokens + key_position] /= sum;
            }

            for (head_column = 0; head_column < head_dimension; ++head_column) {
                float context_value = 0.0f;
                for (key_position = 0; key_position < tokens; ++key_position) {
                    context_value += attention[
                        head * attention_stride + query_position * tokens + key_position]
                        * value[key_position * dimension + head_offset + head_column];
                }
                context[query_position * dimension + head_offset + head_column] = context_value;
            }
        }
    }

    /* Backpropagate the output projection. */
    for (query_position = 0; query_position < tokens; ++query_position) {
        for (output_column = 0; output_column < dimension; ++output_column) {
            const float gradient = output_gradient_data[
                query_position * dimension + output_column];
            output_bias_gradient[output_column] += gradient;
            for (input_column = 0; input_column < dimension; ++input_column) {
                context_gradient[query_position * dimension + input_column] +=
                    gradient * output_weight[input_column * dimension + output_column];
                output_weight_gradient[input_column * dimension + output_column] +=
                    context[query_position * dimension + input_column] * gradient;
            }
        }
    }

    for (head = 0; head < layer->number_of_heads; ++head) {
        const size_t head_offset = head * head_dimension;
        const float scale = 1.0f / sqrtf((float)head_dimension);

        for (query_position = 0; query_position < tokens; ++query_position) {
            float attention_dot_gradient = 0.0f;

            for (key_position = 0; key_position < tokens; ++key_position) {
                float attention_gradient = 0.0f;
                for (head_column = 0; head_column < head_dimension; ++head_column) {
                    attention_gradient += context_gradient[
                        query_position * dimension + head_offset + head_column]
                        * value[key_position * dimension + head_offset + head_column];
                    value_gradient[key_position * dimension + head_offset + head_column] +=
                        attention[head * attention_stride + query_position * tokens + key_position]
                        * context_gradient[query_position * dimension + head_offset + head_column];
                }
                scores[key_position] = attention_gradient;
                attention_dot_gradient += attention_gradient
                    * attention[head * attention_stride + query_position * tokens + key_position];
            }

            for (key_position = 0; key_position < tokens; ++key_position) {
                const float attention_probability =
                    attention[head * attention_stride + query_position * tokens + key_position];
                const float score_gradient =
                    attention_probability * (scores[key_position] - attention_dot_gradient) * scale;

                for (head_column = 0; head_column < head_dimension; ++head_column) {
                    query_gradient[query_position * dimension + head_offset + head_column] +=
                        score_gradient * key[key_position * dimension + head_offset + head_column];
                    key_gradient[key_position * dimension + head_offset + head_column] +=
                        score_gradient * query[query_position * dimension + head_offset + head_column];
                }
            }
        }
    }

    /* Backpropagate Q/K/V projections and form the input gradient. */
    for (query_position = 0; query_position < tokens; ++query_position) {
        for (input_column = 0; input_column < dimension; ++input_column) {
            float input_gradient_value = 0.0f;
            for (output_column = 0; output_column < dimension; ++output_column) {
                const size_t offset = input_column * dimension + output_column;
                const float input_value = input_data[query_position * dimension + input_column];
                input_gradient_value += query_gradient[query_position * dimension + output_column]
                    * query_weight[offset];
                input_gradient_value += key_gradient[query_position * dimension + output_column]
                    * key_weight[offset];
                input_gradient_value += value_gradient[query_position * dimension + output_column]
                    * value_weight[offset];
                query_weight_gradient[offset] += input_value
                    * query_gradient[query_position * dimension + output_column];
                key_weight_gradient[offset] += input_value
                    * key_gradient[query_position * dimension + output_column];
                value_weight_gradient[offset] += input_value
                    * value_gradient[query_position * dimension + output_column];
            }
            input_gradient_data[query_position * dimension + input_column] = input_gradient_value;
        }

        for (output_column = 0; output_column < dimension; ++output_column) {
            query_bias_gradient[output_column] +=
                query_gradient[query_position * dimension + output_column];
            key_bias_gradient[output_column] +=
                key_gradient[query_position * dimension + output_column];
            value_bias_gradient[output_column] +=
                value_gradient[query_position * dimension + output_column];
        }
    }

    for (input_column = 0; input_column < dimension; ++input_column) {
        for (output_column = 0; output_column < dimension; ++output_column) {
            const size_t offset = input_column * dimension + output_column;
            query_weight[offset] -= learning_rate * query_weight_gradient[offset];
            key_weight[offset] -= learning_rate * key_weight_gradient[offset];
            value_weight[offset] -= learning_rate * value_weight_gradient[offset];
            output_weight[offset] -= learning_rate * output_weight_gradient[offset];
        }
        query_bias[input_column] -= learning_rate * query_bias_gradient[input_column];
        key_bias[input_column] -= learning_rate * key_bias_gradient[input_column];
        value_bias[input_column] -= learning_rate * value_bias_gradient[input_column];
        output_bias[input_column] -= learning_rate * output_bias_gradient[input_column];
    }

    free(query);
    free(key);
    free(value);
    free(context);
    free(attention);
    free(query_gradient);
    free(key_gradient);
    free(value_gradient);
    free(context_gradient);
    free(query_weight_gradient);
    free(key_weight_gradient);
    free(value_weight_gradient);
    free(output_weight_gradient);
    free(query_bias_gradient);
    free(key_bias_gradient);
    free(value_bias_gradient);
    free(output_bias_gradient);
    free(scores);
    return input_gradient;
}
