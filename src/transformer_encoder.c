#include "transformer_encoder.h"

#include <stdlib.h>

#include "layer_norm.h"
#include "mlp.h"
#include "multi_head_attention.h"

struct TransformerEncoder {
    size_t embedding_dimension;
    LayerNorm *attention_norm;
    MultiHeadAttention *attention;
    LayerNorm *mlp_norm;
    MlpBlock *mlp;
};

TransformerEncoder *transformer_encoder_create_seeded(
    size_t embedding_dimension,
    size_t number_of_heads,
    size_t mlp_hidden_dimension,
    uint32_t seed)
{
    TransformerEncoder *block;

    if (embedding_dimension == 0 || number_of_heads == 0 || mlp_hidden_dimension == 0) {
        return NULL;
    }

    block = (TransformerEncoder *)calloc(1, sizeof(*block));
    if (block == NULL) {
        return NULL;
    }

    block->embedding_dimension = embedding_dimension;
    block->attention_norm = layer_norm_create(embedding_dimension, 1e-5f);
    block->attention = multi_head_attention_create_seeded(
        embedding_dimension, number_of_heads, seed + 1u);
    block->mlp_norm = layer_norm_create(embedding_dimension, 1e-5f);
    block->mlp = mlp_create_seeded(
        embedding_dimension, mlp_hidden_dimension, seed + 2u);

    if (block->attention_norm == NULL || block->attention == NULL
        || block->mlp_norm == NULL || block->mlp == NULL) {
        transformer_encoder_free(block);
        return NULL;
    }

    return block;
}

TransformerEncoder *transformer_encoder_create(
    size_t embedding_dimension,
    size_t number_of_heads,
    size_t mlp_hidden_dimension)
{
    return transformer_encoder_create_seeded(
        embedding_dimension, number_of_heads, mlp_hidden_dimension, 0x5000u);
}

void transformer_encoder_free(TransformerEncoder *block)
{
    if (block != NULL) {
        layer_norm_free(block->attention_norm);
        multi_head_attention_free(block->attention);
        layer_norm_free(block->mlp_norm);
        mlp_free(block->mlp);
        free(block);
    }
}

Tensor *transformer_encoder_forward(
    const TransformerEncoder *block,
    const Tensor *input)
{
    Tensor *normalized_input;
    Tensor *attention_output;
    Tensor *attention_residual;
    Tensor *normalized_residual;
    Tensor *mlp_output;
    Tensor *output;

    if (block == NULL || input == NULL || tensor_ndim(input) != 2) {
        return NULL;
    }
    if (tensor_shape(input)[1] != block->embedding_dimension) {
        return NULL;
    }

    normalized_input = layer_norm_forward(block->attention_norm, input);
    attention_output = multi_head_attention_forward(block->attention, normalized_input);
    attention_residual = tensor_create(tensor_shape(input), 2);

    if (normalized_input == NULL || attention_output == NULL || attention_residual == NULL) {
        tensor_free(normalized_input);
        tensor_free(attention_output);
        tensor_free(attention_residual);
        return NULL;
    }

    if (tensor_add(input, attention_output, attention_residual) != TENSOR_OK) {
        tensor_free(normalized_input);
        tensor_free(attention_output);
        tensor_free(attention_residual);
        return NULL;
    }

    normalized_residual = layer_norm_forward(block->mlp_norm, attention_residual);
    mlp_output = mlp_forward(block->mlp, normalized_residual);
    output = tensor_create(tensor_shape(input), 2);

    if (normalized_residual == NULL || mlp_output == NULL || output == NULL) {
        tensor_free(normalized_input);
        tensor_free(attention_output);
        tensor_free(attention_residual);
        tensor_free(normalized_residual);
        tensor_free(mlp_output);
        tensor_free(output);
        return NULL;
    }

    if (tensor_add(attention_residual, mlp_output, output) != TENSOR_OK) {
        tensor_free(normalized_input);
        tensor_free(attention_output);
        tensor_free(attention_residual);
        tensor_free(normalized_residual);
        tensor_free(mlp_output);
        tensor_free(output);
        return NULL;
    }

    tensor_free(normalized_input);
    tensor_free(attention_output);
    tensor_free(attention_residual);
    tensor_free(normalized_residual);
    tensor_free(mlp_output);
    return output;
}

Tensor *transformer_encoder_backward(
    TransformerEncoder *block,
    const Tensor *input,
    const Tensor *output_gradient,
    float learning_rate)
{
    Tensor *normalized_input = NULL;
    Tensor *attention_output = NULL;
    Tensor *attention_residual = NULL;
    Tensor *normalized_residual = NULL;
    Tensor *mlp_input_gradient = NULL;
    Tensor *residual_gradient = NULL;
    Tensor *attention_input_gradient = NULL;
    Tensor *input_gradient = NULL;
    size_t i;

    if (block == NULL || input == NULL || output_gradient == NULL
        || learning_rate <= 0.0f || tensor_ndim(input) != 2
        || tensor_ndim(output_gradient) != 2
        || tensor_shape(input)[1] != block->embedding_dimension
        || tensor_shape(output_gradient)[0] != tensor_shape(input)[0]
        || tensor_shape(output_gradient)[1] != tensor_shape(input)[1]) {
        return NULL;
    }

    normalized_input = layer_norm_forward(block->attention_norm, input);
    attention_output = multi_head_attention_forward(block->attention, normalized_input);
    if (normalized_input == NULL || attention_output == NULL) {
        goto cleanup;
    }

    attention_residual = tensor_create(tensor_shape(input), 2);
    if (attention_residual == NULL
        || tensor_add(input, attention_output, attention_residual) != TENSOR_OK) {
        goto cleanup;
    }

    normalized_residual = layer_norm_forward(block->mlp_norm, attention_residual);
    if (normalized_residual == NULL) {
        goto cleanup;
    }

    /* The forward values are recomputed before any parameter in this block changes. */
    mlp_input_gradient = mlp_backward(
        block->mlp, normalized_residual, output_gradient, learning_rate);
    if (mlp_input_gradient == NULL) {
        goto cleanup;
    }

    residual_gradient = layer_norm_backward(
        block->mlp_norm, attention_residual, mlp_input_gradient, learning_rate);
    if (residual_gradient == NULL) {
        goto cleanup;
    }

    attention_input_gradient = multi_head_attention_backward(
        block->attention, normalized_input, residual_gradient, learning_rate);
    if (attention_input_gradient == NULL) {
        goto cleanup;
    }

    input_gradient = layer_norm_backward(
        block->attention_norm, input, attention_input_gradient, learning_rate);
    if (input_gradient == NULL) {
        goto cleanup;
    }

    for (i = 0; i < tensor_numel(input_gradient); ++i) {
        tensor_data_mut(input_gradient)[i] += tensor_data(residual_gradient)[i];
    }

cleanup:
    tensor_free(normalized_input);
    tensor_free(attention_output);
    tensor_free(attention_residual);
    tensor_free(normalized_residual);
    tensor_free(mlp_input_gradient);
    tensor_free(residual_gradient);
    tensor_free(attention_input_gradient);
    return input_gradient;
}

size_t transformer_encoder_embedding_dimension(const TransformerEncoder *block)
{
    return block == NULL ? 0 : block->embedding_dimension;
}

LayerNorm *transformer_encoder_attention_norm_mut(TransformerEncoder *block)
{
    return block == NULL ? NULL : block->attention_norm;
}

MultiHeadAttention *transformer_encoder_attention_mut(TransformerEncoder *block)
{
    return block == NULL ? NULL : block->attention;
}

LayerNorm *transformer_encoder_mlp_norm_mut(TransformerEncoder *block)
{
    return block == NULL ? NULL : block->mlp_norm;
}

MlpBlock *transformer_encoder_mlp_mut(TransformerEncoder *block)
{
    return block == NULL ? NULL : block->mlp;
}
