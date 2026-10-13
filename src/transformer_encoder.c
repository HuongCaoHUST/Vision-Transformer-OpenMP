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

TransformerEncoder *transformer_encoder_create(
    size_t embedding_dimension,
    size_t number_of_heads,
    size_t mlp_hidden_dimension)
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
    block->attention = multi_head_attention_create(embedding_dimension, number_of_heads);
    block->mlp_norm = layer_norm_create(embedding_dimension, 1e-5f);
    block->mlp = mlp_create(embedding_dimension, mlp_hidden_dimension);

    if (block->attention_norm == NULL || block->attention == NULL
        || block->mlp_norm == NULL || block->mlp == NULL) {
        transformer_encoder_free(block);
        return NULL;
    }

    return block;
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

size_t transformer_encoder_embedding_dimension(const TransformerEncoder *block)
{
    return block == NULL ? 0 : block->embedding_dimension;
}
