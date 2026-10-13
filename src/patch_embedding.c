#include "patch_embedding.h"

#include <stdlib.h>

struct PatchEmbedding {
    size_t image_height;
    size_t image_width;
    size_t patch_size;
    size_t channels;
    size_t patch_dimension;
    size_t embedding_dimension;
    size_t patches_per_row;
    size_t patches_per_column;
    size_t number_of_patches;
    Tensor *weights;
    Tensor *bias;
};

static int multiplication_overflows(size_t left, size_t right)
{
    return right != 0 && left > (size_t)-1 / right;
}

PatchEmbedding *patch_embedding_create(
    size_t image_height,
    size_t image_width,
    size_t patch_size,
    size_t channels,
    size_t embedding_dim)
{
    PatchEmbedding *layer;
    size_t weights_shape[2];
    size_t bias_shape[1];

    if (image_height == 0 || image_width == 0 || patch_size == 0 || channels == 0
        || embedding_dim == 0 || image_height % patch_size != 0
        || image_width % patch_size != 0) {
        return NULL;
    }

    layer = (PatchEmbedding *)calloc(1, sizeof(*layer));
    if (layer == NULL) {
        return NULL;
    }

    layer->image_height = image_height;
    layer->image_width = image_width;
    layer->patch_size = patch_size;
    layer->channels = channels;
    layer->embedding_dimension = embedding_dim;
    layer->patches_per_row = image_width / patch_size;
    layer->patches_per_column = image_height / patch_size;

    if (multiplication_overflows(layer->patches_per_row, layer->patches_per_column)
        || multiplication_overflows(patch_size, patch_size)
        || multiplication_overflows(patch_size * patch_size, channels)) {
        free(layer);
        return NULL;
    }

    layer->number_of_patches = layer->patches_per_row * layer->patches_per_column;
    layer->patch_dimension = patch_size * patch_size * channels;

    weights_shape[0] = layer->patch_dimension;
    weights_shape[1] = embedding_dim;
    bias_shape[0] = embedding_dim;

    layer->weights = tensor_zeros(weights_shape, 2);
    layer->bias = tensor_zeros(bias_shape, 1);
    if (layer->weights == NULL || layer->bias == NULL) {
        patch_embedding_free(layer);
        return NULL;
    }

    return layer;
}

void patch_embedding_free(PatchEmbedding *layer)
{
    if (layer != NULL) {
        tensor_free(layer->weights);
        tensor_free(layer->bias);
        free(layer);
    }
}

Tensor *patch_embedding_forward(const PatchEmbedding *layer, const Tensor *image)
{
    size_t image_shape[3];
    size_t output_shape[2];
    const float *input_data;
    const float *weights;
    const float *bias;
    Tensor *output;
    float *output_data;
    size_t patch_row;
    size_t patch_column;
    size_t row;
    size_t column;
    size_t channel;
    size_t patch_index;
    size_t patch_offset;
    size_t embedding_index;

    if (layer == NULL || image == NULL || tensor_ndim(image) != 3) {
        return NULL;
    }

    image_shape[0] = layer->image_height;
    image_shape[1] = layer->image_width;
    image_shape[2] = layer->channels;

    if (tensor_shape(image)[0] != image_shape[0]
        || tensor_shape(image)[1] != image_shape[1]
        || tensor_shape(image)[2] != image_shape[2]) {
        return NULL;
    }

    output_shape[0] = layer->number_of_patches;
    output_shape[1] = layer->embedding_dimension;
    output = tensor_zeros(output_shape, 2);
    if (output == NULL) {
        return NULL;
    }

    input_data = tensor_data(image);
    weights = tensor_data(layer->weights);
    bias = tensor_data(layer->bias);
    output_data = tensor_data_mut(output);

    for (patch_row = 0; patch_row < layer->patches_per_column; ++patch_row) {
        for (patch_column = 0; patch_column < layer->patches_per_row; ++patch_column) {
            patch_index = patch_row * layer->patches_per_row + patch_column;

            for (embedding_index = 0;
                 embedding_index < layer->embedding_dimension;
                 ++embedding_index) {
                float value = bias[embedding_index];

                patch_offset = 0;
                for (row = 0; row < layer->patch_size; ++row) {
                    for (column = 0; column < layer->patch_size; ++column) {
                        size_t image_row = patch_row * layer->patch_size + row;
                        size_t image_column = patch_column * layer->patch_size + column;
                        size_t pixel_offset =
                            (image_row * layer->image_width + image_column) * layer->channels;

                        for (channel = 0; channel < layer->channels; ++channel) {
                            value += input_data[pixel_offset + channel]
                                * weights[patch_offset * layer->embedding_dimension
                                          + embedding_index];
                            ++patch_offset;
                        }
                    }
                }

                output_data[patch_index * layer->embedding_dimension + embedding_index] = value;
            }
        }
    }

    return output;
}

size_t patch_embedding_num_patches(const PatchEmbedding *layer)
{
    return layer == NULL ? 0 : layer->number_of_patches;
}

size_t patch_embedding_patch_dimension(const PatchEmbedding *layer)
{
    return layer == NULL ? 0 : layer->patch_dimension;
}

size_t patch_embedding_embedding_dimension(const PatchEmbedding *layer)
{
    return layer == NULL ? 0 : layer->embedding_dimension;
}

float *patch_embedding_weights_mut(PatchEmbedding *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->weights);
}

float *patch_embedding_bias_mut(PatchEmbedding *layer)
{
    return layer == NULL ? NULL : tensor_data_mut(layer->bias);
}
