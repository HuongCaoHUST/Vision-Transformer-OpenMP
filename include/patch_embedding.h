#ifndef PATCH_EMBEDDING_H
#define PATCH_EMBEDDING_H

#include <stddef.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PatchEmbedding PatchEmbedding;

/*
 * Create a patch embedding layer for channels-last images:
 * input shape  = [image_height, image_width, channels]
 * output shape = [number_of_patches, embedding_dim]
 */
PatchEmbedding *patch_embedding_create(
    size_t image_height,
    size_t image_width,
    size_t patch_size,
    size_t channels,
    size_t embedding_dim);

void patch_embedding_free(PatchEmbedding *layer);

Tensor *patch_embedding_forward(
    const PatchEmbedding *layer,
    const Tensor *image);

size_t patch_embedding_num_patches(const PatchEmbedding *layer);
size_t patch_embedding_patch_dimension(const PatchEmbedding *layer);
size_t patch_embedding_embedding_dimension(const PatchEmbedding *layer);

/* Accessors are provided so training code can update the parameters later. */
float *patch_embedding_weights_mut(PatchEmbedding *layer);
float *patch_embedding_bias_mut(PatchEmbedding *layer);

#ifdef __cplusplus
}
#endif

#endif /* PATCH_EMBEDDING_H */
