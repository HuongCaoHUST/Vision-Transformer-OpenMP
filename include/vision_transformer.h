#ifndef VISION_TRANSFORMER_H
#define VISION_TRANSFORMER_H

#include <stddef.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VisionTransformer VisionTransformer;

VisionTransformer *vision_transformer_create(
    size_t image_height,
    size_t image_width,
    size_t patch_size,
    size_t channels,
    size_t embedding_dimension,
    size_t number_of_heads,
    size_t mlp_hidden_dimension,
    size_t number_of_encoder_blocks,
    size_t number_of_classes);

void vision_transformer_free(VisionTransformer *model);

/* Return logits with shape [1, number_of_classes]. */
Tensor *vision_transformer_forward(
    const VisionTransformer *model,
    const Tensor *image);

size_t vision_transformer_number_of_classes(const VisionTransformer *model);
size_t vision_transformer_number_of_blocks(const VisionTransformer *model);

#ifdef __cplusplus
}
#endif

#endif /* VISION_TRANSFORMER_H */
