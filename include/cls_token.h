#ifndef CLS_TOKEN_H
#define CLS_TOKEN_H

#include <stddef.h>

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ClsToken ClsToken;

ClsToken *cls_token_create(size_t embedding_dimension);
void cls_token_free(ClsToken *layer);

/* Add the learnable CLS token at position zero. */
Tensor *cls_token_forward(const ClsToken *layer, const Tensor *patch_embeddings);

float *cls_token_data_mut(ClsToken *layer);
size_t cls_token_embedding_dimension(const ClsToken *layer);

#ifdef __cplusplus
}
#endif

#endif /* CLS_TOKEN_H */
