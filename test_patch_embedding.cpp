#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

#include "patch_embedding.h"
#include "cls_token.h"
#include "position_embedding.h"
#include "layer_norm.h"
#include "multi_head_attention.h"

using namespace std;

static Tensor *mat_to_tensor(const cv::Mat &image)
{
    size_t shape[3] = {
        static_cast<size_t>(image.rows),
        static_cast<size_t>(image.cols),
        static_cast<size_t>(image.channels())
    };
    Tensor *tensor = tensor_zeros(shape, 3);
    float *data;

    if (tensor == nullptr) {
        return nullptr;
    }

    data = tensor_data_mut(tensor);
    for (int row = 0; row < image.rows; ++row) {
        for (int column = 0; column < image.cols; ++column) {
            const cv::Vec3b pixel = image.at<cv::Vec3b>(row, column);
            size_t offset =
                (static_cast<size_t>(row) * image.cols + column) * image.channels();

            for (int channel = 0; channel < image.channels(); ++channel) {
                data[offset + channel] = pixel[channel] / 255.0f;
            }
        }
    }

    return tensor;
}

int main(int argc, char **argv)
{
    const string image_path = argc > 1 ? argv[1] : "0010.png";
    const int patch_size = 8;
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        cerr << "Unable to read image: " << image_path << '\n';
        return EXIT_FAILURE;
    }

    if (image.rows % patch_size != 0 || image.cols % patch_size != 0) {
        cerr << "Image dimensions must be divisible by patch size "
             << patch_size << '\n';
        return EXIT_FAILURE;
    }

    const int patches_per_row = image.cols / patch_size;
    const int patches_per_column = image.rows / patch_size;
    const int number_of_patches = patches_per_row * patches_per_column;
    const int visualization_scale = 16;

    cv::Mat grid;
    cv::resize(
        image,
        grid,
        cv::Size(image.cols * visualization_scale, image.rows * visualization_scale),
        0.0,
        0.0,
        cv::INTER_NEAREST);
    cv::Mat montage(
        patches_per_column * patch_size,
        patches_per_row * patch_size,
        image.type(),
        cv::Scalar(0, 0, 0));

    for (int patch_row = 0; patch_row < patches_per_column; ++patch_row) {
        for (int patch_column = 0; patch_column < patches_per_row; ++patch_column) {
            const int patch_index = patch_row * patches_per_row + patch_column;
            const cv::Rect area(
                patch_column * patch_size,
                patch_row * patch_size,
                patch_size,
                patch_size);
            const cv::Rect visual_area(
                area.x * visualization_scale,
                area.y * visualization_scale,
                area.width * visualization_scale,
                area.height * visualization_scale);
            cv::Mat patch = image(area);

            patch.copyTo(montage(area));
            cv::rectangle(grid, visual_area, cv::Scalar(0, 255, 0), 2);
            cv::putText(
                grid,
                to_string(patch_index),
                cv::Point(visual_area.x + 4, visual_area.y + 24),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 0, 255),
                1,
                cv::LINE_AA);
        }
    }

    cv::Mat visual_montage;
    cv::resize(
        montage,
        visual_montage,
        cv::Size(montage.cols * visualization_scale, montage.rows * visualization_scale),
        0.0,
        0.0,
        cv::INTER_NEAREST);

    const string grid_path = "build/patches_grid.png";
    const string montage_path = "build/patches_montage.png";
    if (!cv::imwrite(grid_path, grid) || !cv::imwrite(montage_path, visual_montage)) {
        cerr << "Unable to save patch visualization files\n";
        return EXIT_FAILURE;
    }

    Tensor *image_tensor = mat_to_tensor(image);
    PatchEmbedding *layer = patch_embedding_create(
        image.rows,
        image.cols,
        patch_size,
        image.channels(),
        64);
    Tensor *embeddings = nullptr;
    ClsToken *cls_layer = nullptr;
    PositionEmbedding *position_layer = nullptr;
    LayerNorm *norm_layer = nullptr;
    MultiHeadAttention *attention_layer = nullptr;
    Tensor *tokens = nullptr;
    Tensor *positioned_tokens = nullptr;
    Tensor *normalized_tokens = nullptr;
    Tensor *attention_output = nullptr;

    if (image_tensor != nullptr && layer != nullptr) {
        embeddings = patch_embedding_forward(layer, image_tensor);
    }

    if (embeddings == nullptr) {
        cerr << "Patch embedding forward pass failed\n";
        tensor_free(image_tensor);
        patch_embedding_free(layer);
        return EXIT_FAILURE;
    }

    cls_layer = cls_token_create(64);
    tokens = cls_token_forward(cls_layer, embeddings);
    position_layer = position_embedding_create(17, 64);
    positioned_tokens = position_embedding_forward(position_layer, tokens);
    norm_layer = layer_norm_create(64, 1e-5f);
    normalized_tokens = layer_norm_forward(norm_layer, positioned_tokens);
    attention_layer = multi_head_attention_create(64, 4);
    attention_output = multi_head_attention_forward(attention_layer, normalized_tokens);

    if (tokens == nullptr || positioned_tokens == nullptr || normalized_tokens == nullptr
        || attention_output == nullptr) {
        cerr << "Transformer input pipeline failed\n";
        tensor_free(embeddings);
        tensor_free(tokens);
        tensor_free(positioned_tokens);
        tensor_free(normalized_tokens);
        tensor_free(attention_output);
        cls_token_free(cls_layer);
        position_embedding_free(position_layer);
        layer_norm_free(norm_layer);
        multi_head_attention_free(attention_layer);
        tensor_free(image_tensor);
        patch_embedding_free(layer);
        return EXIT_FAILURE;
    }

    cout << "Image: " << image.cols << "x" << image.rows << '\n';
    cout << "Patch size: " << patch_size << "x" << patch_size << '\n';
    cout << "Number of patches: " << number_of_patches << '\n';
    cout << "Embedding shape: [" << tensor_shape(embeddings)[0] << ", "
         << tensor_shape(embeddings)[1] << "]\n";
    cout << "Attention output shape: [" << tensor_shape(attention_output)[0] << ", "
         << tensor_shape(attention_output)[1] << "]\n";
    cout << "Saved: " << grid_path << '\n';
    cout << "Saved: " << montage_path << '\n';

    tensor_free(embeddings);
    tensor_free(tokens);
    tensor_free(positioned_tokens);
    tensor_free(normalized_tokens);
    tensor_free(attention_output);
    cls_token_free(cls_layer);
    position_embedding_free(position_layer);
    layer_norm_free(norm_layer);
    multi_head_attention_free(attention_layer);
    tensor_free(image_tensor);
    patch_embedding_free(layer);
    return EXIT_SUCCESS;
}
