#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "softmax.h"
#include "tensor.h"
#include "vision_transformer.h"

namespace {

const char *const class_names[] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

Tensor *image_to_tensor(const cv::Mat &image)
{
    const size_t shape[3] = {
        static_cast<size_t>(image.rows),
        static_cast<size_t>(image.cols),
        static_cast<size_t>(image.channels())
    };
    Tensor *tensor = tensor_zeros(shape, 3);

    if (tensor == nullptr) {
        return nullptr;
    }

    float *data = tensor_data_mut(tensor);
    for (int row = 0; row < image.rows; ++row) {
        for (int column = 0; column < image.cols; ++column) {
            const cv::Vec3b pixel = image.at<cv::Vec3b>(row, column);
            const size_t offset =
                (static_cast<size_t>(row) * static_cast<size_t>(image.cols)
                 + static_cast<size_t>(column)) * 3;

            for (int channel = 0; channel < 3; ++channel) {
                data[offset + static_cast<size_t>(channel)] =
                    static_cast<float>(pixel[channel]) / 255.0f;
            }
        }
    }

    return tensor;
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <image.png|jpg|jpeg> <weights.vitw>\n";
        return EXIT_FAILURE;
    }

    const cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Unable to read image: " << argv[1] << '\n';
        return EXIT_FAILURE;
    }

    if (image.rows != 32 || image.cols != 32 || image.channels() != 3) {
        std::cerr << "Image must have shape 32x32x3\n";
        return EXIT_FAILURE;
    }

    Tensor *image_tensor = image_to_tensor(image);
    VisionTransformer *model = vision_transformer_load_weights(argv[2]);
    Tensor *logits = nullptr;
    Tensor *probabilities = nullptr;

    if (model == nullptr) {
        std::cerr << "Unable to load weights: " << argv[2] << '\n';
        tensor_free(image_tensor);
        return EXIT_FAILURE;
    }

    if (image_tensor != nullptr) {
        logits = vision_transformer_forward(model, image_tensor);
        probabilities = softmax_forward(logits);
    }

    if (probabilities == nullptr) {
        std::cerr << "Inference failed\n";
        tensor_free(image_tensor);
        tensor_free(logits);
        tensor_free(probabilities);
        vision_transformer_free(model);
        return EXIT_FAILURE;
    }

    const float *probability_data = tensor_data(probabilities);
    size_t predicted_class = 0;
    for (size_t class_index = 1;
         class_index < tensor_shape(probabilities)[1];
         ++class_index) {
        if (probability_data[class_index] > probability_data[predicted_class]) {
            predicted_class = class_index;
        }
    }

    std::cout << "Image: " << argv[1] << '\n'
              << "Predicted label: " << class_names[predicted_class]
              << " (class " << predicted_class << ")\n"
              << "Probability: " << std::fixed << std::setprecision(2)
              << probability_data[predicted_class] * 100.0f << "%\n";

    tensor_free(image_tensor);
    tensor_free(logits);
    tensor_free(probabilities);
    vision_transformer_free(model);
    return EXIT_SUCCESS;
}
