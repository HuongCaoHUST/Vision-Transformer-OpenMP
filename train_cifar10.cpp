#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "cifar10_dataset.hpp"
#include "tensor.h"
#include "vision_transformer.h"

using namespace std;

int main(int argc, char **argv)
{
    const size_t epochs = argc > 1 ? strtoul(argv[1], nullptr, 10) : 1;
    const size_t max_samples = argc > 2 ? strtoul(argv[2], nullptr, 10) : 100;
    Cifar10Dataset dataset("datasets/cifar10", "train", max_samples);
    VisionTransformer *model;
    float learning_rate = 0.01f;

    if (dataset.size() == 0) {
        cerr << "The CIFAR-10 training dataset is empty\n";
        return EXIT_FAILURE;
    }

    model = vision_transformer_create(32, 32, 8, 3, 64, 4, 256, 2, 10);
    if (model == nullptr) {
        cerr << "Unable to create Vision Transformer\n";
        return EXIT_FAILURE;
    }

    cout << "Training samples: " << dataset.size() << '\n';
    cout << "Epochs: " << epochs << '\n';
    cout << "Trainable parameters: classification head only\n";

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        size_t successful_samples = 0;

        dataset.shuffle(static_cast<uint32_t>(epoch + 1));
        for (size_t index = 0; index < dataset.size(); ++index) {
            Tensor *image = nullptr;
            Tensor *features = nullptr;
            size_t label = 0;
            float loss = 0.0f;

            if (!dataset.get(index, &image, &label)) {
                cerr << "Skipping unreadable sample " << index << '\n';
                continue;
            }

            features = vision_transformer_forward_features(model, image);
            if (features == nullptr
                || vision_transformer_train_classifier_step(
                       model, features, label, learning_rate, &loss) != TENSOR_OK) {
                cerr << "Skipping failed training sample " << index << '\n';
                tensor_free(image);
                tensor_free(features);
                continue;
            }

            total_loss += loss;
            ++successful_samples;
            tensor_free(image);
            tensor_free(features);
        }

        const float average_loss = successful_samples == 0
            ? 0.0f : total_loss / static_cast<float>(successful_samples);
        cout << "Epoch " << (epoch + 1)
             << "/" << epochs
             << " - average loss: "
             << fixed << setprecision(6) << average_loss << '\n';
    }

    vision_transformer_free(model);
    return EXIT_SUCCESS;
}
