#include <iostream>

#include "cifar10_dataset.hpp"
#include "tensor.h"

using namespace std;

int main()
{
    Cifar10Dataset dataset("datasets/cifar10", "train", 5);

    cout << "Dataset size: " << dataset.size() << '\n';
    for (size_t index = 0; index < dataset.size(); ++index) {
        Tensor *image = nullptr;
        size_t label = 0;

        if (!dataset.get(index, &image, &label)) {
            cerr << "Unable to load sample " << index << '\n';
            return 1;
        }

        cout << "Sample " << index
             << ": shape=[" << tensor_shape(image)[0]
             << ", " << tensor_shape(image)[1]
             << ", " << tensor_shape(image)[2]
             << "], label=" << label
             << " (" << dataset.class_name(label) << ")\n";
        tensor_free(image);
    }

    return 0;
}
