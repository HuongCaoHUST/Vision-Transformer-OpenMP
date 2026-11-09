# Vision Transformer OpenMP

Triển khai Vision Transformer (ViT) bằng C/C++, có hỗ trợ OpenMP và OpenCV. Project hiện cung cấp pipeline đọc ảnh CIFAR-10 dạng thư mục, forward pass của ViT và bước train classification head.

## 1. Yêu cầu môi trường

- Linux hoặc môi trường có `bash`
- GCC/G++
- GNU Make
- OpenMP
- OpenCV 4 và gói phát triển (`opencv4.pc`)

Trên Ubuntu/Debian có thể cài các gói cần thiết bằng:

```bash
sudo apt update
sudo apt install build-essential pkg-config libopencv-dev
```

Kiểm tra OpenCV:

```bash
pkg-config --modversion opencv4
```

## 2. Chuẩn bị dataset CIFAR-10

`Cifar10Dataset` không đọc file CIFAR-10 binary. Loader yêu cầu ảnh `.png`, `.jpg` hoặc `.jpeg`, kích thước chính xác `32x32`, được sắp xếp theo cấu trúc:

```text
datasets/
└── cifar10/
    ├── train/
    │   ├── airplane/
    │   ├── automobile/
    │   ├── bird/
    │   ├── cat/
    │   ├── deer/
    │   ├── dog/
    │   ├── frog/
    │   ├── horse/
    │   ├── ship/
    │   └── truck/
    └── test/
        ├── airplane/
        ├── automobile/
        ├── bird/
        ├── cat/
        ├── deer/
        ├── dog/
        ├── frog/
        ├── horse/
        ├── ship/
        └── truck/
```

Thứ tự nhãn là:

```text
0 airplane       5 dog
1 automobile     6 frog
2 bird           7 horse
3 cat            8 ship
4 deer           9 truck
```

Ảnh được đọc bằng OpenCV và chuẩn hóa từng giá trị pixel về khoảng `[0, 1]`. Chương trình train hiện dùng split `train` tại đường dẫn cố định `datasets/cifar10/train`.

> Lưu ý: `dowload_datasets.py` hiện là script mẫu dùng Ultralytics/YOLO, không tạo cấu trúc dataset mà chương trình C++ yêu cầu. Cần chuẩn bị và chuyển đổi dataset thành cấu trúc ở trên trước khi train.

## 3. Kiểm tra dataset

Build và chạy bài kiểm tra loader:

```bash
make test-cifar10
```

Kết quả hợp lệ cần hiển thị kích thước ảnh `[32, 32, 3]`, label và tên class. Nếu thấy:

```text
Dataset size: 0
```

hãy kiểm tra lại tên thư mục, đường dẫn và định dạng ảnh.

## 4. Build project

Build chương trình chính:

```bash
make
```

Build riêng chương trình train:

```bash
make train-cifar10
```

Các binary được tạo trong thư mục `build/`.

## 5. Train CIFAR-10

Chương trình train nhận tối đa hai tham số dòng lệnh:

```text
./build/train_cifar10 [số_epoch] [số_sample_tối_đa]
```

Ví dụ chạy 10 epoch trên toàn bộ sample được tìm thấy:

```bash
./build/train_cifar10 10 0
```

Chạy nhanh trên 100 sample trong 3 epoch:

```bash
./build/train_cifar10 3 100
```

Giá trị mặc định là:

```text
epochs      = 1
max_samples = 100
learning_rate = 0.01
```

Vì vậy, sau khi build, lệnh sau sẽ train 1 epoch trên tối đa 100 sample:

```bash
./build/train_cifar10
```

Đặt số thread OpenMP trước khi chạy nếu cần:

```bash
OMP_NUM_THREADS=8 ./build/train_cifar10 10 0
```

Trong mỗi epoch, dataset được shuffle với seed phụ thuộc vào số epoch. Chương trình in loss trung bình, precision, recall, accuracy và thời gian train. Confusion matrix được lưu theo dạng:

```text
runs/confusion_matrix_epoch_1.png
runs/confusion_matrix_epoch_2.png
...
```

## 6. Cấu hình mô hình hiện tại

Mô hình được khởi tạo trong `train_cifar10.cpp` với các tham số:

| Thành phần | Giá trị |
|---|---:|
| Kích thước ảnh | `32x32` |
| Patch size | `8x8` |
| Số channel | `3` |
| Embedding dimension | `64` |
| Số attention head | `4` |
| MLP hidden dimension | `256` |
| Số encoder block | `2` |
| Số class | `10` |
| Learning rate | `0.01` |

Với ảnh `32x32` và patch `8x8`, mô hình tạo `16` patch và thêm `1` CLS token, tổng cộng `17` token.

Hiện tại chỉ classification head được cập nhật trong `vision_transformer_train_classifier_step`; các layer backbone của ViT chưa được backpropagation/update.

## 7. Các lệnh kiểm tra khác

Kiểm tra patch embedding và toàn bộ forward pipeline:

```bash
make test-patches
```

Chạy chương trình demo đọc `0010.png`:

```bash
make run
```

Xóa các file build:

```bash
make clean
```

## 8. Lỗi cần xử lý trước khi train

Ở trạng thái hiện tại, `train_cifar10.cpp` còn các marker conflict của Git:

```text
<<<<<<< HEAD
=======
>>>>>>> 4f94e60 (add confusion matrix)
```

Do đó `make train-cifar10` sẽ lỗi biên dịch cho đến khi conflict được resolve. Sau khi chọn và hợp nhất phần OpenMP với phần confusion matrix, kiểm tra lại bằng:

```bash
rg -n '<<<<<<<|=======|>>>>>>>' train_cifar10.cpp
make clean
make train-cifar10
```

Lệnh `rg` không được in ra dòng nào và binary `build/train_cifar10` phải được tạo thành công trước khi chạy train.

