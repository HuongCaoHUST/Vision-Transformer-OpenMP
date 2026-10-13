CXX       ?= g++
PKG_CONFIG ?= pkg-config

TARGET    := build/vision_transformer
SOURCES   := test.cpp
TENSOR_OBJECTS := build/tensor.o build/patch_embedding.o build/cls_token.o build/position_embedding.o build/layer_norm.o build/softmax.o
PATCH_TEST := build/test_patch_embedding

CPPFLAGS  += $(shell $(PKG_CONFIG) --cflags opencv4 2>/dev/null)
CXXFLAGS  += -std=c++17 -O2 -Wall -Wextra -fopenmp
LDLIBS    += $(shell $(PKG_CONFIG) --libs opencv4 2>/dev/null)

.PHONY: all run test-patches clean help

all: $(TARGET)

$(TARGET): $(SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

run: $(TARGET)
	./$(TARGET)

build/tensor.o: src/tensor.c include/tensor.h
	@mkdir -p $(dir $@)
	gcc -std=c11 -Wall -Wextra -Wpedantic -Iinclude -c $< -o $@

build/patch_embedding.o: src/patch_embedding.c include/patch_embedding.h include/tensor.h
	@mkdir -p $(dir $@)
	gcc -std=c11 -Wall -Wextra -Wpedantic -Iinclude -c $< -o $@

build/cls_token.o: src/cls_token.c include/cls_token.h include/tensor.h
	@mkdir -p $(dir $@)
	gcc -std=c11 -Wall -Wextra -Wpedantic -Iinclude -c $< -o $@

build/position_embedding.o: src/position_embedding.c include/position_embedding.h include/tensor.h
	@mkdir -p $(dir $@)
	gcc -std=c11 -Wall -Wextra -Wpedantic -Iinclude -c $< -o $@

build/layer_norm.o: src/layer_norm.c include/layer_norm.h include/tensor.h
	@mkdir -p $(dir $@)
	gcc -std=c11 -Wall -Wextra -Wpedantic -Iinclude -c $< -o $@

build/softmax.o: src/softmax.c include/softmax.h include/tensor.h
	@mkdir -p $(dir $@)
	gcc -std=c11 -Wall -Wextra -Wpedantic -Iinclude -c $< -o $@

$(PATCH_TEST): test_patch_embedding.cpp $(TENSOR_OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -Iinclude $^ $(LDFLAGS) $(LDLIBS) -o $@

test-patches: $(PATCH_TEST)
	./$(PATCH_TEST)

clean:
	rm -rf build

help:
	@echo "Available commands:"
	@echo "  make        - Build the program"
	@echo "  make run    - Build and run the program"
	@echo "  make clean  - Remove build files"
