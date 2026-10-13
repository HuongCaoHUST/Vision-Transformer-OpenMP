CXX       ?= g++
PKG_CONFIG ?= pkg-config

TARGET    := build/vision_transformer
SOURCES   := test.cpp

CPPFLAGS  += $(shell $(PKG_CONFIG) --cflags opencv4 2>/dev/null)
CXXFLAGS  += -std=c++17 -O2 -Wall -Wextra -fopenmp
LDLIBS    += $(shell $(PKG_CONFIG) --libs opencv4 2>/dev/null)

.PHONY: all run clean help

all: $(TARGET)

$(TARGET): $(SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf build

help:
	@echo "Available commands:"
	@echo "  make        - Build the program"
	@echo "  make run    - Build and run the program"
	@echo "  make clean  - Remove build files"
