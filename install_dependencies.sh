#!/usr/bin/env bash

set -Eeuo pipefail

log() {
    printf '[install] %s\n' "$*"
}

fail() {
    printf '[install] Error: %s\n' "$*" >&2
    exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'USAGE'
Usage: ./install_dependencies.sh

Installs the compiler, Make, pkg-config, and OpenCV development packages
required to build this project on Linux.
USAGE
    exit 0
fi

if [[ "$(uname -s)" != "Linux" ]]; then
    fail "This script currently supports Linux only."
fi

if [[ "${EUID}" -eq 0 ]]; then
    SUDO=()
elif command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo)
else
    fail "Please install sudo or run this script as root."
fi

if command -v apt-get >/dev/null 2>&1; then
    log "Detected Debian/Ubuntu package manager."
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" apt-get install -y \
        build-essential \
        gcc \
        g++ \
        make \
        pkg-config \
        libopencv-dev
elif command -v dnf >/dev/null 2>&1; then
    log "Detected Fedora/RHEL package manager."
    "${SUDO[@]}" dnf install -y \
        gcc \
        gcc-c++ \
        make \
        pkgconf-pkg-config \
        opencv-devel
elif command -v pacman >/dev/null 2>&1; then
    log "Detected Arch Linux package manager."
    "${SUDO[@]}" pacman -Sy --needed --noconfirm \
        base-devel \
        pkgconf \
        opencv
else
    fail "Unsupported distribution: apt-get, dnf, or pacman was not found."
fi

command -v gcc >/dev/null 2>&1 || fail "gcc was not installed."
command -v g++ >/dev/null 2>&1 || fail "g++ was not installed."
command -v make >/dev/null 2>&1 || fail "make was not installed."
command -v pkg-config >/dev/null 2>&1 || fail "pkg-config was not installed."

if ! pkg-config --exists opencv4; then
    fail "OpenCV 4 development files were not found by pkg-config."
fi

log "gcc: $(gcc --version | head -n 1)"
log "g++: $(g++ --version | head -n 1)"
log "OpenCV: $(pkg-config --modversion opencv4)"
log "Dependencies are ready. You can now run: make"
