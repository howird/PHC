{ buildFHSEnv
, cudatoolkit
, config
}:
(buildFHSEnv rec {
  name = "uv";

  targetPkgs = pkgs: (with pkgs; [
    # binaries
    uv
    nodejs pandoc # jupyterlab
    wayland
    gcc ninja stdenv.cc.cc # torch

    # libraries
    zlib # numpy
    # llvmPackages.libcxx
    libGL
    libxcrypt libudev-zero # torch
    glib
    wayland
    libxkbcommon
  ]) ++ (with pkgs.xorg; [
    libX11
    # libXcursor
    # libXrandr
  ]) ++ (if config.cudaSupport then (with pkgs.cudaPackages; [
    cuda_cudart
    # cuda_nvcc
    cuda_cupti

    cudnn
    nccl
    libcusparse
    libcurand
    libcusolver
    libcufft
    libcublas
  ] ++ [pkgs.linuxPackages.nvidia_x11_vulkan_beta]) else []);

  multiPkgs = pkgs: (with pkgs; [
  ]);

  # see https://nixos.org/manual/nixpkgs/stable/#how-to-consume-python-modules-using-pip-in-a-virtual-environment-like-i-am-used-to-on-other-operating-systems
  profile = ''
    SOURCE_DATE_EPOCH=$(date +%s)
    export LIBRARY_PATH=/usr/lib64
    export CPATH=/usr/include
    if [ -d ".venv" ]; then
        echo "Skipping venv creation, '.venv' already exists"
    else
        echo "Creating new venv environment in path: '.venv'"
        uv venv
    fi
    source ".venv/bin/activate"
    # requires uv v0.5.3+, see https://docs.astral.sh/uv/guides/integration/pytorch/
    # uv sync --extra ${if config.cudaSupport then "cuda" else "cpu"}
  '';

  runScript = "bash";
}).env
