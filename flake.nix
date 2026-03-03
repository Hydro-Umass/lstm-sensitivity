{
  description = "LSTM sensitivity";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      libInputs = with pkgs; [
        cudaPackages.cuda_cudart
        cudaPackages.cudatoolkit
        cudaPackages.cudnn
        stdenv.cc.cc
        libuv
        zlib
      ];
      vic = pkgs.stdenv.mkDerivation {
        name = "vic";
        src = pkgs.fetchFromGitHub {
          owner = "kandread";
          repo = "VIC";
          rev = "79c22bd909958ad78a93122c240d93d99f24a307";
          hash = "sha256-1bDqCb39AFQ3LDs3PMzdX+L8I1+QPY1iXgbFxh/Tjko=";
        };
        nativeBuildInputs = [ pkgs.gcc pkgs.automake ];
        buildPhase = ''
          cd src
          sed -i "s|/bin/bash|${pkgs.bash}/bin/bash|" Makefile
          make
        '';
        hardeningDisable = pkgs.lib.optionals (pkgs.stdenv.isAarch64 && pkgs.stdenv.isDarwin) [ "stackprotector" ];
        installPhase = ''
          mkdir -p $out/bin
          cp vicNl $out/bin
        '';
      };
      platypus = pkgs.python3Packages.buildPythonPackage rec {
        pname = "platypus-opt";
        version = "1.4.1";
        src = pkgs.fetchPypi {
          inherit version;
          pname = "platypus_opt";
          hash = "sha256-s0F5C+ZzVMLJZLs0MbmsyKGhR6yp2UZKQ3INxYs4leg=";
        };
        doCheck = false;
        pyproject = true;
        build-system = [
          pkgs.python3Packages.setuptools
        ];
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        name = "lstm";
        buildInputs = with pkgs.python3Packages; [
          matplotlib
          # equinox
          # optax
          # pandas
          # tqdm
          # jax-cuda12-plugin
          # seaborn
        ] ++ libInputs ++ [
          pkgs.cudaPackages.cuda_nvcc
          pkgs.nvtopPackages.nvidia
          pkgs.uv
          pkgs.ty
          pkgs.ruff
          pkgs.rassumfrassum
          vic
          # platypus
        ];

        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath libInputs}:/run/opengl-driver/lib:/run/opengl-driver-32/lib"
          export XLA_FLAGS="--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}"
          export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
          source ./.venv/bin/activate
        '';
      };
    };
}
