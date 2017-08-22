with import <nixpkgs> {};
stdenv.mkDerivation rec {
  name = "env";
  env = buildEnv { name = name; paths = buildInputs; };
  buildInputs = [
    python
    python27Packages.tensorflowWithoutCuda
    python27Packages.ipython
    python27Packages.nose
    python27Packages.urllib3
    python27Packages.pillow
    python27Packages.Keras
    python27Packages.h5py
    jdk
    which
  ];
}