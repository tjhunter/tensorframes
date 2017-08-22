with import <nixpkgs> {};

stdenv.mkDerivation rec {
  name = "env35";
  env = buildEnv { name = name; paths = buildInputs; };
  buildInputs = [
    python
    python35Packages.tensorflowWithoutCuda
    python35Packages.ipython
    python35Packages.nose
    python35Packages.urllib3
    python35Packages.pillow
    python35Packages.Keras
    python35Packages.h5py
    python35Packages.pandas
    jdk
    which
  ];
}