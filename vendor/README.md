# Mobilint SDK packages (vendor-provided)

Place the vendor-provided Mobilint compiler wheels here. They are **not** on public PyPI
and are **not** committed to git (`*.whl` / `*.tar.gz` are ignored).

Required for the compiler image:

- `qbcompiler-*.whl` — ONNX -> `.mxq` quantizing compiler wheel
  - depending on the package version, the Python import may be exposed as `qubee` or `qbcompiler`

```bash
cp /path/to/qbcompiler-*.whl ./vendor/
./build.sh
```

If you keep multiple compiler wheel versions under `vendor/`, `./build.sh` will stop and ask
you to either keep only one or select one explicitly:

```bash
./build.sh --compiler-wheel qbcompiler-1.1.2+aries2-py3-none-any.whl
```

`Dockerfile` installs `qbcompiler-*.whl` from `vendor/` during the image build.
QB runtime is installed separately via pip as `mobilint-qb-runtime`.
See <https://docs.mobilint.com/v1.3/en/introduction.html> and
<https://docs.mobilint.com/v1.3/en/installing_runtime_library.html> for package details.

> Note: use **qbruntime** (QB-RUNTIME) for ARISE — not `maccel`.
