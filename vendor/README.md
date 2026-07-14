# Mobilint SDK packages (vendor-provided)

Place the vendor-provided Mobilint compiler wheels here. They are **not** on public PyPI
and are **not** committed to git (`*.whl` / `*.tar.gz` are ignored).

Required for the compiler image:

- `qubee-*.whl` — ONNX -> `.mxq` quantizing compiler

```bash
cp /path/to/qubee-*.whl ./vendor/
./build.sh
```

`Dockerfile` installs `qubee-*.whl` from `vendor/` during the image build.
QB runtime is installed separately via pip as `mobilint-qb-runtime`.
See <https://docs.mobilint.com/v1.3/en/introduction.html> and
<https://docs.mobilint.com/v1.3/en/installing_runtime_library.html> for package details.

> Note: use **qbruntime** (QB-RUNTIME) for ARISE — not `maccel`.
