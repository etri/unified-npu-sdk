# Mobilint SDK packages (vendor-provided)

Place the vendor-provided Mobilint wheels here. They are **not** on public PyPI
and are **not** committed to git (`*.whl` / `*.tar.gz` are ignored).

Required for a full build/runtime image:

- `qubee-*.whl`      — ONNX -> `.mxq` quantizing compiler
- `qbruntime-*.whl`  — QB-RUNTIME (`.mxq` inference) + `mobilint-cli`

```bash
cp /path/to/qubee-*.whl     ./vendor/
cp /path/to/qbruntime-*.whl ./vendor/
./build.sh
```

`Dockerfile` installs every `vendor/*.whl` during the image build.
See <https://docs.mobilint.com/v1.2/en/introduction.html> for package details.

> Note: use **qbruntime** (QB-RUNTIME) for ARISE — not `maccel`.
