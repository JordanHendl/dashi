# Dashi Tests

## Hello Triangle
[Drawing a moving triangle](https://github.com/JordanHendl/dashi/blob/main/tests/hello_triangle/bin.rs)

## Minifb Triangle
This test demonstrates rendering using the lightweight `minifb` window backend.
It is ignored by default because it opens a native window.

Build and run it with:

```bash
cargo test --no-default-features --features dashi-minifb --test minifb_triangle -- --ignored
```

You can also run all tests (without executing the ignored one) via `cargo test`.

## Framebuffer Comparison
The `framebuffer_compare` test demonstrates capturing GPU output to a CPU visible buffer and comparing it against a reference PNG image using the helper functions in `image_utils`.
Reference PNGs are not stored in the repo. Place them under `tests/reference` before running.

Run all tests with:

```bash
cargo test
```
