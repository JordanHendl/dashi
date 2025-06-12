# Dashi Tests

## Hello Triangle
[Drawing a moving triangle](https://github.com/JordanHendl/dashi/blob/main/tests/hello_triangle/bin.rs)

## Minifb Triangle
This test demonstrates rendering using the lightweight `minifb` window backend.
It is ignored by default because it opens a native window.

Build and run it with:

```bash
cargo test --features dashi-minifb --test minifb_triangle -- --ignored
```

You can also run all tests (without executing the ignored one) via `cargo test`.
