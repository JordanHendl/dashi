# Dashi

Dashi is a low-level graphics backend written in Rust. It provides efficient abstractions for rendering tasks, while giving developers direct control over GPU resources, making it suitable for performance-critical applications like game engines, real-time visualizations, and rendering tools.

## Features

- **Low-Level Control**: Offers a thin layer of abstraction over graphics APIs, allowing you to manage GPU resources efficiently without unnecessary overhead.
- **Cross-Platform**: Designed to be compatible across multiple platforms, providing seamless experiences on Windows and Linux.
- **Rust Safety**: Combines the power of Rust's ownership and borrowing system to ensure memory and thread safety, while dealing with low-level graphics.
- **Extensible**: Built with extensibility in mind, enabling integration with higher-level frameworks or custom rendering pipelines.

## Getting Started

### Prerequisites

To start using Dashi, you'll need:

- [Rust](https://www.rust-lang.org/tools/install) (version 1.50 or higher recommended)
- A supported GPU with up-to-date drivers

### Installation

Add Dashi to your project's `Cargo.toml`:

```toml
[dependencies]
dashi = {git = "https://github.com/JordanHendl/dashi"}
```

### Example Usage

See the [tests](https://github.com/JordanHendl/dashi/tree/main/tests) for example usage.

### Window Backends

Dashi ships with multiple window backends. The default `dashi-sdl2` feature
uses SDL2 for a full event system. If build times are a concern you can instead
enable the `dashi-minifb` feature which relies on `minifb`. It compiles much
faster but only offers very basic input handling.
Only one of these window features can be enabled at a time.

## Documentation

Detailed documentation is available [here](https://github.com/JordanHendl/dashi/wiki). It includes guides, API references, and examples to help you get the most out of Dashi.

## Roadmap

- [x] Vulkan backend support
- [x] Bump Allocation (using https://github.com/sebbbi/OffsetAllocator/tree/main)
- [ ] Support for compute workloads
- [ ] Direct3D integration

## License

Dashi is licensed under the [MIT License](LICENSE).

## Contact

For questions or suggestions, feel free to reach out via [GitHub Issues](https://github.com/JordanHendl/dashi/issues).


