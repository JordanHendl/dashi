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

## OpenXR and multiview

The optional `dashi-openxr` feature coexists with ordinary window displays.
Load `XrGraphics` before Vulkan initialization and pass it in `ContextInfo.xr`.
This uses the runtime-selected physical device and `XR_KHR_vulkan_enable2` to
create Vulkan objects with the caller's allocation callbacks. Then call
`create_xr_session`. Applications own the session/event/frame loop and must
release session children before destroying the graphics context.

`import_xr_image` registers borrowed runtime images. Acquiring/waiting, layout
transitions, queue synchronization and releasing each image remain the caller's
responsibility. Destroy registrations before the runtime destroys its images.
This replaces the experimental `make_xr_display` API, which replaced windows
and submitted no projection layers.

`make_render_pass_multiview` and `make_graphics_pipeline_multiview` accept matching
view masks; ordinary methods use zero. Use `Type2DArray` attachment views and
check `multiview_max_views`. Mesh/geometry/tessellation multiview is not enabled.

```text
cargo test --features dashi-openxr --test multiview --test openxr_triangle
cargo test --features dashi-openxr --test openxr_triangle -- --ignored
```

The first command runs deterministic two-layer GPU readback and startup opt-in
checks without a headset. The ignored test requires a runtime/headset and checks
graphics/session creation only; it does not establish presentation correctness.

## Framebuffer Comparison
The `framebuffer_compare` test demonstrates capturing GPU output to a CPU visible buffer and comparing it against a reference PNG image using the helper functions in `image_utils`.
Reference PNGs are not stored in the repo. Place them under `tests/reference` before running.

Run all tests with:

```bash
cargo test
```

Verify the window-control path with:

```bash
cargo check
cargo test
cargo test --test winit_window_controls -- --ignored
```

The `winit_window_controls` test is Windows-only and ignored by default because
it opens a native window and exercises minimize/restore behavior.
