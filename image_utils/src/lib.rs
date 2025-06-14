use std::path::Path;
use image::ImageError;

/// Load a PNG file and return raw RGBA pixels and dimensions.
pub fn load_png(path: &Path) -> Result<(Vec<u8>, u32, u32), ImageError> {
    let img = image::open(path)?.to_rgba8();
    let (width, height) = img.dimensions();
    Ok((img.into_raw(), width, height))
}

/// Compare two RGBA buffers with a per-channel tolerance.
pub fn compare_rgba(actual: &[u8], expected: &[u8], width: u32, height: u32, tolerance: u8) -> bool {
    if actual.len() != expected.len() || actual.len() != (width * height * 4) as usize {
        return false;
    }
    for (a, e) in actual.iter().zip(expected.iter()) {
        let diff = if a > e { a - e } else { e - a };
        if diff > tolerance {
            return false;
        }
    }
    true
}

/// Optionally save a diff image visualizing the differences between two buffers.
pub fn save_diff(path: &Path, actual: &[u8], expected: &[u8], width: u32, height: u32) -> Result<(), ImageError> {
    let len = (width * height * 4) as usize;
    if actual.len() < len || expected.len() < len {
        return Err(ImageError::IoError(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "buffer size mismatch",
        )));
    }
    let mut diff_buf = Vec::with_capacity(len);
    for (a, e) in actual.iter().zip(expected.iter()) {
        let diff = if a > e { a - e } else { e - a };
        diff_buf.push(diff);
    }
    image::save_buffer(path, &diff_buf, width, height, image::ColorType::Rgba8)
}
