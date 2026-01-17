# Rasterizer

A high-performance Python library for converting clouds of complex-numbered points into stunning bilevel (black & white) and color-annotated images. Combines NumPy array operations with Numba JIT compilation for point rendering and PyVIPS for advanced image composition and output.

## Features

- **Fast rasterization** of complex-plane point clouds using Numba-accelerated parallel stamping
- **Multiple colorization schemes** including RGB/HSV interpolation, histogram equalization, and multi-palette field blending
- **Auto-levels and tone mapping** for professional-quality JPEG output
- **Advanced composition**: text footers, rounded passepartout frames, grid mosaics
- **Flexible output formats**: 1-bit PNG, 8-bit RGB PNG, JPEG with metadata embedding
- **Extensible palette system** with 100+ named tri-palettes and gradient presets

## Installation

### Using uv (recommended)

```bash
uv add rasterizer
```

### Using pip

```bash
pip install rasterizer
```

### Development installation

```bash
git clone https://github.com/yourusername/rasterizer.git
cd rasterizer
uv sync --dev
```

## Quick Start

```python
import numpy as np
from rasterizer import canvas, export

# Generate complex points (e.g., from a dynamical system)
z = np.random.randn(10000) + 1j * np.random.randn(10000)

# Project to pixel coordinates
px, py = canvas.project_to_canvas(z, pix=1024, margin_frac=0.1)

# Create canvas and stamp points with circular dots
img = np.zeros((1024, 1024), dtype=np.uint8)
dy, dx = canvas.make_disc_offsets(r=2)
canvas.stamp_points(img, py, px, dy, dx, value=255)

# Save as 1-bit PNG
export.save_png_bilevel(img, "output.png", invert=False)
```

## Architecture

The package follows a modular pipeline design:

```
Input (complex array z)
    |
    v
canvas.project_to_canvas() ---> pixel coordinates (px, py)
    |
    v
canvas.stamp_points() ---------> bilevel canvas (Numba JIT)
    |
    v
colors.rgb_scheme_*() ---------> RGB colorization (optional)
    |
    v
compose.add_footer_label() ----> text annotation (optional)
compose.add_rounded_passepartout_bilevel_pct() -> framing (optional)
    |
    v
export.save_*() ---------------> file output (PNG/JPEG/VIPS)
```

## Module Reference

| Module | Purpose |
|--------|---------|
| `canvas` | Low-level rasterization: point stamping, radius bucketing, coordinate projection |
| `colors` | Colorization schemes: RGB/HSV interpolation, histogram equalization, palette blending |
| `formats` | NumPy <-> PyVIPS image conversions |
| `compose` | Image composition: footers, rounded frames, padding |
| `export` | File output: PNG, JPEG, VIPS native format, mosaics |
| `autolevels` | Tone mapping and auto-levels for JPEG export |
| `footer` | Text glyph rendering with morphological blending |
| `image2spec` | XMP/EXIF metadata embedding for reproducibility |

---

## canvas - Rasterization & Point Stamping

High-performance point rendering with Numba JIT compilation.

### Functions

#### `make_disc_offsets(r: int) -> tuple[np.ndarray, np.ndarray]`

Generate circular stamp coordinates for radius `r`.

```python
dy, dx = canvas.make_disc_offsets(5)  # 5-pixel radius disc
```

#### `stamp_points(canvas, ys, xs, dy, dx, value=255)`

Stamp points onto a canvas using Numba-compiled parallel loops.

```python
canvas_array = np.zeros((512, 512), dtype=np.uint8)
canvas.stamp_points(canvas_array, ys, xs, dy, dx, value=255)
```

#### `bucket_by_radius(r_px, r_min, r_max) -> tuple`

Sort points by radius for efficient batch stamping with different dot sizes.

```python
order, r_vals, starts, counts = canvas.bucket_by_radius(radii, 1, 10)
```

#### `project_to_canvas(z: np.ndarray, pix: int, margin_frac: float) -> tuple`

Project complex coordinates to pixel coordinates.

```python
px, py = canvas.project_to_canvas(z, pix=1024, margin_frac=0.1)
```

#### `render_to_canvas(z, pix, margin_frac) -> np.ndarray`

Direct rendering of complex points to a canvas (convenience function).

#### `warmup_raster_kernels()`

Pre-compile Numba kernels to avoid first-call latency.

---

## colors - Colorization Schemes

Sophisticated colorization for Lyapunov exponent fields and similar data.

### Color Parsing

```python
from rasterizer.colors import parse_color_spec

# Hex formats
r, g, b = parse_color_spec("FF0000", default=(0, 0, 0))      # Red
r, g, b = parse_color_spec("#00FF00", default=(0, 0, 0))     # Green

# Named colors
r, g, b = parse_color_spec("blue", default=(0, 0, 0))        # Blue
r, g, b = parse_color_spec("coral", default=(0, 0, 0))       # Coral
```

### RGB Colorization Schemes

#### `rgb_scheme_mh(lyap, params) -> np.ndarray`

Markus & Hess style with linear normalization.

```python
rgb = rgb_scheme_mh(lyap_field, {
    'neg_color': 'FFFF00',   # Yellow for negative values
    'zero_color': '000000',  # Black for zero
    'pos_color': 'FF0000',   # Red for positive values
})
```

#### `rgb_scheme_mh_eq(lyap, params) -> np.ndarray`

Markus & Hess with histogram equalization for better contrast.

#### `rgb_scheme_palette_eq(lyap, params) -> np.ndarray`

Use a named tri-palette from the built-in palette library.

```python
rgb = rgb_scheme_palette_eq(lyap_field, {
    'palette': 'rg',      # Red-Green palette
    'gamma': 1.2,         # Tone curve adjustment
})
```

#### `rgb_scheme_multipoint(lyap, params) -> np.ndarray`

N-stop gradient colorization.

```python
rgb = rgb_scheme_multipoint(lyap_field, {
    'color_string': 'FF0000:FFFF00:00FFFF:0000FF',  # Red->Yellow->Cyan->Blue
    'gamma': 1.0,
})
```

### HSV Colorization Schemes

HSV interpolation uses shortest-arc hue blending to avoid discontinuities at the red/magenta boundary.

#### `hsv_scheme_mh_eq(lyap, params) -> np.ndarray`

HSV interpolation with histogram equalization.

#### `hsv_scheme_palette_eq(lyap, params) -> np.ndarray`

HSV interpolation with named palette.

### Per-Pixel Palette Field Blending

Blend two palettes per-pixel based on image features (gradient, texture, etc.).

#### `rgb_scheme_palette_field(lyap, params) -> np.ndarray`

```python
rgb = rgb_scheme_palette_field(lyap_field, {
    'paletteA': 'rg',           # Base palette
    'paletteB': 'gc',           # Blend palette
    'w_feature': 'grad',        # Weight by gradient magnitude
    'w_sigma': 2.0,             # Smooth the weight field
    'w_strength': 0.8,          # Blend strength [0, 1]
    'norm': 'eq',               # Histogram equalization
})
```

**Available features for `w_feature`:**

| Feature | Description |
|---------|-------------|
| `grad` | Gradient magnitude |
| `gradx`, `grady` | Directional gradients |
| `grad_dir` | Oriented energy along specific angle |
| `lap` | Laplacian (edges) |
| `lvar` | Local variance (texture) |
| `dog` | Difference of Gaussians |
| `st_coh` | Structure tensor coherence |
| `gabor_max` | Maximum Gabor response across orientations |
| `gabor_theta` | Gabor response at specific orientation |
| `ms_ratio` | Multiscale band energy ratio |

#### `hsv_scheme_palette_field(lyap, params) -> np.ndarray`

Same as above but with HSV interpolation and saturation-aware hue pinning.

### Palette Visualization

```python
from rasterizer.colors import create_palette_field

# Visualize palette blending space
img = create_palette_field(
    'rg', 'gc',           # Two palettes to blend
    pix=512,              # Output size
    interp='hsv',         # Interpolation mode
    gamma=1.0,
)
```

---

## formats - Array Conversions

Bridge between NumPy arrays and PyVIPS images.

#### `np_to_vips_rgb_u8(arr: np.ndarray) -> vips.Image`

Convert RGB array to PyVIPS. Accepts both `HxWx3` and `3xHxW` layouts.

```python
from rasterizer.formats import np_to_vips_rgb_u8

vips_img = np_to_vips_rgb_u8(rgb_array)
```

#### `np_to_vips_gray_u8(arr: np.ndarray) -> vips.Image`

Convert grayscale array to PyVIPS.

```python
from rasterizer.formats import np_to_vips_gray_u8

vips_img = np_to_vips_gray_u8(gray_array)
```

---

## compose - Image Composition

Add text labels, frames, and artistic effects.

#### `add_footer_label(base, text, ...) -> vips.Image`

Render text footer with automatic font sizing and word wrapping.

```python
from rasterizer.compose import add_footer_label

result = add_footer_label(
    vips_img,
    "Lyapunov Exponent Field, x in [-1,1], y in [-2,2]",
    footer_frac=0.02,       # Text height as fraction of image
    pad_lr_px=48,           # Left/right padding
    dpi=300,
    align="centre",
    font_family="PT Mono",
    font_weight="Regular",
    invert=False,           # True for dark backgrounds
)
```

#### `add_rounded_passepartout_bilevel_pct(img, ...) -> vips.Image`

Add a rounded rectangular mat frame (passepartout).

```python
from rasterizer.compose import add_rounded_passepartout_bilevel_pct

result = add_rounded_passepartout_bilevel_pct(
    vips_img,
    margin_frac=0.10,       # 10% margin
    radius_frac=0.04,       # 4% corner radius
    auto_white_bg=True,     # Auto-detect mat color from edges
)
```

#### `pad_to_square(im, px) -> vips.Image`

Center-pad image to a square canvas.

```python
from rasterizer.compose import pad_to_square

result = pad_to_square(vips_img, 1024)
```

---

## export - File Output

Save images in various formats with optional processing.

#### `save_png_bilevel(canvas, out_path, invert, footer_text=None, passepartout=False)`

Save as 1-bit PNG (smallest file size for bilevel images).

```python
from rasterizer.export import save_png_bilevel

save_png_bilevel(canvas_array, "output.png", invert=False)
```

#### `save_png_rgb(rgb, out_path, invert=False, footer_text=None)`

Save as 8-bit RGB PNG.

#### `save_jpg_rgb(rgb, out_path, footer_text=None, quality=95, spec=None, autolvl=True, resize=None)`

Save as JPEG with optional auto-levels, footer, and metadata.

```python
from rasterizer.export import save_jpg_rgb

save_jpg_rgb(
    rgb_array,
    "output.jpg",
    footer_text="My Image",
    quality=95,
    autolvl=True,           # Apply auto-levels
    spec="param1=value1",   # Embed metadata
    resize=2048,            # Resize to max dimension
)
```

#### `save_mosaic_png_rgb(tiles, titles, cols, gap, out_path, ...)`

Compose multiple RGB tiles into a grid mosaic.

```python
from rasterizer.export import save_mosaic_png_rgb

save_mosaic_png_rgb(
    tiles=[tile1, tile2, tile3, tile4],
    titles=["A", "B", "C", "D"],
    cols=2,
    gap=16,
    out_path="mosaic.png",
    invert=False,
    thumbnail=1024,         # Optional: resize output
)
```

#### `save_mosaic_png_bilevel(tiles, titles, cols, gap, out_path, invert, ...)`

Same as above for bilevel tiles.

#### `add_suffix_number(path, n, width=5) -> str`

Insert a number before the file extension.

```python
from rasterizer.export import add_suffix_number

path = add_suffix_number("image.png", 42)  # "image_00042.png"
```

---

## autolevels - Tone Mapping

Professional auto-levels processing for JPEG output.

```python
from rasterizer.autolevels import AutoLevelsRGBConfig, process_image

config = AutoLevelsRGBConfig(
    bins=256,
    clip_low=0.0,
    clip_high=1.0,
    peak_factor=8.0,
    gamma=1.0,
    auto_gamma="median",
    target=0.55,
    sigmoid_strength=3,
    vibrance=0.05,
)

# Apply to float [0,1] image
result = process_image(vips_img_float, config)
```

---

## Color Palettes

### Tri-Palettes

Tri-palettes define three colors for negative, zero, and positive values:

```
NEG_COLOR:ZERO_COLOR:POS_COLOR
```

**Built-in examples:**

| Name | Spec | Description |
|------|------|-------------|
| `rg` | `FFFF00:000000:FF0000` | Yellow-Black-Red |
| `gc` | `00FF00:000000:00FFFF` | Green-Black-Cyan |
| `fire` | `FFFF00:800000:FF0000` | Yellow-Maroon-Red |
| `ice` | `00FFFF:000080:0000FF` | Cyan-Navy-Blue |

### Multi-Stop Gradients

For `rgb_scheme_multipoint`, use colon-separated hex colors:

```python
'color_string': 'FF0000:FFFF00:00FF00:00FFFF:0000FF'
```

### CLI Palette Tools

List available palettes:

```bash
python -m rasterizer.colors --export tri --format json
python -m rasterizer.colors --export tri --only warm --values
python -m rasterizer.colors --export color  # Named colors
```

Generate palette visualization:

```bash
python -m rasterizer.colors field rg,gc,1.2,1024,hsv -o palette.jpg
```

---

## CLI Reference

### colors.py

```bash
# Export palette lists
python -m rasterizer.colors --export tri                    # Tri-palette names
python -m rasterizer.colors --export tri --values           # With hex specs
python -m rasterizer.colors --export tri --only warm        # Filter warm palettes
python -m rasterizer.colors --export color --format json    # Named colors as JSON

# Generate palette field visualization
python -m rasterizer.colors field PALETTE_A,PALETTE_B[,gamma[,pix[,interp]]] -o out.jpg
python -m rasterizer.colors field rg,gc,1.2,1024,hsv -o field.jpg
```

### footer.py (Test Harness)

```bash
python -m rasterizer.footer --pix 5000 --ratio 20.0 --pad 10.0 --out test.jpg
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **numba** | >=0.62,<0.64 | JIT compilation for `stamp_points`, `bucket_by_radius` |
| **numpy** | >=2.3,<2.4 | Array operations, histogram equalization |
| **pillow** | >=12.1.0 | JPEG encoding |
| **pyvips** | >=3.1.1 | Image I/O, composition, lazy evaluation |
| **scipy** | >=1.17.0 | Sobel, Laplacian, Gaussian filters, convolution |

**Note:** PyVIPS requires [libvips](https://www.libvips.org/) to be installed on your system.

---

## Testing

Run the test suite:

```bash
uv run python -m pytest tests/ -v
```

With coverage:

```bash
uv run python -m pytest tests/ -v --cov=rasterizer --cov-report=html
```

The test suite covers:

- `test_canvas.py` - Stamping, bucketing, coordinate projection
- `test_colors.py` - Color parsing, histogram equalization, colorization
- `test_compose.py` - Footer rendering, passepartout, padding
- `test_export.py` - File I/O, mosaic assembly
- `test_formats.py` - NumPy <-> PyVIPS conversions

---

## Examples

### Full Colorization Pipeline

```python
import numpy as np
from rasterizer import canvas, export
from rasterizer.colors import hsv_scheme_palette_eq
from rasterizer.compose import add_footer_label, add_rounded_passepartout_bilevel_pct
from rasterizer.formats import np_to_vips_rgb_u8

# 1. Generate Lyapunov exponent field (example: random data)
lyap = np.random.randn(2048, 2048) * 0.5

# 2. Colorize with HSV palette
rgb = hsv_scheme_palette_eq(lyap, {
    'palette': 'fire',
    'gamma': 1.1,
})

# 3. Convert to PyVIPS for composition
vips_img = np_to_vips_rgb_u8(rgb)

# 4. Add footer label
vips_img = add_footer_label(
    vips_img,
    "Lyapunov Exponent Field",
    footer_frac=0.015,
)

# 5. Save with auto-levels
export.save_jpg_rgb(rgb, "lyapunov.jpg", autolvl=True, quality=95)
```

### Batch Processing with Mosaics

```python
from rasterizer.export import save_mosaic_png_bilevel, add_suffix_number

# Process multiple frames
frames = []
for i, z in enumerate(point_clouds):
    img = np.zeros((512, 512), dtype=np.uint8)
    px, py = canvas.project_to_canvas(z, pix=512, margin_frac=0.1)
    dy, dx = canvas.make_disc_offsets(2)
    canvas.stamp_points(img, py, px, dy, dx)
    frames.append(img)

# Save as mosaic
save_mosaic_png_bilevel(
    frames,
    titles=[f"t={i}" for i in range(len(frames))],
    cols=4,
    gap=8,
    out_path="animation_mosaic.png",
    invert=False,
)
```

---

## License

MIT License

## Author

Nick Nassuphis (nicknassuphis@gmail.com)
