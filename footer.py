import argparse
from pathlib import Path
import pyvips as vips

# -------------------------------------
#
#
#
# -------------------------------------

txt=(
    "slot:1,"
    "map:nn14:AB:-40:-40:40:40,"
    "m:j1s(cos(l**5-l**4+l**3-l**2+l))*j1s(ei(l*l-l))*j0s(si(l*x+l+x))*j1s(sin(l*l-l))*exp(aira(x)+1j*frec(l)),"
    "rgb:pfm:beauty_gucci_taupe_peach:pastel_warmwhite_forest:1.2:s1=0.868:s2=0.63:s3=0.977:se=1.0,"
    "hist:6:6:5120"
)

# -------------------------------------
#
# text is wrapped before image is
# generated
#
# -------------------------------------

def wrap_after_commas(s: str, N: int) -> str:
    if N <= 0: return s
    out, count = [], 0
    for ch in s:
        out.append(ch)
        count += 1
        if ch == "," and count >= N:
            out.append("\n")
            count = 0
    return "".join(out)

def text2glyph(txt,max_chars,border):
    wrapped_txt = wrap_after_commas(txt,max_chars)
    glyph = vips.Image.text(wrapped_txt, dpi=300, font="PT Mono Regular", align="centre",spacing=1)
    bilevel_glyph = (glyph>0).ifthenelse(255, 0)
    w = bilevel_glyph.width
    h = bilevel_glyph.height
    b = int(h*border)
    return bilevel_glyph.embed(b,b,w+2*b,h+2*b)

# -------------------------------------
#
# ratio is horizontal location in image
# pad is fraction of image width
#
# -------------------------------------

def scale_to_fit(overlay,base,ratio,pad):
    width_ratio = (base.width*(1-pad))/overlay.width
    wscaled_overlay = overlay.resize(width_ratio,kernel="nearest")
    height_ratio = min(1,base.height*ratio/wscaled_overlay.height)
    hscaled_overlay = wscaled_overlay.resize(height_ratio,kernel="nearest")
    return hscaled_overlay

def insertion_coords(base, overlay, ratio: float) -> tuple[int, int]:
    """
    Return (x, y) to insert `overlay` into `base` so it is:
      - horizontally centered
      - vertically centered within the bottom `ratio` fraction of `base` height

    ratio: 0..1 (e.g. 0.2 for bottom 20%)
    """
    H, W = int(base.height), int(base.width)
    h, w = int(overlay.height), int(overlay.width)

    r = float(ratio)
    if not (0.0 < r <= 1.0):
        raise ValueError("ratio must be in (0, 1].")

    band_h = max(1, int(round(r * H)))
    band_y0 = H - band_h

    x = (W - w) // 2
    y = band_y0 + (band_h - h) // 2
    return x, y

# -------------------------------------
#
# insert: destructive
# overlay: just the white areas
# fade: darken image close to letters
#
# -------------------------------------

def insert_glyph(base,overlay,ratio,pad):
    scaled_overlay = scale_to_fit(overlay,base,ratio,pad)
    x,y = insertion_coords(base,scaled_overlay, ratio)
    out = base.insert(scaled_overlay, x, y)
    return out

def overlay_glyph(base,overlay,ratio,pad):
    scaled_overlay = scale_to_fit(overlay,base,ratio,pad)
    w, h = scaled_overlay.width, scaled_overlay.height
    x,y = insertion_coords(base,scaled_overlay, ratio)
    base_region = base.crop(x,y,w,h)
    out = base.insert(base_region | scaled_overlay, x, y)
    return out

def fade_glyph(base,overlay,ratio,pad):
    scaled_overlay = scale_to_fit(overlay,base,ratio,pad)
    w, h = scaled_overlay.width, scaled_overlay.height
    x,y = insertion_coords(base,scaled_overlay, ratio)
    base_region = base.crop(x,y,w,h)
    mask = scaled_overlay
    morph_mask = [[255,255,255],[255,255,255],[255,255,255]]
    for _ in range(20): mask = mask.dilate(morph_mask)
    mask = mask.gaussblur(20)  
    black = base_region.new_from_image(0)
    white = base_region.new_from_image(255)
    blended_base_region = mask.ifthenelse(black, base_region, blend=True)
    insert_region = scaled_overlay.ifthenelse(white,blended_base_region)
    out = base.insert(insert_region, x, y)
    return out

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Footer renderer test harness.")
    ap.add_argument(
        "footer",
        nargs="?",
        type=str,
        default=txt,
        help="Footer text to render (default: built-in test string).",
    )
    ap.add_argument("--pix", type=int, default=5000)
    ap.add_argument("--ratio", type=float, default=20.0)
    ap.add_argument("--pad", type=float, default=10.0)
    ap.add_argument("--out", type=str, default="footer_test.jpg")
    return ap.parse_args()

def main() -> None:
    args = _parse_args()
    ratio = args.ratio / 100.0
    pad = args.pad / 100.0

    # make an interesting-looking image
    img = vips.Image.\
        perlin(args.pix,args.pix, cell_size=args.pix/30, uchar=True, seed=42).\
        scharr().\
        sobel().\
        linear(-1,255)
    
    img = img.bandjoin([(img%64)*4,((int(255)-img)%64)*4])
        
    #img = vips.Image.black(args.pix,args.pix).new_from_image(255)
    glyph = text2glyph(args.footer,40,1.0)
    fade_glyph(img,glyph,ratio,pad).write_to_file(args.out)

if __name__ == "__main__":
    main()

