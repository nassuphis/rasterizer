#!/usr/bin/env swift
import Foundation
import CoreImage
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

func die(_ msg: String) -> Never {
    fputs("error: \(msg)\n", stderr)
    exit(1)
}

enum AutoGammaMode: String {
    case none
    case median
}

struct Opts {
    var dump = false
    var bins = 256

    // Percentile clipping (percent of pixels)
    var clipLow: Double  = 0.0
    var clipHigh: Double = 1.0

    // Peak limiting: cap any bin to peakFactor*(1/bins) and redistribute excess.
    // 0 disables.
    var peakFactor: Double = 0.0

    // Gamma
    var gamma: Double = 1.0
    var autoGamma: AutoGammaMode = .none
    var target: Double = 0.5   // target median after stretch if autoGamma=median

    // Optional sigmoid on normalized values (S-curve)
    var sigmoidStrength: Double = 0.0
    var sigmoidMid: Double = 0.5

    // Optional vibrance (after tone curve)
    var vibrance: Double = 0.0

    // NEW: final pooled RGB stretch to quantiles q and 1-q (applied LAST)
    // q in [0, 0.5). If user passes >0.5 and <=50 treat as percent (e.g. 1 -> 1%).
    var pooledRGBQuantile: Double? = nil
}

func parseDouble(_ s: String, _ name: String) -> Double {
    guard let v = Double(s) else { die("bad \(name): \(s)") }
    return v
}
func parseInt(_ s: String, _ name: String) -> Int {
    guard let v = Int(s) else { die("bad \(name): \(s)") }
    return v
}

func peakLimitPDF(_ pdfIn: [Double], factor: Double) -> [Double] {
    guard factor > 0 else { return pdfIn }
    let bins = pdfIn.count
    let avg = 1.0 / Double(bins)
    let limit = factor * avg

    var pdf = pdfIn
    var excess = 0.0
    for i in 0..<bins {
        if pdf[i] > limit {
            excess += (pdf[i] - limit)
            pdf[i] = limit
        }
    }
    let add = excess / Double(bins)
    for i in 0..<bins { pdf[i] += add }

    // Renormalize
    let total = pdf.reduce(0.0, +)
    if total > 0 {
        for i in 0..<bins { pdf[i] /= total }
    }
    return pdf
}

func cdfFromPDF(_ pdf: [Double]) -> [Double] {
    var cdf = [Double](repeating: 0.0, count: pdf.count)
    var acc = 0.0
    for i in 0..<pdf.count {
        acc += pdf[i]
        cdf[i] = acc
    }
    return cdf
}

func findBin(_ cdf: [Double], target: Double) -> Int {
    for i in 0..<cdf.count {
        if cdf[i] >= target { return i }
    }
    return cdf.count - 1
}

func computeGammaFromMedian(pdf: [Double], black: Double, white: Double, target: Double) -> Double {
    let bins = pdf.count
    let cdf = cdfFromPDF(pdf)
    let medBin = findBin(cdf, target: 0.5)
    let x = Double(medBin) / Double(bins - 1)
    let denom = max(white - black, 1e-9)
    let s = max(1e-6, min(1.0 - 1e-6, (x - black) / denom))
    let t = max(1e-6, min(1.0 - 1e-6, target))

    var g = log(t) / log(s)
    if !g.isFinite { g = 1.0 }
    g = max(0.5, min(2.0, g))
    return g
}

func areaHistogramRGBAf(image: CIImage, extent: CGRect, bins: Int, ctx: CIContext) -> [Float] {
    guard let h = CIFilter(name: "CIAreaHistogram") else { die("CIAreaHistogram not available") }
    h.setValue(image, forKey: kCIInputImageKey)
    h.setValue(CIVector(cgRect: extent), forKey: kCIInputExtentKey)
    h.setValue(bins, forKey: "inputCount")
    h.setValue(1.0, forKey: "inputScale")
    guard let histImage = h.outputImage else { die("failed to compute histogram") }

    let he = histImage.extent.integral
    var buf = [Float](repeating: 0, count: bins * 4)
    buf.withUnsafeMutableBytes { raw in
        ctx.render(histImage,
                   toBitmap: raw.baseAddress!,
                   rowBytes: bins * 4 * MemoryLayout<Float>.size,
                   bounds: he,
                   format: .RGBAf,
                   colorSpace: nil)
    }
    return buf
}

func pooledRGBPDF(fromHistRGBAf hist: [Float], bins: Int) -> [Double] {
    // Build pooled PDF = avg(normalized R, normalized G, normalized B)
    var rTot = 0.0, gTot = 0.0, bTot = 0.0
    for i in 0..<bins {
        rTot += Double(max(0.0, hist[i * 4 + 0]))
        gTot += Double(max(0.0, hist[i * 4 + 1]))
        bTot += Double(max(0.0, hist[i * 4 + 2]))
    }
    if rTot <= 0 || gTot <= 0 || bTot <= 0 { die("histogram totals were zero") }

    var pdf = [Double](repeating: 0.0, count: bins)
    for i in 0..<bins {
        let r = Double(max(0.0, hist[i * 4 + 0])) / rTot
        let g = Double(max(0.0, hist[i * 4 + 1])) / gTot
        let b = Double(max(0.0, hist[i * 4 + 2])) / bTot
        pdf[i] = (r + g + b) / 3.0
    }
    // Normalize (should already sum to 1, but guard numerical drift)
    let total = pdf.reduce(0.0, +)
    if total > 0 {
        for i in 0..<bins { pdf[i] /= total }
    }
    return pdf
}

// ---------------- CLI ----------------

var argv = Array(CommandLine.arguments.dropFirst())
var opts = Opts()

func usage() -> Never {
    print("""
    usage:
      preview_autolevels_rgb.swift [options] input output

    options:
      --dump, -d
      --bins N                 (default 256)
      --clip-low P             (default 0.0)
      --clip-high P            (default 1.0)
      --peak-factor F          (0 disables; try 4..12)

      --gamma G                (default 1.0)
      --auto-gamma median      (compute gamma from median to hit --target)
      --target T               (default 0.5; used with auto-gamma)

      --sigmoid SxM            e.g. 3x0.5 or 4x0.45 (0 disables)
      --vibrance A             e.g. 0.05 (0 disables)

      --pooled_rgb q           FINAL STEP: pooled RGB stretch to quantiles q and 1-q.
                              q can be fraction (0.01) or percent (1 -> 1%).
                              This is applied AFTER all other adjustments.

    """)
    exit(1)
}

while let a = argv.first, a.hasPrefix("-") {
    argv.removeFirst()
    switch a {
    case "--dump", "-d":
        opts.dump = true

    case "--bins":
        guard let v = argv.first else { usage() }
        argv.removeFirst()
        opts.bins = parseInt(v, "bins")

    case "--clip-low":
        guard let v = argv.first else { usage() }
        argv.removeFirst()
        opts.clipLow = parseDouble(v, "clip-low")

    case "--clip-high":
        guard let v = argv.first else { usage() }
        argv.removeFirst()
        opts.clipHigh = parseDouble(v, "clip-high")

    case "--peak-factor":
        guard let v = argv.first else { usage() }
        argv.removeFirst()
        opts.peakFactor = parseDouble(v, "peak-factor")

    case "--gamma":
        guard let v = argv.first else { usage() }
        argv.removeFirst()
        opts.gamma = parseDouble(v, "gamma")
        opts.autoGamma = .none

    case "--auto-gamma":
        guard let v = argv.first else { usage() }
        argv.removeFirst()
        guard let m = AutoGammaMode(rawValue: v) else { die("bad --auto-gamma \(v) (use median)") }
        opts.autoGamma = m

    case "--target":
        guard let v = argv.first else { usage() }
        argv.removeFirst()
        opts.target = parseDouble(v, "target")

    case "--sigmoid":
        guard let v = argv.first else { usage() }
        argv.removeFirst()
        let parts = v.split(separator: "x", omittingEmptySubsequences: false)
        guard parts.count == 2,
              let s = Double(parts[0]),
              let m = Double(parts[1]) else {
            die("bad --sigmoid format. Use SxM (e.g. 3x0.5)")
        }
        opts.sigmoidStrength = s
        opts.sigmoidMid = m

    case "--vibrance":
        guard let v = argv.first else { usage() }
        argv.removeFirst()
        opts.vibrance = parseDouble(v, "vibrance")

    case "--pooled_rgb", "--pooled-rgb":
        guard let v = argv.first else { usage() }
        argv.removeFirst()
        var q = parseDouble(v, "pooled_rgb q")
        // Allow "1" meaning 1% for convenience
        if q > 0.5 && q <= 50.0 { q /= 100.0 }
        if q < 0 || q >= 0.5 { die("--pooled_rgb must be in [0,0.5) as fraction (0.01) or percent (1 = 1%)") }
        opts.pooledRGBQuantile = q

    case "--help", "-h":
        usage()

    default:
        die("unknown option: \(a)")
    }
}

guard argv.count == 2 else { usage() }

let inURL  = URL(fileURLWithPath: argv[0])
let outURL = URL(fileURLWithPath: argv[1])

guard let ciIn = CIImage(contentsOf: inURL) else {
    die("cannot read input: \(inURL.path)")
}

// EXIF orientation
let exifOrientation: Int32 = {
    guard let src = CGImageSourceCreateWithURL(inURL as CFURL, nil),
          let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
    else { return 1 }

    if let n = props[kCGImagePropertyOrientation] as? NSNumber { return n.int32Value }
    if let i = props[kCGImagePropertyOrientation] as? Int { return Int32(i) }
    return 1
}()

let oriented = ciIn.oriented(forExifOrientation: exifOrientation)
let extent = oriented.extent.integral

let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
let ctx = CIContext(options: [
    CIContextOption.workingColorSpace: sRGB,
    CIContextOption.outputColorSpace:  sRGB
])

// ---- STEP 1: initial pooled RGB histogram (from input) ----
let histIn = areaHistogramRGBAf(image: oriented, extent: extent, bins: opts.bins, ctx: ctx)
var pdf = pooledRGBPDF(fromHistRGBAf: histIn, bins: opts.bins)

// Spike suppression (optional)
pdf = peakLimitPDF(pdf, factor: opts.peakFactor)

let cdf = cdfFromPDF(pdf)
let lowTarget  = max(0.0, min(1.0, opts.clipLow / 100.0))
let highTarget = max(0.0, min(1.0, 1.0 - (opts.clipHigh / 100.0)))

let blackBin = findBin(cdf, target: lowTarget)
let whiteBin = findBin(cdf, target: highTarget)

let black = Double(blackBin) / Double(opts.bins - 1)
let white = max(Double(whiteBin) / Double(opts.bins - 1), black + 1e-6)

var gammaToUse = opts.gamma
if opts.autoGamma == .median {
    gammaToUse = computeGammaFromMedian(pdf: pdf, black: black, white: white, target: opts.target)
}

if opts.dump {
    print("INPUT pooled endpoints:")
    print("  bins=\(opts.bins) clipLow=\(opts.clipLow)% clipHigh=\(opts.clipHigh)% peakFactor=\(opts.peakFactor)")
    print(String(format: "  blackBin=%d black=%.6f  whiteBin=%d white=%.6f", blackBin, black, whiteBin, white))
    print(String(format: "  gamma=%.6f autoGamma=%@ target=%.3f", gammaToUse, opts.autoGamma.rawValue, opts.target))
    print(String(format: "  sigmoid=%.3fx%.3f  vibrance=%.3f", opts.sigmoidStrength, opts.sigmoidMid, opts.vibrance))
    if let q = opts.pooledRGBQuantile {
        print(String(format: "  pooled_rgb(final) q=%.6f", q))
    }
}

// ---- STEP 2: apply per-channel curve (same mapping on R,G,B) ----
guard let kernel = CIColorKernel(source: """
kernel vec4 autoLevelsRGB(__sample s,
                          float black, float white,
                          float gamma,
                          float sigStrength, float sigMid) {

  float denom = max(white - black, 1e-6);

  vec3 v = (s.rgb - vec3(black)) / denom;
  v = clamp(v, 0.0, 1.0);

  v = pow(v, vec3(gamma));

  if (sigStrength > 0.0) {
    float s0 = 1.0 / (1.0 + exp(sigStrength * (sigMid - 0.0)));
    float s1 = 1.0 / (1.0 + exp(sigStrength * (sigMid - 1.0)));
    vec3 ss = 1.0 / (1.0 + exp(sigStrength * (sigMid - v)));
    v = clamp((ss - s0) / max(s1 - s0, 1e-6), 0.0, 1.0);
  }

  return vec4(clamp(v, 0.0, 1.0), s.a);
}
""") else {
    die("failed to compile kernel (CI kernel language is deprecated but still works)")
}

guard var ciOut = kernel.apply(extent: extent, arguments: [
    oriented,
    Float(black), Float(white),
    Float(gammaToUse),
    Float(opts.sigmoidStrength), Float(opts.sigmoidMid)
]) else {
    die("failed to apply RGB levels")
}

// Optional vibrance (usually keep small)
if abs(opts.vibrance) > 1e-9 {
    guard let vib = CIFilter(name: "CIVibrance") else { die("CIVibrance not available") }
    vib.setValue(ciOut, forKey: kCIInputImageKey)
    vib.setValue(opts.vibrance, forKey: "inputAmount")
    if let o = vib.outputImage { ciOut = o }
}

// ---- STEP 3 (NEW): final pooled RGB stretch to quantiles q and 1-q ----
if let qIn = opts.pooledRGBQuantile, qIn > 0 {
    let histOut = areaHistogramRGBAf(image: ciOut, extent: extent, bins: opts.bins, ctx: ctx)
    var pdfOut = pooledRGBPDF(fromHistRGBAf: histOut, bins: opts.bins)

    // NOTE: no peak limiting here unless you want it — user asked for quantiles directly.
    let cdfOut = cdfFromPDF(pdfOut)

    let loBin = findBin(cdfOut, target: qIn)
    let hiBin = findBin(cdfOut, target: 1.0 - qIn)

    let lo = Double(loBin) / Double(opts.bins - 1)
    let hi = max(Double(hiBin) / Double(opts.bins - 1), lo + 1e-6)

    if opts.dump {
        print("FINAL pooled_rgb stretch (computed on output):")
        print(String(format: "  q=%.6f  loBin=%d lo=%.6f  hiBin=%d hi=%.6f",
                     qIn, loBin, lo, hiBin, hi))
    }

    guard let finalKernel = CIColorKernel(source: """
kernel vec4 finalPooledStretch(__sample s, float lo, float hi) {
  float denom = max(hi - lo, 1e-6);
  vec3 v = (s.rgb - vec3(lo)) / denom;
  v = clamp(v, 0.0, 1.0);
  return vec4(v, s.a);
}
""") else {
        die("failed to compile final stretch kernel")
    }

    guard let stretched = finalKernel.apply(extent: extent, arguments: [ciOut, Float(lo), Float(hi)]) else {
        die("failed to apply final pooled stretch")
    }
    ciOut = stretched
}

// ---- Render + write ----
guard let cg = ctx.createCGImage(ciOut, from: extent) else {
    die("failed to render output")
}

let outType = UTType(filenameExtension: outURL.pathExtension) ?? .png
guard let dest = CGImageDestinationCreateWithURL(outURL as CFURL,
                                                outType.identifier as CFString,
                                                1, nil) else {
    die("cannot create output: \(outURL.path)")
}

var props: [CFString: Any] = [:]
if outType.conforms(to: .jpeg) {
    props[kCGImageDestinationLossyCompressionQuality] = 0.95
}
CGImageDestinationAddImage(dest, cg, props as CFDictionary)

guard CGImageDestinationFinalize(dest) else {
    die("failed writing: \(outURL.path)")
}
