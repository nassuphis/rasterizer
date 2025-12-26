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

struct Opts {
    var dump = false
    var bins = 256

    // percentile clipping (in percent of pixels)
    var clipLow: Double  = 0.0
    var clipHigh: Double = 1.0

    // Peak limiting factor relative to average bin mass (1/bins).
    // 0 disables peak limiting.
    var peakFactor: Double = 0.0

    // Midtone curve
    var gamma: Double = 1.0 // applies on normalized luma
    var sigmoidStrength: Double = 0.0 // 0 disables sigmoid
    var sigmoidMid: Double = 0.5      // midpoint for sigmoid in normalized space

    // Optional vibrance (Core Image filter)
    var vibrance: Double = 0.0
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

    // renormalize
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

// ---- CLI ----
var argv = Array(CommandLine.arguments.dropFirst())
var opts = Opts()

func usage() -> Never {
    print("""
    usage:
      preview_autolevels_like.swift [options] input output

    options:
      --dump, -d
      --bins N                 (default 256)
      --clip-low P             (default 0.0)
      --clip-high P            (default 1.0)

      --peak-factor F          Peak-limit histogram bins to F*(1/bins). 0 disables.
                              Try 4..12 for CG images with huge flat regions.

      --gamma G                (default 1.0)
      --sigmoid SxM            Apply S-curve after gamma on normalized luma.
                              Example: --sigmoid 6x0.5  (strength=6, midpoint=0.5)
                              Use S=0 to disable.

      --vibrance A             Apply CIVibrance with inputAmount=A (default 0.0)
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
    case "--sigmoid":
        guard let v = argv.first else { usage() }
        argv.removeFirst()
        let parts = v.split(separator: "x", omittingEmptySubsequences: false)
        guard parts.count == 2,
              let s = Double(parts[0]),
              let m = Double(parts[1]) else {
            die("bad --sigmoid format. Use SxM, e.g. 6x0.5")
        }
        opts.sigmoidStrength = s
        opts.sigmoidMid = m
    case "--vibrance":
        guard let v = argv.first else { usage() }
        argv.removeFirst()
        opts.vibrance = parseDouble(v, "vibrance")
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

// ---- Build luma image using CIColorMatrix (no kernels needed here) ----
// IMPORTANT: all three vectors must be (wR, wG, wB, 0) to compute luma correctly.
guard let cm = CIFilter(name: "CIColorMatrix") else { die("CIColorMatrix not available") }
cm.setValue(oriented, forKey: kCIInputImageKey)

let wR: CGFloat = 0.2126, wG: CGFloat = 0.7152, wB: CGFloat = 0.0722
let lvec = CIVector(x: wR, y: wG, z: wB, w: 0)

cm.setValue(lvec, forKey: "inputRVector")
cm.setValue(lvec, forKey: "inputGVector")
cm.setValue(lvec, forKey: "inputBVector")
cm.setValue(CIVector(x: 0, y: 0, z: 0, w: 1), forKey: "inputAVector")
cm.setValue(CIVector(x: 0, y: 0, z: 0, w: 0), forKey: "inputBiasVector")

guard let lumaImage = cm.outputImage else { die("failed to build luma image") }

// ---- Histogram (CIAreaHistogram) ----
guard let h = CIFilter(name: "CIAreaHistogram") else { die("CIAreaHistogram not available") }
h.setValue(lumaImage, forKey: kCIInputImageKey)
h.setValue(CIVector(cgRect: extent), forKey: kCIInputExtentKey)
h.setValue(opts.bins, forKey: "inputCount")
h.setValue(1.0, forKey: "inputScale")
guard let histImage = h.outputImage else { die("failed to compute histogram") }

let he = histImage.extent.integral
var buf = [Float](repeating: 0, count: opts.bins * 4)
buf.withUnsafeMutableBytes { raw in
    ctx.render(histImage,
               toBitmap: raw.baseAddress!,
               rowBytes: opts.bins * 4 * MemoryLayout<Float>.size,
               bounds: he,
               format: .RGBAf,
               colorSpace: nil)
}

// pdf from red channel
var pdf = [Double](repeating: 0, count: opts.bins)
var total = 0.0
for i in 0..<opts.bins {
    let v = Double(buf[i * 4 + 0])
    pdf[i] = max(0.0, v)
    total += pdf[i]
}
if total <= 0 { die("histogram total was zero") }
for i in 0..<opts.bins { pdf[i] /= total }

// Peak limiting  ✅ FIXED CALL (no pdfIn: label)
pdf = peakLimitPDF(pdf, factor: opts.peakFactor)

let cdf = cdfFromPDF(pdf)

let lowTarget  = max(0.0, min(1.0, opts.clipLow / 100.0))
let highTarget = max(0.0, min(1.0, 1.0 - (opts.clipHigh / 100.0)))

let blackBin = findBin(cdf, target: lowTarget)
let whiteBin = findBin(cdf, target: highTarget)

let black = Double(blackBin) / Double(opts.bins - 1)
let white = max(Double(whiteBin) / Double(opts.bins - 1), black + 1e-6)

if opts.dump {
    print("bins=\(opts.bins) clipLow=\(opts.clipLow)% clipHigh=\(opts.clipHigh)% peakFactor=\(opts.peakFactor)")
    print(String(format: "blackBin=%d black=%.6f  whiteBin=%d white=%.6f", blackBin, black, whiteBin, white))
    print(String(format: "gamma=%.6f  sigmoid=%.3fx%.3f  vibrance=%.3f",
                 opts.gamma, opts.sigmoidStrength, opts.sigmoidMid, opts.vibrance))
}

// ---- Apply mapping (kernel, deprecated but works) ----
guard let kernel = CIColorKernel(source: """
kernel vec4 autoLevelsLike(__sample s,
                           float black, float white,
                           float gamma,
                           float sigStrength, float sigMid) {
  float y = dot(s.rgb, vec3(0.2126, 0.7152, 0.0722));
  float denom = max(white - black, 1e-6);
  float t = clamp((y - black) / denom, 0.0, 1.0);

  // gamma on normalized luma
  t = pow(t, gamma);

  // optional sigmoid (S-curve) on normalized luma, with endpoint normalization
  if (sigStrength > 0.0) {
    float s0 = 1.0 / (1.0 + exp(sigStrength * (sigMid - 0.0)));
    float s1 = 1.0 / (1.0 + exp(sigStrength * (sigMid - 1.0)));
    float ss = 1.0 / (1.0 + exp(sigStrength * (sigMid - t)));
    t = clamp((ss - s0) / max(s1 - s0, 1e-6), 0.0, 1.0);
  }

  float scale = (y > 1e-6) ? (t / y) : 0.0;
  vec3 rgb = clamp(s.rgb * scale, 0.0, 1.0);
  return vec4(rgb, s.a);
}
""") else {
    die("failed to compile kernel")
}

guard var ciOut = kernel.apply(extent: extent, arguments: [
    oriented,
    Float(black), Float(white),
    Float(opts.gamma),
    Float(opts.sigmoidStrength), Float(opts.sigmoidMid)
]) else {
    die("failed to apply levels kernel")
}

// Optional vibrance
if abs(opts.vibrance) > 1e-9 {
    guard let vib = CIFilter(name: "CIVibrance") else { die("CIVibrance not available") }
    vib.setValue(ciOut, forKey: kCIInputImageKey)
    vib.setValue(opts.vibrance, forKey: "inputAmount")
    if let o = vib.outputImage { ciOut = o }
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

