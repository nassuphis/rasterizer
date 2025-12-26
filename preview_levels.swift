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

enum Mode: String {
    case luma
    case rgb
    case perchannel
}

enum AutoGamma: String {
    case none
    case mean
    case median
}

struct Opts {
    var dump = false
    var bins = 256
    var mode: Mode = .luma

    // Winsorization (% of pixels clipped)
    var clipLowPercent: Double = 0.0
    var clipHighPercent: Double = 1.0

    // Ignore N bins at each extreme before percentile selection
    var ignoreLowBins: Int = 0
    var ignoreHighBins: Int = 0

    // Gamma handling
    var gamma: Double = 1.0
    var autoGamma: AutoGamma = .none
    var target: Double = 0.5
}

func parseDouble(_ s: String, _ name: String) -> Double {
    guard let v = Double(s) else { die("bad \(name): \(s)") }
    return v
}
func parseInt(_ s: String, _ name: String) -> Int {
    guard let v = Int(s) else { die("bad \(name): \(s)") }
    return v
}

func percentileBlackWhite(pdfIn: [Double],
                          clipLowPercent: Double,
                          clipHighPercent: Double,
                          ignoreLowBins: Int,
                          ignoreHighBins: Int) -> (blackBin: Int, whiteBin: Int, black: Double, white: Double, cdf: [Double], pdf: [Double]) {
    let bins = pdfIn.count
    var pdf = pdfIn

    // Zero out ignored extremes
    if ignoreLowBins > 0 {
        for i in 0..<min(ignoreLowBins, bins) { pdf[i] = 0.0 }
    }
    if ignoreHighBins > 0 {
        for i in 0..<min(ignoreHighBins, bins) { pdf[bins - 1 - i] = 0.0 }
    }

    // Normalize
    var total = pdf.reduce(0.0, +)
    if total <= 0 {
        // Fall back: use original pdf if we nuked everything
        pdf = pdfIn
        total = pdf.reduce(0.0, +)
        if total <= 0 { die("histogram total was zero") }
    }
    for i in 0..<bins { pdf[i] /= total }

    // CDF
    var cdf = [Double](repeating: 0.0, count: bins)
    var acc = 0.0
    for i in 0..<bins {
        acc += pdf[i]
        cdf[i] = acc
    }

    let lowTarget = max(0.0, min(1.0, clipLowPercent / 100.0))
    let highTarget = max(0.0, min(1.0, 1.0 - (clipHighPercent / 100.0)))

    func findBin(_ target: Double) -> Int {
        for i in 0..<bins {
            if cdf[i] >= target { return i }
        }
        return bins - 1
    }

    let blackBin = findBin(lowTarget)
    let whiteBin = findBin(highTarget)

    let black = Double(blackBin) / Double(bins - 1)
    let white = Double(whiteBin) / Double(bins - 1)

    return (blackBin, whiteBin, black, white, cdf, pdf)
}

func computeAutoGamma(stat: AutoGamma, target: Double, pdf: [Double], black: Double, white: Double) -> Double {
    guard stat != .none else { return 1.0 }
    let bins = pdf.count
    let denom = max(white - black, 1e-9)

    // CDF for median
    var cdf = [Double](repeating: 0.0, count: bins)
    var acc = 0.0
    for i in 0..<bins {
        acc += pdf[i]
        cdf[i] = acc
    }

    func u(atBin i: Int) -> Double {
        let x = Double(i) / Double(bins - 1)
        return max(0.0, min(1.0, (x - black) / denom))
    }

    let s: Double
    switch stat {
    case .mean:
        var m = 0.0
        for i in 0..<bins { m += u(atBin: i) * pdf[i] }
        s = m
    case .median:
        var medBin = 0
        for i in 0..<bins {
            if cdf[i] >= 0.5 { medBin = i; break }
        }
        s = u(atBin: medBin)
    case .none:
        s = 0.5
    }

    // gamma = log(target) / log(s)
    // Guard against s in {0,1}
    let eps = 1e-6
    let ss = max(eps, min(1.0 - eps, s))
    let tt = max(eps, min(1.0 - eps, target))

    var g = log(tt) / log(ss)

    // Clamp to sane range (tweak if you want)
    if !g.isFinite { g = 1.0 }
    g = max(0.5, min(2.0, g))
    return g
}

// ---------------- CLI ----------------

var argv = Array(CommandLine.arguments.dropFirst())
var opts = Opts()

while let a = argv.first, a.hasPrefix("-") {
    argv.removeFirst()
    switch a {
    case "--dump", "-d":
        opts.dump = true

    case "--mode":
        guard let v = argv.first else { die("missing value for --mode (luma|rgb|perchannel)") }
        argv.removeFirst()
        guard let m = Mode(rawValue: v) else { die("bad --mode \(v) (use luma|rgb|perchannel)") }
        opts.mode = m

    case "--bins":
        guard let v = argv.first else { die("missing value for --bins") }
        argv.removeFirst()
        opts.bins = parseInt(v, "bins")

    case "--clip-low":
        guard let v = argv.first else { die("missing value for --clip-low") }
        argv.removeFirst()
        opts.clipLowPercent = parseDouble(v, "clip-low")

    case "--clip-high":
        guard let v = argv.first else { die("missing value for --clip-high") }
        argv.removeFirst()
        opts.clipHighPercent = parseDouble(v, "clip-high")

    case "--ignore-extremes":
        guard let v = argv.first else { die("missing value for --ignore-extremes") }
        argv.removeFirst()
        let n = parseInt(v, "ignore-extremes")
        opts.ignoreLowBins = n
        opts.ignoreHighBins = n

    case "--ignore-low":
        guard let v = argv.first else { die("missing value for --ignore-low") }
        argv.removeFirst()
        opts.ignoreLowBins = parseInt(v, "ignore-low")

    case "--ignore-high":
        guard let v = argv.first else { die("missing value for --ignore-high") }
        argv.removeFirst()
        opts.ignoreHighBins = parseInt(v, "ignore-high")

    case "--gamma":
        guard let v = argv.first else { die("missing value for --gamma") }
        argv.removeFirst()
        opts.gamma = parseDouble(v, "gamma")
        opts.autoGamma = .none

    case "--auto-gamma":
        guard let v = argv.first else { die("missing value for --auto-gamma (mean|median)") }
        argv.removeFirst()
        guard let ag = AutoGamma(rawValue: v) else { die("bad --auto-gamma \(v) (use mean|median)") }
        opts.autoGamma = ag

    case "--target":
        guard let v = argv.first else { die("missing value for --target") }
        argv.removeFirst()
        opts.target = parseDouble(v, "target")

    case "--help", "-h":
        print("""
        usage:
          preview_levels_plus.swift [options] input output

        modes:
          --mode luma        Hue-preserving levels on luminance (your current approach)
          --mode rgb         Composite RGB: pick black/white from pooled RGB values, apply to all channels
          --mode perchannel  Per-channel levels (R,G,B separately)

        key options:
          --clip-low P       percent clipped in shadows (default 0.0)
          --clip-high P      percent clipped in highlights (default 1.0)
          --ignore-extremes N  ignore N bins at both ends before picking percentiles (helps with huge 0/255 spikes)
          --auto-gamma mean|median  compute gamma to hit --target after the linear stretch
          --gamma G          fixed gamma (disables auto-gamma)
          --dump             print chosen parameters

        examples:
          ./preview_levels_plus.swift --mode rgb --ignore-extremes 1 --clip-low 0 --clip-high 1 --auto-gamma median --target 0.5 in.jpg out.jpg
          ./preview_levels_plus.swift --mode perchannel --clip-low 0 --clip-high 0.5 in.jpg out.jpg
        """)
        exit(0)

    default:
        die("unknown option: \(a) (run with --help)")
    }
}

guard argv.count == 2 else {
    die("usage: preview_levels_plus.swift [options] input output (run with --help)")
}

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

func areaHistogram(of image: CIImage, bins: Int) -> [Float] {
    guard let f = CIFilter(name: "CIAreaHistogram") else { die("CIAreaHistogram not available") }
    f.setValue(image, forKey: kCIInputImageKey)
    f.setValue(CIVector(cgRect: extent), forKey: kCIInputExtentKey)
    f.setValue(bins, forKey: "inputCount")
    f.setValue(1.0, forKey: "inputScale")

    guard let histImage = f.outputImage else { die("failed to compute histogram") }

    let he = histImage.extent.integral
    guard Int(he.width) == bins, Int(he.height) == 1 else {
        die("unexpected histogram extent: \(he)")
    }

    var buf = [Float](repeating: 0, count: bins * 4)
    buf.withUnsafeMutableBytes { raw in
        ctx.render(
            histImage,
            toBitmap: raw.baseAddress!,
            rowBytes: bins * 4 * MemoryLayout<Float>.size,
            bounds: he,
            format: .RGBAf,
            colorSpace: nil
        )
    }
    return buf
}

// Build PDFs depending on mode
var gammaToUse = opts.gamma

var black: Double = 0
var white: Double = 1

var blackRGB = (r: 0.0, g: 0.0, b: 0.0)
var whiteRGB = (r: 1.0, g: 1.0, b: 1.0)

// For auto-gamma we need the pdf we used for endpoint selection (normalized after ignore-extremes)
var pdfForGamma: [Double] = []

switch opts.mode {

case .luma:
    // Luminance image via CIColorMatrix (no kernel needed here)
    guard let cm = CIFilter(name: "CIColorMatrix") else { die("CIColorMatrix not available") }
    cm.setValue(oriented, forKey: kCIInputImageKey)
    // Output RGB = y = wR*R + wG*G + wB*B
    let wR: CGFloat = 0.2126, wG: CGFloat = 0.7152, wB: CGFloat = 0.0722
    cm.setValue(CIVector(x: wR, y: wR, z: wR, w: 0), forKey: "inputRVector")
    cm.setValue(CIVector(x: wG, y: wG, z: wG, w: 0), forKey: "inputGVector")
    cm.setValue(CIVector(x: wB, y: wB, z: wB, w: 0), forKey: "inputBVector")
    cm.setValue(CIVector(x: 0, y: 0, z: 0, w: 1), forKey: "inputAVector")
    cm.setValue(CIVector(x: 0, y: 0, z: 0, w: 0), forKey: "inputBiasVector")

    guard let lumaImage = cm.outputImage else { die("failed to build luma image") }

    let hist = areaHistogram(of: lumaImage, bins: opts.bins)

    // pdf from red channel
    var pdf = [Double](repeating: 0, count: opts.bins)
    var total = 0.0
    for i in 0..<opts.bins {
        let v = Double(hist[i * 4 + 0])
        pdf[i] = max(0.0, v)
        total += pdf[i]
    }
    if total <= 0 { die("luma histogram total was zero") }
    for i in 0..<opts.bins { pdf[i] /= total }

    let res = percentileBlackWhite(pdfIn: pdf,
                                   clipLowPercent: opts.clipLowPercent,
                                   clipHighPercent: opts.clipHighPercent,
                                   ignoreLowBins: opts.ignoreLowBins,
                                   ignoreHighBins: opts.ignoreHighBins)

    black = res.black
    white = max(res.white, black + 1e-6)
    pdfForGamma = res.pdf

    if opts.autoGamma != .none {
        gammaToUse = computeAutoGamma(stat: opts.autoGamma, target: opts.target, pdf: pdfForGamma, black: black, white: white)
    }

    if opts.dump {
        print("mode=luma bins=\(opts.bins)")
        print(String(format: "clipLow=%.4f%% clipHigh=%.4f%% ignoreLowBins=%d ignoreHighBins=%d",
                     opts.clipLowPercent, opts.clipHighPercent, opts.ignoreLowBins, opts.ignoreHighBins))
        print(String(format: "black=%.6f white=%.6f gamma=%.6f", black, white, gammaToUse))
    }

case .rgb, .perchannel:
    let hist = areaHistogram(of: oriented, bins: opts.bins)

    // Build per-channel pdfs
    func channelPDF(_ ch: Int) -> [Double] {
        var pdf = [Double](repeating: 0, count: opts.bins)
        var total = 0.0
        for i in 0..<opts.bins {
            let v = Double(hist[i * 4 + ch])
            pdf[i] = max(0.0, v)
            total += pdf[i]
        }
        if total <= 0 { die("histogram total was zero") }
        for i in 0..<opts.bins { pdf[i] /= total }
        return pdf
    }

    let rPdf = channelPDF(0)
    let gPdf = channelPDF(1)
    let bPdf = channelPDF(2)

    // Combined distribution = pooled RGB values
    var comb = [Double](repeating: 0, count: opts.bins)
    for i in 0..<opts.bins {
        comb[i] = (rPdf[i] + gPdf[i] + bPdf[i]) / 3.0
    }

    // Use combined for gamma (single midtone)
    let combRes = percentileBlackWhite(pdfIn: comb,
                                       clipLowPercent: opts.clipLowPercent,
                                       clipHighPercent: opts.clipHighPercent,
                                       ignoreLowBins: opts.ignoreLowBins,
                                       ignoreHighBins: opts.ignoreHighBins)
    pdfForGamma = combRes.pdf
    if opts.autoGamma != .none {
        gammaToUse = computeAutoGamma(stat: opts.autoGamma, target: opts.target, pdf: pdfForGamma, black: combRes.black, white: max(combRes.white, combRes.black + 1e-6))
    }

    if opts.mode == .rgb {
        black = combRes.black
        white = max(combRes.white, black + 1e-6)

        if opts.dump {
            print("mode=rgb(composite) bins=\(opts.bins)")
            print(String(format: "clipLow=%.4f%% clipHigh=%.4f%% ignoreLowBins=%d ignoreHighBins=%d",
                         opts.clipLowPercent, opts.clipHighPercent, opts.ignoreLowBins, opts.ignoreHighBins))
            print(String(format: "black=%.6f white=%.6f gamma=%.6f", black, white, gammaToUse))
        }

    } else {
        // per-channel endpoints
        let rRes = percentileBlackWhite(pdfIn: rPdf,
                                        clipLowPercent: opts.clipLowPercent,
                                        clipHighPercent: opts.clipHighPercent,
                                        ignoreLowBins: opts.ignoreLowBins,
                                        ignoreHighBins: opts.ignoreHighBins)
        let gRes = percentileBlackWhite(pdfIn: gPdf,
                                        clipLowPercent: opts.clipLowPercent,
                                        clipHighPercent: opts.clipHighPercent,
                                        ignoreLowBins: opts.ignoreLowBins,
                                        ignoreHighBins: opts.ignoreHighBins)
        let bRes = percentileBlackWhite(pdfIn: bPdf,
                                        clipLowPercent: opts.clipLowPercent,
                                        clipHighPercent: opts.clipHighPercent,
                                        ignoreLowBins: opts.ignoreLowBins,
                                        ignoreHighBins: opts.ignoreHighBins)

        blackRGB = (rRes.black, gRes.black, bRes.black)
        whiteRGB = (max(rRes.white, rRes.black + 1e-6),
                    max(gRes.white, gRes.black + 1e-6),
                    max(bRes.white, bRes.black + 1e-6))

        if opts.dump {
            print("mode=perchannel bins=\(opts.bins)")
            print(String(format: "clipLow=%.4f%% clipHigh=%.4f%% ignoreLowBins=%d ignoreHighBins=%d",
                         opts.clipLowPercent, opts.clipHighPercent, opts.ignoreLowBins, opts.ignoreHighBins))
            print(String(format: "blackR=%.6f whiteR=%.6f", blackRGB.r, whiteRGB.r))
            print(String(format: "blackG=%.6f whiteG=%.6f", blackRGB.g, whiteRGB.g))
            print(String(format: "blackB=%.6f whiteB=%.6f", blackRGB.b, whiteRGB.b))
            print(String(format: "gamma=%.6f (autoGamma=%@ target=%.3f)", gammaToUse, String(describing: opts.autoGamma), opts.target))
        }
    }
}

// ---------------- Apply mapping ----------------

let kernelSource: String
switch opts.mode {
case .luma:
    kernelSource = """
    kernel vec4 applyLevelsLuma(__sample s, float black, float white, float gamma) {
      float y = dot(s.rgb, vec3(0.2126, 0.7152, 0.0722));
      float denom = max(white - black, 1e-6);
      float t = clamp((y - black) / denom, 0.0, 1.0);
      t = pow(t, gamma);
      float scale = (y > 1e-6) ? (t / y) : 0.0;
      vec3 rgb = clamp(s.rgb * scale, 0.0, 1.0);
      return vec4(rgb, s.a);
    }
    """
case .rgb:
    kernelSource = """
    kernel vec4 applyLevelsRGB(__sample s, float black, float white, float gamma) {
      float denom = max(white - black, 1e-6);
      vec3 v = (s.rgb - vec3(black)) / denom;
      v = clamp(v, 0.0, 1.0);
      v = pow(v, vec3(gamma));
      return vec4(v, s.a);
    }
    """
case .perchannel:
    kernelSource = """
    kernel vec4 applyLevelsPerChannel(__sample s,
                                      float br, float wr,
                                      float bg, float wg,
                                      float bb, float wb,
                                      float gamma) {
      float dr = max(wr - br, 1e-6);
      float dg = max(wg - bg, 1e-6);
      float db = max(wb - bb, 1e-6);

      float r = clamp((s.r - br) / dr, 0.0, 1.0);
      float g = clamp((s.g - bg) / dg, 0.0, 1.0);
      float b = clamp((s.b - bb) / db, 0.0, 1.0);

      vec3 v = pow(vec3(r, g, b), vec3(gamma));
      return vec4(clamp(v, 0.0, 1.0), s.a);
    }
    """
}

guard let kernel = CIColorKernel(source: kernelSource) else {
    die("failed to compile kernel (note: CI kernel language is deprecated but still works)")
}

let ciOut: CIImage
switch opts.mode {
case .luma:
    guard let out = kernel.apply(extent: extent, arguments: [oriented, Float(black), Float(white), Float(gammaToUse)]) else {
        die("failed to apply luma levels")
    }
    ciOut = out
case .rgb:
    guard let out = kernel.apply(extent: extent, arguments: [oriented, Float(black), Float(white), Float(gammaToUse)]) else {
        die("failed to apply rgb levels")
    }
    ciOut = out
case .perchannel:
    guard let out = kernel.apply(extent: extent, arguments: [
        oriented,
        Float(blackRGB.r), Float(whiteRGB.r),
        Float(blackRGB.g), Float(whiteRGB.g),
        Float(blackRGB.b), Float(whiteRGB.b),
        Float(gammaToUse)
    ]) else {
        die("failed to apply per-channel levels")
    }
    ciOut = out
}

// Render + write
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



