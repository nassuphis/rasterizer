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

func usageAndExit(_ code: Int32 = 1) -> Never {
    let u = """
    usage:
      autolevels.swift [--dump|-d] [--tone-only] [--curve-only] input output

    what it does:
      Uses Core Image autoAdjustmentFilters() (Apple's auto-enhance suggestions),
      then optionally restricts to tonal-only filters to better match Preview "Auto Levels".

    flags:
      --dump, -d       Print which filters were chosen + their configured parameters
      --tone-only      Keep only: CIToneCurve, CIHighlightShadowAdjust
      --curve-only     Keep only: CIToneCurve (strongest "levels only" approximation)

    examples:
      ./autolevels.swift in.jpg out.jpg
      ./autolevels.swift --dump in.jpg out.jpg
      ./autolevels.swift --curve-only in.jpg out.jpg
      ./autolevels.swift --tone-only in.jpg out.jpg
    """
    print(u)
    exit(code)
}

var argv = Array(CommandLine.arguments.dropFirst())

var dump = false
var toneOnly = false
var curveOnly = false

while let a = argv.first, a.hasPrefix("-") {
    switch a {
    case "--dump", "-d":
        dump = true
    case "--tone-only":
        toneOnly = true
    case "--curve-only":
        curveOnly = true
    case "--help", "-h":
        usageAndExit(0)
    default:
        die("unknown option \(a)\n(run with --help for usage)")
    }
    argv.removeFirst()
}

guard argv.count == 2 else {
    usageAndExit(1)
}

let inURL  = URL(fileURLWithPath: argv[0])
let outURL = URL(fileURLWithPath: argv[1])

// Load input
guard let ciIn = CIImage(contentsOf: inURL) else {
    die("cannot read input: \(inURL.path)")
}

// Read EXIF orientation as Int32 (what CIImage.oriented(forExifOrientation:) expects)
let exifOrientation: Int32 = {
    guard let src = CGImageSourceCreateWithURL(inURL as CFURL, nil),
          let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
    else { return 1 }

    // Orientation is typically an NSNumber in the metadata dictionary
    if let n = props[kCGImagePropertyOrientation] as? NSNumber {
        return n.int32Value
    }
    if let i = props[kCGImagePropertyOrientation] as? Int {
        return Int32(i)
    }
    return 1
}()

// Apply orientation to pixels (so output matches what Preview shows)
let oriented = ciIn.oriented(forExifOrientation: exifOrientation)

// Ask Core Image for its recommended auto-adjustment filter chain.
//
// If you ever see unexpected rotation/cropping, you can disable those behaviors by using options,
// but keeping it simple first tends to be easiest for matching Preview-like behavior.
var filters = oriented.autoAdjustmentFilters()

// Optional: restrict to tonal-only filters (closer to Preview "Auto Levels")
if curveOnly {
    filters = filters.filter { $0.name == "CIToneCurve" }
} else if toneOnly {
    let keep = Set(["CIToneCurve", "CIHighlightShadowAdjust"])
    filters = filters.filter { keep.contains($0.name) }
}

// Dump chosen filters + their configured params
if dump {
    if filters.isEmpty {
        print("(no auto-adjustment filters returned)")
    } else {
        for f in filters {
            print(f.name)
            for key in f.inputKeys where key != kCIInputImageKey {
                if let v = f.value(forKey: key) {
                    print("  \(key): \(v)")
                }
            }
        }
    }
}

// Apply filter chain
var ciOut = oriented
for f in filters {
    f.setValue(ciOut, forKey: kCIInputImageKey)
    if let out = f.outputImage {
        ciOut = out
    }
}

// Render + write output
let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!
let ctx = CIContext(options: [
    CIContextOption.workingColorSpace: sRGB,
    CIContextOption.outputColorSpace:  sRGB
])

guard let cg = ctx.createCGImage(ciOut, from: ciOut.extent) else {
    die("failed to render output")
}

let outType = UTType(filenameExtension: outURL.pathExtension) ?? .png

guard let dest = CGImageDestinationCreateWithURL(
    outURL as CFURL,
    outType.identifier as CFString,
    1,
    nil
) else {
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

