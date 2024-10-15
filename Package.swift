// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ElevenLabsSwift",
    platforms: [
        .iOS(.v13)
    ],
    products: [
        .library(
            name: "ElevenLabsSwift",
            targets: ["ElevenLabsSwift"]
        ),
    ],
    targets: [
        .target(
            name: "ElevenLabsSwift"
        )
    ]
)
