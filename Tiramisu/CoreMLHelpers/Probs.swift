//
//  Probs.swift
//  Tiramisu
//
//  Created by James Kauten on 10/16/18.
//  Copyright © 2018 Kautenja. All rights reserved.
//

import Foundation
import UIKit
import CoreML

/// Convert probability tensor into an image
func probsToImage(_ _probs: MLMultiArray) -> UIImage? {
    // TODO: dynamically load a label map instead of hard coding
    // can this bonus data be included in the model file?
    let label_map = [
        0:  [0, 128, 192],
        1:  [128, 0, 0],
        2:  [64, 0, 128],
        3:  [192, 192, 128],
        4:  [64, 64, 128],
        5:  [64, 64, 0],
        6:  [128, 64, 128],
        7:  [0, 0, 192],
        8:  [192, 128, 128],
        9:  [128, 128, 128],
        10: [192, 192, 0],
        11: [0, 0, 0]
    ]
    // convert the MLMultiArray to a MultiArray
    var probs = MultiArray<Double>(_probs)
    // get the shape information from the probs
    let classes = probs.shape[0]
    let height = probs.shape[1]
    let width = probs.shape[2]
    // initialize some bytes to store the image in
    var bytes = [UInt8](repeating: 0, count: height * width * 4)
    // iterate over the pixels in the output probs
    for h in 0 ..< height {
        for w in 0 ..< width {
            // store the highest probability and the corresponding class
            var max_prob: Double = 0
            var max_c: Int = 0
            // iterate over class labels to extract the highest probability
            for c in 0 ..< classes {
                // replace the highest prob and index if this prob is greater
                if probs[c, h, w] > max_prob {
                    max_prob = probs[c, h, w]
                    max_c = c
                }
            }
            // get the array offset for this word
            let offset = h * width * 4 + w * 4
            // get the RGB value for the highest probability class
            let rgb = label_map[max_c]
            // set the bytes to the RGB value and alpha of 1.0 (255)
            bytes[offset + 0] = UInt8(rgb![0])
            bytes[offset + 1] = UInt8(rgb![1])
            bytes[offset + 2] = UInt8(rgb![2])
            bytes[offset + 3] = 255
        }
    }
    // create a UIImage from the byte array
    return UIImage.fromByteArray(bytes, width: width, height: height,
                                 scale: 0, orientation: .up,
                                 bytesPerRow: width * 4,
                                 colorSpace: CGColorSpaceCreateDeviceRGB(),
                                 alphaInfo: .premultipliedLast)
}