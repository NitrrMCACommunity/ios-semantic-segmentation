//
//  ViewController.swift
//  Tiramisu
//
//  Created by James Kauten on 10/15/18.
//  Copyright © 2018 Kautenja. All rights reserved.
//

import UIKit
import AVFoundation
import Vision
import Metal
import MetalPerformanceShaders

/// A view controller to pass camera inputs through a vision model
class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    /// a local reference to time to update the framerate
    var time = Date()
    
    var ready: Bool = true

    /// the view to preview raw RGB data from the camera
    @IBOutlet weak var preview: UIView!
    /// the view for showing the segmentation
    @IBOutlet weak var segmentation: UIImageView!
    /// a label to show the framerate of the model
    @IBOutlet weak var framerate: UILabel!
    
    /// the camera session for streaming data from the camera
    var captureSession: AVCaptureSession!
    /// the video preview layer
    var videoPreviewLayer: AVCaptureVideoPreviewLayer!
    
    /// TODO:
    private var _device: MTLDevice?
    /// TODO:
    var device: MTLDevice! {
        get {
            // try to unwrap the private device instance
            if let device = _device {
                return device
            }
            _device = MTLCreateSystemDefaultDevice()
            return _device
        }
    }
    
    var _queue: MTLCommandQueue?
    
    var queue: MTLCommandQueue! {
        get {
            // try to unwrap the private queue instance
            if let queue = _queue {
                return queue
            }
            _queue = device.makeCommandQueue()
            return _queue
        }
    }

    /// the model for the view controller to apss camera data through
    private var _model: VNCoreMLModel?
    /// the model for the view controller to apss camera data through
    var model: VNCoreMLModel! {
        get {
            // try to unwrap the private model instance
            if let model = _model {
                return model
            }
            // try to create a new model and fail gracefully
            do {
                _model = try VNCoreMLModel(for: Tiramisu45().model)
            } catch let error {
                let message = "failed to load model: \(error.localizedDescription)"
                popup_alert(self, title: "Model Error", message: message)
            }
            return _model
        }
    }
    
    /// the request and handler for the model
    private var _request: VNCoreMLRequest?
    /// the request and handler for the model
    var request: VNCoreMLRequest! {
        get {
            // try to unwrap the private request instance
            if let request = _request {
                return request
            }
            // create the request
            _request = VNCoreMLRequest(model: model) { (finishedRequest, error) in
                // handle an error from the inference engine
                if let error = error {
                    print("inference error: \(error.localizedDescription)")
                    return
                }
                // make sure the UI is ready for another frame
                guard self.ready else { return }
                // get the outputs from the model
                let outputs = finishedRequest.results as? [VNCoreMLFeatureValueObservation]
                
                // get the probabilities as the first output of the model
                guard let softmax = outputs?[0].featureValue.multiArrayValue else {
                    print("failed to extract output from model")
                    return
                }
                print("Out has arrivied")
               /* let height = softmax.shape[1].intValue
                let width = softmax.shape[2].intValue
                let rawPtr = OpaquePointer(softmax.dataPointer)
                let features  = UnsafeMutablePointer<Float32>(rawPtr)
                for i in 0...width{
                    for j in 0...height{
                        var val = features[i*width + j];
                        
                        //softmax.dataPointer[i*width + j]
                    }
                }
                print("Final output: finished")*/
                
                
                
                
                
                
                // get the dimensions of the probability tensor
                let channels = softmax.shape[0].intValue
                let height = softmax.shape[1].intValue
                let width = softmax.shape[2].intValue
                
                // create an image for the softmax outputs
                let desc = MPSImageDescriptor(channelFormat: .float32,
                                              width: width,
                                              height: height,
                                              featureChannels: channels)
                let probs = MPSImage(device: self.device, imageDescriptor: desc)
                probs.writeBytes(softmax.dataPointer,
                                 dataLayout: .featureChannelsxHeightxWidth,
                                 imageIndex: 0)
                
                let arrayMl = try! MLMultiArray(shape: [12,softmax.shape[1],softmax.shape[2]], dataType: .float32)
                probs.readBytes(arrayMl.dataPointer, dataLayout: .featureChannelsxHeightxWidth, imageIndex: 0)
                let images = self.processSoftmax(arrayMl)
                // update the image on the UI thread
                DispatchQueue.main.async {
                    self.segmentation.image = images
                    let fps = -1 / self.time.timeIntervalSinceNow
                    self.time = Date()
                    self.framerate.text = "\(fps)"
                }
            self.ready = true
            //self.ready = false
                
                // create an output image for the Arg Max output
               /* let desc1 = MPSImageDescriptor(channelFormat: .float32,
                                               width: width,
                                               height: height,
                                               featureChannels: 1)
                let classes = MPSImage(device: self.device, imageDescriptor: desc1)

                // create a buffer and pass the inputs through the filter to the outputs
                let buffer = self.queue.makeCommandBuffer()
                let filter = MPSNNReduceFeatureChannelsArgumentMax(device: self.device)
                filter.encode(commandBuffer: buffer!, sourceImage: probs, destinationImage: classes)
                
                // add a callback to handle the buffer's completion and commit the buffer
                buffer?.addCompletedHandler({ (_buffer) in
                    let argmax = try! MLMultiArray(shape: [1, softmax.shape[1], softmax.shape[2]], dataType: .float32)
                    classes.readBytes(argmax.dataPointer,
                                      dataLayout: .featureChannelsxHeightxWidth,
                                      imageIndex: 0)
    
                    // unmap the discrete segmentation to RGB pixels
                    let image = codesToImage(argmax)
                    // update the image on the UI thread
                    DispatchQueue.main.async {
                        self.segmentation.image = image
                        let fps = -1 / self.time.timeIntervalSinceNow
                        self.time = Date()
                        self.framerate.text = "\(fps)"
                    }
                    self.ready = true
                })
                self.ready = false
                buffer?.commit()*/

            }
            // set the input image size to be a scaled version
            // of the image
            _request?.imageCropAndScaleOption = .scaleFill
            return _request
        }
    }
    
    /// Respond to a memory warning from the OS
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        popup_alert(self, title: "Memory Warning", message: "received memory warning")
    }
    
    /// Handle the view appearing
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        // setup the AV session
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .hd1280x720
        // get a handle on the back camera
        guard let camera = AVCaptureDevice.default(for: AVMediaType.video) else {
            let message = "Unable to access the back camera!"
            popup_alert(self, title: "Camera Error", message: message)
            return
        }
        // create an input device from the back camera and handle
        // any errors (i.e., privacy request denied)
        do {
            // setup the camera input and video output
            let input = try AVCaptureDeviceInput(device: camera)
            let videoOutput = AVCaptureVideoDataOutput()
            videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            // add the inputs and ouptuts to the sessionr and start the preview
            if captureSession.canAddInput(input) && captureSession.canAddOutput(videoOutput) {
                captureSession.addInput(input)
                captureSession.addOutput(videoOutput)
                setupCameraPreview()
            }
        }
        catch let error  {
            let message = "failed to intialize camera: \(error.localizedDescription)"
            popup_alert(self, title: "Camera Error", message: message)
            return
        }
    }

    /// Setup the live preview from the camera
    func setupCameraPreview() {
        // create a video preview layer for the view controller
        videoPreviewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        // set the metadata of the video preview
        videoPreviewLayer.videoGravity = .resizeAspect
        videoPreviewLayer.connection?.videoOrientation = .landscapeRight
        // add the preview layer as a sublayer of the preview view
        preview.layer.addSublayer(videoPreviewLayer)
        // start the capture session asyncrhonously
        DispatchQueue.global(qos: .userInitiated).async {
            // start the capture session in the background thread
            self.captureSession.startRunning()
            // set the frame of the video preview to the bounds of the
            // preview view
            DispatchQueue.main.async {
                self.videoPreviewLayer.frame = self.preview.bounds
            }
        }
    }
    
    /// Handle a frame from the camera video stream
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        // create a Core Video pixel buffer which is an image buffer that holds pixels in main memory
        // Applications generating frames, compressing or decompressing video, or using Core Image
        // can all make use of Core Video pixel buffers
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            let message = "failed to create pixel buffer from video input"
            popup_alert(self, title: "Inference Error", message: message)
            return
        }
        // execute the request
        do {
            try VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
        } catch let error {
            let message = "failed to perform inference: \(error.localizedDescription)"
            popup_alert(self, title: "Inference Error", message: message)
        }
    }
    
    
    private var mm = false
    func processSoftmax(_ _probs: MLMultiArray)->UIImage{
        let label_map = [
            0:  [255, 0, 0],
            1:  [70, 70, 70],
            2:  [0, 0, 142],
            3:  [153, 153, 153],
            4:  [190, 153, 153],
            5:  [220, 20, 60],
            6:  [128, 64, 128],
            7:  [244, 35, 232],
            8:  [220, 220, 0],
            9:  [70, 130, 180],
            10: [107, 142, 35],
            11: [0, 0, 0]
        ]
        
        // convert the MLMultiArray to a MultiArray
        var codes = MultiArray<Float32>(_probs)
        // get the shape information from the probs
        let height = codes.shape[1]
        let width = codes.shape[2]
        // initialize some bytes to store the image in
        var bytes = [UInt8](repeating: 255, count: height * width * 4)
        // iterate over the pixels in the output probs
        for h in 0 ..< height {
            for w in 0 ..< width {
                // get the array offset for this word
                let offset = h * width * 4 + w * 4
                var ch1 = codes[0, h, w]
                //ch1 = ch1/255.0
                let ch2 = 1.0-ch1
                
                if(ch1 < ch2){
                    let rgb = label_map[0]
                    bytes[offset + 0] = UInt8(rgb![0])
                    bytes[offset + 1] = UInt8(rgb![1])
                    bytes[offset + 2] = UInt8(rgb![2])
                }
                else{
                    let rgb = label_map[1]
                    bytes[offset + 0] = UInt8(rgb![0])
                    bytes[offset + 1] = UInt8(rgb![1])
                    bytes[offset + 2] = UInt8(rgb![2])
                }
            
            }
        }
        print("Finish Frame: finished")
        // create a UIImage from the byte array
        return UIImage.fromByteArray(bytes, width: width, height: height,
                                     scale: 0, orientation: .up,
                                     bytesPerRow: width * 4,
                                     colorSpace: CGColorSpaceCreateDeviceRGB(),
                                     alphaInfo: .premultipliedLast)!
        
        
        
    }
    
    
    
    
}
