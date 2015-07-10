//
//  ViewController.swift
//  MetalKitReactionDiffusion
//
//  Created by Simon Gladman on 09/07/2015.
//  Copyright © 2015 Simon Gladman. All rights reserved.
//

import UIKit
import AVFoundation
import CoreMedia
import MetalKit
import MetalPerformanceShaders

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate
{
    var device:MTLDevice!
    let metalView = ReactionDiffusionMetalView(frame: CGRectZero)
    
    var videoTextureCache : Unmanaged<CVMetalTextureCacheRef>?
    
    override func viewDidLoad()
    {
        super.viewDidLoad()
   
        let captureSession = AVCaptureSession()
        captureSession.sessionPreset = AVCaptureSessionPresetPhoto
        
        let backCamera = AVCaptureDevice.defaultDeviceWithMediaType(AVMediaTypeVideo)
        
        do
        {
            let input = try AVCaptureDeviceInput(device: backCamera)
            
            captureSession.addInput(input)
        }
        catch
        {
            print("can't access camera")
            return
        }
        
        // although we don't use this, it's required to get captureOutput invoked
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        view.layer.addSublayer(previewLayer)
        
        let videoOutput = AVCaptureVideoDataOutput()
        
        videoOutput.setSampleBufferDelegate(self, queue: dispatch_queue_create("sample buffer delegate", DISPATCH_QUEUE_SERIAL))
        if captureSession.canAddOutput(videoOutput)
        {
            captureSession.addOutput(videoOutput)
        }
        
        setUpMetal()
        
        captureSession.startRunning()
        
        view.addSubview(metalView)
    }
    
    private func setUpMetal()
    {
        guard let device = MTLCreateSystemDefaultDevice() else
        {
            return
        }
        
        self.device = device
        
        metalView.framebufferOnly = false

        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &videoTextureCache)
    }

    func captureOutput(captureOutput: AVCaptureOutput!, didOutputSampleBuffer sampleBuffer: CMSampleBuffer!, fromConnection connection: AVCaptureConnection!)
    {
        let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        
        var yTextureRef : Unmanaged<CVMetalTextureRef>?
        
        let yWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer!, 0);
        let yHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer!, 0);
        
        CVMetalTextureCacheCreateTextureFromImage(kCFAllocatorDefault,
            videoTextureCache!.takeUnretainedValue(),
            pixelBuffer!,
            nil,
            MTLPixelFormat.R8Unorm,
            yWidth, yHeight, 0,
            &yTextureRef)

        metalView.parameterGradientTexture = CVMetalTextureGetTexture((yTextureRef?.takeUnretainedValue())!)
        
        yTextureRef?.release()
    }
    
    override func viewDidLayoutSubviews()
    {
        metalView.frame = CGRect(x: 0, y: 0, width: 1024, height: 768)
    }
}

