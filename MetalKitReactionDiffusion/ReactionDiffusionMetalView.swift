//
//  ReactionDiffusionMetalView.swift
//  MetalKitReactionDiffusion
//
//  Created by Simon Gladman on 09/07/2015.
//  Copyright © 2015 Simon Gladman. All rights reserved.
//

import Metal
import MetalKit
import MetalPerformanceShaders

let Width = 2048
let Height = 1536

class ReactionDiffusionMetalView: MTKView
{
    var pipelineState: MTLComputePipelineState!
    var defaultLibrary: MTLLibrary!
    var commandQueue: MTLCommandQueue!
    var threadsPerThreadgroup:MTLSize!
    var threadgroupsPerGrid: MTLSize!
    
    var resetFlag = true
    
    let bitsPerComponent = Int(8)
    let bytesPerRow = Int(4 * Width)
    let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo.ByteOrder32Big.rawValue | CGImageAlphaInfo.PremultipliedLast.rawValue
    
    var textureA: MTLTexture!
    var textureB: MTLTexture!
    var useTextureAForInput = true
    var blur: MPSImageGaussianBlur!
    
    override init(frame: CGRect)
    {
        super.init(frame: frame)
        
        device = MTLCreateSystemDefaultDevice()
        
        defaultLibrary = device!.newDefaultLibrary()!
        commandQueue = device!.newCommandQueue()
        
        let kernelFunction = defaultLibrary.newFunctionWithName("fitzhughNagumoShader")
        
        do
        {
            pipelineState = try device!.newComputePipelineStateWithFunction(kernelFunction!)
        }
        catch
        {
            fatalError("Unable to create pipeline state")
        }
        
        threadsPerThreadgroup = MTLSizeMake(16, 16, 1)
        threadgroupsPerGrid = MTLSizeMake(Width / threadsPerThreadgroup.width, Height / threadsPerThreadgroup.height, 1)
        
        // set up texture
   
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptorWithPixelFormat(MTLPixelFormat.RGBA8Unorm,
            width: Width, height: Height,
            mipmapped: false)
        
        textureA = device!.newTextureWithDescriptor(textureDescriptor)

        textureB = device!.newTextureWithDescriptor(textureDescriptor)
        
        blur = MPSImageGaussianBlur(device: device!, sigma: 3)
    }
    
    var parameterGradientTexture: MTLTexture?
    
    override func drawRect(dirtyRect: CGRect)
    {
        guard let drawable = currentDrawable, parameterGradientTexture = parameterGradientTexture, device = device else
        {
            print("not ready! ")
            return
        }
        
        if resetFlag
        {
            resetFlag = false
            
            let noise = MDLNoiseTexture(scalarNoiseWithSmoothness: 0.75,
                name: nil,
                textureDimensions: [Int32(Width), Int32(Height)],
                channelCount: 4,
                channelEncoding: MDLTextureChannelEncoding.UInt8,
                grayscale: false)
            
            let noiseData = noise.texelDataWithBottomLeftOrigin()
            
            let region = MTLRegionMake2D(0, 0, Int(Width), Int(Height))
            
            if let noiseData = noiseData
            {
                textureA.replaceRegion(region, mipmapLevel: 0, withBytes: noiseData.bytes, bytesPerRow: Int(bytesPerRow))
                textureB.replaceRegion(region, mipmapLevel: 0, withBytes: noiseData.bytes, bytesPerRow: Int(bytesPerRow))
            }
        }
  
        let commandBuffer = commandQueue.commandBuffer()
        let commandEncoder = commandBuffer.computeCommandEncoder()
        
         commandEncoder.setComputePipelineState(pipelineState)
        
        var reactionDiffusionStruct = ReactionDiffusionParameters()
        
        let buffer: MTLBuffer = device.newBufferWithBytes(&reactionDiffusionStruct, length: sizeof(ReactionDiffusionParameters), options: MTLResourceOptions.CPUCacheModeDefaultCache)
        
        commandEncoder.setBuffer(buffer, offset: 0, atIndex: 0)
        
        for _ in 0 ... 4
        {
            if useTextureAForInput
            {
                commandEncoder.setTexture(textureA, atIndex: 0)
                commandEncoder.setTexture(textureB, atIndex: 1)
            }
            else
            {
                commandEncoder.setTexture(textureB, atIndex: 0)
                commandEncoder.setTexture(textureA, atIndex: 1)
            }
            
            commandEncoder.setTexture(parameterGradientTexture, atIndex: 2)
            
            commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            
            useTextureAForInput = !useTextureAForInput
        }
        
        commandEncoder.endEncoding()
        
        if !useTextureAForInput
        {
            blur.encodeToCommandBuffer(commandBuffer, sourceTexture: textureB, destinationTexture: drawable.texture)
        }
        else
        {
            blur.encodeToCommandBuffer(commandBuffer, sourceTexture: textureA, destinationTexture: drawable.texture)
        }
        
        commandBuffer.presentDrawable(drawable)
        
        commandBuffer.commit();
        
        

    }
    
    required init?(coder aDecoder: NSCoder)
    {
        fatalError("init(coder:) has not been implemented")
    }
}




struct ReactionDiffusionParameters
{
    // Fitzhugh-Nagumo
    
    var timestep: Float = 0.1
    var a0: Float = 0.219900
    var a1: Float = 0.7
    var epsilon: Float = 0.638700
    var delta: Float = 2.54
    var k1: Float = 2.055
    var k2: Float = 2.00920
    var k3: Float =  0.5563
    
}