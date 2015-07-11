//
//  ReactionDiffusion.metal
//  MetalKitReactionDiffusion
//
//  Created by Simon Gladman on 09/07/2015.
//  Copyright © 2015 Simon Gladman. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

struct ReactionDiffusionParameters
{
    // Fitzhugh-Nagumo
    
    float timestep;
    float a0;
    float a1;
    float epsilon;
    float delta;
    float k1;
    float k2;
    float k3;

};

kernel void fitzhughNagumoShader(texture2d<float, access::read> inTexture [[texture(0)]],
                                 texture2d<float, access::write> outTexture [[texture(1)]],
                                 constant ReactionDiffusionParameters &params [[buffer(0)]],
                                 texture2d<float, access::read> parameterGradientTexture [[texture(2)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    const uint2 northIndex(gid.x, gid.y - 1);
    const uint2 southIndex(gid.x, gid.y + 1);
    const uint2 westIndex(gid.x - 1, gid.y);
    const uint2 eastIndex(gid.x + 1, gid.y);
    
    const float2 northColor = inTexture.read(northIndex).rb;
    const float2 southColor = inTexture.read(southIndex).rb;
    const float2 westColor = inTexture.read(westIndex).rb;
    const float2 eastColor = inTexture.read(eastIndex).rb;
    
    const float2 thisColor = inTexture.read(gid).rb;
    
    const float2 laplacian = (northColor.rg + southColor.rg + westColor.rg + eastColor.rg) - (4.0 * thisColor.rg);
    const float laplacian_a = laplacian.r;
    const float laplacian_b = laplacian.g;
    
    const float a = thisColor.r;
    const float b = thisColor.g;
    
    const float tweakedTimestep = params.timestep + parameterGradientTexture.read(gid).r * 0.05;
    
    const float tweakedA0 = params.a0 + parameterGradientTexture.read(gid).r * 0.06;
    const float tweakedA1 = params.a1 - sqrt(parameterGradientTexture.read(gid).r) * 0.85;
    
    const float tweakedEpsilon = params.epsilon - parameterGradientTexture.read(gid).r * 0.4;
    const float tweakedDelta = params.delta - parameterGradientTexture.read(gid).r * 1.25 ;
    
    const float tweakedk1= params.k1 - parameterGradientTexture.read(gid).r * 0.04;
    const float tweakedk2 = params.k2 - parameterGradientTexture.read(gid).r * 1.75;
    const float tweakedk3 = params.k3 + parameterGradientTexture.read(gid).r * 1.5;
    
    const float delta_a = (tweakedk1 * a) - (tweakedk2 * a * a) - (a * a * a) - b + laplacian_a;
    const float delta_b = tweakedEpsilon * (tweakedk3 * a - tweakedA1 * b - tweakedA0) + tweakedDelta * laplacian_b;
    
    const float4 outColor(0.04 * parameterGradientTexture.read(gid).r + a + (tweakedTimestep * delta_a),
                          (b + (tweakedTimestep * delta_b)) * parameterGradientTexture.read(gid).r * (a + (tweakedTimestep * delta_a)),
                          b + (tweakedTimestep * delta_b),
                          1);
    
    outTexture.write(outColor, gid);
}

