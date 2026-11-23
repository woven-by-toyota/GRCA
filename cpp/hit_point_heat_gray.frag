#version 450
layout(location = 0) in float vDistNorm;
layout(location = 0) out vec4 outColor;

// Grayscale depth pass (heat-map mode, hit_depth_colormap "gray"): linear camera depth,
// near = white (255) -> far = black (0), like a depth map; vDistNorm comes from
// hit_point_camdepth.vert.  Pixel value v maps back to depth = near + (1 - v/255) * (far - near).
void main() {
    if (vDistNorm < 0.0) discard;  // miss
    float g = 1.0 - clamp(vDistNorm, 0.0, 1.0);   // linear in depth (sRGB byte = 255 * g)
    // The swapchain is sRGB: write the linear value whose sRGB encoding is g.
    float lin = (g <= 0.04045) ? g / 12.92 : pow((g + 0.055) / 1.055, 2.4);
    outColor = vec4(vec3(lin), 1.0);
}
