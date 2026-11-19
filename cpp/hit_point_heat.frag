#version 450
layout(location = 0) in float vDistNorm;
layout(location = 0) out vec4 outColor;

// RViz "Rainbow" colour transformer (getRainbowColor in rviz point_cloud_transformers):
// 0 = magenta -> blue -> cyan -> green -> yellow -> 1 = red.
vec3 rvizRainbow(float value) {
    value = clamp(value, 0.0, 1.0);
    float h = value * 5.0 + 1.0;
    int   i = int(floor(h));
    float f = h - float(i);
    if ((i & 1) == 0) f = 1.0 - f;  // i even
    float n = 1.0 - f;
    if      (i <= 1) return vec3(n,   0.0, 1.0);
    else if (i == 2) return vec3(0.0, n,   1.0);
    else if (i == 3) return vec3(0.0, 1.0, n);
    else if (i == 4) return vec3(n,   1.0, 0.0);
    else             return vec3(1.0, n,   0.0);
}

// Heat-map mode (mesh_vis_on == 4) only: hit points coloured by depth.
// RViz writes these values straight to the screen; the swapchain is sRGB, so
// convert to linear first to get the same on-screen colours.
vec3 srgbToLinear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
}

void main() {
    if (vDistNorm < 0.0) discard;  // miss
    // vDistNorm: hit distance normalised over this frame's nearest..farthest hit.
    outColor = vec4(srgbToLinear(rvizRainbow(vDistNorm)), 1.0);
}
