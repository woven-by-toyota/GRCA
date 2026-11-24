#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in float inDist;

layout(push_constant) uniform PushConsts {
    mat4 mvp;
    float minDist;   // here: nearest camera depth of the colour range
    float maxDist;   // here: farthest camera depth of the colour range
} pc;

layout(location = 0) out float vDistNorm;

// Grayscale depth pass (heat-map mode, hit_depth_colormap "gray"): like hit_point.vert, but the
// normalised value is the point's camera (view-space) depth, gl_Position.w, not its LiDAR range.
void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    gl_PointSize = 5.0;
    if (inDist < 0.0)
        vDistNorm = -1.0; // miss
    else
        vDistNorm = clamp((gl_Position.w - pc.minDist) / (pc.maxDist - pc.minDist), 0.0, 1.0);
}
