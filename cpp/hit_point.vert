
#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in float inDist;

layout(push_constant) uniform PushConsts {
    mat4 mvp;
    float minDist;
    float maxDist;
} pc;

layout(location = 0) out float vDistNorm;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    gl_PointSize = 5.0;
    if (inDist < 0.0)
        vDistNorm = -1.0; // miss sentinel — passed to frag as-is
    else
        vDistNorm = clamp((inDist - pc.minDist) / (pc.maxDist - pc.minDist), 0.0, 1.0);
}
