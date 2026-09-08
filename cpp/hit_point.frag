#version 450
layout(location = 0) in float vDistNorm;
layout(location = 0) out vec4 outColor;
void main() {
    if (vDistNorm < 0.0) discard;
    outColor = vec4(1.0, 0.0, 0.0, 1.0);
}
