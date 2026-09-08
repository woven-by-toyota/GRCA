#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

layout(push_constant) uniform PushConsts {
	mat4 mvp;
	float minDist;
	float maxDist;
} pc;

void main() {
	fragColor = inColor;
	gl_Position = pc.mvp * vec4(inPosition, 1.0);
}
