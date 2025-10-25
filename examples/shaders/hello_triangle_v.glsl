#version 450
layout(location = 0) in vec2 inPosition;
layout(location = 0) out vec2 frag_color;

layout(binding = 0) uniform position_offset {
    vec2 pos;
};

void main() {
    frag_color = inPosition;
    gl_Position = vec4(inPosition + pos, 0.0, 1.0);
}
