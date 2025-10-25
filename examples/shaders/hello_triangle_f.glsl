#version 450 core
layout(location = 0) in vec2 frag_color;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(frag_color.xy, 0.0, 1.0);
}
