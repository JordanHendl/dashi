#version 450 core
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive    : enable
layout (location = 0) in vec2 in_position;
layout (location = 1) in vec2 in_tex_coord;

layout(location = 0) out vec2 frag_coord;
void main()
{
  frag_coord = in_tex_coord;
  gl_Position = vec4(in_position.x, in_position.y, 0.0, 1.0);
}
