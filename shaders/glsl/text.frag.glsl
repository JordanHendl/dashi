#version 450 core
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive    : enable
layout(location = 0) in vec2 frag_coord;

layout(location = 0) out vec4 frag_color;

layout(binding = 11) uniform sampler2D font;

void main()
{
  vec4 sampled_color = texture(font, frag_coord);
  if(sampled_color.a < 0.1f) discard;
  frag_color = vec4(vec3(sampled_color.r), 1.0f);
} 
