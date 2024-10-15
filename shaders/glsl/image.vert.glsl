#version 450 core
#extension GL_ARB_separate_shader_objects : enable

layout ( location = 0 ) in  vec4 vertex     ; ///< Input in the following format: vec2 position, vec2 texCoords>
layout ( location = 0 ) out vec2 tex_coords ; ///< Texture coordinate output of this stage.
layout( binding = 1) uniform tex_offset
{
  vec2 offset;
};

void main() {
  tex_coords = vertex.zw ;
  gl_Position = vec4( vertex.xy + offset, 0.0, 1.0 ) ;
}
