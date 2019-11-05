layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec3 g_position[];
in vec3 g_normal[];
in vec4 g_color[];
uniform mat4 view_matrix;
uniform mat4 window_matrix;
uniform float size[];
out vec3 f_position;
out vec3 f_normal;
out vec4 f_color;

void main()
{
    f_position = g_position[0];
    f_normal = g_normal[0];
    f_color = g_color[0];
    
    float s = size[0] / 2;
    
    vec4 pos = view_matrix * gl_in[0].gl_Position;
    vec2 xyloc = vec2(-1 * s, -1 * s);
    gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
    EmitVertex();
    
    xyloc = vec2(1 * s, -1 * s);
    gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
    EmitVertex();
    
    xyloc = vec2(-1 * s, 1 * s);
    gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
    EmitVertex();
    
    xyloc = vec2(1 * s, 1 * s);
    gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
    EmitVertex();
    
    EndPrimitive();
}