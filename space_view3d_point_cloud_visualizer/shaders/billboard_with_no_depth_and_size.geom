layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec4 vcolor[];
in float valpha[];
in float vsizef[];

uniform mat4 view_matrix;
uniform mat4 window_matrix;

uniform float size[];

out vec4 fcolor;
out float falpha;

void main()
{
    fcolor = vcolor[0];
    falpha = valpha[0];
    
    // value is diameter, i need radius, then multiply by individual point size
    float s = (size[0] / 2) * vsizef[0];
    
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