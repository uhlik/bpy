layout (points) in;
// 3 * 16 = 48
layout (triangle_strip, max_vertices = 48) out;

in vec4 vcolor[];
in float valpha[];

uniform mat4 view_matrix;
uniform mat4 window_matrix;

uniform float size[];

out vec4 fcolor;
out float falpha;

vec2 disc_coords(float radius, int step, int steps)
{
    const float PI = 3.1415926535897932384626433832795;
    float angstep = 2 * PI / steps;
    float x = sin(step * angstep) * radius;
    float y = cos(step * angstep) * radius;
    return vec2(x, y);
}

void main()
{
    fcolor = vcolor[0];
    falpha = valpha[0];
    float s = size[0];
    
    vec4 pos = view_matrix * gl_in[0].gl_Position;
    float r = s / 2;
    int steps = 16;
    
    for(int i = 0; i < steps; i++)
    {
        
        gl_Position = window_matrix * (pos);
        EmitVertex();
        
        vec2 xyloc = disc_coords(r, i, steps);
        gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
        EmitVertex();
        
        xyloc = disc_coords(r, i + 1, steps);
        gl_Position = window_matrix * (pos + vec4(xyloc, 0, 0));
        EmitVertex();
        
        EndPrimitive();
    }
    
}