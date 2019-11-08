layout (points) in;
layout (triangle_strip, max_vertices = 48) out;

in vec3 g_position[];
in vec3 g_normal[];
in vec4 g_color[];

uniform mat4 view_matrix;
uniform mat4 window_matrix;
uniform float size[];

out vec3 f_position;
out vec3 f_normal;
out vec4 f_color;

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
    f_position = g_position[0];
    f_normal = g_normal[0];
    f_color = g_color[0];
    
    float s = size[0];
    vec4 pos = view_matrix * gl_in[0].gl_Position;
    float r = s / 2;
    // 3 * 16 = max_vertices 48
    int steps = 16;
    for (int i = 0; i < steps; i++)
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