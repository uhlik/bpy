layout(points) in;
layout(line_strip, max_vertices = 2) out;

uniform mat4 perspective_matrix;
uniform mat4 object_matrix;
uniform float length = 1.0;
uniform vec4 color = vec4(1.0, 1.0, 1.0, 1.0);

in vec3 vertex_normal[];

out vec4 vertex_color;

void main()
{
    vec3 normal = vertex_normal[0];
    
    vertex_color = color;
    
    vec4 v0 = gl_in[0].gl_Position;
    gl_Position = perspective_matrix * object_matrix * v0;
    EmitVertex();
    
    vec4 v1 = v0 + vec4(normal * length, 0.0);
    gl_Position = perspective_matrix * object_matrix * v1;
    EmitVertex();
    
    EndPrimitive();
}