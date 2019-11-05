layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out vec3 vertex_normal;

void main()
{
    vertex_normal = normal;
    gl_Position = vec4(position, 1.0);
}