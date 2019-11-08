layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec4 color;

uniform mat4 model;

out vec3 g_position;
out vec3 g_normal;
out vec4 g_color;

void main()
{
    gl_Position = model * vec4(position, 1.0);
    g_position = vec3(model * vec4(position, 1.0));
    g_normal = mat3(transpose(inverse(model))) * normal;
    g_color = color;
}