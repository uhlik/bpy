layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec4 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float point_size;
uniform float alpha_radius;

out vec3 f_position;
out vec3 f_normal;
out vec4 f_color;
out float f_alpha_radius;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    gl_PointSize = point_size;
    f_position = vec3(model * vec4(position, 1.0));
    f_normal = mat3(transpose(inverse(model))) * normal;
    f_color = color;
    f_alpha_radius = alpha_radius;
}