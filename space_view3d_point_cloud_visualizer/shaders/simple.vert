in vec3 position;
in vec4 color;
uniform mat4 perspective_matrix;
uniform mat4 object_matrix;
uniform float point_size;
uniform float alpha_radius;
uniform float global_alpha;
out vec4 f_color;
out float f_alpha_radius;
void main()
{
    gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
    gl_PointSize = point_size;
    // f_color = color;
    f_color = vec4(color[0], color[1], color[2], global_alpha);
    f_alpha_radius = alpha_radius;
}