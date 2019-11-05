in vec3 position;
uniform mat4 perspective_matrix;
uniform mat4 object_matrix;
uniform float point_size;
out vec3 f_color;
void main()
{
    gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
    gl_PointSize = point_size;
    // f_color = position * 0.5 + 0.5;
    f_color = position;
}