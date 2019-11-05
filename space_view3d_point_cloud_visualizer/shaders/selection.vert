in vec3 position;
uniform mat4 perspective_matrix;
uniform mat4 object_matrix;
uniform float point_size;
void main()
{
    gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
    gl_PointSize = point_size;
}