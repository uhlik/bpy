in vec3 position;
in vec3 color;

uniform mat4 perspective_matrix;
uniform mat4 object_matrix;
uniform float size;

out vec3 f_color;

void main()
{
    gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
    gl_PointSize = size;
    f_color = color;
}