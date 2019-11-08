in vec3 position;
in vec4 color;

uniform mat4 perspective_matrix;
uniform mat4 object_matrix;
uniform float point_size;
uniform float global_alpha;

out vec3 f_color;
out float f_alpha;

void main()
{
    gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
    gl_PointSize = point_size;
    f_color = color.rgb;
    f_alpha = global_alpha;
}