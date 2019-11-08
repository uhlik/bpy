in vec3 position;
in vec4 color;
// in float size;
in int size;

uniform mat4 perspective_matrix;
uniform mat4 object_matrix;
uniform float global_alpha;

out vec3 f_color;
out float f_alpha;

void main()
{
    gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
    gl_PointSize = size;
    f_color = color.rgb;
    f_alpha = global_alpha;
}