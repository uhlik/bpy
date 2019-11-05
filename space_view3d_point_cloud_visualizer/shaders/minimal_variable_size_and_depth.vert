in vec3 position;
in vec4 color;
in int size;
uniform mat4 perspective_matrix;
uniform mat4 object_matrix;
uniform float global_alpha;

uniform vec3 center;
uniform float maxdist;

out vec3 f_color;
out float f_alpha;

out float f_depth;

void main()
{
    gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
    gl_PointSize = size;
    f_color = color.rgb;
    f_alpha = global_alpha;
    
    vec4 pp = perspective_matrix * object_matrix * vec4(position, 1.0);
    vec4 op = perspective_matrix * object_matrix * vec4(center, 1.0);
    float d = op.z - pp.z;
    f_depth = ((d - (-maxdist)) / (maxdist - d)) / 2;
}