in vec3 position;
uniform mat4 perspective_matrix;
uniform mat4 object_matrix;
uniform vec3 center;
uniform float point_size;
uniform float maxdist;
out float f_depth;
void main()
{
    gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0);
    gl_PointSize = point_size;
    vec4 pp = perspective_matrix * object_matrix * vec4(position, 1.0);
    vec4 op = perspective_matrix * object_matrix * vec4(center, 1.0);
    float d = op.z - pp.z;
    f_depth = ((d - (-maxdist)) / (maxdist - d)) / 2;
}