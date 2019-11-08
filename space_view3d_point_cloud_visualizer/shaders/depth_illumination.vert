in vec3 position;
in vec3 normal;

uniform mat4 perspective_matrix;
uniform mat4 object_matrix;
uniform vec3 center;
uniform float point_size;
uniform float maxdist;
uniform vec3 light_direction;
uniform vec3 light_intensity;
uniform vec3 shadow_direction;
uniform vec3 shadow_intensity;

out float f_depth;
out vec3 f_light_direction;
out vec3 f_light_intensity;
out vec3 f_shadow_direction;
out vec3 f_shadow_intensity;
out vec3 f_normal;

void main()
{
    gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0);
    gl_PointSize = point_size;
    vec4 pp = perspective_matrix * object_matrix * vec4(position, 1.0);
    vec4 op = perspective_matrix * object_matrix * vec4(center, 1.0);
    float d = op.z - pp.z;
    f_depth = ((d - (-maxdist)) / (maxdist - d)) / 2;
    f_normal = normal;
    f_light_direction = light_direction;
    f_light_intensity = light_intensity;
    f_shadow_direction = shadow_direction;
    f_shadow_intensity = shadow_intensity;
}