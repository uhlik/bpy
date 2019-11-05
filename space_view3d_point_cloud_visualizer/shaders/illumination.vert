in vec3 position;
in vec3 normal;
in vec4 color;

// uniform float show_illumination;
uniform vec3 light_direction;
uniform vec3 light_intensity;
uniform vec3 shadow_direction;
uniform vec3 shadow_intensity;
// uniform float show_normals;

uniform mat4 perspective_matrix;
uniform mat4 object_matrix;
uniform float point_size;
uniform float alpha_radius;
uniform float global_alpha;

out vec4 f_color;
out float f_alpha_radius;
out vec3 f_normal;

out vec3 f_light_direction;
out vec3 f_light_intensity;
out vec3 f_shadow_direction;
out vec3 f_shadow_intensity;
// out float f_show_normals;
// out float f_show_illumination;

void main()
{
    gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
    gl_PointSize = point_size;
    f_normal = normal;
    // f_color = color;
    f_color = vec4(color[0], color[1], color[2], global_alpha);
    f_alpha_radius = alpha_radius;
    
    // f_light_direction = normalize(vec3(inverse(object_matrix) * vec4(light_direction, 1.0)));
    f_light_direction = light_direction;
    f_light_intensity = light_intensity;
    // f_shadow_direction = normalize(vec3(inverse(object_matrix) * vec4(shadow_direction, 1.0)));
    f_shadow_direction = shadow_direction;
    f_shadow_intensity = shadow_intensity;
    // f_show_normals = show_normals;
    // f_show_illumination = show_illumination;
}