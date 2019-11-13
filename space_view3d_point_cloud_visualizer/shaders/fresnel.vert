in vec3 position;
in vec3 normal;
in vec4 color;

// uniform mat4 perspective_matrix;
// uniform mat4 object_matrix;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform float point_size;
uniform float alpha_radius;
uniform float global_alpha;

out vec3 f_position;
out vec3 f_normal;
out vec3 f_color;
out float f_alpha;
out float f_alpha_radius;

void main()
{
    // gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
    gl_Position = projection * view * model * vec4(position, 1.0);
    
    gl_PointSize = point_size;
    // f_position = gl_Position;
    // f_normal = normal;
    f_position = vec3(model * vec4(position, 1.0));
    f_normal = mat3(transpose(inverse(model))) * normal;
    
    f_color = color.rgb;
    f_alpha = global_alpha;
    f_alpha_radius = alpha_radius;
}