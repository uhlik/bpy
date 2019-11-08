in vec3 position;
in vec4 color;

uniform mat4 perspective_matrix;
uniform mat4 object_matrix;
uniform float point_size;
uniform float alpha_radius;
uniform float global_alpha;

uniform float exposure;
uniform float gamma;
uniform float brightness;
uniform float contrast;
uniform float hue;
uniform float saturation;
uniform float value;
uniform float invert;

out vec4 f_color;
out float f_alpha_radius;

out float f_exposure;
out float f_gamma;
out float f_brightness;
out float f_contrast;
out float f_hue;
out float f_saturation;
out float f_value;
out float f_invert;

void main()
{
    gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
    gl_PointSize = point_size;
    f_color = vec4(color[0], color[1], color[2], global_alpha);
    f_alpha_radius = alpha_radius;
    
    f_exposure = exposure;
    f_gamma = gamma;
    f_brightness = brightness;
    f_contrast = contrast;
    f_hue = hue;
    f_saturation = saturation;
    f_value = value;
    f_invert = invert;
}