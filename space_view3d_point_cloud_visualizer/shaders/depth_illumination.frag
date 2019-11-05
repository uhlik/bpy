in float f_depth;
in vec3 f_normal;
in vec3 f_light_direction;
in vec3 f_light_intensity;
in vec3 f_shadow_direction;
in vec3 f_shadow_intensity;
uniform float alpha_radius;
uniform float global_alpha;
uniform float brightness;
uniform float contrast;
uniform vec3 color_a;
uniform vec3 color_b;
out vec4 fragColor;
void main()
{
    float r = 0.0f;
    float a = 1.0f;
    vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
    r = dot(cxy, cxy);
    if(r > alpha_radius){
        discard;
    }
    vec3 l = vec3(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity);
    vec3 s = vec3(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity);
    vec3 color = mix(color_b, color_a, f_depth);
    // brightness/contrast after illumination
    // vec3 c = color + l - s;
    // vec3 cc = (c - 0.5) * contrast + 0.5 + brightness;
    // fragColor = vec4(cc, global_alpha) * a;
    
    // brightness/contrast before illumination
    vec3 cc = (color - 0.5) * contrast + 0.5 + brightness;
    vec3 c = cc + l - s;
    fragColor = vec4(c, global_alpha) * a;
}