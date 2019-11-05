in vec4 f_color;
in vec3 f_normal;
in float f_alpha_radius;

in vec3 f_light_direction;
in vec3 f_light_intensity;
in vec3 f_shadow_direction;
in vec3 f_shadow_intensity;
// in float f_show_normals;
// in float f_show_illumination;

out vec4 fragColor;

void main()
{
    float r = 0.0f;
    float a = 1.0f;
    vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
    r = dot(cxy, cxy);
    if(r > f_alpha_radius){
        discard;
    }
    // fragColor = f_color * a;
    
    vec4 col;
    
    // if(f_show_normals > 0.5){
    //     col = vec4(f_normal, 1.0) * a;
    // }else if(f_show_illumination > 0.5){
    
    // if(f_show_illumination > 0.5){
    //     vec4 light = vec4(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity, 1);
    //     vec4 shadow = vec4(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity, 1);
    //     col = (f_color + light - shadow) * a;
    // }else{
    //     col = f_color * a;
    // }
    
    vec4 light = vec4(max(dot(f_light_direction, -f_normal), 0) * f_light_intensity, 1);
    vec4 shadow = vec4(max(dot(f_shadow_direction, -f_normal), 0) * f_shadow_intensity, 1);
    col = (f_color + light - shadow) * a;
    
    fragColor = col;
}