in vec3 f_position;
in vec3 f_normal;
in vec3 f_color;
in float f_alpha;
in float f_alpha_radius;

uniform vec3 view_position;
uniform float fresnel_sharpness;
uniform float use_colors;
uniform float use_invert;

out vec4 fragColor;

void main()
{
    vec3 color = f_color;
    // vec3 V = normalize(view_position.xyz - f_position.xyz);
    vec3 V = normalize(view_position.xyz - f_position);
    vec3 N = normalize(f_normal);
    float fresnel = pow(1.0 - dot(N, V), fresnel_sharpness);
    vec3 fresnel_color = vec3(fresnel, fresnel, fresnel);
    
    float r = 0.0f;
    float a = 1.0f;
    vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
    r = dot(cxy, cxy);
    if(r > f_alpha_radius){
        discard;
    }
    
    vec3 col = fresnel_color;
    if(use_invert > 0.0){
        col = 1.0 - col;
    }
    if(use_colors > 0.0){
        col = vec3(f_color * col);
    }
    fragColor = vec4(col, f_alpha) * a;
}