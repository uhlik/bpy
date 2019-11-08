in float f_depth;

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
    vec3 color = mix(color_b, color_a, f_depth);
    color = (color - 0.5) * contrast + 0.5 + brightness;
    fragColor = vec4(color, global_alpha) * a;
}