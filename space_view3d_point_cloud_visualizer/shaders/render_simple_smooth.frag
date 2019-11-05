in vec4 f_color;
in float f_alpha_radius;
out vec4 fragColor;
void main()
{
    float r = 0.0f;
    float d = 0.0f;
    float a = 1.0f;
    vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
    r = dot(cxy, cxy);
    d = fwidth(r);
    a = 1.0 - smoothstep(1.0 - (d / 2), 1.0 + (d / 2), r);
    //fragColor = f_color * a;
    fragColor = vec4(f_color.rgb, f_color.a * a);
}