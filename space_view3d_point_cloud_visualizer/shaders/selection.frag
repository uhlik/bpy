uniform vec4 color;
uniform float alpha_radius;
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
    fragColor = color * a;
}