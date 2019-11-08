in vec3 f_color;
in float f_alpha;

out vec4 fragColor;

void main()
{
    fragColor = vec4(f_color, f_alpha);
}