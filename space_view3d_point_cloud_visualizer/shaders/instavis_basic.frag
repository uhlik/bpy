in vec3 f_color;

uniform float alpha = 1.0;

out vec4 fragColor;

void main()
{
    fragColor = vec4(f_color, alpha);
}