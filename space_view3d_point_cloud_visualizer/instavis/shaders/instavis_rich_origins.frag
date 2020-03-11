layout (location = 0) out vec4 frag_color;

in vec3 f_position;
in vec3 f_color;

uniform float alpha = 1.0;

void main()
{
    frag_color = vec4(f_color.rgb, alpha);
}