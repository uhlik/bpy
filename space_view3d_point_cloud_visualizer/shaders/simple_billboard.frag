layout(location = 0) out vec4 frag_color;

in vec4 fcolor;
in float falpha;

void main()
{
    frag_color = vec4(fcolor.rgb, falpha);
}