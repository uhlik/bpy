layout(location = 0) out vec4 frag_color;

in vec4 fcolor;
in float falpha;

in float fdepth;
uniform float brightness;
uniform float contrast;
uniform float blend;

void main()
{
    vec3 depth_color = vec3(fdepth, fdepth, fdepth);
    depth_color = (depth_color - 0.5) * contrast + 0.5 + brightness;
    depth_color = mix(depth_color, vec3(1.0, 1.0, 1.0), blend);
    frag_color = vec4(fcolor.rgb * depth_color, falpha);
}