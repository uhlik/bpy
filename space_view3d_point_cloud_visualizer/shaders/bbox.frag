layout(location = 0) out vec4 frag_color;

uniform float global_alpha;

in vec4 vertex_color;

void main()
{
    frag_color = vec4(vertex_color.rgb, vertex_color[3] * global_alpha);
}