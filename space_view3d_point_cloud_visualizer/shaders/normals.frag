layout(location = 0) out vec4 frag_color;

uniform float global_alpha;

in vec4 vertex_color;

void main()
{
    // frag_color = vertex_color;
    frag_color = vec4(vertex_color[0], vertex_color[1], vertex_color[2], vertex_color[3] * global_alpha);
}