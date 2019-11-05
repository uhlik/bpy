layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

uniform mat4 object_matrix;
uniform float alpha;

out vec4 vcolor;
out float valpha;

void main()
{
    gl_Position = object_matrix * vec4(position, 1.0);
    vcolor = color;
    valpha = alpha;
}