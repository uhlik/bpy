layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 2) in float sizef;

uniform mat4 object_matrix;
uniform mat4 perspective_matrix;

uniform float alpha;
uniform vec3 center;
uniform float maxdist;

out vec4 vcolor;
out float valpha;
out float vsizef;
out float vdepth;

void main()
{
    gl_Position = object_matrix * vec4(position, 1.0);
    vcolor = color;
    valpha = alpha;
    vsizef = sizef;
    
    vec4 pp = perspective_matrix * object_matrix * vec4(position, 1.0);
    vec4 op = perspective_matrix * object_matrix * vec4(center, 1.0);
    float d = op.z - pp.z;
    vdepth = ((d - (-maxdist)) / (maxdist - d)) / 2;
}