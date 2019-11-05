layout(points) in;
layout(line_strip, max_vertices = 256) out;

uniform mat4 perspective_matrix;
uniform mat4 object_matrix;
uniform vec4 color = vec4(1.0, 1.0, 1.0, 1.0);

uniform float length = 0.1;
uniform vec3 center = vec3(0.0, 0.0, 0.0);
uniform vec3 dimensions = vec3(1.0, 1.0, 1.0);

out vec4 vertex_color;

void line();

void line(vec4 o, vec3 a, vec3 b)
{
    gl_Position = perspective_matrix * object_matrix * (o + vec4(a, 0.0));
    EmitVertex();
    gl_Position = perspective_matrix * object_matrix * (o + vec4(b, 0.0));
    EmitVertex();
    EndPrimitive();
}

void main()
{
    vertex_color = color;
    
    vec4 o = vec4(center, 1.0);
    
    float w = dimensions[0] / 2;
    float h = dimensions[1] / 2;
    float d = dimensions[2] / 2;
    float l = length;
    
    vec3 p00 = vec3(-(w - l),       -h,       -d);
    vec3 p01 = vec3(      -w,       -h,       -d);
    vec3 p02 = vec3(      -w,       -h, -(d - l));
    vec3 p03 = vec3(      -w, -(h - l),       -d);
    vec3 p04 = vec3(-(w - l),       -h,        d);
    vec3 p05 = vec3(      -w,       -h,        d);
    vec3 p06 = vec3(      -w, -(h - l),        d);
    vec3 p07 = vec3(      -w,       -h,  (d - l));
    vec3 p08 = vec3(      -w,  (h - l),       -d);
    vec3 p09 = vec3(      -w,        h,       -d);
    vec3 p10 = vec3(      -w,        h, -(d - l));
    vec3 p11 = vec3(-(w - l),        h,       -d);
    vec3 p12 = vec3(-(w - l),        h,        d);
    vec3 p13 = vec3(      -w,        h,        d);
    vec3 p14 = vec3(      -w,        h,  (d - l));
    vec3 p15 = vec3(      -w,  (h - l),        d);
    vec3 p16 = vec3(       w, -(h - l),       -d);
    vec3 p17 = vec3(       w,       -h,       -d);
    vec3 p18 = vec3(       w,       -h, -(d - l));
    vec3 p19 = vec3( (w - l),       -h,       -d);
    vec3 p20 = vec3( (w - l),       -h,        d);
    vec3 p21 = vec3(       w,       -h,        d);
    vec3 p22 = vec3(       w,       -h,  (d - l));
    vec3 p23 = vec3(       w, -(h - l),        d);
    vec3 p24 = vec3( (w - l),        h,       -d);
    vec3 p25 = vec3(       w,        h,       -d);
    vec3 p26 = vec3(       w,        h, -(d - l));
    vec3 p27 = vec3(       w,  (h - l),       -d);
    vec3 p28 = vec3(       w,  (h - l),        d);
    vec3 p29 = vec3(       w,        h,        d);
    vec3 p30 = vec3(       w,        h,  (d - l));
    vec3 p31 = vec3( (w - l),        h,        d);
    
    line(o, p00, p01);
    line(o, p01, p03);
    line(o, p02, p01);
    line(o, p04, p05);
    line(o, p05, p07);
    line(o, p06, p05);
    line(o, p08, p09);
    line(o, p09, p11);
    line(o, p10, p09);
    line(o, p12, p13);
    line(o, p13, p15);
    line(o, p14, p13);
    line(o, p16, p17);
    line(o, p17, p19);
    line(o, p18, p17);
    line(o, p20, p21);
    line(o, p21, p23);
    line(o, p22, p21);
    line(o, p24, p25);
    line(o, p25, p27);
    line(o, p26, p25);
    line(o, p28, p29);
    line(o, p29, p31);
    line(o, p30, p29);
    
}