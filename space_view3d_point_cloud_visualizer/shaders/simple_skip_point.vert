in vec3 position;
in vec4 color;
in int index;
uniform mat4 perspective_matrix;
uniform mat4 object_matrix;
uniform float point_size;
uniform float alpha_radius;
uniform float global_alpha;
uniform float skip_index;
out vec4 f_color;
out float f_alpha_radius;
void main()
{
    gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
    
    if(skip_index <= index){
        gl_Position = vec4(2.0, 0.0, 0.0, 1.0);
    }
    
    gl_PointSize = point_size;
    f_color = vec4(color[0], color[1], color[2], global_alpha);
    f_alpha_radius = alpha_radius;
}