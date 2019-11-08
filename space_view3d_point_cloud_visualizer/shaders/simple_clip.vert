in vec3 position;
in vec4 color;

uniform mat4 perspective_matrix;
uniform mat4 object_matrix;
uniform float point_size;
uniform float alpha_radius;
uniform float global_alpha;

out vec4 f_color;
out float f_alpha_radius;

uniform vec4 clip_plane0;
uniform vec4 clip_plane1;
uniform vec4 clip_plane2;
uniform vec4 clip_plane3;
uniform vec4 clip_plane4;
uniform vec4 clip_plane5;

void main()
{
    gl_Position = perspective_matrix * object_matrix * vec4(position, 1.0f);
    gl_PointSize = point_size;
    f_color = vec4(color[0], color[1], color[2], global_alpha);
    f_alpha_radius = alpha_radius;
    
    vec4 pos = vec4(position, 1.0f);
    gl_ClipDistance[0] = dot(clip_plane0, pos);
    gl_ClipDistance[1] = dot(clip_plane1, pos);
    gl_ClipDistance[2] = dot(clip_plane2, pos);
    gl_ClipDistance[3] = dot(clip_plane3, pos);
    gl_ClipDistance[4] = dot(clip_plane4, pos);
    gl_ClipDistance[5] = dot(clip_plane5, pos);
}