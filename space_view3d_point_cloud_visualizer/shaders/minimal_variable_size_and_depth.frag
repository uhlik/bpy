in vec3 f_color;
in float f_alpha;

in float f_depth;
uniform float brightness;
uniform float contrast;
uniform float blend;

out vec4 fragColor;
void main()
{
    // fragColor = vec4(f_color, f_alpha);
    
    vec3 depth_color = vec3(f_depth, f_depth, f_depth);
    depth_color = (depth_color - 0.5) * contrast + 0.5 + brightness;
    // fragColor = vec4(depth_color, global_alpha) * a;
    
    depth_color = mix(depth_color, vec3(1.0, 1.0, 1.0), blend);
    
    fragColor = vec4(f_color * depth_color, f_alpha);
    
}