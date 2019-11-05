in vec3 f_position;
in vec3 f_normal;
in vec4 f_color;
in float f_alpha_radius;

uniform float alpha;
uniform vec3 light_position;
uniform vec3 light_color;
uniform vec3 view_position;
uniform float ambient_strength;
uniform float specular_strength;
uniform float specular_exponent;

out vec4 frag_color;

void main()
{
    vec3 ambient = ambient_strength * light_color;
    
    vec3 nor = normalize(f_normal);
    vec3 light_direction = normalize(light_position - f_position);
    vec3 diffuse = max(dot(nor, light_direction), 0.0) * light_color;
    
    vec3 view_direction = normalize(view_position - f_position);
    vec3 reflection_direction = reflect(-light_direction, nor);
    float spec = pow(max(dot(view_direction, reflection_direction), 0.0), specular_exponent);
    vec3 specular = specular_strength * spec * light_color;
    
    vec3 col = (ambient + diffuse + specular) * f_color.rgb;
    
    float r = 0.0f;
    float a = 1.0f;
    vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
    r = dot(cxy, cxy);
    if(r > f_alpha_radius){
        discard;
    }
    
    frag_color = vec4(col, alpha) * a;
}