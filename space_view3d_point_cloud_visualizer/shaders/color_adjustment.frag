in vec4 f_color;
in float f_alpha_radius;

out vec4 fragColor;

in float f_exposure;
in float f_gamma;
in float f_brightness;
in float f_contrast;
in float f_hue;
in float f_saturation;
in float f_value;
in float f_invert;

// https://stackoverflow.com/questions/15095909/from-rgb-to-hsv-in-opengl-glsl
vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main()
{
    float r = 0.0f;
    float a = 1.0f;
    vec2 cxy = 2.0f * gl_PointCoord - 1.0f;
    r = dot(cxy, cxy);
    if(r > f_alpha_radius){
        discard;
    }
    fragColor = f_color * a;
    
    // adjustments
    vec3 rgb = fragColor.rgb;
    float alpha = fragColor.a;
    vec3 color = rgb;
    
    // exposure
    color = clamp(color * pow(2, f_exposure), 0.0, 1.0);
    // gamma
    color = clamp(vec3(pow(color[0], 1 / f_gamma), pow(color[1], 1 / f_gamma), pow(color[2], 1 / f_gamma)), 0.0, 1.0);
    
    // brightness/contrast
    color = clamp((color - 0.5) * f_contrast + 0.5 + f_brightness, 0.0, 1.0);
    
    // hue/saturation/value
    vec3 hsv = rgb2hsv(color);
    float hue = f_hue;
    if(hue > 1.0){
        hue = mod(hue, 1.0);
    }
    hsv[0] = mod((hsv[0] + hue), 1.0);
    hsv[1] += f_saturation;
    hsv[2] += f_value;
    hsv = clamp(hsv, 0.0, 1.0);
    color = hsv2rgb(hsv);
    
    if(f_invert > 0.0){
        color = vec3(1.0 - color[0], 1.0 - color[1], 1.0 - color[2]);
    }
    
    fragColor = vec4(color, alpha);
    
}