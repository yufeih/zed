struct GpuiShaderFrame {
    viewport: vec4<f32>,
    bounds: vec4<f32>,
    clip: vec4<f32>,
    parameters: array<vec4<f32>, 4>,
    flags: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> gpui_shader_frame: GpuiShaderFrame;

@group(0) @binding(1)
var gpui_input_texture0: texture_2d<f32>;

@group(0) @binding(2)
var gpui_input_texture1: texture_2d<f32>;

fn gpui_read_input(image: texture_2d<f32>, pixel: vec2<i32>) -> vec4<f32> {
    if (any(pixel < vec2<i32>(0)) || any(pixel >= vec2<i32>(textureDimensions(image)))) {
        return vec4<f32>(0.0);
    }
    return textureLoad(image, pixel, 0);
}

fn gpui_sample_input(image: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let position = uv * vec2<f32>(textureDimensions(image)) - 0.5;
    let pixel = vec2<i32>(floor(position));
    let fraction = fract(position);
    return mix(
        mix(gpui_read_input(image, pixel), gpui_read_input(image, pixel + vec2<i32>(1, 0)), fraction.x),
        mix(gpui_read_input(image, pixel + vec2<i32>(0, 1)), gpui_read_input(image, pixel + vec2<i32>(1, 1)), fraction.x),
        fraction.y);
}

fn gpui_sample_input0(uv: vec2<f32>) -> vec4<f32> {
    return gpui_sample_input(gpui_input_texture0, uv);
}

fn gpui_sample_input1(uv: vec2<f32>) -> vec4<f32> {
    return gpui_sample_input(gpui_input_texture1, uv);
}

@vertex
fn gpui_shader_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let unit = vec2<f32>(f32(index & 1u), f32((index >> 1u) & 1u));
    let position = gpui_shader_frame.bounds.xy + unit * gpui_shader_frame.bounds.zw;
    return vec4<f32>(
        position / gpui_shader_frame.viewport.xy * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0),
        0.0, 1.0);
}

@fragment
fn gpui_shader_fragment(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let scale = gpui_shader_frame.viewport.zw;
    let color = paint((position.xy - gpui_shader_frame.bounds.xy) / scale,
        gpui_shader_frame.bounds.zw / scale, gpui_shader_frame.parameters);
    if (any(position.xy < gpui_shader_frame.clip.xy) || any(position.xy >= gpui_shader_frame.clip.zw)) {
        discard;
    }
    if (gpui_shader_frame.flags.z != 0.0) {
        return color;
    }
    let alpha = clamp(color.a, 0.0, 1.0) * gpui_shader_frame.flags.w;
    var rgb = max(color.rgb, vec3<f32>(0.0));
    if (gpui_shader_frame.flags.x != 0.0) {
        rgb = select(pow((rgb + 0.055) / 1.055, vec3<f32>(2.4)),
            rgb / 12.92, rgb <= vec3<f32>(0.04045));
    }
    if (gpui_shader_frame.flags.y != 0.0) {
        rgb *= alpha;
    }
    return vec4<f32>(rgb, alpha);
}
