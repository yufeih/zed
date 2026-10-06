use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use anyhow::{Context as _, Result, anyhow, ensure};
use collections::FxHashMap;

use crate::{PaintSurface, RenderImage, RenderImageParams, SharedString};

/// A validated portable fragment shader. Define
/// `fn paint(position: vec2<f32>, size: vec2<f32>, parameters: array<vec4<f32>, 4>) -> vec4<f32>`.
/// Coordinates are logical pixels; return straight-alpha sRGB. GPUI supplies clipping,
/// element opacity, color conversion and the vertex stage. `gpui_sample_input0(uv)` and
/// `gpui_sample_input1(uv)` sample optional inputs with bilinear filtering and transparent
/// borders. Intermediate passes preserve raw floating-point values; only the final pass
/// applies display conversion and opacity. Additional bindings and entry points are unsupported.
#[derive(Debug)]
pub struct CustomShader {
    /// Stable identity used by renderer pipeline caches.
    id: u64,
    /// Label included in compilation diagnostics.
    label: SharedString,
    /// Complete WGSL program including GPUI's entry points.
    wgsl: String,
    /// Translated DirectX program.
    hlsl: ShaderProgram,
    /// Translated Metal program.
    msl: ShaderProgram,
}

/// A native translation of a portable shader.
#[derive(Debug)]
pub struct ShaderProgram {
    /// Native shader source.
    pub source: String,
    /// Native vertex entry point.
    pub vertex_entry: String,
    /// Native fragment entry point.
    pub fragment_entry: String,
}

impl CustomShader {
    /// Stable identity of this immutable program.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Label used in renderer diagnostics.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// WGSL source including the standard entry points.
    pub fn wgsl(&self) -> &str {
        &self.wgsl
    }

    /// Native DirectX translation.
    pub fn hlsl(&self) -> &ShaderProgram {
        &self.hlsl
    }

    /// Native Metal translation.
    pub fn msl(&self) -> &ShaderProgram {
        &self.msl
    }

    /// Validate and translate once, then retain the returned shader across frames.
    pub fn new(label: impl Into<SharedString>, source: &str) -> Result<Arc<Self>> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let label = label.into();
        let wgsl = format!("{source}\n{}", include_str!("custom_shader.wgsl"));
        let module = naga::front::wgsl::parse_str(&wgsl)
            .map_err(|error| anyhow!("{label}: {}", error.emit_to_string(&wgsl)))?;
        ensure!(
            module.entry_points.len() == 2,
            "{label}: additional shader entry points are not supported"
        );
        ensure!(
            module
                .global_variables
                .iter()
                .filter(|(_, variable)| variable.binding.is_some())
                .count()
                == 3,
            "{label}: additional shader resource bindings are not supported"
        );
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .map_err(|error| anyhow!("{label}: {}", error.emit_to_string(&wgsl)))?;
        let binding = naga::ResourceBinding {
            group: 0,
            binding: 0,
        };
        let mut hlsl_options = naga::back::hlsl::Options {
            shader_model: naga::back::hlsl::ShaderModel::V5_0,
            fake_missing_bindings: false,
            ..Default::default()
        };
        hlsl_options.binding_map.insert(
            binding,
            naga::back::hlsl::BindTarget {
                register: 0,
                ..Default::default()
            },
        );
        for index in 0..2 {
            hlsl_options.binding_map.insert(
                naga::ResourceBinding {
                    group: 0,
                    binding: index + 1,
                },
                naga::back::hlsl::BindTarget {
                    register: index,
                    ..Default::default()
                },
            );
        }
        let mut hlsl_source = String::new();
        let hlsl_info =
            naga::back::hlsl::Writer::new(&mut hlsl_source, &hlsl_options, &Default::default())
                .write(&module, &info, None)
                .with_context(|| format!("{label}: translating WGSL to HLSL"))?;
        let hlsl_names = hlsl_info
            .entry_point_names
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let [vertex_entry, fragment_entry]: [String; 2] = hlsl_names
            .try_into()
            .map_err(|_| anyhow!("{label}: missing HLSL entry points"))?;
        let hlsl = ShaderProgram {
            source: hlsl_source,
            vertex_entry,
            fragment_entry,
        };
        let mut msl_options = naga::back::msl::Options {
            lang_version: (2, 0),
            fake_missing_bindings: false,
            ..Default::default()
        };
        for entry in &module.entry_points {
            let mut resources = naga::back::msl::EntryPointResources::default();
            resources.resources.insert(
                binding,
                naga::back::msl::BindTarget {
                    buffer: Some(0),
                    ..Default::default()
                },
            );
            for index in 0..2 {
                resources.resources.insert(
                    naga::ResourceBinding {
                        group: 0,
                        binding: index + 1,
                    },
                    naga::back::msl::BindTarget {
                        texture: Some(index as u8),
                        ..Default::default()
                    },
                );
            }
            msl_options
                .per_entry_point_map
                .insert(entry.name.clone(), resources);
        }
        let (msl_source, msl_info) =
            naga::back::msl::write_string(&module, &info, &msl_options, &Default::default())
                .with_context(|| format!("{label}: translating WGSL to MSL"))?;
        let msl_names = msl_info
            .entry_point_names
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let [vertex_entry, fragment_entry]: [String; 2] = msl_names
            .try_into()
            .map_err(|_| anyhow!("{label}: missing MSL entry points"))?;
        Ok(Arc::new(Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            label,
            wgsl,
            hlsl,
            msl: ShaderProgram {
                source: msl_source,
                vertex_entry,
                fragment_entry,
            },
        }))
    }
}

/// A shader invocation whose inputs can be images or earlier shader invocations.
#[derive(Clone, Debug)]
pub struct ShaderPass {
    /// Program shared by all invocations of this effect.
    pub shader: Arc<CustomShader>,
    /// Application-defined values for this invocation.
    pub parameters: [[f32; 4]; 4],
    /// Missing inputs sample transparent black.
    pub inputs: [Option<ShaderInput>; 2],
}

/// An image or an offscreen shader result.
#[derive(Clone, Debug)]
pub enum ShaderInput {
    /// A frame of an existing GPUI image.
    Image(ShaderImage),
    /// Render a pass at window resolution divided by `2.pow(downsample)`, at least one pixel.
    Pass {
        /// The invocation to render.
        pass: Arc<ShaderPass>,
        /// Power-of-two resolution divisor, relative to the final surface.
        downsample: u8,
    },
}

/// A shader-readable image frame.
#[derive(Clone, Debug)]
pub struct ShaderImage {
    /// Image retained for as long as it is used by the graph.
    pub image: Arc<RenderImage>,
    /// Frame to sample.
    pub frame_index: usize,
}

impl ShaderImage {
    /// Cache key independent of renderer and GPU device.
    pub fn key(&self) -> RenderImageParams {
        RenderImageParams {
            image_id: self.image.id,
            frame_index: self.frame_index,
        }
    }

    /// Convert GPUI's BGRA image storage into a validated RGBA upload.
    pub fn rgba(&self) -> Result<([u32; 2], Vec<u8>)> {
        let size = self.image.size(self.frame_index);
        ensure!(
            size.width.0 > 0 && size.height.0 > 0,
            "shader image frame is empty"
        );
        let mut bytes = self
            .image
            .as_bytes(self.frame_index)
            .context("shader image frame is missing")?
            .to_vec();
        for pixel in bytes.as_chunks_mut::<4>().0 {
            pixel.swap(0, 2);
        }
        Ok(([size.width.0 as u32, size.height.0 as u32], bytes))
    }
}

/// A resolved shader input; pass indices always refer to preceding nodes.
#[derive(Clone, Debug)]
pub enum ShaderNodeInput {
    /// CPU-backed image.
    Image(ShaderImage),
    /// A preceding offscreen result.
    Pass(usize),
}

/// One invocation in dependency order. The last node is the final, full-resolution pass.
#[derive(Clone, Debug)]
pub struct ShaderNode {
    /// Program and parameters for this invocation.
    pub pass: Arc<ShaderPass>,
    /// Resolved input dependencies.
    pub inputs: [Option<ShaderNodeInput>; 2],
    /// Resolution divisor for intermediate results.
    pub downsample: u8,
}

impl ShaderNode {
    /// Physical target dimensions, including very small downsampled surfaces.
    pub fn size(&self, surface: &PaintSurface) -> [u32; 2] {
        [
            (surface.bounds.size.width.0.ceil() as u32)
                .checked_shr(self.downsample as u32)
                .unwrap_or(0)
                .max(1),
            (surface.bounds.size.height.0.ceil() as u32)
                .checked_shr(self.downsample as u32)
                .unwrap_or(0)
                .max(1),
        ]
    }

    /// Raw intermediate-pass uniforms; display opacity is applied only to the final result.
    pub fn uniforms(&self, surface: &PaintSurface, scale_factor: f32) -> [[f32; 4]; 8] {
        let [width, height] = self.size(surface).map(|value| value as f32);
        [
            [
                width,
                height,
                width * scale_factor / surface.bounds.size.width.0,
                height * scale_factor / surface.bounds.size.height.0,
            ],
            [0.0, 0.0, width, height],
            [0.0, 0.0, width, height],
            self.pass.parameters[0],
            self.pass.parameters[1],
            self.pass.parameters[2],
            self.pass.parameters[3],
            [0.0, 0.0, 1.0, 1.0],
        ]
    }
}

/// Per-paint shader data, separate from the cached program.
#[derive(Clone, Debug)]
pub struct ShaderSurface {
    /// Final invocation and its input graph.
    pub pass: Arc<ShaderPass>,
    /// Logical-to-device pixel scale.
    pub scale_factor: f32,
    /// Inherited element opacity.
    pub opacity: f32,
}

impl ShaderSurface {
    /// Flatten shared inputs once per surface without recursive traversal.
    pub fn graph(&self) -> Result<Vec<ShaderNode>> {
        let mut nodes = Vec::new();
        let mut indices = FxHashMap::default();
        let mut stack = vec![(self.pass.clone(), 0, false)];
        while let Some((pass, downsample, visited)) = stack.pop() {
            let key = (Arc::as_ptr(&pass), downsample);
            if indices.contains_key(&key) {
                continue;
            }
            if !visited {
                stack.push((pass.clone(), downsample, true));
                for input in pass.inputs.iter().rev() {
                    if let Some(ShaderInput::Pass { pass, downsample }) = input {
                        stack.push((pass.clone(), *downsample, false));
                    }
                }
                continue;
            }
            let mut inputs = [None, None];
            for (index, input) in pass.inputs.iter().enumerate() {
                inputs[index] = match input {
                    Some(ShaderInput::Image(image)) => Some(ShaderNodeInput::Image(image.clone())),
                    Some(ShaderInput::Pass { pass, downsample }) => Some(ShaderNodeInput::Pass(
                        *indices
                            .get(&(Arc::as_ptr(pass), *downsample))
                            .context("shader pass dependency is missing")?,
                    )),
                    None => None,
                };
            }
            indices.insert(key, nodes.len());
            nodes.push(ShaderNode {
                pass,
                inputs,
                downsample,
            });
        }
        Ok(nodes)
    }

    /// Uniform layout shared by the native and wgpu renderers.
    pub fn uniforms(
        &self,
        surface: &PaintSurface,
        viewport: [f32; 2],
        linear_color: bool,
        premultiplied_alpha: bool,
    ) -> [[f32; 4]; 8] {
        [
            [
                viewport[0],
                viewport[1],
                self.scale_factor,
                self.scale_factor,
            ],
            [
                surface.bounds.origin.x.0,
                surface.bounds.origin.y.0,
                surface.bounds.size.width.0,
                surface.bounds.size.height.0,
            ],
            [
                surface.content_mask.bounds.origin.x.0,
                surface.content_mask.bounds.origin.y.0,
                surface.content_mask.bounds.right().0,
                surface.content_mask.bounds.bottom().0,
            ],
            self.pass.parameters[0],
            self.pass.parameters[1],
            self.pass.parameters[2],
            self.pass.parameters[3],
            [
                u8::from(linear_color) as f32,
                u8::from(premultiplied_alpha) as f32,
                0.0,
                self.opacity,
            ],
        ]
    }
}
