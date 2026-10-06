use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use anyhow::{Context as _, Result, anyhow, ensure};

use crate::{PaintSurface, SharedString};

/// A validated portable fragment shader. Define
/// `fn paint(position: vec2<f32>, size: vec2<f32>, parameters: array<vec4<f32>, 4>) -> vec4<f32>`.
/// Coordinates are logical pixels; return straight-alpha sRGB. GPUI supplies clipping,
/// element opacity, color conversion and the vertex stage. Resource bindings and additional
/// entry points are not supported.
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
                == 1,
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

/// Per-paint shader data, separate from the cached program.
#[derive(Clone, Debug)]
pub struct ShaderSurface {
    /// Validated shader shared across paints.
    pub shader: Arc<CustomShader>,
    /// Four application-defined vectors, unchanged by opacity or DPI scaling.
    pub parameters: [[f32; 4]; 4],
    /// Logical-to-device pixel scale.
    pub scale_factor: f32,
    /// Inherited element opacity.
    pub opacity: f32,
}

impl ShaderSurface {
    /// Uniform layout shared by the native and wgpu renderers.
    pub fn uniforms(
        &self,
        surface: &PaintSurface,
        viewport: [f32; 2],
        linear_color: bool,
        premultiplied_alpha: bool,
    ) -> [[f32; 4]; 8] {
        [
            [viewport[0], viewport[1], self.scale_factor, self.opacity],
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
            self.parameters[0],
            self.parameters[1],
            self.parameters[2],
            self.parameters[3],
            [
                u8::from(linear_color) as f32,
                u8::from(premultiplied_alpha) as f32,
                0.0,
                0.0,
            ],
        ]
    }
}
