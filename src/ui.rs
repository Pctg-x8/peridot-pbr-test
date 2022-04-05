use std::sync::Arc;

use crate::{CapturingComponent, ScreenResources};
use bedrock as br;
use br::MemoryBound;
use peridot::{DefaultRenderCommands, ModelData};
use peridot_vg::{FlatPathBuilder, PathBuilder};

pub struct UIRenderingBuffers {
    render_params: peridot_vg::RendererParams,
    buffer: peridot::Buffer,
    transform_buffer_view: br::BufferView,
}
impl UIRenderingBuffers {
    pub fn new(
        e: &peridot::Graphics,
        context: &peridot_vg::Context,
        tfb: &mut peridot::TransferBatch,
    ) -> br::Result<Self> {
        let mut bp = peridot::BufferPrealloc::new(e);
        let offsets = context.prealloc(&mut bp);
        let buffer = bp.build_transferred()?;
        let mreq = buffer.requirements();
        let memory = br::DeviceMemory::allocate(
            e,
            mreq.size as _,
            e.memory_type_manager
                .device_local_index(mreq.memoryTypeBits)
                .expect("no suitable memory for ui rendering meshes")
                .index(),
        )?;
        let buffer = peridot::Buffer::bound(buffer, &Arc::new(memory.into()), 0)?;
        let transform_buffer_view = buffer.create_view(
            br::vk::VK_FORMAT_R32G32B32A32_SFLOAT,
            offsets.transforms_byterange(),
        )?;

        let stg_buffer = bp.build_upload()?;
        let mreq = stg_buffer.requirements();
        let stg_mty = e
            .memory_type_manager
            .host_visible_index(mreq.memoryTypeBits, br::MemoryPropertyFlags::HOST_COHERENT)
            .or_else(|| {
                e.memory_type_manager
                    .host_visible_index(mreq.memoryTypeBits, br::MemoryPropertyFlags::EMPTY)
            })
            .expect("no suitable memory for staging");
        let memory = br::DeviceMemory::allocate(e, mreq.size as _, stg_mty.index())?;
        let mut stg_buffer = peridot::Buffer::bound(stg_buffer, &Arc::new(memory.into()), 0)?;
        let render_params = stg_buffer.guard_map(0..bp.total_size(), move |mem| {
            context.stage_data_into(mem, offsets)
        })?;
        tfb.add_mirroring_buffer(&stg_buffer, &buffer, 0, bp.total_size());
        tfb.add_buffer_graphics_ready(
            br::PipelineStageFlags::VERTEX_INPUT.vertex_shader(),
            &buffer,
            0..bp.total_size(),
            br::AccessFlags::VERTEX_ATTRIBUTE_READ | br::AccessFlags::UNIFORM_READ,
        );

        Ok(Self {
            render_params,
            buffer,
            transform_buffer_view,
        })
    }

    pub fn transform_buffer_view(&self) -> &br::BufferView {
        &self.transform_buffer_view
    }

    pub fn populate_render_commands(
        &self,
        engine: &peridot::Engine<impl peridot::NativeLinker>,
        cmd: &mut br::CmdRecord,
        screen_res: &ScreenResources,
        ui_transform_desc: br::DescriptorSet,
        target_pixels: peridot::math::Vector2F32,
    ) {
        self.render_params.default_render_commands(
            engine,
            cmd,
            &self.buffer,
            peridot_vg::RendererExternalInstances {
                interior_pipeline: &screen_res.vg_interior_inv_pipeline,
                curve_pipeline: &screen_res.vg_curve_inv_pipeline,
                transform_buffer_descriptor_set: ui_transform_desc,
                target_pixels,
            },
        );
    }
}

pub const UI_SLIDER_WIDTH: f32 = 204.0;
pub const UI_SLIDER_HEIGHT: f32 = 32.0;
pub const UI_SLIDER_LABEL_HEIGHT: f32 = 28.0;
pub const UI_SLIDER_VALUE_LABEL_LEFT: f32 = 20.0;
pub const UI_SLIDER_VALUE_LABEL_TOP_OFFSET: f32 = 2.0;
pub const UI_LEFT_MARGIN: f32 = 8.0;

pub const UI_ROUGHNESS_TOP: f32 = 48.0;
pub const UI_ANISOTROPIC_TOP: f32 = 108.0;
pub const UI_METALLIC_TOP: f32 = 168.0;
pub const UI_REFLECTANCE_TOP: f32 = 228.0;

pub trait UIRenderable {
    #[allow(unused_variables)]
    fn render(&self, context: &mut peridot_vg::Context) {}
    #[allow(unused_variables)]
    fn render_mask(&self, context: &mut peridot_vg::Context) {}
    #[allow(unused_variables)]
    fn render_dynamic(&self, context: &mut peridot_vg::Context) {}
    #[allow(unused_variables)]
    fn render_dynamic_mesh(&self, vertices: &mut [peridot::math::Vector2F32]) {}
}

pub struct UIStaticLabel {
    position: peridot::math::Vector2F32,
    text: String,
    font: std::rc::Rc<peridot_vg::Font>,
}
impl UIStaticLabel {
    pub fn new(
        position: peridot::math::Vector2F32,
        text: String,
        font: std::rc::Rc<peridot_vg::Font>,
    ) -> Self {
        Self {
            position,
            text,
            font,
        }
    }
}
impl UIRenderable for UIStaticLabel {
    fn render(&self, context: &mut peridot_vg::Context) {
        context.set_transform(euclid::Transform2D::create_translation(
            self.position.0,
            self.position.1,
        ));
        context
            .text(&self.font, &self.text)
            .expect("Text rendering failed");
    }
}

pub struct UISlider {
    position: peridot::math::Vector2F32,
    size: peridot::math::Vector2F32,
    label_font: std::rc::Rc<peridot_vg::Font>,
    value: f32,
    capture_id: CapturingComponent,
    mesh_vertex_offset: usize,
}
impl UISlider {
    const OUTLINE_THICKNESS: f32 = 2.0;

    pub fn new(
        position: peridot::math::Vector2F32,
        size: peridot::math::Vector2F32,
        label_font: std::rc::Rc<peridot_vg::Font>,
        init_value: f32,
        capture_id: CapturingComponent,
        mesh_vertex_offset: usize,
    ) -> Self {
        Self {
            position,
            size,
            label_font,
            value: init_value,
            capture_id,
            mesh_vertex_offset,
        }
    }

    pub fn try_capture_input(&self, p: peridot::math::Vector2F32) -> Option<CapturingComponent> {
        if (self.position.0..=self.position.0 + self.size.0).contains(&p.0)
            && (-self.position.1..=-self.position.1 + self.size.1).contains(&p.1)
        {
            Some(self.capture_id)
        } else {
            None
        }
    }

    pub fn update_capturing_input(&mut self, e: &peridot::Engine<impl peridot::NativeLinker>) {
        if let Some((px, _)) = e.input().get_plane_position(0) {
            let lx = px - self.position.0;
            self.value = (lx / self.size.0).min(1.0).max(0.0);
        }
    }

    pub fn value(&self) -> f32 {
        self.value
    }
}

impl UIRenderable for UISlider {
    fn render(&self, context: &mut peridot_vg::Context) {
        context.set_transform(euclid::Transform2D::create_translation(
            self.position.0,
            self.position.1,
        ));

        let mut f = context.begin_figure(peridot_vg::FillRule::EvenOdd);
        // outer
        f.move_to(peridot::math::Vector2(8.0, 0.0).into());
        f.line_to(peridot::math::Vector2(UI_SLIDER_WIDTH - 8.0, 0.0).into());
        f.quadratic_bezier_to(
            peridot::math::Vector2(UI_SLIDER_WIDTH, 0.0).into(),
            peridot::math::Vector2(UI_SLIDER_WIDTH, -8.0).into(),
        );
        f.line_to(peridot::math::Vector2(UI_SLIDER_WIDTH, -UI_SLIDER_HEIGHT + 8.0).into());
        f.quadratic_bezier_to(
            peridot::math::Vector2(UI_SLIDER_WIDTH, -UI_SLIDER_HEIGHT).into(),
            peridot::math::Vector2(UI_SLIDER_WIDTH - 8.0, -UI_SLIDER_HEIGHT).into(),
        );
        f.line_to(peridot::math::Vector2(8.0, -UI_SLIDER_HEIGHT).into());
        f.quadratic_bezier_to(
            peridot::math::Vector2(0.0, -UI_SLIDER_HEIGHT).into(),
            peridot::math::Vector2(0.0, -UI_SLIDER_HEIGHT + 8.0).into(),
        );
        f.line_to(peridot::math::Vector2(0.0, -8.0).into());
        f.quadratic_bezier_to(
            peridot::math::Vector2(0.0, 0.0).into(),
            peridot::math::Vector2(8.0, 0.0).into(),
        );
        // inner
        f.move_to(peridot::math::Vector2(8.0, -Self::OUTLINE_THICKNESS).into());
        f.line_to(peridot::math::Vector2(UI_SLIDER_WIDTH - 8.0, -Self::OUTLINE_THICKNESS).into());
        f.quadratic_bezier_to(
            peridot::math::Vector2(
                UI_SLIDER_WIDTH - Self::OUTLINE_THICKNESS,
                -Self::OUTLINE_THICKNESS,
            )
            .into(),
            peridot::math::Vector2(UI_SLIDER_WIDTH - Self::OUTLINE_THICKNESS, -8.0).into(),
        );
        f.line_to(
            peridot::math::Vector2(
                UI_SLIDER_WIDTH - Self::OUTLINE_THICKNESS,
                -UI_SLIDER_HEIGHT + 8.0,
            )
            .into(),
        );
        f.quadratic_bezier_to(
            peridot::math::Vector2(
                UI_SLIDER_WIDTH - Self::OUTLINE_THICKNESS,
                -UI_SLIDER_HEIGHT + Self::OUTLINE_THICKNESS,
            )
            .into(),
            peridot::math::Vector2(
                UI_SLIDER_WIDTH - 8.0,
                -UI_SLIDER_HEIGHT + Self::OUTLINE_THICKNESS,
            )
            .into(),
        );
        f.line_to(peridot::math::Vector2(8.0, -UI_SLIDER_HEIGHT + Self::OUTLINE_THICKNESS).into());
        f.quadratic_bezier_to(
            peridot::math::Vector2(
                Self::OUTLINE_THICKNESS,
                -UI_SLIDER_HEIGHT + Self::OUTLINE_THICKNESS,
            )
            .into(),
            peridot::math::Vector2(Self::OUTLINE_THICKNESS, -UI_SLIDER_HEIGHT + 8.0).into(),
        );
        f.line_to(peridot::math::Vector2(Self::OUTLINE_THICKNESS, -8.0).into());
        f.quadratic_bezier_to(
            peridot::math::Vector2(Self::OUTLINE_THICKNESS, -Self::OUTLINE_THICKNESS).into(),
            peridot::math::Vector2(8.0, -Self::OUTLINE_THICKNESS).into(),
        );
        f.end();
    }

    fn render_mask(&self, context: &mut peridot_vg::Context) {
        context.set_transform(euclid::Transform2D::create_translation(
            self.position.0,
            self.position.1,
        ));

        let mut f = context.begin_figure(peridot_vg::FillRule::Winding);
        // mask
        f.move_to(peridot::math::Vector2(8.0, -Self::OUTLINE_THICKNESS).into());
        f.line_to(peridot::math::Vector2(UI_SLIDER_WIDTH - 8.0, -Self::OUTLINE_THICKNESS).into());
        f.quadratic_bezier_to(
            peridot::math::Vector2(
                UI_SLIDER_WIDTH - Self::OUTLINE_THICKNESS,
                -Self::OUTLINE_THICKNESS,
            )
            .into(),
            peridot::math::Vector2(UI_SLIDER_WIDTH - Self::OUTLINE_THICKNESS, -8.0).into(),
        );
        f.line_to(
            peridot::math::Vector2(
                UI_SLIDER_WIDTH - Self::OUTLINE_THICKNESS,
                -UI_SLIDER_HEIGHT + 8.0,
            )
            .into(),
        );
        f.quadratic_bezier_to(
            peridot::math::Vector2(
                UI_SLIDER_WIDTH - Self::OUTLINE_THICKNESS,
                -UI_SLIDER_HEIGHT + Self::OUTLINE_THICKNESS,
            )
            .into(),
            peridot::math::Vector2(
                UI_SLIDER_WIDTH - 8.0,
                -UI_SLIDER_HEIGHT + Self::OUTLINE_THICKNESS,
            )
            .into(),
        );
        f.line_to(peridot::math::Vector2(8.0, -UI_SLIDER_HEIGHT + Self::OUTLINE_THICKNESS).into());
        f.quadratic_bezier_to(
            peridot::math::Vector2(
                Self::OUTLINE_THICKNESS,
                -UI_SLIDER_HEIGHT + Self::OUTLINE_THICKNESS,
            )
            .into(),
            peridot::math::Vector2(Self::OUTLINE_THICKNESS, -UI_SLIDER_HEIGHT + 8.0).into(),
        );
        f.line_to(peridot::math::Vector2(Self::OUTLINE_THICKNESS, -8.0).into());
        f.quadratic_bezier_to(
            peridot::math::Vector2(Self::OUTLINE_THICKNESS, -Self::OUTLINE_THICKNESS).into(),
            peridot::math::Vector2(8.0, -Self::OUTLINE_THICKNESS).into(),
        );
        f.end();
    }

    fn render_dynamic(&self, context: &mut peridot_vg::Context) {
        context.set_transform(euclid::Transform2D::create_translation(
            self.position.0 + 16.0,
            self.position.1 - UI_SLIDER_VALUE_LABEL_TOP_OFFSET,
        ));
        context
            .text(&self.label_font, &format!("{:.2}", self.value))
            .expect("Text rendering failed");
    }

    fn render_dynamic_mesh(&self, vertices: &mut [peridot::math::Vector2F32]) {
        vertices[self.mesh_vertex_offset + 0] =
            peridot::math::Vector2(self.position.0, -self.position.1);
        vertices[self.mesh_vertex_offset + 1] =
            peridot::math::Vector2(self.position.0 + self.size.0 * self.value, -self.position.1);
        vertices[self.mesh_vertex_offset + 2] =
            peridot::math::Vector2(self.position.0, -self.position.1 + self.size.1);
        vertices[self.mesh_vertex_offset + 3] = peridot::math::Vector2(
            self.position.0 + self.size.0 * self.value,
            -self.position.1 + self.size.1,
        );
    }
}
