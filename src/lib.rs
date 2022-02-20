use std::rc::Rc;
use std::sync::Arc;

use bedrock as br;
use br::{MemoryBound, VkHandle};
use peridot::math::One;
use peridot::{self, DefaultRenderCommands, ModelData};
use peridot_vertex_processing_pack::*;
use peridot_vg::{FlatPathBuilder, PathBuilder};

mod mesh;
mod staging;
use self::staging::DynamicStagingBuffer;

#[repr(C)]
pub struct ObjectTransform {
    mvp: peridot::math::Matrix4F32,
    model_transform: peridot::math::Matrix4F32,
}
#[repr(C)]
pub struct RasterizationCameraInfo {
    pos: peridot::math::Vector4F32,
}
#[repr(C)]
pub struct RasterizationDirectionalLightInfo {
    dir: peridot::math::Vector4F32,
    intensity: peridot::math::Vector4F32,
}
#[repr(C)]
#[derive(Clone, Debug)]
pub struct MaterialInfo {
    base_color: peridot::math::Vector4F32,
    roughness: f32,
    anisotropic: f32,
    metallic: f32,
    reflectance: f32,
}

pub struct Descriptors {
    _pool: br::DescriptorPool,
    descriptors: Vec<br::DescriptorSet>,
}
impl Descriptors {
    pub fn new(
        e: &peridot::Engine<impl peridot::NativeLinker>,
        const_res: &ConstResources,
    ) -> Self {
        let mut dp = br::DescriptorPool::new(
            e.graphics_device(),
            8,
            &[
                br::DescriptorPoolSize(br::DescriptorType::UniformBuffer, 6),
                br::DescriptorPoolSize(br::DescriptorType::UniformTexelBuffer, 3),
            ],
            false,
        )
        .expect("Failed to create descriptor pool");
        let descriptors = dp
            .alloc(&[
                &const_res.dsl_ub1,
                &const_res.dsl_ub1,
                &const_res.dsl_ub2_f,
                &const_res.dsl_ub1_f,
                &const_res.dsl_utb1,
                &const_res.dsl_utb1,
                &const_res.dsl_utb1,
                &const_res.dsl_ub1,
            ])
            .expect("Failed to allocate descriptors");

        Self {
            _pool: dp,
            descriptors,
        }
    }

    pub fn grid_transform(&self) -> br::DescriptorSet {
        self.descriptors[0]
    }

    pub fn object_transform(&self) -> br::DescriptorSet {
        self.descriptors[1]
    }

    pub fn rasterization_scene_info(&self) -> br::DescriptorSet {
        self.descriptors[2]
    }

    pub fn material_info(&self) -> br::DescriptorSet {
        self.descriptors[3]
    }

    pub fn ui_transform_buffer(&self) -> br::DescriptorSet {
        self.descriptors[4]
    }

    pub fn ui_mask_transform_buffer(&self) -> br::DescriptorSet {
        self.descriptors[5]
    }

    pub fn ui_dynamic_transform_buffer(&self) -> br::DescriptorSet {
        self.descriptors[6]
    }

    pub fn ui_fill_rect_transform(&self) -> br::DescriptorSet {
        self.descriptors[7]
    }
}

pub struct ConstResources {
    render_pass: br::RenderPass,
    dsl_ub1: br::DescriptorSetLayout,
    dsl_ub1_f: br::DescriptorSetLayout,
    dsl_ub2_f: br::DescriptorSetLayout,
    dsl_utb1: br::DescriptorSetLayout,
    unlit_colored_shader: PvpShaderModules<'static>,
    unlit_colored_pipeline_layout: Arc<br::PipelineLayout>,
    pbr_shader: PvpShaderModules<'static>,
    pbr_pipeline_layout: Arc<br::PipelineLayout>,
    vg_interior_color_fixed_shader: PvpShaderModules<'static>,
    vg_curve_color_fixed_shader: PvpShaderModules<'static>,
    vg_pipeline_layout: Arc<br::PipelineLayout>,
    unlit_colored_ext_shader: PvpShaderModules<'static>,
    unlit_colored_ext_pipeline: Arc<br::PipelineLayout>,
}
impl ConstResources {
    const RENDER_STENCIL_PREPASS: u32 = 0;
    const RENDER_MAIN_PASS: u32 = 1;

    pub fn new(e: &peridot::Engine<impl peridot::NativeLinker>) -> Self {
        let main_attachment = br::AttachmentDescription::new(
            e.backbuffer_format(),
            e.requesting_backbuffer_layout().0,
            e.requesting_backbuffer_layout().0,
        )
        .load_op(br::LoadOp::Clear)
        .store_op(br::StoreOp::Store);
        let depth_attachment = br::AttachmentDescription::new(
            br::vk::VK_FORMAT_D24_UNORM_S8_UINT,
            br::ImageLayout::DepthStencilAttachmentOpt,
            br::ImageLayout::DepthStencilAttachmentOpt,
        )
        .load_op(br::LoadOp::Clear)
        .store_op(br::StoreOp::DontCare)
        .stencil_load_op(br::LoadOp::Clear)
        .stencil_store_op(br::StoreOp::DontCare);
        let stencil_prepass = br::SubpassDescription::new()
            .depth_stencil(1, br::ImageLayout::DepthStencilAttachmentOpt);
        let main_pass = br::SubpassDescription::new()
            .add_color_output(0, br::ImageLayout::ColorAttachmentOpt, None)
            .depth_stencil(1, br::ImageLayout::DepthStencilAttachmentOpt);
        let passdep = br::vk::VkSubpassDependency {
            srcSubpass: br::vk::VK_SUBPASS_EXTERNAL,
            dstSubpass: 1,
            srcStageMask: br::PipelineStageFlags::TOP_OF_PIPE.0,
            dstStageMask: br::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.0,
            srcAccessMask: 0,
            dstAccessMask: br::AccessFlags::COLOR_ATTACHMENT.write,
            dependencyFlags: br::vk::VK_DEPENDENCY_BY_REGION_BIT,
        };
        let stencil_prepass_dep = br::vk::VkSubpassDependency {
            srcSubpass: 0,
            dstSubpass: 1,
            srcStageMask: br::PipelineStageFlags::LATE_FRAGMENT_TESTS.0,
            dstStageMask: br::PipelineStageFlags::EARLY_FRAGMENT_TESTS.0,
            srcAccessMask: br::AccessFlags::DEPTH_STENCIL_ATTACHMENT.write,
            dstAccessMask: br::AccessFlags::DEPTH_STENCIL_ATTACHMENT.read,
            dependencyFlags: br::vk::VK_DEPENDENCY_BY_REGION_BIT,
        };
        let render_pass = br::RenderPassBuilder::new()
            .add_attachments([main_attachment, depth_attachment])
            .add_subpasses([stencil_prepass, main_pass])
            .add_dependencies([stencil_prepass_dep, passdep])
            .create(e.graphics_device())
            .expect("Failed to create render pass");

        let dsl_ub1 = br::DescriptorSetLayout::new(
            e.graphics_device(),
            &[br::DescriptorSetLayoutBinding::UniformBuffer(
                1,
                br::ShaderStage::VERTEX,
            )],
        )
        .expect("Failed to create descriptor set layout");
        let dsl_ub1_f = br::DescriptorSetLayout::new(
            e.graphics_device(),
            &[br::DescriptorSetLayoutBinding::UniformBuffer(
                1,
                br::ShaderStage::FRAGMENT,
            )],
        )
        .expect("Failed to create ub1f descriptor set layout");
        let dsl_ub2_f = br::DescriptorSetLayout::new(
            e.graphics_device(),
            &[
                br::DescriptorSetLayoutBinding::UniformBuffer(1, br::ShaderStage::FRAGMENT),
                br::DescriptorSetLayoutBinding::UniformBuffer(1, br::ShaderStage::FRAGMENT),
            ],
        )
        .expect("Failed to create ub2f descriptor set layout");
        let dsl_utb1 = br::DescriptorSetLayout::new(
            e.graphics_device(),
            &[br::DescriptorSetLayoutBinding::UniformTexelBuffer(
                1,
                br::ShaderStage::VERTEX,
            )],
        )
        .expect("Failed to create utb1 descriptor set layout");

        let unlit_colored_shader = PvpShaderModules::new(
            e.graphics_device(),
            e.load("shaders.unlit_colored")
                .expect("Failed to load unlit_colored shader"),
        )
        .expect("Failed to create shader modules");
        let unlit_colored_pipeline_layout =
            br::PipelineLayout::new(e.graphics_device(), &[&dsl_ub1], &[])
                .expect("Failed to create unlit_colored pipeline layout")
                .into();

        let pbr_shader = PvpShaderModules::new(
            e.graphics_device(),
            e.load("shaders.pbr").expect("Failed to load pbr shader"),
        )
        .expect("Failed to create pbr shader modules");
        let pbr_pipeline_layout = br::PipelineLayout::new(
            e.graphics_device(),
            &[&dsl_ub1, &dsl_ub2_f, &dsl_ub1_f],
            &[],
        )
        .expect("Failed to create pbr pipeline layout")
        .into();
        let unlit_colored_ext_shader = PvpShaderModules::new(
            e.graphics_device(),
            e.load("shaders.unlit_colored_ext")
                .expect("Failed to load unlit_colored_ext shader"),
        )
        .expect("Failed to create unlit_colored_ext shader modules");
        let unlit_colored_ext_pipeline = br::PipelineLayout::new(
            e.graphics_device(),
            &[&dsl_ub1],
            &[(br::ShaderStage::FRAGMENT, 0..16)],
        )
        .expect("Failed to create unlit_colored_ext pipeline layout")
        .into();

        let vg_interior_color_fixed_shader = PvpShaderModules::new(
            e.graphics_device(),
            e.load("shaders.vg.interiorColorFixed")
                .expect("Failed to load vg interior color shader"),
        )
        .expect("Failed to create vg interior color shader modules");
        let vg_curve_color_fixed_shader = PvpShaderModules::new(
            e.graphics_device(),
            e.load("shaders.vg.curveColorFixed")
                .expect("Failed to load vg curve color shader"),
        )
        .expect("Failed to create vg curve color shader modules");
        let vg_pipeline_layout = br::PipelineLayout::new(
            e.graphics_device(),
            &[&dsl_utb1],
            &[(br::ShaderStage::VERTEX, 0..4 * 4)],
        )
        .expect("Failed to create vg pipeline layout")
        .into();

        Self {
            render_pass,
            dsl_ub1,
            dsl_ub1_f,
            dsl_ub2_f,
            dsl_utb1,
            unlit_colored_shader,
            unlit_colored_pipeline_layout,
            pbr_shader,
            pbr_pipeline_layout,
            vg_interior_color_fixed_shader,
            vg_curve_color_fixed_shader,
            vg_pipeline_layout,
            unlit_colored_ext_shader,
            unlit_colored_ext_pipeline,
        }
    }
}

pub struct ScreenResources {
    depth_texture: peridot::Image,
    depth_texture_view: br::ImageView,
    frame_buffers: Vec<br::Framebuffer>,
    grid_render_pipeline: peridot::LayoutedPipeline,
    pbr_pipeline: peridot::LayoutedPipeline,
    vg_interior_pipeline: peridot::LayoutedPipeline,
    vg_curve_pipeline: peridot::LayoutedPipeline,
    vg_interior_inv_pipeline: peridot::LayoutedPipeline,
    vg_curve_inv_pipeline: peridot::LayoutedPipeline,
    vg_interior_mask_pipeline: peridot::LayoutedPipeline,
    vg_curve_mask_pipeline: peridot::LayoutedPipeline,
    ui_fill_rect_pipeline: peridot::LayoutedPipeline,
    ui_border_line_pipeline: peridot::LayoutedPipeline,
}
impl ScreenResources {
    pub fn new(
        e: &mut peridot::Engine<impl peridot::NativeLinker>,
        const_res: &ConstResources,
    ) -> Self {
        let bb0 = e.backbuffer(0).expect("no backbuffers?");

        let depth_image = br::ImageDesc::new(
            AsRef::<br::vk::VkExtent2D>::as_ref(bb0.size()),
            br::vk::VK_FORMAT_D24_UNORM_S8_UINT,
            br::ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            br::ImageLayout::Undefined,
        )
        .create(e.graphics_device())
        .expect("Failed to create depth image object");
        let mreq = depth_image.requirements();
        let depth_mem = br::DeviceMemory::allocate(
            e.graphics_device(),
            mreq.size as _,
            e.graphics()
                .memory_type_manager
                .device_local_index(mreq.memoryTypeBits)
                .expect("No suitable memory for depth buffer")
                .index(),
        )
        .expect("Failed to allocate depth buffer memory");
        let depth_texture = peridot::Image::bound(depth_image, &Arc::new(depth_mem.into()), 0)
            .expect("Failed to bind depth buffer memory");
        let depth_texture_view = depth_texture
            .create_view(
                None,
                None,
                &br::ComponentMapping::default(),
                &br::ImageSubresourceRange::depth_stencil(0..1, 0..1),
            )
            .expect("Failed to create depth buffer view");
        e.submit_commands(|r| {
            let image_barriers = [br::ImageMemoryBarrier::new_raw(
                &depth_texture,
                &br::ImageSubresourceRange::depth_stencil(0..1, 0..1),
                br::ImageLayout::Undefined,
                br::ImageLayout::DepthStencilAttachmentOpt,
            )];

            r.pipeline_barrier(
                br::PipelineStageFlags::TOP_OF_PIPE,
                br::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                true,
                &[],
                &[],
                &image_barriers,
            );
        })
        .expect("Failed to submit initial barrier");

        let frame_buffers: Vec<_> = e
            .iter_backbuffers()
            .map(|b| {
                br::Framebuffer::new(
                    &const_res.render_pass,
                    &[&b, &depth_texture_view],
                    b.size().as_ref(),
                    1,
                )
                .expect("Failed to create framebuffer")
            })
            .collect();

        let area = AsRef::<br::vk::VkExtent2D>::as_ref(bb0.size())
            .clone()
            .into_rect(br::vk::VkOffset2D { x: 0, y: 0 });
        let viewport = br::vk::VkViewport::from_rect_with_depth_range(&area, 0.0..1.0);

        let unlit_colored_vps = const_res
            .unlit_colored_shader
            .generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
        let mut pb = br::GraphicsPipelineBuilder::new(
            &const_res.unlit_colored_pipeline_layout,
            (&const_res.render_pass, 1),
            unlit_colored_vps,
        );
        pb.viewport_scissors(
            br::DynamicArrayState::Static(&[viewport]),
            br::DynamicArrayState::Static(&[area]),
        );
        pb.add_attachment_blend(br::AttachmentColorBlendState::premultiplied());
        pb.multisample_state(Some(br::MultisampleState::new()));
        pb.depth_test_settings(Some(br::CompareOp::LessOrEqual), true);
        let grid_render_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(e.graphics_device(), None)
                .expect("Failed to create grid render pipeline"),
            &const_res.unlit_colored_pipeline_layout,
        );

        let pbr_vps = const_res
            .pbr_shader
            .generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        pb.layout(&const_res.pbr_pipeline_layout)
            .vertex_processing(pbr_vps);
        let pbr_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(e.graphics_device(), None)
                .expect("Failed to create pbr pipeline"),
            &const_res.pbr_pipeline_layout,
        );

        let stencil_simple_matching = br::vk::VkStencilOpState {
            compareOp: br::CompareOp::Equal as _,
            compareMask: 0x01,
            writeMask: 0,
            reference: 1,
            failOp: br::StencilOp::Keep as _,
            depthFailOp: br::StencilOp::Keep as _,
            passOp: br::StencilOp::Keep as _,
        };
        let ui_fill_rect_vps = const_res
            .unlit_colored_ext_shader
            .generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        pb.layout(&const_res.unlit_colored_ext_pipeline)
            .vertex_processing(ui_fill_rect_vps)
            .stencil_test_enable(true)
            .stencil_control_front(stencil_simple_matching.clone())
            .stencil_control_back(stencil_simple_matching);
        let ui_fill_rect_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(e.graphics_device(), None)
                .expect("Failed to create ui_fill_rect pipeline"),
            &const_res.unlit_colored_ext_pipeline,
        );
        let ui_border_line_vps = const_res
            .unlit_colored_ext_shader
            .generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
        pb.vertex_processing(ui_border_line_vps);
        let ui_border_line_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(e.graphics_device(), None)
                .expect("Failed to create ui_border_line pipeline"),
            &const_res.unlit_colored_ext_pipeline,
        );

        pb.stencil_test_enable(false);
        let mut vg_interior_vps = const_res
            .vg_interior_color_fixed_shader
            .generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        let mut vg_curve_vps = const_res
            .vg_curve_color_fixed_shader
            .generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        let spc_map = &[
            br::vk::VkSpecializationMapEntry {
                constantID: 0,
                offset: 0,
                size: 4,
            },
            br::vk::VkSpecializationMapEntry {
                constantID: 1,
                offset: 4,
                size: 4,
            },
        ];
        vg_interior_vps.vertex_shader_mut().specinfo = Some((
            std::borrow::Cow::Borrowed(spc_map),
            br::DynamicDataCell::from_slice(&peridot_vg::renderer_pivot::LEFT_TOP),
        ));
        vg_curve_vps.vertex_shader_mut().specinfo = Some((
            std::borrow::Cow::Borrowed(spc_map),
            br::DynamicDataCell::from_slice(&peridot_vg::renderer_pivot::LEFT_TOP),
        ));
        pb.layout(&const_res.vg_pipeline_layout);
        pb.vertex_processing(vg_interior_vps.clone());
        let vg_interior_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(e.graphics_device(), None)
                .expect("Failed to create vg interior pipeline"),
            &const_res.vg_pipeline_layout,
        );
        pb.vertex_processing(vg_curve_vps.clone());
        let vg_curve_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(e.graphics_device(), None)
                .expect("Failed to create vg curve pipeline"),
            &const_res.vg_pipeline_layout,
        );

        pb.clear_attachment_blends();
        let mut inv_blend = br::AttachmentColorBlendState::noblend();
        inv_blend
            .enable()
            .color_blend(
                br::BlendFactor::OneMinusDestColor,
                br::BlendOp::Add,
                br::BlendFactor::Zero,
            )
            .alpha_blend(
                br::BlendFactor::Zero,
                br::BlendOp::Add,
                br::BlendFactor::One,
            );
        pb.add_attachment_blend(inv_blend);
        pb.vertex_processing(vg_interior_vps.clone());
        let vg_interior_inv_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(e.graphics_device(), None)
                .expect("Failed to create vg interior inv color pipeline"),
            &const_res.vg_pipeline_layout,
        );
        pb.vertex_processing(vg_curve_vps);
        let vg_curve_inv_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(e.graphics_device(), None)
                .expect("Failed to create vg curve inv color pipeline"),
            &const_res.vg_pipeline_layout,
        );

        let stencil_simple_write = br::vk::VkStencilOpState {
            compareOp: br::CompareOp::Always as _,
            compareMask: 0xffffffff,
            writeMask: 0x01,
            reference: 1,
            failOp: br::StencilOp::Keep as _,
            depthFailOp: br::StencilOp::Keep as _,
            passOp: br::StencilOp::Replace as _,
        };
        pb.render_pass(&const_res.render_pass, 0);
        pb.depth_write_enable(false);
        pb.stencil_test_enable(true);
        pb.stencil_control_front(stencil_simple_write.clone());
        pb.stencil_control_back(stencil_simple_write);
        let vg_curve_mask_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(e.graphics_device(), None)
                .expect("Failed to create vg curve mask pipeline"),
            &const_res.vg_pipeline_layout,
        );
        pb.layout(&const_res.vg_pipeline_layout);
        pb.vertex_processing(vg_interior_vps);
        let vg_interior_mask_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(e.graphics_device(), None)
                .expect("Failed to create vg interior pipeline"),
            &const_res.vg_pipeline_layout,
        );

        Self {
            depth_texture,
            depth_texture_view,
            frame_buffers,
            grid_render_pipeline,
            pbr_pipeline,
            vg_interior_pipeline,
            vg_curve_pipeline,
            vg_interior_inv_pipeline,
            vg_curve_inv_pipeline,
            vg_interior_mask_pipeline,
            vg_curve_mask_pipeline,
            ui_fill_rect_pipeline,
            ui_border_line_pipeline,
        }
    }
}

pub struct MutableBufferOffsets {
    grid_mvp: u64,
    object_mvp: u64,
    camera_info: u64,
    directional_light_info: u64,
    material: u64,
    ui_fill_rects: u64,
    ui_transform: u64,
}
pub struct StaticBufferOffsets {
    grid: u64,
    icosphere_vertices: u64,
    icosphere_indices: u64,
    ui: peridot_vg::ContextPreallocOffsets,
    ui_mask: peridot_vg::ContextPreallocOffsets,
    ui_border_line_indices: u64,
    ui_fill_rect_indices: u64,
}
pub struct StaticBufferInitializer<'o> {
    offsets: &'o StaticBufferOffsets,
    icosphere: &'o mesh::UnitIcosphere,
    ui: &'o peridot_vg::Context,
    ui_mask: &'o peridot_vg::Context,
    ui_render_params: Option<peridot_vg::RendererParams>,
    ui_mask_render_params: Option<peridot_vg::RendererParams>,
}
impl peridot::FixedBufferInitializer for StaticBufferInitializer<'_> {
    fn stage_data(&mut self, m: &br::MappedMemoryRange) {
        unsafe {
            let grid_range = m.slice_mut::<[peridot::ColoredVertex; 2]>(
                self.offsets.grid as _,
                mesh::GRID_MESH_LINE_COUNT,
            );
            mesh::build_grid_mesh_into(grid_range);

            m.slice_mut(
                self.offsets.icosphere_vertices as _,
                self.icosphere.vertices.len(),
            )
            .clone_from_slice(&self.icosphere.vertices);
            m.slice_mut(
                self.offsets.icosphere_indices as _,
                self.icosphere.indices.len(),
            )
            .clone_from_slice(&self.icosphere.indices);
            self.ui_render_params = Some(self.ui.stage_data_into(m, self.offsets.ui.clone()));
            self.ui_mask_render_params = Some(
                self.ui_mask
                    .stage_data_into(m, self.offsets.ui_mask.clone()),
            );

            m.slice_mut(
                self.offsets.ui_border_line_indices as _,
                mesh::UI_FILL_RECT_BORDER_INDEX_COUNT,
            )
            .copy_from_slice(mesh::UI_FILL_RECT_BORDER_INDICES);
            m.slice_mut(
                self.offsets.ui_fill_rect_indices as _,
                mesh::UI_FILL_RECT_INDEX_COUNT,
            )
            .copy_from_slice(mesh::UI_FILL_RECT_INDICES);
        }
    }

    fn buffer_graphics_ready(
        &self,
        tfb: &mut peridot::TransferBatch,
        buf: &peridot::Buffer,
        range: std::ops::Range<u64>,
    ) {
        tfb.add_buffer_graphics_ready(
            br::PipelineStageFlags::VERTEX_INPUT,
            buf,
            self.offsets.grid..range.end,
            br::AccessFlags::VERTEX_ATTRIBUTE_READ,
        );
    }
}

pub struct UpdateSets {
    grid_mvp_stg_offset: Option<u64>,
    object_mvp_stg_offset: Option<u64>,
    camera_info_stg_offset: Option<u64>,
    directional_light_info_stg_offset: Option<u64>,
    material_stg_offset: Option<u64>,
    ui_fill_rects: Option<u64>,
    ui_transform: Option<u64>,
}
impl UpdateSets {
    pub fn new() -> Self {
        Self {
            grid_mvp_stg_offset: None,
            object_mvp_stg_offset: None,
            camera_info_stg_offset: None,
            directional_light_info_stg_offset: None,
            material_stg_offset: None,
            ui_fill_rects: None,
            ui_transform: None,
        }
    }
}

pub struct Memory {
    mem: peridot::FixedMemory,
    static_offsets: StaticBufferOffsets,
    mutable_offsets: MutableBufferOffsets,
    dynamic_stg: DynamicStagingBuffer,
    update_sets: UpdateSets,
    icosphere_vertex_count: usize,
    ui_render_params: peridot_vg::RendererParams,
    ui_mask_render_params: peridot_vg::RendererParams,
    ui_transform_buffer_view: br::BufferView,
    ui_mask_transform_buffer_view: br::BufferView,
}
impl Memory {
    pub fn new(
        e: &peridot::Engine<impl peridot::NativeLinker>,
        tfb: &mut peridot::TransferBatch,
        ui: &peridot_vg::Context,
        ui_mask: &peridot_vg::Context,
    ) -> Self {
        let icosphere = mesh::UnitIcosphere::base()
            .subdivide()
            .subdivide()
            .subdivide();

        let mut static_bp = peridot::BufferPrealloc::new(e.graphics());
        let offsets = StaticBufferOffsets {
            grid: static_bp.add(peridot::BufferContent::vertices::<
                [peridot::ColoredVertex; 2],
            >(mesh::GRID_MESH_LINE_COUNT)),
            icosphere_vertices: static_bp.add(peridot::BufferContent::vertices::<
                mesh::VertexWithNormals,
            >(icosphere.vertices.len())),
            icosphere_indices: static_bp.add(peridot::BufferContent::indices::<u16>(
                icosphere.indices.len(),
            )),
            ui_border_line_indices: static_bp.add(peridot::BufferContent::index::<
                [u16; mesh::UI_FILL_RECT_BORDER_INDEX_COUNT],
            >()),
            ui_fill_rect_indices: static_bp.add(peridot::BufferContent::index::<
                [u16; mesh::UI_FILL_RECT_INDEX_COUNT],
            >()),
            ui: ui.prealloc(&mut static_bp),
            ui_mask: ui_mask.prealloc(&mut static_bp),
        };
        let textures = peridot::TextureInitializationGroup::new(e.graphics_device());
        let mut mutable_bp = peridot::BufferPrealloc::new(e.graphics());
        let mutable_offsets = MutableBufferOffsets {
            grid_mvp: mutable_bp
                .add(peridot::BufferContent::uniform::<peridot::math::Matrix4F32>()),
            object_mvp: mutable_bp.add(peridot::BufferContent::uniform::<ObjectTransform>()),
            camera_info: mutable_bp
                .add(peridot::BufferContent::uniform::<RasterizationCameraInfo>()),
            directional_light_info: mutable_bp.add(peridot::BufferContent::uniform::<
                RasterizationDirectionalLightInfo,
            >()),
            material: mutable_bp.add(peridot::BufferContent::uniform::<MaterialInfo>()),
            ui_transform: mutable_bp
                .add(peridot::BufferContent::uniform::<peridot::math::Matrix4F32>()),
            ui_fill_rects: mutable_bp.add(peridot::BufferContent::vertex::<
                [peridot::math::Vector2F32; mesh::UI_FILL_RECT_COUNT],
            >()),
        };

        let mut initializer = StaticBufferInitializer {
            offsets: &offsets,
            icosphere: &icosphere,
            ui: &ui,
            ui_mask: &ui_mask,
            ui_render_params: None,
            ui_mask_render_params: None,
        };
        let mem = peridot::FixedMemory::new(
            e.graphics(),
            static_bp,
            mutable_bp,
            textures,
            &mut initializer,
            tfb,
        )
        .expect("Failed to initialize fixed memory");
        let ui_render_params = initializer.ui_render_params.unwrap();
        let ui_mask_render_params = initializer.ui_mask_render_params.unwrap();

        Self {
            ui_transform_buffer_view: mem
                .buffer
                .0
                .create_view(
                    br::vk::VK_FORMAT_R32G32B32A32_SFLOAT,
                    ui_render_params.transforms_byterange(),
                )
                .expect("Failed to create buffer view of transforms"),
            ui_mask_transform_buffer_view: mem
                .buffer
                .0
                .create_view(
                    br::vk::VK_FORMAT_R32G32B32A32_SFLOAT,
                    ui_mask_render_params.transforms_byterange(),
                )
                .expect("Failed to create buffer view of mask transforms"),
            mem,
            ui_render_params,
            ui_mask_render_params,
            static_offsets: offsets,
            mutable_offsets,
            dynamic_stg: DynamicStagingBuffer::new(e.graphics())
                .expect("Failed to create dynamic staging buffer"),
            update_sets: UpdateSets::new(),
            icosphere_vertex_count: icosphere.indices.len(),
        }
    }

    pub fn apply_main_camera(
        &mut self,
        e: &mut peridot::Graphics,
        camera: &peridot::math::Camera,
        aspect: f32,
    ) {
        let camera_matrix = camera.view_projection_matrix(aspect);

        self.update_sets.grid_mvp_stg_offset =
            Some(self.dynamic_stg.push(e, camera_matrix.clone()));
        self.update_sets.object_mvp_stg_offset = Some(self.dynamic_stg.push(
            e,
            ObjectTransform {
                mvp: camera_matrix,
                model_transform: peridot::math::Matrix4::ONE,
            },
        ));
    }

    pub fn set_camera_info(&mut self, e: &mut peridot::Graphics, info: RasterizationCameraInfo) {
        self.update_sets.camera_info_stg_offset = Some(self.dynamic_stg.push(e, info));
    }

    pub fn set_directional_light_info(
        &mut self,
        e: &mut peridot::Graphics,
        info: RasterizationDirectionalLightInfo,
    ) {
        self.update_sets.directional_light_info_stg_offset = Some(self.dynamic_stg.push(e, info));
    }

    pub fn set_material(&mut self, e: &mut peridot::Graphics, info: MaterialInfo) {
        self.update_sets.material_stg_offset = Some(self.dynamic_stg.push(e, info));
    }

    pub fn update_ui_fill_rect_vertices(
        &mut self,
        e: &mut peridot::Graphics,
        vertices: &[peridot::math::Vector2F32; mesh::UI_FILL_RECT_COUNT],
    ) {
        self.update_sets.ui_fill_rects = Some(self.dynamic_stg.push_multiple_values(e, vertices));
    }

    pub fn construct_new_ui_fill_rect_vertices(
        &mut self,
        e: &mut peridot::Graphics,
        ctor: impl FnMut(&mut [peridot::math::Vector2F32]),
    ) {
        self.update_sets.ui_fill_rects = Some(self.dynamic_stg.construct_multiple_values_inplace(
            e,
            mesh::UI_FILL_RECT_COUNT,
            ctor,
        ));
    }

    pub fn set_ui_transform(
        &mut self,
        e: &mut peridot::Graphics,
        transform: peridot::math::Matrix4F32,
    ) {
        self.update_sets.ui_transform = Some(self.dynamic_stg.push(e, transform));
    }

    pub fn ready_transfer(&mut self, e: &peridot::Graphics, tfb: &mut peridot::TransferBatch) {
        self.dynamic_stg.end_mapped(e);

        if let Some(o) = self.update_sets.grid_mvp_stg_offset.take() {
            let target_offset = self.mem.mut_buffer_placement + self.mutable_offsets.grid_mvp;

            tfb.add_copying_buffer(
                self.dynamic_stg.buffer().with_dev_offset(o),
                self.mem.buffer.0.with_dev_offset(target_offset),
                std::mem::size_of::<peridot::math::Matrix4F32>() as _,
            );
            tfb.add_buffer_graphics_ready(
                br::PipelineStageFlags::VERTEX_SHADER,
                &self.mem.buffer.0,
                self.grid_transform_range(),
                br::AccessFlags::UNIFORM_READ,
            );
        }

        if let Some(o) = self.update_sets.object_mvp_stg_offset.take() {
            let r = self.object_transform_range();

            tfb.add_copying_buffer(
                self.dynamic_stg.buffer().with_dev_offset(o),
                self.mem.buffer.0.with_dev_offset(r.start),
                std::mem::size_of::<ObjectTransform>() as _,
            );
            tfb.add_buffer_graphics_ready(
                br::PipelineStageFlags::VERTEX_SHADER,
                &self.mem.buffer.0,
                r,
                br::AccessFlags::UNIFORM_READ,
            )
        }

        if let Some(o) = self.update_sets.camera_info_stg_offset.take() {
            let r = self.camera_info_range();

            tfb.add_copying_buffer(
                self.dynamic_stg.buffer().with_dev_offset(o),
                self.mem.buffer.0.with_dev_offset(r.start),
                std::mem::size_of::<RasterizationCameraInfo>() as _,
            );
            tfb.add_buffer_graphics_ready(
                br::PipelineStageFlags::FRAGMENT_SHADER,
                &self.mem.buffer.0,
                r,
                br::AccessFlags::UNIFORM_READ,
            );
        }

        if let Some(o) = self.update_sets.directional_light_info_stg_offset.take() {
            let r = self.directional_light_info_range();

            tfb.add_copying_buffer(
                self.dynamic_stg.buffer().with_dev_offset(o),
                self.mem.buffer.0.with_dev_offset(r.start),
                std::mem::size_of::<RasterizationDirectionalLightInfo>() as _,
            );
            tfb.add_buffer_graphics_ready(
                br::PipelineStageFlags::FRAGMENT_SHADER,
                &self.mem.buffer.0,
                r,
                br::AccessFlags::UNIFORM_READ,
            );
        }

        if let Some(o) = self.update_sets.material_stg_offset.take() {
            let r = self.material_range();

            tfb.add_copying_buffer(
                self.dynamic_stg.buffer().with_dev_offset(o),
                self.mem.buffer.0.with_dev_offset(r.start),
                std::mem::size_of::<MaterialInfo>() as _,
            );
            tfb.add_buffer_graphics_ready(
                br::PipelineStageFlags::FRAGMENT_SHADER,
                &self.mem.buffer.0,
                r,
                br::AccessFlags::UNIFORM_READ,
            );
        }

        if let Some(o) = self.update_sets.ui_fill_rects.take() {
            let r = self.ui_fill_rects_range();

            tfb.add_copying_buffer(
                self.dynamic_stg.buffer().with_dev_offset(o),
                self.mem.buffer.0.with_dev_offset(r.start),
                std::mem::size_of::<[peridot::math::Vector2F32; mesh::UI_FILL_RECT_COUNT]>() as _,
            );
            tfb.add_buffer_graphics_ready(
                br::PipelineStageFlags::VERTEX_INPUT,
                &self.mem.buffer.0,
                r,
                br::AccessFlags::VERTEX_ATTRIBUTE_READ,
            );
        }

        if let Some(o) = self.update_sets.ui_transform.take() {
            let r = self.ui_transform_range();

            tfb.add_copying_buffer(
                self.dynamic_stg.buffer().with_dev_offset(o),
                self.mem.buffer.0.with_dev_offset(r.start),
                std::mem::size_of::<peridot::math::Matrix4F32>() as _,
            );
            tfb.add_buffer_graphics_ready(
                br::PipelineStageFlags::VERTEX_SHADER,
                &self.mem.buffer.0,
                r,
                br::AccessFlags::UNIFORM_READ,
            );
        }
    }

    pub fn grid_transform_range(&self) -> std::ops::Range<u64> {
        let target_offset = self.mem.mut_buffer_placement + self.mutable_offsets.grid_mvp;

        target_offset..target_offset + std::mem::size_of::<peridot::math::Matrix4F32>() as u64
    }

    pub fn object_transform_range(&self) -> std::ops::Range<u64> {
        let target_offset = self.mem.mut_buffer_placement + self.mutable_offsets.object_mvp;

        target_offset..target_offset + std::mem::size_of::<ObjectTransform>() as u64
    }

    pub fn camera_info_range(&self) -> std::ops::Range<u64> {
        self.mem.range_in_mut_buffer(
            self.mutable_offsets.camera_info
                ..self.mutable_offsets.camera_info
                    + std::mem::size_of::<RasterizationCameraInfo>() as u64,
        )
    }

    pub fn directional_light_info_range(&self) -> std::ops::Range<u64> {
        self.mem.range_in_mut_buffer(
            self.mutable_offsets.directional_light_info
                ..self.mutable_offsets.directional_light_info
                    + std::mem::size_of::<RasterizationDirectionalLightInfo>() as u64,
        )
    }

    pub fn material_range(&self) -> std::ops::Range<u64> {
        self.mem.range_in_mut_buffer(
            self.mutable_offsets.material
                ..self.mutable_offsets.material + std::mem::size_of::<MaterialInfo>() as u64,
        )
    }

    pub fn ui_fill_rects_range(&self) -> std::ops::Range<u64> {
        self.mem.range_in_mut_buffer(
            self.mutable_offsets.ui_fill_rects
                ..self.mutable_offsets.ui_fill_rects
                    + std::mem::size_of::<[peridot::math::Vector2F32; mesh::UI_FILL_RECT_COUNT]>()
                        as u64,
        )
    }

    pub fn ui_transform_range(&self) -> std::ops::Range<u64> {
        self.mem.range_in_mut_buffer(
            self.mutable_offsets.ui_transform
                ..self.mutable_offsets.ui_transform
                    + std::mem::size_of::<peridot::math::Matrix4F32>() as u64,
        )
    }
}

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
}

pub fn range_cast_u64_usize(r: std::ops::Range<u64>) -> std::ops::Range<usize> {
    r.start as _..r.end as _
}

pub struct DirtyTracker<T> {
    value: T,
    dirty: bool,
}
impl<T> DirtyTracker<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            dirty: false,
        }
    }

    pub fn get(&self) -> &T {
        &self.value
    }

    pub fn modify(&mut self) -> &mut T {
        self.dirty = true;
        &mut self.value
    }

    pub fn take_dirty_flag(&mut self) -> bool {
        let d = self.dirty;
        self.dirty = false;
        d
    }

    pub fn try_set_dirty_neq(&mut self, value: T)
    where
        T: PartialEq,
    {
        if self.value != value {
            *self.modify() = value;
        }
    }
}

pub struct EdgeTrigger<T: Eq> {
    current: T,
}
impl<T: Eq> EdgeTrigger<T> {
    pub fn new(init: T) -> Self {
        Self { current: init }
    }

    pub fn update(&mut self, new_value: T) -> Option<&T> {
        let triggered = self.current != new_value;
        self.current = new_value;
        Some(&self.current).filter(|_| triggered)
    }
}

const ID_PLANE_PRESS: u16 = 0;
const ID_CAMERA_MOVE_AX_X: u8 = 0;
const ID_CAMERA_MOVE_AX_Y: u8 = 1;
const ID_CAMERA_MOVE_AX_Z: u8 = 2;
const ID_CAMERA_ROT_AX_X: u8 = 3;
const ID_CAMERA_ROT_AX_Y: u8 = 4;

#[derive(Clone, Copy)]
pub enum CapturingComponent {
    Roughness,
    Anisotropic,
    Metallic,
    Reflectance,
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

trait UIRenderable {
    #[allow(unused_variables)]
    fn render(&self, context: &mut peridot_vg::Context) {}
    #[allow(unused_variables)]
    fn render_mask(&self, context: &mut peridot_vg::Context) {}
    #[allow(unused_variables)]
    fn render_dynamic(&self, context: &mut peridot_vg::Context) {}
    #[allow(unused_variables)]
    fn render_dynamic_mesh(&self, vertices: &mut [peridot::math::Vector2F32]) {}
}

struct UIStaticLabel {
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

struct UISlider {
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

pub struct RenderBundle {
    pool: br::CommandPool,
    buffers: Vec<br::CommandBuffer>,
}
impl RenderBundle {
    pub fn new(g: &peridot::Graphics, count: u32) -> br::Result<Self> {
        let mut pool = br::CommandPool::new(&g, g.graphics_queue_family_index(), false, false)?;

        Ok(Self {
            buffers: pool.alloc(count, false)?,
            pool,
        })
    }

    pub fn reset(&mut self) -> br::Result<()> {
        self.pool.reset(false)
    }

    pub fn synchronized(&mut self, index: usize) -> br::SynchronizedCommandBuffer {
        unsafe { self.buffers[index].synchronize_with(&mut self.pool) }
    }
}

pub struct Differential<T>
where
    T: std::ops::Sub<T, Output = T> + Copy,
{
    current_value: T,
}
impl<T> Differential<T>
where
    T: std::ops::Sub<T, Output = T> + Copy,
{
    pub fn new(init_value: T) -> Self {
        Self {
            current_value: init_value,
        }
    }

    pub fn update(&mut self, new_value: T) -> T {
        let d = self.current_value - new_value;
        self.current_value = new_value;
        d
    }
}

pub struct FreeCameraView {
    xrot: DirtyTracker<f32>,
    yrot: DirtyTracker<f32>,
    rot_x_input_diff: Differential<f32>,
    rot_y_input_diff: Differential<f32>,
    camera: DirtyTracker<(peridot::math::Camera, f32)>,
}
impl FreeCameraView {
    pub fn new(
        init_xrot: f32,
        init_yrot: f32,
        init_camera_distance: f32,
        init_aspect_value: f32,
    ) -> Self {
        let init_rot = peridot::math::Quaternion::new(init_yrot, peridot::math::Vector3::RIGHT)
            * peridot::math::Quaternion::new(init_xrot, peridot::math::Vector3::DOWN);

        Self {
            xrot: DirtyTracker::new(init_xrot),
            yrot: DirtyTracker::new(init_yrot),
            rot_x_input_diff: Differential::new(0.0),
            rot_y_input_diff: Differential::new(0.0),
            camera: DirtyTracker::new((
                peridot::math::Camera {
                    projection: Some(peridot::math::ProjectionMethod::Perspective {
                        fov: 60.0f32.to_radians(),
                    }),
                    depth_range: 0.1..100.0,
                    position: peridot::math::Matrix3::from(init_rot)
                        * peridot::math::Vector3::back()
                        * init_camera_distance,
                    rotation: init_rot,
                },
                init_aspect_value,
            )),
        }
    }

    pub fn update(
        &mut self,
        e: &peridot::Engine<impl peridot::NativeLinker>,
        dt: std::time::Duration,
        current_capturing: Option<CapturingComponent>,
    ) {
        let rdx = self
            .rot_x_input_diff
            .update(e.input().analog_value_abs(ID_CAMERA_ROT_AX_X));
        let rdy = self
            .rot_y_input_diff
            .update(e.input().analog_value_abs(ID_CAMERA_ROT_AX_Y));

        if current_capturing.is_none() && !e.input().button_pressing_time(ID_PLANE_PRESS).is_zero()
        {
            self.xrot
                .try_set_dirty_neq(self.xrot.get() + rdx.to_radians() * 0.125);
            self.yrot.try_set_dirty_neq(
                (self.yrot.get() + rdy.to_radians() * 0.125)
                    .clamp(-85.0f32.to_radians(), 85.0f32.to_radians()),
            );
        }

        let dx = self.xrot.take_dirty_flag();
        let dy = self.yrot.take_dirty_flag();
        if dx || dy {
            self.camera.modify().0.rotation =
                peridot::math::Quaternion::new(*self.yrot.get(), peridot::math::Vector3::RIGHT)
                    * peridot::math::Quaternion::new(
                        *self.xrot.get(),
                        peridot::math::Vector3::DOWN,
                    );
        }

        let mx = e.input().analog_value_abs(ID_CAMERA_MOVE_AX_X);
        let my = e.input().analog_value_abs(ID_CAMERA_MOVE_AX_Y);
        let mz = e.input().analog_value_abs(ID_CAMERA_MOVE_AX_Z);

        if mx != 0.0 || my != 0.0 || mz != 0.0 {
            let xzv = peridot::math::Matrix3::from(peridot::math::Quaternion::new(
                *self.xrot.get(),
                peridot::math::Vector3::DOWN,
            )) * peridot::math::Vector3(mx, 0.0, mz);
            self.camera.modify().0.position +=
                (xzv + peridot::math::Vector3(0.0, my, 0.0)) * 2.0 * dt.as_secs_f32();
        }
    }
}

pub struct Game<NL: peridot::NativeLinker> {
    const_res: ConstResources,
    descriptors: Descriptors,
    mem: Memory,
    screen_res: ScreenResources,
    ui_dynamic_buffers: UIRenderingBuffers,
    command_buffers: peridot::CommandBundle,
    update_commands: peridot::CommandBundle,
    render_bundles: Vec<RenderBundle>,
    main_camera: FreeCameraView,
    material_data: DirtyTracker<MaterialInfo>,
    capturing_component: Option<CapturingComponent>,
    ui_roughness_slider: UISlider,
    ui_anisotropic_slider: UISlider,
    ui_metallic_slider: UISlider,
    ui_reflectance_slider: UISlider,
    plane_touch_edge: EdgeTrigger<bool>,
    last_frame_tfb: peridot::TransferBatch,
    ph: std::marker::PhantomData<*const NL>,
}
impl<NL: peridot::NativeLinker> Game<NL> {
    pub const NAME: &'static str = "PBR test";
    pub const VERSION: (u32, u32, u32) = (0, 1, 0);
}
impl<NL: peridot::NativeLinker> peridot::FeatureRequests for Game<NL> {}
impl<NL: peridot::NativeLinker + Sync> peridot::EngineEvents<NL> for Game<NL>
where
    NL::Presenter: Sync,
{
    fn init(e: &mut peridot::Engine<NL>) -> Self {
        e.input_mut()
            .map(peridot::NativeButtonInput::Mouse(0), ID_PLANE_PRESS);
        e.input_mut().map(
            peridot::AxisKey {
                positive_key: peridot::NativeButtonInput::Character('D'),
                negative_key: peridot::NativeButtonInput::Character('A'),
            },
            ID_CAMERA_MOVE_AX_X,
        );
        e.input_mut().map(
            peridot::AxisKey {
                positive_key: peridot::NativeButtonInput::Character('W'),
                negative_key: peridot::NativeButtonInput::Character('S'),
            },
            ID_CAMERA_MOVE_AX_Z,
        );
        e.input_mut().map(
            peridot::AxisKey {
                positive_key: peridot::NativeButtonInput::Character('Q'),
                negative_key: peridot::NativeButtonInput::Character('Z'),
            },
            ID_CAMERA_MOVE_AX_Y,
        );
        e.input_mut()
            .map(peridot::NativeAnalogInput::MouseX, ID_CAMERA_ROT_AX_X);
        e.input_mut()
            .map(peridot::NativeAnalogInput::MouseY, ID_CAMERA_ROT_AX_Y);

        let material_data = DirtyTracker::new(MaterialInfo {
            base_color: peridot::math::Vector4(1.0, 0.0, 0.0, 1.0),
            roughness: 0.4,
            anisotropic: 0.0,
            metallic: 0.0,
            reflectance: 0.5,
        });

        let bb0 = e.backbuffer(0).expect("no backbuffers?");
        let render_area = AsRef::<br::vk::VkExtent2D>::as_ref(bb0.size())
            .clone()
            .into_rect(br::vk::VkOffset2D { x: 0, y: 0 });
        let aspect = bb0.size().width as f32 / bb0.size().height as f32;

        let mut ui = peridot_vg::Context::new(e.rendering_precision());
        let mut ui_control_mask = peridot_vg::Context::new(e.rendering_precision());
        let mut ui_dynamic_texts = peridot_vg::Context::new(e.rendering_precision());
        let font = Rc::new(
            peridot_vg::FontProvider::new()
                .expect("Failed to create FontProvider")
                .best_match("Yu Gothic UI", &peridot_vg::FontProperties::default(), 18.0)
                .expect("Failed to find best match font"),
        );
        let font_sm = Rc::new(
            peridot_vg::FontProvider::new()
                .expect("Failed to create FontProvider")
                .best_match("Yu Gothic UI", &peridot_vg::FontProperties::default(), 14.0)
                .expect("Failed to find best match font"),
        );

        let ui_heading_label = UIStaticLabel::new(
            peridot::math::Vector2(8.0, -8.0),
            String::from("Peridot PBR Test Controls"),
            font.clone(),
        );
        let ui_roughness_label = UIStaticLabel::new(
            peridot::math::Vector2(UI_LEFT_MARGIN, -UI_ROUGHNESS_TOP),
            String::from("roughness"),
            font_sm.clone(),
        );
        let ui_roughness_slider = UISlider::new(
            peridot::math::Vector2(UI_LEFT_MARGIN, -UI_ROUGHNESS_TOP - UI_SLIDER_LABEL_HEIGHT),
            peridot::math::Vector2(UI_SLIDER_WIDTH, UI_SLIDER_HEIGHT),
            font_sm.clone(),
            material_data.get().roughness,
            CapturingComponent::Roughness,
            0,
        );
        let ui_anisotropic_label = UIStaticLabel::new(
            peridot::math::Vector2(UI_LEFT_MARGIN, -UI_ANISOTROPIC_TOP),
            String::from("anisotropic"),
            font_sm.clone(),
        );
        let ui_anisotropic_slider = UISlider::new(
            peridot::math::Vector2(UI_LEFT_MARGIN, -UI_ANISOTROPIC_TOP - UI_SLIDER_LABEL_HEIGHT),
            peridot::math::Vector2(UI_SLIDER_WIDTH, UI_SLIDER_HEIGHT),
            font_sm.clone(),
            material_data.get().anisotropic,
            CapturingComponent::Anisotropic,
            4,
        );
        let ui_metallic_label = UIStaticLabel::new(
            peridot::math::Vector2(UI_LEFT_MARGIN, -UI_METALLIC_TOP),
            String::from("metallic"),
            font_sm.clone(),
        );
        let ui_metallic_slider = UISlider::new(
            peridot::math::Vector2(UI_LEFT_MARGIN, -UI_METALLIC_TOP - UI_SLIDER_LABEL_HEIGHT),
            peridot::math::Vector2(UI_SLIDER_WIDTH, UI_SLIDER_HEIGHT),
            font_sm.clone(),
            material_data.get().metallic,
            CapturingComponent::Metallic,
            8,
        );
        let ui_reflectance_label = UIStaticLabel::new(
            peridot::math::Vector2(UI_LEFT_MARGIN, -UI_REFLECTANCE_TOP),
            String::from("reflectance"),
            font_sm.clone(),
        );
        let ui_reflectance_slider = UISlider::new(
            peridot::math::Vector2(UI_LEFT_MARGIN, -UI_REFLECTANCE_TOP - UI_SLIDER_LABEL_HEIGHT),
            peridot::math::Vector2(UI_SLIDER_WIDTH, UI_SLIDER_HEIGHT),
            font_sm.clone(),
            material_data.get().reflectance,
            CapturingComponent::Reflectance,
            12,
        );

        let renderables = [
            &ui_heading_label as &dyn UIRenderable,
            &ui_roughness_label,
            &ui_roughness_slider,
            &ui_anisotropic_label,
            &ui_anisotropic_slider,
            &ui_metallic_label,
            &ui_metallic_slider,
            &ui_reflectance_label,
            &ui_reflectance_slider,
        ];
        for r in &renderables {
            r.render(&mut ui);
            r.render_mask(&mut ui_control_mask);
            r.render_dynamic(&mut ui_dynamic_texts);
        }

        let const_res = ConstResources::new(e);
        let mut tfb = peridot::TransferBatch::new();
        let mut mem = Memory::new(e, &mut tfb, &ui, &ui_control_mask);
        let main_camera = FreeCameraView::new(0.0, -5.0f32.to_radians(), 5.0, aspect);
        mem.apply_main_camera(e.graphics_mut(), &main_camera.camera.get().0, aspect);
        mem.set_camera_info(
            e.graphics_mut(),
            RasterizationCameraInfo {
                pos: peridot::math::Vector4(
                    main_camera.camera.get().0.position.0,
                    main_camera.camera.get().0.position.1,
                    main_camera.camera.get().0.position.2,
                    1.0,
                ),
            },
        );
        mem.set_directional_light_info(
            e.graphics_mut(),
            RasterizationDirectionalLightInfo {
                dir: -peridot::math::Vector4(0.2f32, 0.3, 0.5, 0.0).normalize(),
                intensity: peridot::math::Vector4(2.0, 2.0, 2.0, 1.0),
            },
        );
        mem.set_material(e.graphics_mut(), material_data.get().clone());
        mem.construct_new_ui_fill_rect_vertices(e.graphics_mut(), |vs| {
            for r in renderables {
                r.render_dynamic_mesh(vs);
            }
        });
        mem.set_ui_transform(
            e.graphics_mut(),
            peridot::math::Camera {
                projection: Some(peridot::math::ProjectionMethod::UI {
                    design_width: bb0.size().width as _,
                    design_height: bb0.size().height as _,
                }),
                ..Default::default()
            }
            .projection_matrix(aspect),
        );
        mem.ready_transfer(e.graphics(), &mut tfb);

        let ui_dynamic_buffers = UIRenderingBuffers::new(e.graphics(), &ui_dynamic_texts, &mut tfb)
            .expect("Failed to allocate ui dynamic buffers");

        e.submit_commands(|r| {
            tfb.sink_transfer_commands(r);
            tfb.sink_graphics_ready_commands(r);
        })
        .expect("Failed to initialize resources");

        let descriptors = Descriptors::new(e, &const_res);
        let mut dub = peridot::DescriptorSetUpdateBatch::new();
        dub.write(
            descriptors.grid_transform(),
            0,
            br::DescriptorUpdateInfo::UniformBuffer(vec![(
                mem.mem.buffer.0.native_ptr(),
                range_cast_u64_usize(mem.grid_transform_range()),
            )]),
        );
        dub.write(
            descriptors.object_transform(),
            0,
            br::DescriptorUpdateInfo::UniformBuffer(vec![(
                mem.mem.buffer.0.native_ptr(),
                range_cast_u64_usize(mem.object_transform_range()),
            )]),
        );
        dub.write(
            descriptors.rasterization_scene_info(),
            0,
            br::DescriptorUpdateInfo::UniformBuffer(vec![(
                mem.mem.buffer.0.native_ptr(),
                range_cast_u64_usize(mem.camera_info_range()),
            )]),
        );
        dub.write(
            descriptors.rasterization_scene_info(),
            1,
            br::DescriptorUpdateInfo::UniformBuffer(vec![(
                mem.mem.buffer.0.native_ptr(),
                range_cast_u64_usize(mem.directional_light_info_range()),
            )]),
        );
        dub.write(
            descriptors.material_info(),
            0,
            br::DescriptorUpdateInfo::UniformBuffer(vec![(
                mem.mem.buffer.0.native_ptr(),
                range_cast_u64_usize(mem.material_range()),
            )]),
        );
        dub.write(
            descriptors.ui_transform_buffer(),
            0,
            br::DescriptorUpdateInfo::UniformTexelBuffer(vec![mem
                .ui_transform_buffer_view
                .native_ptr()]),
        );
        dub.write(
            descriptors.ui_mask_transform_buffer(),
            0,
            br::DescriptorUpdateInfo::UniformTexelBuffer(vec![mem
                .ui_mask_transform_buffer_view
                .native_ptr()]),
        );
        dub.write(
            descriptors.ui_dynamic_transform_buffer(),
            0,
            br::DescriptorUpdateInfo::UniformTexelBuffer(vec![ui_dynamic_buffers
                .transform_buffer_view
                .native_ptr()]),
        );
        dub.write(
            descriptors.ui_fill_rect_transform(),
            0,
            br::DescriptorUpdateInfo::UniformBuffer(vec![(
                mem.mem.buffer.0.native_ptr(),
                range_cast_u64_usize(mem.ui_transform_range()),
            )]),
        );
        dub.submit(e.graphics_device());

        let screen_res = ScreenResources::new(e, &const_res);

        let mut render_bundles = (0..5)
            .map(|_| {
                RenderBundle::new(e.graphics(), e.backbuffer_count() as _)
                    .expect("Failed to create render bundle")
            })
            .collect::<Vec<_>>();
        rayon::scope(|s| {
            let (rb0, rb1, rb2, rb3, rb4) = match &mut render_bundles[..] {
                &mut [ref mut rb0, ref mut rb1, ref mut rb2, ref mut rb3, ref mut rb4] => {
                    (rb0, rb1, rb2, rb3, rb4)
                }
                _ => unreachable!(),
            };

            s.spawn(|_| {
                for n in 0..e.backbuffer_count() {
                    Self::repopulate_ui_mask_render_commands(
                        e,
                        rb0.synchronized(n),
                        &const_res.render_pass,
                        &descriptors,
                        &screen_res,
                        n,
                        &mem.mem.buffer.0,
                        &mem.ui_mask_render_params,
                    );
                }
            });
            s.spawn(|_| {
                for n in 0..e.backbuffer_count() {
                    Self::repopulate_grid_render_commands(
                        e,
                        rb1.synchronized(n),
                        &const_res.render_pass,
                        &descriptors,
                        &screen_res,
                        n,
                        &mem.mem.buffer.0,
                        &mem.static_offsets,
                    );
                }
            });
            s.spawn(|_| {
                for n in 0..e.backbuffer_count() {
                    Self::repopulate_pbr_object_render_commands(
                        e,
                        rb2.synchronized(n),
                        &const_res.render_pass,
                        &descriptors,
                        &screen_res,
                        n,
                        &mem.mem.buffer.0,
                        &mem.static_offsets,
                        mem.icosphere_vertex_count as _,
                    );
                }
            });
            s.spawn(|_| {
                for n in 0..e.backbuffer_count() {
                    Self::repopulate_static_ui_render_commands(
                        e,
                        rb3.synchronized(n),
                        &const_res.render_pass,
                        &descriptors,
                        &screen_res,
                        n,
                        &mem.ui_render_params,
                        &mem.mem.buffer.0,
                        &mem.static_offsets,
                        &mem.mutable_offsets,
                        mem.mem.mut_buffer_placement,
                    );
                }
            });
            s.spawn(|_| {
                for n in 0..e.backbuffer_count() {
                    Self::repopulate_dynamic_ui_render_commands(
                        e,
                        rb4.synchronized(n),
                        &const_res.render_pass,
                        &descriptors,
                        &screen_res,
                        n,
                        &ui_dynamic_buffers,
                    );
                }
            });
        });

        let mut command_buffers = peridot::CommandBundle::new(
            e.graphics(),
            peridot::CBSubmissionType::Graphics,
            e.backbuffer_count(),
        )
        .expect("Failed to alloc command bundle");
        Self::repopulate_screen_commands(
            e,
            render_area,
            &mut command_buffers,
            &const_res,
            &screen_res,
            &render_bundles,
        );
        let update_commands =
            peridot::CommandBundle::new(e.graphics(), peridot::CBSubmissionType::Transfer, 1)
                .expect("Failed to alloc update command buffers");

        Self {
            const_res,
            descriptors,
            mem,
            ui_dynamic_buffers,
            main_camera,
            screen_res,
            command_buffers,
            update_commands,
            render_bundles,
            material_data,
            capturing_component: None,
            plane_touch_edge: EdgeTrigger::new(false),
            ui_roughness_slider,
            ui_anisotropic_slider,
            ui_metallic_slider,
            ui_reflectance_slider,
            last_frame_tfb: peridot::TransferBatch::new(),
            ph: std::marker::PhantomData,
        }
    }

    fn update(
        &mut self,
        e: &mut peridot::Engine<NL>,
        on_backbuffer_of: u32,
        delta_time: std::time::Duration,
    ) -> (Option<br::SubmissionBatch>, br::SubmissionBatch) {
        self.last_frame_tfb = peridot::TransferBatch::new();

        let press_inframe = self
            .plane_touch_edge
            .update(!e.input().button_pressing_time(ID_PLANE_PRESS).is_zero());
        if let Some(&p) = press_inframe {
            if p {
                self.capturing_component = e.input().get_plane_position(0).and_then(|(px, py)| {
                    let pv = peridot::math::Vector2(px, py);

                    [
                        &self.ui_roughness_slider,
                        &self.ui_anisotropic_slider,
                        &self.ui_metallic_slider,
                        &self.ui_reflectance_slider,
                    ]
                    .into_iter()
                    .fold(None, |a, c| a.or_else(|| c.try_capture_input(pv)))
                });
            } else {
                self.capturing_component = None;
            }
        }
        let mut ui_mesh_dirty = false;
        if self.plane_touch_edge.current {
            // dragging
            match self.capturing_component {
                Some(CapturingComponent::Roughness) => {
                    self.ui_roughness_slider.update_capturing_input(e);
                    self.material_data.modify().roughness = self.ui_roughness_slider.value;
                    ui_mesh_dirty = true;
                }
                Some(CapturingComponent::Anisotropic) => {
                    self.ui_anisotropic_slider.update_capturing_input(e);
                    self.material_data.modify().anisotropic = self.ui_anisotropic_slider.value;
                    ui_mesh_dirty = true;
                }
                Some(CapturingComponent::Metallic) => {
                    self.ui_metallic_slider.update_capturing_input(e);
                    self.material_data.modify().metallic = self.ui_metallic_slider.value;
                    ui_mesh_dirty = true;
                }
                Some(CapturingComponent::Reflectance) => {
                    self.ui_reflectance_slider.update_capturing_input(e);
                    self.material_data.modify().reflectance = self.ui_reflectance_slider.value;
                    ui_mesh_dirty = true;
                }
                None => (),
            }
        }

        self.main_camera
            .update(e, delta_time, self.capturing_component);

        if self.main_camera.camera.take_dirty_flag() {
            self.mem.apply_main_camera(
                e.graphics_mut(),
                &self.main_camera.camera.get().0,
                self.main_camera.camera.get().1,
            );
        }

        if self.material_data.take_dirty_flag() {
            self.mem
                .set_material(e.graphics_mut(), self.material_data.get().clone());
        }

        if ui_mesh_dirty {
            let mut ui_dynamic_texts = peridot_vg::Context::new(e.rendering_precision());
            self.mem
                .construct_new_ui_fill_rect_vertices(e.graphics_mut(), |vs| {
                    for c in &[
                        &self.ui_roughness_slider,
                        &self.ui_anisotropic_slider,
                        &self.ui_metallic_slider,
                        &self.ui_reflectance_slider,
                    ] {
                        c.render_dynamic(&mut ui_dynamic_texts);
                        c.render_dynamic_mesh(vs);
                    }
                });
            let ui_dynamic_buffers =
                UIRenderingBuffers::new(e.graphics(), &ui_dynamic_texts, &mut self.last_frame_tfb)
                    .expect("Failed to allocate ui dynamic buffers");
            let mut dub = peridot::DescriptorSetUpdateBatch::new();
            dub.write(
                self.descriptors.ui_dynamic_transform_buffer(),
                0,
                br::DescriptorUpdateInfo::UniformTexelBuffer(vec![ui_dynamic_buffers
                    .transform_buffer_view
                    .native_ptr()]),
            );
            dub.submit(&e.graphics());
            self.render_bundles[4]
                .reset()
                .expect("Failed to reset dynamic ui render bundles");
            for n in 0..e.backbuffer_count() {
                Self::repopulate_dynamic_ui_render_commands(
                    e,
                    self.render_bundles[4].synchronized(n),
                    &self.const_res.render_pass,
                    &self.descriptors,
                    &self.screen_res,
                    n,
                    &ui_dynamic_buffers,
                );
            }
            self.command_buffers
                .reset()
                .expect("Failed to reset command buffers");
            Self::repopulate_screen_commands(
                e,
                self.screen_res.frame_buffers[0]
                    .size()
                    .clone()
                    .into_rect(br::vk::VkOffset2D { x: 0, y: 0 }),
                &mut self.command_buffers,
                &self.const_res,
                &self.screen_res,
                &self.render_bundles,
            );
            self.ui_dynamic_buffers = ui_dynamic_buffers;
        }

        self.mem
            .ready_transfer(e.graphics(), &mut self.last_frame_tfb);
        self.mem.dynamic_stg.clear();
        let update_submission =
            if self.last_frame_tfb.has_copy_ops() || self.last_frame_tfb.has_ready_barrier_ops() {
                self.update_commands
                    .reset()
                    .expect("Failed to reset update commands");
                unsafe {
                    let mut r = self.update_commands[0]
                        .begin()
                        .expect("Failed to begin recording update commands");
                    self.last_frame_tfb.sink_transfer_commands(&mut r);
                    self.last_frame_tfb.sink_graphics_ready_commands(&mut r);
                }

                Some(br::SubmissionBatch {
                    command_buffers: std::borrow::Cow::Borrowed(&self.update_commands[..]),
                    ..Default::default()
                })
            } else {
                None
            };

        (
            update_submission,
            br::SubmissionBatch {
                command_buffers: std::borrow::Cow::Borrowed(
                    &self.command_buffers[on_backbuffer_of as usize..on_backbuffer_of as usize + 1],
                ),
                ..Default::default()
            },
        )
    }

    fn discard_backbuffer_resources(&mut self) {
        self.command_buffers
            .reset()
            .expect("Failed to reset screen commands");
        self.screen_res.frame_buffers.clear();
    }
    fn on_resize(&mut self, e: &mut peridot::Engine<NL>, new_size: peridot::math::Vector2<usize>) {
        self.screen_res = ScreenResources::new(e, &self.const_res);

        rayon::scope(|s| {
            let (rb0, rb1, rb2, rb3, rb4) = match &mut self.render_bundles[..] {
                &mut [ref mut rb0, ref mut rb1, ref mut rb2, ref mut rb3, ref mut rb4] => {
                    (rb0, rb1, rb2, rb3, rb4)
                }
                _ => unreachable!(),
            };

            s.spawn(|_| {
                rb0.reset()
                    .expect("Failed to reset ui mask render commands");

                for n in 0..e.backbuffer_count() {
                    Self::repopulate_ui_mask_render_commands(
                        e,
                        rb0.synchronized(n),
                        &self.const_res.render_pass,
                        &self.descriptors,
                        &self.screen_res,
                        n,
                        &self.mem.mem.buffer.0,
                        &self.mem.ui_mask_render_params,
                    );
                }
            });
            s.spawn(|_| {
                rb1.reset().expect("Failed to reset grid render commands");

                for n in 0..e.backbuffer_count() {
                    Self::repopulate_grid_render_commands(
                        e,
                        rb1.synchronized(n),
                        &self.const_res.render_pass,
                        &self.descriptors,
                        &self.screen_res,
                        n,
                        &self.mem.mem.buffer.0,
                        &self.mem.static_offsets,
                    );
                }
            });
            s.spawn(|_| {
                rb2.reset()
                    .expect("Failed to reset pbr object render commands");

                for n in 0..e.backbuffer_count() {
                    Self::repopulate_pbr_object_render_commands(
                        e,
                        rb2.synchronized(n),
                        &self.const_res.render_pass,
                        &self.descriptors,
                        &self.screen_res,
                        n,
                        &self.mem.mem.buffer.0,
                        &self.mem.static_offsets,
                        self.mem.icosphere_vertex_count as _,
                    );
                }
            });
            s.spawn(|_| {
                rb3.reset()
                    .expect("Failed to reset static ui render commands");

                for n in 0..e.backbuffer_count() {
                    Self::repopulate_static_ui_render_commands(
                        e,
                        rb3.synchronized(n),
                        &self.const_res.render_pass,
                        &self.descriptors,
                        &self.screen_res,
                        n,
                        &self.mem.ui_render_params,
                        &self.mem.mem.buffer.0,
                        &self.mem.static_offsets,
                        &self.mem.mutable_offsets,
                        self.mem.mem.mut_buffer_placement,
                    );
                }
            });
            s.spawn(|_| {
                rb4.reset()
                    .expect("Failed to reset dynamic ui render commands");

                for n in 0..e.backbuffer_count() {
                    Self::repopulate_dynamic_ui_render_commands(
                        e,
                        rb4.synchronized(n),
                        &self.const_res.render_pass,
                        &self.descriptors,
                        &self.screen_res,
                        n,
                        &self.ui_dynamic_buffers,
                    );
                }
            });
        });

        Self::repopulate_screen_commands(
            e,
            br::vk::VkExtent2D {
                width: new_size.0 as _,
                height: new_size.1 as _,
            }
            .into_rect(br::vk::VkOffset2D { x: 0, y: 0 }),
            &mut self.command_buffers,
            &self.const_res,
            &self.screen_res,
            &self.render_bundles,
        );

        let new_aspect = new_size.0 as f32 / new_size.1 as f32;
        self.main_camera.camera.modify().1 = new_aspect;
        self.mem.set_ui_transform(
            e.graphics_mut(),
            peridot::math::Camera {
                projection: Some(peridot::math::ProjectionMethod::UI {
                    design_width: new_size.0 as _,
                    design_height: new_size.1 as _,
                }),
                ..Default::default()
            }
            .projection_matrix(new_aspect),
        );
    }
}
impl<NL: peridot::NativeLinker> Game<NL> {
    fn repopulate_ui_mask_render_commands(
        engine: &peridot::Engine<NL>,
        mut command_buffer: br::SynchronizedCommandBuffer,
        renderpass: &br::RenderPass,
        descriptors: &Descriptors,
        screen_res: &ScreenResources,
        frame_buffer_index: usize,
        device_buffer: &peridot::Buffer,
        render_params: &peridot_vg::RendererParams,
    ) {
        let fb = &screen_res.frame_buffers[frame_buffer_index];
        let mut rec = command_buffer
            .begin_inherit(
                Some((fb, renderpass, ConstResources::RENDER_STENCIL_PREPASS)),
                None,
            )
            .expect("Failed to initiate ui mask render bundle recording");

        render_params.default_render_commands(
            engine,
            &mut rec,
            device_buffer,
            peridot_vg::RendererExternalInstances {
                interior_pipeline: &screen_res.vg_interior_mask_pipeline,
                curve_pipeline: &screen_res.vg_curve_mask_pipeline,
                transform_buffer_descriptor_set: descriptors.ui_mask_transform_buffer(),
                target_pixels: peridot::math::Vector2(fb.size().width as _, fb.size().height as _),
            },
        );
    }

    fn repopulate_grid_render_commands(
        engine: &peridot::Engine<NL>,
        mut command_buffer: br::SynchronizedCommandBuffer,
        renderpass: &br::RenderPass,
        descriptors: &Descriptors,
        screen_res: &ScreenResources,
        frame_buffer_index: usize,
        device_buffer: &br::Buffer,
        static_offsets: &StaticBufferOffsets,
    ) {
        let mut rec = command_buffer
            .begin_inherit(
                Some((
                    &screen_res.frame_buffers[frame_buffer_index],
                    renderpass,
                    ConstResources::RENDER_MAIN_PASS,
                )),
                None,
            )
            .expect("Failed to initiate grid render bundle recording");

        screen_res.grid_render_pipeline.bind(&mut rec);
        rec.bind_graphics_descriptor_sets(0, &[descriptors.grid_transform().into()], &[]);
        rec.bind_vertex_buffers(0, &[(device_buffer, static_offsets.grid as _)]);
        rec.draw(mesh::GRID_MESH_LINE_COUNT as _, 1, 0, 0);
    }

    fn repopulate_pbr_object_render_commands(
        engine: &peridot::Engine<NL>,
        mut command_buffer: br::SynchronizedCommandBuffer,
        renderpass: &br::RenderPass,
        descriptors: &Descriptors,
        screen_res: &ScreenResources,
        frame_buffer_index: usize,
        device_buffer: &br::Buffer,
        static_offsets: &StaticBufferOffsets,
        icosphere_vertex_count: u32,
    ) {
        let mut rec = command_buffer
            .begin_inherit(
                Some((
                    &screen_res.frame_buffers[frame_buffer_index],
                    renderpass,
                    ConstResources::RENDER_MAIN_PASS,
                )),
                None,
            )
            .expect("Failed to initiate pbr object render bundle recording");

        screen_res.pbr_pipeline.bind(&mut rec);
        rec.bind_graphics_descriptor_sets(
            0,
            &[
                descriptors.object_transform().into(),
                descriptors.rasterization_scene_info().into(),
                descriptors.material_info().into(),
            ],
            &[],
        );
        rec.bind_vertex_buffers(
            0,
            &[(device_buffer, static_offsets.icosphere_vertices as _)],
        );
        rec.bind_index_buffer(
            device_buffer,
            static_offsets.icosphere_indices as _,
            br::IndexType::U16,
        );
        rec.draw_indexed(icosphere_vertex_count, 1, 0, 0, 0);
    }

    fn repopulate_static_ui_render_commands(
        engine: &peridot::Engine<NL>,
        mut command_buffer: br::SynchronizedCommandBuffer,
        renderpass: &br::RenderPass,
        descriptors: &Descriptors,
        screen_res: &ScreenResources,
        frame_buffer_index: usize,
        render_params: &peridot_vg::RendererParams,
        device_buffer: &peridot::Buffer,
        static_offsets: &StaticBufferOffsets,
        mutable_offsets: &MutableBufferOffsets,
        mut_buffer_placement: u64,
    ) {
        let fb = &screen_res.frame_buffers[frame_buffer_index];
        let mut rec = command_buffer
            .begin_inherit(
                Some((fb, renderpass, ConstResources::RENDER_MAIN_PASS)),
                None,
            )
            .expect("Failed to initiate static ui render bundle recording");

        screen_res.ui_fill_rect_pipeline.bind(&mut rec);
        rec.bind_graphics_descriptor_sets(0, &[descriptors.ui_fill_rect_transform().into()], &[]);
        rec.bind_vertex_buffers(
            0,
            &[(
                device_buffer,
                (mutable_offsets.ui_fill_rects + mut_buffer_placement) as _,
            )],
        );
        rec.bind_index_buffer(
            device_buffer,
            static_offsets.ui_fill_rect_indices as _,
            br::IndexType::U16,
        );
        rec.push_graphics_constant(br::ShaderStage::FRAGMENT, 0, &[1.0f32, 1.0, 1.0, 0.25]);
        rec.draw_indexed(mesh::UI_FILL_RECT_INDEX_COUNT as _, 1, 0, 0, 0);

        screen_res.ui_border_line_pipeline.bind(&mut rec);
        rec.bind_index_buffer(
            device_buffer,
            static_offsets.ui_border_line_indices as _,
            br::IndexType::U16,
        );
        rec.push_graphics_constant(br::ShaderStage::FRAGMENT, 0, &[1.0f32, 1.0, 1.0, 1.0]);
        rec.draw_indexed(mesh::UI_FILL_RECT_BORDER_INDEX_COUNT as _, 1, 0, 0, 0);

        render_params.default_render_commands(
            engine,
            &mut rec,
            device_buffer,
            peridot_vg::RendererExternalInstances {
                interior_pipeline: &screen_res.vg_interior_pipeline,
                curve_pipeline: &screen_res.vg_curve_pipeline,
                transform_buffer_descriptor_set: descriptors.ui_transform_buffer(),
                target_pixels: peridot::math::Vector2(fb.size().width as _, fb.size().height as _),
            },
        );
    }

    fn repopulate_dynamic_ui_render_commands(
        engine: &peridot::Engine<NL>,
        mut command_buffer: br::SynchronizedCommandBuffer,
        renderpass: &br::RenderPass,
        descriptors: &Descriptors,
        screen_res: &ScreenResources,
        frame_buffer_index: usize,
        ui_dynamic_buffers: &UIRenderingBuffers,
    ) {
        let fb = &screen_res.frame_buffers[frame_buffer_index];
        let mut rec = command_buffer
            .begin_inherit(
                Some((fb, renderpass, ConstResources::RENDER_MAIN_PASS)),
                None,
            )
            .expect("Failed to initiate static ui render bundle recording");

        ui_dynamic_buffers.render_params.default_render_commands(
            engine,
            &mut rec,
            &ui_dynamic_buffers.buffer,
            peridot_vg::RendererExternalInstances {
                interior_pipeline: &screen_res.vg_interior_inv_pipeline,
                curve_pipeline: &screen_res.vg_curve_inv_pipeline,
                transform_buffer_descriptor_set: descriptors.ui_dynamic_transform_buffer(),
                target_pixels: peridot::math::Vector2(fb.size().width as _, fb.size().height as _),
            },
        );
    }

    fn repopulate_screen_commands(
        engine: &peridot::Engine<NL>,
        render_area: br::vk::VkRect2D,
        command_buffers: &mut peridot::CommandBundle,
        const_res: &ConstResources,
        screen_res: &ScreenResources,
        render_bundles: &[RenderBundle],
    ) {
        for (n, (cb, fb)) in command_buffers
            .iter_mut()
            .zip(&screen_res.frame_buffers)
            .enumerate()
        {
            let mut r = unsafe { cb.begin().expect("Failed to begin command recording") };
            r.begin_render_pass(
                &const_res.render_pass,
                fb,
                render_area.clone(),
                &[
                    br::ClearValue::color_f32([0.25 * 0.25, 0.5 * 0.25, 1.0 * 0.25, 1.0]),
                    br::ClearValue::depth_stencil(1.0, 0),
                ],
                false,
            );
            unsafe {
                r.execute_commands(&[render_bundles[0].buffers[n].native_ptr()]);
            }
            r.next_subpass(false);
            unsafe {
                r.execute_commands(
                    &render_bundles[1..]
                        .iter()
                        .map(|b| b.buffers[n].native_ptr())
                        .collect::<Vec<_>>(),
                );
            }
            r.end_render_pass();
        }
    }
}
