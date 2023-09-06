use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::sync::Arc;

use bedrock as br;
use br::{
    Buffer, CommandBuffer, CommandPool, DescriptorPool, Device, DeviceMemory, Image, ImageChild,
    ImageSubresourceSlice, MemoryBound, SubmissionBatch, VkHandle,
};
use peridot::math::One;
use peridot::mthelper::{DynamicMutabilityProvider, SharedRef};
use peridot::{self, DefaultRenderCommands, ModelData};
use peridot_command_object::{
    BeginRenderPass, BindGraphicsDescriptorSets, Blending, BufferUsage,
    BufferUsageTransitionBarrier, ColorAttachmentBlending, CopyBuffer, DescriptorSets,
    EndRenderPass, GraphicsCommand, GraphicsCommandCombiner, GraphicsCommandSubmission,
    IndexedMesh, Mesh, NextSubpass, PipelineBarrier, PipelineBarrierEntry, PushConstant,
    RangedBuffer, RangedImage, SimpleDraw, StandardIndexedMesh, StandardMesh,
    ViewportWithScissorRect,
};
use peridot_memory_manager::MemoryManager;
use peridot_vertex_processing_pack::*;
use peridot_vg::{FontProvider, FontProviderConstruct};
use ui::{
    UIRenderable, UIRenderingBuffers, UISlider, UIStaticLabel, UI_ANISOTROPIC_TOP, UI_LEFT_MARGIN,
    UI_METALLIC_TOP, UI_REFLECTANCE_TOP, UI_ROUGHNESS_TOP, UI_SLIDER_HEIGHT,
    UI_SLIDER_LABEL_HEIGHT, UI_SLIDER_WIDTH,
};

mod mesh;
mod staging;
mod ui;
use self::staging::DynamicStagingBuffer;

#[repr(C)]
pub struct ObjectTransform {
    mvp: peridot::math::Matrix4F32,
    model_transform: peridot::math::Matrix4F32,
    view_projection: peridot::math::Matrix4F32,
}
#[repr(C)]
pub struct RasterizationCameraInfo {
    pos: peridot::math::Vector4F32,
}
#[repr(C)]
pub struct RasterizationDirectionalLightInfo {
    /// 原点から見た光源の方向（ライト本体の向きとは逆）
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

pub struct DetailedDescriptorSetLayout {
    pub object: br::DescriptorSetLayoutObject<peridot::DeviceObject>,
    pub pool_requirements: Vec<br::vk::VkDescriptorPoolSize>,
}
impl DetailedDescriptorSetLayout {
    pub fn new(
        g: &peridot::Graphics,
        bindings: Vec<br::DescriptorSetLayoutBinding>,
    ) -> br::Result<Self> {
        let mut pool_requirements = BTreeMap::new();
        for b in bindings.iter() {
            *pool_requirements.entry(b.ty).or_insert(0) += b.count;
        }
        let object =
            br::DescriptorSetLayoutBuilder::with_bindings(bindings).create(g.device().clone())?;

        Ok(Self {
            object,
            pool_requirements: pool_requirements
                .into_iter()
                .map(|(ty, n)| ty.with_count(n))
                .collect(),
        })
    }
}

pub struct DescriptorStore {
    _pool: br::DescriptorPoolObject<peridot::DeviceObject>,
    descriptors: Vec<br::DescriptorSet>,
}
impl DescriptorStore {
    pub fn new<'a>(
        g: &peridot::Graphics,
        layouts: impl IntoIterator<Item = &'a DetailedDescriptorSetLayout>,
    ) -> br::Result<Self> {
        let layouts = layouts.into_iter();

        let mut pool_sizes = BTreeMap::new();
        let mut objects = Vec::with_capacity(layouts.size_hint().0);
        for l in layouts {
            objects.push(&l.object);
            for s in l.pool_requirements.iter() {
                *pool_sizes
                    .entry(unsafe { core::mem::transmute::<_, br::DescriptorType>(s._type) })
                    .or_insert(0) += s.descriptorCount;
            }
        }

        let mut dp = br::DescriptorPoolBuilder::new(objects.len() as _)
            .reserve_all(pool_sizes.into_iter().map(|(ty, n)| ty.with_count(n)))
            .create(g.device().clone())?;
        let descriptors = dp.alloc(&objects)?;

        Ok(Self {
            _pool: dp,
            descriptors,
        })
    }

    pub fn raw_descriptor(&self, index: usize) -> Option<br::DescriptorSet> {
        self.descriptors.get(index).copied()
    }

    pub fn descriptor(&self, index: usize) -> Option<br::DescriptorPointer> {
        self.raw_descriptor(index)
            .map(|d| br::DescriptorPointer::new(d.into(), 0))
    }
}

pub struct ConstResources {
    render_pass: br::RenderPassObject<peridot::DeviceObject>,
    dsl_ub1: DetailedDescriptorSetLayout,
    dsl_ub1_f: DetailedDescriptorSetLayout,
    dsl_ub2_f: DetailedDescriptorSetLayout,
    dsl_utb1: DetailedDescriptorSetLayout,
    dsl_ics1_f: DetailedDescriptorSetLayout,
    #[allow(dead_code)]
    linear_sampler: br::SamplerObject<peridot::DeviceObject>,
    unlit_colored_shader: PvpShaderModules<'static, peridot::DeviceObject>,
    unlit_colored_pipeline_layout: Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
    pbr_shader: PvpShaderModules<'static, peridot::DeviceObject>,
    pbr_pipeline_layout: Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
    vg_interior_color_fixed_shader: PvpShaderModules<'static, peridot::DeviceObject>,
    vg_curve_color_fixed_shader: PvpShaderModules<'static, peridot::DeviceObject>,
    vg_pipeline_layout: Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
    unlit_colored_ext_shader: PvpShaderModules<'static, peridot::DeviceObject>,
    unlit_colored_ext_pipeline: Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
    skybox_shader: PvpShaderModules<'static, peridot::DeviceObject>,
    skybox_shader_layout: Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
}
impl ConstResources {
    const RENDER_STENCIL_PREPASS: u32 = 0;
    const RENDER_MAIN_PASS: u32 = 1;

    pub fn new(e: &peridot::Engine<impl peridot::NativeLinker>) -> Self {
        let main_attachment = e
            .back_buffer_attachment_desc()
            .color_memory_op(br::LoadOp::Clear, br::StoreOp::Store);
        let depth_attachment = br::AttachmentDescription::new(
            br::vk::VK_FORMAT_D24_UNORM_S8_UINT,
            br::ImageLayout::DepthStencilAttachmentOpt,
            br::ImageLayout::DepthStencilAttachmentOpt,
        )
        .load_op(br::LoadOp::Clear)
        .stencil_load_op(br::LoadOp::Clear);
        let stencil_prepass = br::SubpassDescription::new()
            .depth_stencil(1, br::ImageLayout::DepthStencilAttachmentOpt);
        let main_pass = br::SubpassDescription::new()
            .add_color_output(0, br::ImageLayout::ColorAttachmentOpt, None)
            .depth_stencil(1, br::ImageLayout::DepthStencilAttachmentOpt);
        let pass_dep = br::vk::VkSubpassDependency {
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
            .add_dependencies([stencil_prepass_dep, pass_dep])
            .create(e.graphics().device().clone())
            .expect("Failed to create render pass");

        let linear_sampler = br::SamplerBuilder::default()
            .create(e.graphics().device().clone())
            .expect("Failed to create DefaultSampler");
        let dsl_ub1 = DetailedDescriptorSetLayout::new(
            e.graphics(),
            vec![br::DescriptorType::UniformBuffer
                .make_binding(1)
                .only_for_vertex()],
        )
        .expect("Failed to create descriptor set layout");
        let dsl_ub1_f = DetailedDescriptorSetLayout::new(
            e.graphics(),
            vec![br::DescriptorType::UniformBuffer
                .make_binding(1)
                .only_for_fragment()],
        )
        .expect("Failed to create ub1f descriptor set layout");
        let dsl_ub2_f = DetailedDescriptorSetLayout::new(
            e.graphics(),
            vec![
                br::DescriptorType::UniformBuffer
                    .make_binding(1)
                    .only_for_fragment(),
                br::DescriptorType::UniformBuffer
                    .make_binding(1)
                    .only_for_fragment(),
            ],
        )
        .expect("Failed to create ub2f descriptor set layout");
        let dsl_utb1 = DetailedDescriptorSetLayout::new(
            e.graphics(),
            vec![br::DescriptorType::UniformTexelBuffer
                .make_binding(1)
                .only_for_vertex()],
        )
        .expect("Failed to create utb1 descriptor set layout");
        let dsl_ics1_f = DetailedDescriptorSetLayout::new(
            e.graphics(),
            vec![br::DescriptorType::CombinedImageSampler
                .make_binding(1)
                .only_for_fragment()
                .with_immutable_samplers(vec![br::SamplerObjectRef::new(&linear_sampler)])],
        )
        .expect("Failed to create ics1_f descriptor set layout");

        let unlit_colored_shader = PvpShaderModules::new(
            e.graphics().device(),
            e.load("shaders.unlit_colored")
                .expect("Failed to load unlit_colored shader"),
        )
        .expect("Failed to create shader modules");
        let unlit_colored_pipeline_layout =
            br::PipelineLayoutBuilder::new(vec![&dsl_ub1.object], vec![])
                .create(e.graphics().device().clone())
                .expect("Failed to create unlit_colored pipeline layout")
                .into();

        let pbr_shader = PvpShaderModules::new(
            e.graphics().device(),
            e.load("shaders.pbr").expect("Failed to load pbr shader"),
        )
        .expect("Failed to create pbr shader modules");
        let pbr_pipeline_layout = br::PipelineLayoutBuilder::new(
            vec![
                &dsl_ub1.object,
                &dsl_ub2_f.object,
                &dsl_ub1_f.object,
                &dsl_ics1_f.object,
            ],
            vec![],
        )
        .create(e.graphics().device().clone())
        .expect("Failed to create pbr pipeline layout")
        .into();
        let unlit_colored_ext_shader = PvpShaderModules::new(
            e.graphics().device(),
            e.load("shaders.unlit_colored_ext")
                .expect("Failed to load unlit_colored_ext shader"),
        )
        .expect("Failed to create unlit_colored_ext shader modules");
        let unlit_colored_ext_pipeline = br::PipelineLayoutBuilder::new(
            vec![&dsl_ub1.object],
            vec![(br::ShaderStage::FRAGMENT, 0..16)],
        )
        .create(e.graphics().device().clone())
        .expect("Failed to create unlit_colored_ext pipeline layout")
        .into();

        let vg_interior_color_fixed_shader = PvpShaderModules::new(
            e.graphics().device(),
            e.load("shaders.vg.interiorColorFixed")
                .expect("Failed to load vg interior color shader"),
        )
        .expect("Failed to create vg interior color shader modules");
        let vg_curve_color_fixed_shader = PvpShaderModules::new(
            e.graphics().device(),
            e.load("shaders.vg.curveColorFixed")
                .expect("Failed to load vg curve color shader"),
        )
        .expect("Failed to create vg curve color shader modules");
        let vg_pipeline_layout = br::PipelineLayoutBuilder::new(
            vec![&dsl_utb1.object],
            vec![(br::ShaderStage::VERTEX, 0..4 * 4)],
        )
        .create(e.graphics().device().clone())
        .expect("Failed to create vg pipeline layout")
        .into();

        let skybox_shader = PvpShaderModules::new(
            e.graphics().device(),
            e.load("shaders.skybox")
                .expect("Failed to load skybox shader"),
        )
        .expect("Failed to create skybox shader modules");
        let skybox_shader_layout =
            br::PipelineLayoutBuilder::new(vec![&dsl_ub1.object, &dsl_ics1_f.object], vec![])
                .create(e.graphics().device().clone())
                .expect("Failed to create skybox pipeline layout")
                .into();

        Self {
            render_pass,
            linear_sampler,
            dsl_ub1,
            dsl_ub1_f,
            dsl_ub2_f,
            dsl_utb1,
            dsl_ics1_f,
            unlit_colored_shader,
            unlit_colored_pipeline_layout,
            pbr_shader,
            pbr_pipeline_layout,
            vg_interior_color_fixed_shader,
            vg_curve_color_fixed_shader,
            vg_pipeline_layout,
            unlit_colored_ext_shader,
            unlit_colored_ext_pipeline,
            skybox_shader,
            skybox_shader_layout,
        }
    }
}

pub struct ScreenResources {
    frame_buffers: Vec<
        br::FramebufferObject<
            peridot::DeviceObject,
            SharedRef<dyn br::ImageView<ConcreteDevice = peridot::DeviceObject> + Sync + Send>,
        >,
    >,
    grid_render_pipeline: peridot::LayoutedPipeline<
        br::PipelineObject<peridot::DeviceObject>,
        Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
    >,
    pbr_pipeline: peridot::LayoutedPipeline<
        br::PipelineObject<peridot::DeviceObject>,
        Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
    >,
    vg_interior_pipeline: peridot::LayoutedPipeline<
        br::PipelineObject<peridot::DeviceObject>,
        Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
    >,
    vg_curve_pipeline: peridot::LayoutedPipeline<
        br::PipelineObject<peridot::DeviceObject>,
        Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
    >,
    vg_interior_inv_pipeline: peridot::LayoutedPipeline<
        br::PipelineObject<peridot::DeviceObject>,
        Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
    >,
    vg_curve_inv_pipeline: peridot::LayoutedPipeline<
        br::PipelineObject<peridot::DeviceObject>,
        Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
    >,
    vg_interior_mask_pipeline: peridot::LayoutedPipeline<
        br::PipelineObject<peridot::DeviceObject>,
        Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
    >,
    vg_curve_mask_pipeline: peridot::LayoutedPipeline<
        br::PipelineObject<peridot::DeviceObject>,
        Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
    >,
    ui_fill_rect_pipeline: peridot::LayoutedPipeline<
        br::PipelineObject<peridot::DeviceObject>,
        Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
    >,
    ui_border_line_pipeline: peridot::LayoutedPipeline<
        br::PipelineObject<peridot::DeviceObject>,
        Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
    >,
    skybox_render_pipeline: peridot::LayoutedPipeline<
        br::PipelineObject<peridot::DeviceObject>,
        Arc<br::PipelineLayoutObject<peridot::DeviceObject>>,
    >,
}
impl ScreenResources {
    pub fn new<NL: peridot::NativeLinker>(
        e: &mut peridot::Engine<NL>,
        memory_manager: &mut MemoryManager,
        const_res: &ConstResources,
    ) -> Self
    where
        <NL::Presenter as peridot::PlatformPresenter>::BackBuffer: Send + Sync,
    {
        let bb0 = e.back_buffer(0).expect("no backbuffers?");

        let depth_texture = memory_manager
            .allocate_device_local_image(
                e.graphics(),
                br::ImageDesc::new(
                    bb0.image().size().wh(),
                    br::vk::VK_FORMAT_D24_UNORM_S8_UINT,
                    br::ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                    br::ImageLayout::Undefined,
                ),
            )
            .expect("Failed to allocate depth texture");
        let depth_texture_view = SharedRef::new(
            depth_texture
                .subresource_range(br::AspectMask::DEPTH.stencil(), 0..1, 0..1)
                .view_builder()
                .create()
                .expect("Failed to create depth buffer view"),
        );
        {
            let depth_texture = RangedImage::single_depth_stencil_plane(depth_texture_view.image());

            PipelineBarrier::new()
                .by_region()
                .with_barrier(depth_texture.barrier(
                    br::ImageLayout::Undefined,
                    br::ImageLayout::DepthStencilAttachmentOpt,
                ))
                .submit(e)
                .expect("Failed to submit initial barrier");
        }

        let frame_buffers: Vec<_> = e
            .iter_back_buffers()
            .map(|b| {
                e.graphics()
                    .device()
                    .clone()
                    .new_framebuffer(
                        &const_res.render_pass,
                        vec![
                            b.clone()
                                as Arc<
                                    dyn br::ImageView<ConcreteDevice = peridot::DeviceObject>
                                        + Send
                                        + Sync,
                                >,
                            depth_texture_view.clone(),
                        ],
                        b.image().size().as_ref(),
                        1,
                    )
                    .expect("Failed to create framebuffer")
            })
            .collect();

        let area = bb0.image().size().wh().into_rect(br::vk::VkOffset2D::ZERO);
        let viewport = area.make_viewport(0.0..1.0);

        let unlit_colored_vps = const_res
            .unlit_colored_shader
            .generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
        let mut pb = br::GraphicsPipelineBuilder::<
            _,
            br::PipelineObject<peridot::DeviceObject>,
            _,
            _,
            _,
            _,
            _,
            _,
        >::new(
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
            pb.create(
                e.graphics().device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )
            .expect("Failed to create grid render pipeline"),
            const_res.unlit_colored_pipeline_layout.clone(),
        );

        let pbr_vps = const_res
            .pbr_shader
            .generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
        pb.layout(&const_res.pbr_pipeline_layout)
            .vertex_processing(pbr_vps);
        let pbr_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(
                e.graphics().device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )
            .expect("Failed to create pbr pipeline"),
            const_res.pbr_pipeline_layout.clone(),
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
            pb.create(
                e.graphics().device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )
            .expect("Failed to create ui_fill_rect pipeline"),
            const_res.unlit_colored_ext_pipeline.clone(),
        );
        let ui_border_line_vps = const_res
            .unlit_colored_ext_shader
            .generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
        pb.vertex_processing(ui_border_line_vps);
        let ui_border_line_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(
                e.graphics().device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )
            .expect("Failed to create ui_border_line pipeline"),
            const_res.unlit_colored_ext_pipeline.clone(),
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
            pb.create(
                e.graphics().device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )
            .expect("Failed to create vg interior pipeline"),
            const_res.vg_pipeline_layout.clone(),
        );
        pb.vertex_processing(vg_curve_vps.clone());
        let vg_curve_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(
                e.graphics().device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )
            .expect("Failed to create vg curve pipeline"),
            const_res.vg_pipeline_layout.clone(),
        );

        let inv_blend = ColorAttachmentBlending::new(
            Blending::source_only(br::BlendFactor::OneMinusDestColor),
            Blending::STRAIGHT_DEST,
        );
        pb.set_attachment_blends(vec![inv_blend.into_vk()]);
        pb.vertex_processing(vg_interior_vps.clone());
        let vg_interior_inv_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(
                e.graphics().device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )
            .expect("Failed to create vg interior inv color pipeline"),
            const_res.vg_pipeline_layout.clone(),
        );
        pb.vertex_processing(vg_curve_vps);
        let vg_curve_inv_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(
                e.graphics().device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )
            .expect("Failed to create vg curve inv color pipeline"),
            const_res.vg_pipeline_layout.clone(),
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
            pb.create(
                e.graphics().device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )
            .expect("Failed to create vg curve mask pipeline"),
            const_res.vg_pipeline_layout.clone(),
        );
        pb.layout(&const_res.vg_pipeline_layout);
        pb.vertex_processing(vg_interior_vps);
        let vg_interior_mask_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(
                e.graphics().device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )
            .expect("Failed to create vg interior pipeline"),
            const_res.vg_pipeline_layout.clone(),
        );

        pb.stencil_test_enable(false);
        pb.depth_write_enable(false);
        pb.render_pass(&const_res.render_pass, 1);
        pb.layout(&const_res.skybox_shader_layout);
        pb.vertex_processing(
            const_res
                .skybox_shader
                .generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST),
        );
        pb.set_attachment_blends(vec![ColorAttachmentBlending::Disabled.into_vk()]);
        pb.depth_compare_op(br::CompareOp::LessOrEqual);
        let skybox_render_pipeline = peridot::LayoutedPipeline::combine(
            pb.create(
                e.graphics().device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )
            .expect("Failed to create skybox render pipeline"),
            const_res.skybox_shader_layout.clone(),
        );

        Self {
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
            skybox_render_pipeline,
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
    camera_vp_separated: u64,
}
pub struct StaticBufferOffsets {
    grid: u64,
    icosphere_vertices: u64,
    icosphere_indices: u64,
    ui: peridot_vg::ContextPreallocOffsets,
    ui_mask: peridot_vg::ContextPreallocOffsets,
    ui_border_line_indices: u64,
    ui_fill_rect_indices: u64,
    skybox_cube: u64,
    skybox_cube_indices: u64,
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
    fn stage_data(
        &mut self,
        m: &br::MappedMemoryRange<impl br::DeviceMemory + br::VkHandleMut + ?Sized>,
    ) {
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
            m.slice_mut::<peridot::math::Vector4F32>(self.offsets.skybox_cube as _, 8)
                .clone_from_slice(&[
                    peridot::math::Vector4(-1.0, -1.0, -1.0, 1.0),
                    peridot::math::Vector4(1.0, -1.0, -1.0, 1.0),
                    peridot::math::Vector4(1.0, 1.0, -1.0, 1.0),
                    peridot::math::Vector4(-1.0, 1.0, -1.0, 1.0),
                    peridot::math::Vector4(-1.0, -1.0, 1.0, 1.0),
                    peridot::math::Vector4(1.0, -1.0, 1.0, 1.0),
                    peridot::math::Vector4(1.0, 1.0, 1.0, 1.0),
                    peridot::math::Vector4(-1.0, 1.0, 1.0, 1.0),
                ]);
            m.slice_mut::<u16>(self.offsets.skybox_cube_indices as _, 36)
                .copy_from_slice(&[
                    0, 1, 3, 1, 3, 2, 4, 5, 7, 5, 7, 6, 0, 3, 4, 3, 4, 7, 1, 2, 5, 2, 5, 6, 0, 1,
                    4, 1, 4, 5, 2, 3, 6, 3, 6, 7,
                ]);
        }
    }

    fn buffer_graphics_ready<Device: br::Device + 'static>(
        &self,
        tfb: &mut peridot::TransferBatch,
        buf: &Arc<
            peridot::Buffer<
                impl br::Buffer<ConcreteDevice = Device> + 'static,
                impl br::DeviceMemory<ConcreteDevice = Device> + 'static,
            >,
        >,
        range: std::ops::Range<u64>,
    ) {
        tfb.add_buffer_graphics_ready(
            br::PipelineStageFlags::VERTEX_INPUT,
            buf.clone(),
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
    camera_vp_separated_offset: Option<u64>,
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
            camera_vp_separated_offset: None,
        }
    }
}

const PREFILTERED_ENVMAP_MIP_LEVELS: u32 = 5;
const PREFILTERED_ENVMAP_SIZE: u32 = 128;

pub struct Memory {
    manager: MemoryManager,
    mem: peridot::FixedMemory<
        peridot::DeviceObject,
        peridot::Buffer<
            br::BufferObject<peridot::DeviceObject>,
            br::DeviceMemoryObject<peridot::DeviceObject>,
        >,
    >,
    static_offsets: StaticBufferOffsets,
    mutable_offsets: MutableBufferOffsets,
    dynamic_stg: DynamicStagingBuffer,
    update_sets: UpdateSets,
    icosphere_vertex_count: usize,
    ui_render_params: peridot_vg::RendererParams,
    ui_mask_render_params: peridot_vg::RendererParams,
    ui_transform_buffer_view: br::BufferViewObject<
        SharedRef<
            peridot::Buffer<
                br::BufferObject<peridot::DeviceObject>,
                br::DeviceMemoryObject<peridot::DeviceObject>,
            >,
        >,
    >,
    ui_mask_transform_buffer_view: br::BufferViewObject<
        SharedRef<
            peridot::Buffer<
                br::BufferObject<peridot::DeviceObject>,
                br::DeviceMemoryObject<peridot::DeviceObject>,
            >,
        >,
    >,
    dwt_ibl_cubemap: br::ImageViewObject<peridot_memory_manager::Image>,
    dwt_irradiance_cubemap: br::ImageViewObject<peridot_memory_manager::Image>,
    dwt_prefiltered_envmap: peridot_memory_manager::Image,
}
impl Memory {
    pub fn new(
        e: &peridot::Engine<impl peridot::NativeLinker>,
        tfb: &mut peridot::TransferBatch,
        ui: &peridot_vg::Context,
        ui_mask: &peridot_vg::Context,
    ) -> Self {
        let mut manager = MemoryManager::new(e.graphics());

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
            skybox_cube: static_bp
                .add(peridot::BufferContent::vertices::<peridot::math::Vector4F32>(8)),
            skybox_cube_indices: static_bp.add(peridot::BufferContent::indices::<u16>(36)),
            ui: ui.prealloc(&mut static_bp),
            ui_mask: ui_mask.prealloc(&mut static_bp),
        };
        let textures = peridot::TextureInitializationGroup::new(e.graphics().device().clone());
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
            camera_vp_separated: mutable_bp.add(peridot::BufferContent::uniform::<
                [peridot::math::Matrix4F32; 2],
            >()),
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

        let dwts = manager
            .allocate_multiple_device_local_images(
                e.graphics(),
                [
                    br::ImageDesc::new(
                        peridot::math::Vector2(512u32, 512),
                        peridot::PixelFormat::RGBA64F as _,
                        br::ImageUsage::COLOR_ATTACHMENT.sampled(),
                        br::ImageLayout::Undefined,
                    )
                    .flags(br::ImageFlags::CUBE_COMPATIBLE)
                    .array_layers(6),
                    br::ImageDesc::new(
                        peridot::math::Vector2(32u32, 32),
                        peridot::PixelFormat::RGBA64F as _,
                        br::ImageUsage::COLOR_ATTACHMENT.sampled(),
                        br::ImageLayout::Undefined,
                    )
                    .flags(br::ImageFlags::CUBE_COMPATIBLE)
                    .array_layers(6),
                    br::ImageDesc::new(
                        peridot::math::Vector2(PREFILTERED_ENVMAP_SIZE, PREFILTERED_ENVMAP_SIZE),
                        peridot::PixelFormat::RGBA64F as _,
                        br::ImageUsage::COLOR_ATTACHMENT.sampled(),
                        br::ImageLayout::Undefined,
                    )
                    .flags(br::ImageFlags::CUBE_COMPATIBLE)
                    .array_layers(6)
                    .mip_levels(PREFILTERED_ENVMAP_MIP_LEVELS),
                ],
            )
            .expect("Failed to allocate dwts");
        let Ok::<[_; 3], _>([dwt_ibl_cubemap, dwt_irradiance_cubemap, dwt_prefiltered_envmap]) = dwts.try_into() else {
            unreachable!("unknown combination");
        };

        Self {
            dynamic_stg: DynamicStagingBuffer::new(e.graphics(), &mut manager)
                .expect("Failed to create dynamic staging buffer"),
            manager,
            ui_transform_buffer_view: mem
                .buffer
                .object
                .clone()
                .create_view(
                    br::vk::VK_FORMAT_R32G32B32A32_SFLOAT,
                    ui_render_params.transforms_byterange(),
                )
                .expect("Failed to create buffer view of transforms"),
            ui_mask_transform_buffer_view: mem
                .buffer
                .object
                .clone()
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
            update_sets: UpdateSets::new(),
            icosphere_vertex_count: icosphere.indices.len(),
            dwt_ibl_cubemap: dwt_ibl_cubemap
                .subresource_range(br::AspectMask::COLOR, 0..1, 0..6)
                .view_builder()
                .with_dimension(br::vk::VK_IMAGE_VIEW_TYPE_CUBE)
                .create()
                .expect("Failed to create IBL cubemap view"),
            dwt_irradiance_cubemap: dwt_irradiance_cubemap
                .subresource_range(br::AspectMask::COLOR, 0..1, 0..6)
                .view_builder()
                .with_dimension(br::vk::VK_IMAGE_VIEW_TYPE_CUBE)
                .create()
                .expect("Failed to create irradiance cubemap view"),
            dwt_prefiltered_envmap,
        }
    }

    pub fn apply_main_camera(
        &mut self,
        e: &mut peridot::Engine<impl peridot::NativeLinker>,
        camera: &peridot::math::Camera,
        aspect: f32,
    ) {
        let camera_matrix = camera.view_projection_matrix(aspect);

        self.update_sets.grid_mvp_stg_offset = Some(self.dynamic_stg.push(
            e,
            &mut self.manager,
            camera_matrix.clone(),
        ));
        self.update_sets.object_mvp_stg_offset = Some(self.dynamic_stg.push(
            e,
            &mut self.manager,
            ObjectTransform {
                mvp: camera_matrix,
                model_transform: peridot::math::Matrix4::ONE,
                view_projection: camera.view_matrix(),
            },
        ));
        self.update_sets.camera_vp_separated_offset = Some(self.dynamic_stg.push(
            e,
            &mut self.manager,
            [camera.projection_matrix(aspect), camera.view_matrix()],
        ))
    }

    pub fn set_camera_info(
        &mut self,
        e: &mut peridot::Engine<impl peridot::NativeLinker>,
        info: RasterizationCameraInfo,
    ) {
        self.update_sets.camera_info_stg_offset =
            Some(self.dynamic_stg.push(e, &mut self.manager, info));
    }

    pub fn set_directional_light_info(
        &mut self,
        e: &mut peridot::Engine<impl peridot::NativeLinker>,
        info: RasterizationDirectionalLightInfo,
    ) {
        self.update_sets.directional_light_info_stg_offset =
            Some(self.dynamic_stg.push(e, &mut self.manager, info));
    }

    pub fn set_material(
        &mut self,
        e: &mut peridot::Engine<impl peridot::NativeLinker>,
        info: MaterialInfo,
    ) {
        self.update_sets.material_stg_offset =
            Some(self.dynamic_stg.push(e, &mut self.manager, info));
    }

    pub fn update_ui_fill_rect_vertices(
        &mut self,
        e: &mut peridot::Engine<impl peridot::NativeLinker>,
        vertices: &[peridot::math::Vector2F32; mesh::UI_FILL_RECT_COUNT],
    ) {
        self.update_sets.ui_fill_rects = Some(self.dynamic_stg.push_multiple_values(
            e,
            &mut self.manager,
            vertices,
        ));
    }

    pub fn construct_new_ui_fill_rect_vertices(
        &mut self,
        e: &mut peridot::Engine<impl peridot::NativeLinker>,
        ctor: impl FnOnce(&mut [peridot::math::Vector2F32]),
    ) {
        self.update_sets.ui_fill_rects = Some(self.dynamic_stg.construct_multiple_values_inplace(
            e,
            &mut self.manager,
            mesh::UI_FILL_RECT_COUNT,
            ctor,
        ));
    }

    pub fn set_ui_transform(
        &mut self,
        e: &mut peridot::Engine<impl peridot::NativeLinker>,
        transform: peridot::math::Matrix4F32,
    ) {
        self.update_sets.ui_transform =
            Some(self.dynamic_stg.push(e, &mut self.manager, transform));
    }

    pub fn ready_transfer<'s>(
        &'s mut self,
        e: &peridot::Graphics,
        tfb: &mut peridot::TransferBatch2,
    ) {
        self.dynamic_stg.end_mapped(e);
        let get_dynamic_stg_buffer = || self.dynamic_stg.buffer().borrow();

        if let Some(o) = self.update_sets.grid_mvp_stg_offset.take() {
            let target_offset = self.mem.mut_buffer_placement + self.mutable_offsets.grid_mvp;

            tfb.copy_buffer(
                self.dynamic_stg.buffer().clone(),
                o,
                self.mem.buffer.object.clone(),
                target_offset,
                core::mem::size_of::<peridot::math::Matrix4F32>() as _,
            );
            tfb.register_outer_usage(
                br::PipelineStageFlags::VERTEX_SHADER.0,
                self.mem.buffer.object.clone(),
                target_offset
                    ..target_offset + core::mem::size_of::<peridot::math::Matrix4F32>() as u64,
                br::AccessFlags::UNIFORM_READ,
            );
        }

        if let Some(o) = self.update_sets.object_mvp_stg_offset.take() {
            let r = self.object_transform_buffer().clone_inner_ref();

            r.clone()
                .batched_copy_from(tfb, self.dynamic_stg.buffer().clone(), o);
            r.batching_set_outer_usage(tfb, BufferUsage::VERTEX_UNIFORM);
        }

        if let Some(o) = self.update_sets.camera_info_stg_offset.take() {
            let r = self.camera_info_buffer().clone_inner_ref();

            r.clone()
                .batched_copy_from(tfb, self.dynamic_stg.buffer().clone(), o);
            r.batching_set_outer_usage(tfb, BufferUsage::FRAGMENT_UNIFORM);
        }

        if let Some(o) = self.update_sets.directional_light_info_stg_offset.take() {
            let r = self.directional_light_info_buffer().clone_inner_ref();

            r.clone()
                .batched_copy_from(tfb, self.dynamic_stg.buffer().clone(), o);
            r.batching_set_outer_usage(tfb, BufferUsage::FRAGMENT_UNIFORM);
        }

        if let Some(o) = self.update_sets.material_stg_offset.take() {
            let r = self.material_buffer().clone_inner_ref();

            r.clone()
                .batched_copy_from(tfb, self.dynamic_stg.buffer().clone(), o);
            r.batching_set_outer_usage(tfb, BufferUsage::FRAGMENT_UNIFORM);
        }

        if let Some(o) = self.update_sets.ui_fill_rects.take() {
            let r = self.ui_fill_rects_buffer().clone_inner_ref();

            r.clone()
                .batched_copy_from(tfb, self.dynamic_stg.buffer().clone(), o);
            r.batching_set_outer_usage(tfb, BufferUsage::VERTEX_BUFFER);
        }

        if let Some(o) = self.update_sets.ui_transform.take() {
            let r = self.ui_transform_buffer().clone_inner_ref();

            r.clone()
                .batched_copy_from(tfb, self.dynamic_stg.buffer().clone(), o);
            r.batching_set_outer_usage(tfb, BufferUsage::VERTEX_UNIFORM);
        }

        if let Some(o) = self.update_sets.camera_vp_separated_offset.take() {
            let r = self.camera_vp_separated_buffer().clone_inner_ref();

            r.clone()
                .batched_copy_from(tfb, self.dynamic_stg.buffer().clone(), o);
            r.batching_set_outer_usage(tfb, BufferUsage::VERTEX_UNIFORM);
        }
    }

    pub fn grid_transform_range(&self) -> std::ops::Range<u64> {
        let target_offset = self.mem.mut_buffer_placement + self.mutable_offsets.grid_mvp;

        target_offset..target_offset + std::mem::size_of::<peridot::math::Matrix4F32>() as u64
    }
    pub fn grid_transform_buffer<'s>(
        &'s self,
    ) -> RangedBuffer<&'s SharedRef<impl br::Buffer + peridot::TransferrableBufferResource>> {
        RangedBuffer(&self.mem.buffer.object, self.grid_transform_range())
    }

    pub fn object_transform_range(&self) -> std::ops::Range<u64> {
        let target_offset = self.mem.mut_buffer_placement + self.mutable_offsets.object_mvp;

        target_offset..target_offset + std::mem::size_of::<ObjectTransform>() as u64
    }
    pub fn object_transform_buffer<'s>(
        &'s self,
    ) -> RangedBuffer<&'s SharedRef<impl br::Buffer + peridot::TransferrableBufferResource>> {
        RangedBuffer(&self.mem.buffer.object, self.object_transform_range())
    }

    pub fn camera_info_range(&self) -> std::ops::Range<u64> {
        self.mem.range_in_mut_buffer(
            self.mutable_offsets.camera_info
                ..self.mutable_offsets.camera_info
                    + std::mem::size_of::<RasterizationCameraInfo>() as u64,
        )
    }
    pub fn camera_info_buffer<'s>(
        &'s self,
    ) -> RangedBuffer<&'s SharedRef<impl br::Buffer + peridot::TransferrableBufferResource>> {
        RangedBuffer(&self.mem.buffer.object, self.camera_info_range())
    }

    pub fn directional_light_info_range(&self) -> std::ops::Range<u64> {
        self.mem.range_in_mut_buffer(
            self.mutable_offsets.directional_light_info
                ..self.mutable_offsets.directional_light_info
                    + std::mem::size_of::<RasterizationDirectionalLightInfo>() as u64,
        )
    }
    pub fn directional_light_info_buffer<'s>(
        &'s self,
    ) -> RangedBuffer<&'s SharedRef<impl br::Buffer + peridot::TransferrableBufferResource>> {
        RangedBuffer(&self.mem.buffer.object, self.directional_light_info_range())
    }

    pub fn material_range(&self) -> std::ops::Range<u64> {
        self.mem.range_in_mut_buffer(
            self.mutable_offsets.material
                ..self.mutable_offsets.material + std::mem::size_of::<MaterialInfo>() as u64,
        )
    }
    pub fn material_buffer<'s>(
        &'s self,
    ) -> RangedBuffer<&'s SharedRef<impl br::Buffer + peridot::TransferrableBufferResource>> {
        RangedBuffer(&self.mem.buffer.object, self.material_range())
    }

    pub fn ui_fill_rects_range(&self) -> std::ops::Range<u64> {
        self.mem.range_in_mut_buffer(
            self.mutable_offsets.ui_fill_rects
                ..self.mutable_offsets.ui_fill_rects
                    + std::mem::size_of::<[peridot::math::Vector2F32; mesh::UI_FILL_RECT_COUNT]>()
                        as u64,
        )
    }
    pub fn ui_fill_rects_buffer<'s>(
        &'s self,
    ) -> RangedBuffer<&'s SharedRef<impl br::Buffer + peridot::TransferrableBufferResource>> {
        RangedBuffer(&self.mem.buffer.object, self.ui_fill_rects_range())
    }

    pub fn ui_transform_range(&self) -> std::ops::Range<u64> {
        self.mem.range_in_mut_buffer(
            self.mutable_offsets.ui_transform
                ..self.mutable_offsets.ui_transform
                    + std::mem::size_of::<peridot::math::Matrix4F32>() as u64,
        )
    }
    pub fn ui_transform_buffer<'s>(
        &'s self,
    ) -> RangedBuffer<&'s SharedRef<impl br::Buffer + peridot::TransferrableBufferResource>> {
        RangedBuffer(&self.mem.buffer.object, self.ui_transform_range())
    }

    pub fn camera_vp_separated_range(&self) -> std::ops::Range<u64> {
        self.mem.range_in_mut_buffer(
            self.mutable_offsets.camera_vp_separated
                ..self.mutable_offsets.camera_vp_separated
                    + std::mem::size_of::<[peridot::math::Matrix4F32; 2]>() as u64,
        )
    }
    pub fn camera_vp_separated_buffer<'s>(
        &'s self,
    ) -> RangedBuffer<&'s SharedRef<impl br::Buffer + peridot::TransferrableBufferResource>> {
        RangedBuffer(&self.mem.buffer.object, self.camera_vp_separated_range())
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

pub struct RenderBundle {
    pool: br::CommandPoolObject<peridot::DeviceObject>,
    buffers: Vec<br::CommandBufferObject<peridot::DeviceObject>>,
}
impl RenderBundle {
    pub fn new(g: &peridot::Graphics, count: u32) -> br::Result<Self> {
        let mut pool = br::CommandPoolBuilder::new(g.graphics_queue_family_index())
            .create(g.device().clone())?;

        Ok(Self {
            buffers: pool.alloc(count, false)?,
            pool,
        })
    }

    pub fn reset(&mut self) -> br::Result<()> {
        self.pool.reset(false)
    }

    pub fn synchronized(
        &mut self,
        index: usize,
    ) -> br::SynchronizedCommandBuffer<
        br::CommandPoolObject<peridot::DeviceObject>,
        br::CommandBufferObject<peridot::DeviceObject>,
    > {
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
        let init_rot = peridot::math::Quaternion::new(init_yrot, peridot::math::Vector3::right())
            * peridot::math::Quaternion::new(init_xrot, peridot::math::Vector3::up());

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
                peridot::math::Quaternion::new(*self.yrot.get(), peridot::math::Vector3::right())
                    * peridot::math::Quaternion::new(
                        *self.xrot.get(),
                        peridot::math::Vector3::up(),
                    );
        }

        let mx = e.input().analog_value_abs(ID_CAMERA_MOVE_AX_X);
        let my = e.input().analog_value_abs(ID_CAMERA_MOVE_AX_Y);
        let mz = e.input().analog_value_abs(ID_CAMERA_MOVE_AX_Z);

        if mx != 0.0 || my != 0.0 || mz != 0.0 {
            let xzv = peridot::math::Matrix3::from(peridot::math::Quaternion::new(
                *self.xrot.get(),
                peridot::math::Vector3::up(),
            )) * peridot::math::Vector3(mx, 0.0, mz);
            self.camera.modify().0.position +=
                (xzv + peridot::math::Vector3(0.0, my, 0.0)) * 2.0 * dt.as_secs_f32();
        }
    }
}

fn init_controls(e: &mut peridot::Engine<impl peridot::NativeLinker>) {
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
}

pub struct Game<NL: peridot::NativeLinker> {
    const_res: ConstResources,
    descriptors: DescriptorStore,
    mem: Memory,
    screen_res: ScreenResources,
    ui_dynamic_buffers: UIRenderingBuffers,
    command_buffers: peridot::CommandBundle<peridot::DeviceObject>,
    update_commands: peridot::CommandBundle<peridot::DeviceObject>,
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
    last_frame_tfb2: peridot::TransferBatch2,
    ph: std::marker::PhantomData<*const NL>,
}
impl<NL: peridot::NativeLinker> peridot::FeatureRequests for Game<NL> {}
impl<NL: peridot::NativeLinker + Sync> peridot::EngineEvents<NL> for Game<NL>
where
    NL::Presenter: Sync,
    <NL::Presenter as peridot::PlatformPresenter>::BackBuffer: Send + Sync,
{
    fn init(e: &mut peridot::Engine<NL>) -> Self {
        init_controls(e);

        let material_data = DirtyTracker::new(MaterialInfo {
            base_color: peridot::math::Vector4(1.0, 1.0, 1.0, 1.0),
            roughness: 0.4,
            anisotropic: 0.0,
            metallic: 0.0,
            reflectance: 0.5,
        });

        let bb0 = e.back_buffer(0).expect("no back-buffers?");
        let render_area = bb0.image().size().wh().into_rect(br::vk::VkOffset2D::ZERO);
        let aspect = render_area.extent.width as f32 / render_area.extent.height as f32;

        let mut ui = peridot_vg::Context::new(e.rendering_precision());
        let mut ui_control_mask = peridot_vg::Context::new(e.rendering_precision());
        let mut ui_dynamic_texts = peridot_vg::Context::new(e.rendering_precision());
        let font = Rc::new(
            peridot_vg::DefaultFontProvider::new()
                .expect("Failed to create FontProvider")
                .best_match("Yu Gothic UI", &peridot_vg::FontProperties::default(), 18.0)
                .expect("Failed to find best match font"),
        );
        let font_sm = Rc::new(
            peridot_vg::DefaultFontProvider::new()
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
        let mut tfb2 = peridot::TransferBatch2::new();
        let mut mem = Memory::new(e, &mut tfb, &ui, &ui_control_mask);
        let main_camera = FreeCameraView::new(0.0, -5.0f32.to_radians(), 5.0, aspect);
        mem.apply_main_camera(e, &main_camera.camera.get().0, aspect);
        mem.set_camera_info(
            e,
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
            e,
            RasterizationDirectionalLightInfo {
                dir: peridot::math::Vector4(0.2f32, 0.3, -0.5, 0.0).normalize(),
                intensity: peridot::math::Vector4(2.0, 2.0, 2.0, 1.0),
            },
        );
        mem.set_material(e, material_data.get().clone());
        mem.construct_new_ui_fill_rect_vertices(e, |vs| {
            for r in renderables {
                r.render_dynamic_mesh(vs);
            }
        });
        mem.set_ui_transform(
            e,
            peridot::math::Camera {
                projection: Some(peridot::math::ProjectionMethod::UI {
                    design_width: bb0.image().size().width as _,
                    design_height: bb0.image().size().height as _,
                }),
                ..Default::default()
            }
            .projection_matrix(aspect),
        );
        mem.ready_transfer(e.graphics(), &mut tfb2);

        let ui_dynamic_buffers =
            UIRenderingBuffers::new(e.graphics(), &mut mem.manager, &ui_dynamic_texts, &mut tfb)
                .expect("Failed to allocate ui dynamic buffers");

        let background_asset: peridot_image::HDR = e
            .load("background.GCanyon_C_YumaPoint_3k")
            .expect("Failed to load background image");
        let mut tmp_loaded_image = br::ImageDesc::new(
            peridot::math::Vector2(background_asset.info.width, background_asset.info.height),
            br::vk::VK_FORMAT_R16G16B16A16_SFLOAT,
            br::ImageUsage::SAMPLED,
            br::ImageLayout::Preinitialized,
        )
        .use_linear_tiling()
        .create(e.graphics().device().clone())
        .expect("Failed to create tmp image data");
        let tmp_loaded_image_mreq = tmp_loaded_image.requirements();

        let mut bp = peridot::BufferPrealloc::new(e.graphics());
        let fillrect_offset = bp.add(peridot::BufferContent::vertex::<
            [peridot::math::Vector4F32; 4],
        >());
        let cube_ref_positions_offset = bp.add(peridot::BufferContent::vertex::<
            [[peridot::math::Vector4F32; 4]; 6],
        >());
        let buffer_data_size = bp.total_size();
        let mut buffer = bp
            .build_custom_usage(br::BufferUsage::VERTEX_BUFFER)
            .expect("Failed to build hdr import buffer");
        let buffer_mreq = buffer.requirements();
        let buffer_placement_offset = ((tmp_loaded_image_mreq.size + (buffer_mreq.alignment - 1))
            / buffer_mreq.alignment)
            * buffer_mreq.alignment;

        let mx = e
            .graphics()
            .memory_type_manager
            .host_visible_index(
                tmp_loaded_image_mreq.memoryTypeBits & buffer_mreq.memoryTypeBits,
                br::MemoryPropertyFlags::HOST_COHERENT,
            )
            .expect("No suitable memory location for background image initialization");
        let mut tmp_loaded_image_mem = br::DeviceMemoryRequest::allocate(
            (buffer_placement_offset + buffer_mreq.size) as _,
            mx.index(),
        )
        .execute(e.graphics().device().clone())
        .expect("Failed to allocate background image mmeory");
        tmp_loaded_image
            .bind(&tmp_loaded_image_mem, 0)
            .expect("Failed to bind memory");
        buffer
            .bind(&tmp_loaded_image_mem, buffer_placement_offset as _)
            .expect("Failed to bind memory");
        let tmp_loaded_image_view = tmp_loaded_image
            .subresource_range(br::AspectMask::COLOR, 0..1, 0..1)
            .view_builder()
            .create()
            .expect("Failed to create background image view");

        let m0 = tmp_loaded_image_mem
            .map(0..(buffer_placement_offset + buffer_data_size) as _)
            .expect("Failed to map memory");
        let row_stride = tmp_loaded_image_view
            .image()
            .subresource(br::AspectMask::COLOR, 0, 0)
            .layout_info()
            .rowPitch;
        for r in 0..background_asset.info.height {
            let row_source = background_asset.pixels[r as usize
                * background_asset.info.width as usize
                ..(r + 1) as usize * background_asset.info.width as usize]
                .iter()
                .flat_map(|c| {
                    c.to_hdr()
                        .0
                        .into_iter()
                        .map(half::f16::from_f32)
                        .chain(std::iter::once(half::f16::from_f32(1.0)))
                        .flat_map(|v| v.to_le_bytes())
                })
                .collect::<Vec<_>>();
            unsafe {
                m0.slice_mut::<u8>(r as usize * row_stride as usize, row_stride as _)
                    .copy_from_slice(&row_source);
            }
        }
        unsafe {
            m0.slice_mut::<peridot::math::Vector4F32>(
                (buffer_placement_offset + fillrect_offset) as _,
                4,
            )
            .clone_from_slice(&[
                peridot::math::Vector4(-1.0, -1.0, 0.0, 1.0),
                peridot::math::Vector4(1.0, -1.0, 0.0, 1.0),
                peridot::math::Vector4(-1.0, 1.0, 0.0, 1.0),
                peridot::math::Vector4(1.0, 1.0, 0.0, 1.0),
            ]);
            m0.slice_mut::<[peridot::math::Vector4F32; 4]>(
                (buffer_placement_offset + cube_ref_positions_offset) as _,
                6,
            )
            .clone_from_slice(&[
                [
                    peridot::math::Vector4(1.0, 1.0, 1.0, 1.0),
                    peridot::math::Vector4(1.0, 1.0, -1.0, 1.0),
                    peridot::math::Vector4(1.0, -1.0, 1.0, 1.0),
                    peridot::math::Vector4(1.0, -1.0, -1.0, 1.0),
                ],
                [
                    peridot::math::Vector4(-1.0, 1.0, -1.0, 1.0),
                    peridot::math::Vector4(-1.0, 1.0, 1.0, 1.0),
                    peridot::math::Vector4(-1.0, -1.0, -1.0, 1.0),
                    peridot::math::Vector4(-1.0, -1.0, 1.0, 1.0),
                ],
                [
                    peridot::math::Vector4(-1.0, 1.0, -1.0, 1.0),
                    peridot::math::Vector4(1.0, 1.0, -1.0, 1.0),
                    peridot::math::Vector4(-1.0, 1.0, 1.0, 1.0),
                    peridot::math::Vector4(1.0, 1.0, 1.0, 1.0),
                ],
                [
                    peridot::math::Vector4(-1.0, -1.0, 1.0, 1.0),
                    peridot::math::Vector4(1.0, -1.0, 1.0, 1.0),
                    peridot::math::Vector4(-1.0, -1.0, -1.0, 1.0),
                    peridot::math::Vector4(1.0, -1.0, -1.0, 1.0),
                ],
                [
                    peridot::math::Vector4(-1.0, 1.0, 1.0, 1.0),
                    peridot::math::Vector4(1.0, 1.0, 1.0, 1.0),
                    peridot::math::Vector4(-1.0, -1.0, 1.0, 1.0),
                    peridot::math::Vector4(1.0, -1.0, 1.0, 1.0),
                ],
                [
                    peridot::math::Vector4(1.0, 1.0, -1.0, 1.0),
                    peridot::math::Vector4(-1.0, 1.0, -1.0, 1.0),
                    peridot::math::Vector4(1.0, -1.0, -1.0, 1.0),
                    peridot::math::Vector4(-1.0, -1.0, -1.0, 1.0),
                ],
            ]);
        }
        m0.end();

        let precompute_rp = br::RenderPassBuilder::new()
            .add_attachment(
                br::AttachmentDescription::new(
                    br::vk::VK_FORMAT_R16G16B16A16_SFLOAT,
                    br::ImageLayout::Undefined,
                    br::ImageLayout::ShaderReadOnlyOpt,
                )
                .store_op(br::StoreOp::Store),
            )
            .add_subpass(br::SubpassDescription::new().add_color_output(
                0,
                br::ImageLayout::ColorAttachmentOpt,
                None,
            ))
            .add_dependency(br::vk::VkSubpassDependency {
                srcSubpass: 0,
                dstSubpass: br::vk::VK_SUBPASS_EXTERNAL,
                srcStageMask: br::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.0,
                dstStageMask: br::PipelineStageFlags::FRAGMENT_SHADER.0,
                srcAccessMask: br::AccessFlags::COLOR_ATTACHMENT.write,
                dstAccessMask: br::AccessFlags::SHADER.read,
                dependencyFlags: br::vk::VK_DEPENDENCY_BY_REGION_BIT,
            })
            .create(e.graphics().device().clone())
            .expect("Failed to create equirectangular to cubemap render pass");
        let equirect_to_cubemap_fbs = (0..6)
            .map(|l| {
                let iv = mem
                    .dwt_ibl_cubemap
                    .by_ref()
                    .subresource_range(br::AspectMask::COLOR, 0..1, l..l + 1)
                    .view_builder()
                    .create()?;
                let s = iv.size().as_ref();

                e.graphics()
                    .device()
                    .clone()
                    .new_framebuffer(&precompute_rp, vec![iv], s, 1)
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to create equirect to cubemap frame buffer");
        let irradiance_precompute_fbs = (0..6)
            .map(|l| {
                let iv = mem
                    .dwt_irradiance_cubemap
                    .by_ref()
                    .subresource_range(br::AspectMask::COLOR, 0..1, l..l + 1)
                    .view_builder()
                    .create()?;
                let s = iv.size().as_ref();

                e.graphics()
                    .device()
                    .clone()
                    .new_framebuffer(&precompute_rp, vec![iv], s, 1)
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to create irradiance precompute frame buffer");
        let prefiltered_envmap_fbs = (0..6)
            .flat_map(|d| (0..PREFILTERED_ENVMAP_MIP_LEVELS).map(move |ml| (d, ml)))
            .map(|(d, ml)| {
                let iv = mem
                    .dwt_prefiltered_envmap
                    .by_ref()
                    .subresource_range(br::AspectMask::COLOR, ml..ml + 1, d..d + 1)
                    .view_builder()
                    .create()?;

                e.graphics().device().clone().new_framebuffer(
                    &precompute_rp,
                    vec![iv],
                    &br::vk::VkExtent2D {
                        width: (PREFILTERED_ENVMAP_SIZE as f32 * 0.5f32.powi(ml as _)) as _,
                        height: (PREFILTERED_ENVMAP_SIZE as f32 * 0.5f32.powi(ml as _)) as _,
                    },
                    1,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to create prefiltered envmap frame buffer");

        let equirect_to_cubemap_shader = peridot_vertex_processing_pack::PvpShaderModules::new(
            e.graphics().device(),
            e.load("shaders.equirectangular_to_cubemap")
                .expect("Failed to load equirectangular to cubemap shader"),
        )
        .expect("Failed to create equirectangular to cubemap shader modules");
        let irradiance_precompute_shader = PvpShaderModules::new(
            e.graphics().device(),
            e.load("shaders.irradiance_convolution")
                .expect("Failed to load irradiance convolution shader"),
        )
        .expect("Failed to create irradiance convolution shader modules");
        let prefilter_envmap_shader = PvpShaderModules::new(
            e.graphics().device(),
            e.load("shaders.prefilter_env")
                .expect("Failed to load envmap prefilter shader"),
        )
        .expect("Failed to create envmap prefilter shader modules");
        let linear_smp = br::SamplerBuilder::default()
            .create(e.graphics().device().clone())
            .expect("Failed to default sampler");
        let dsl = DetailedDescriptorSetLayout::new(
            e.graphics(),
            vec![br::DescriptorType::CombinedImageSampler
                .make_binding(1)
                .only_for_fragment()
                .with_immutable_samplers(vec![br::SamplerObjectRef::new(&linear_smp)])],
        )
        .expect("Failed to create equirectangular to cubemap descriptor set layout");
        let precompute_common_pl = br::PipelineLayoutBuilder::new(vec![&dsl.object], vec![])
            .create(e.graphics().device().clone())
            .expect("Failed to create precompute common pipeline layout");
        let prefilter_envmap_pl = br::PipelineLayoutBuilder::new(
            vec![&dsl.object],
            vec![(br::ShaderStage::FRAGMENT, 0..4)],
        )
        .create(e.graphics().device().clone())
        .expect("Failed to create envmap prefilter pipeline layout");
        let precompute_ds = DescriptorStore::new(e.graphics(), [&dsl, &dsl])
            .expect("Failed to allocate equirectangular to cubemap descriptor sets");
        e.graphics().update_descriptor_sets(
            &[
                precompute_ds.descriptor(0).unwrap().write(
                    br::DescriptorContents::CombinedImageSampler(vec![
                        br::DescriptorImageRef::new(
                            &tmp_loaded_image_view,
                            br::ImageLayout::ShaderReadOnlyOpt,
                        ),
                    ]),
                ),
                precompute_ds.descriptor(1).unwrap().write(
                    br::DescriptorContents::CombinedImageSampler(vec![
                        br::DescriptorImageRef::new(
                            &mem.dwt_ibl_cubemap,
                            br::ImageLayout::ShaderReadOnlyOpt,
                        ),
                    ]),
                ),
            ],
            &[],
        );
        let mut precompute_pipeline = br::GraphicsPipelineBuilder::<
            _,
            br::PipelineObject<peridot::DeviceObject>,
            _,
            _,
            _,
            _,
            _,
            _,
        >::new(
            &precompute_common_pl,
            (&precompute_rp, 0),
            equirect_to_cubemap_shader.generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP),
        );
        let equirect_to_cubemap_render_rect =
            br::vk::VkExtent2D::spread1(512).into_rect(br::vk::VkOffset2D::ZERO);
        let irradiance_precompute_render_rect =
            br::vk::VkExtent2D::spread1(32).into_rect(br::vk::VkOffset2D::ZERO);
        precompute_pipeline
            .viewport_scissors(
                br::DynamicArrayState::Static(&[
                    equirect_to_cubemap_render_rect.make_viewport(0.0..1.0)
                ]),
                br::DynamicArrayState::Static(&[equirect_to_cubemap_render_rect.clone()]),
            )
            .add_attachment_blend(br::AttachmentColorBlendState::noblend())
            .multisample_state(Some(br::MultisampleState::new()));
        let equirect_to_cubemap_pipeline = precompute_pipeline
            .create(
                e.graphics().device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )
            .expect("Failed to create equirectangular to cubemap pipeline");
        precompute_pipeline
            .viewport_scissors(
                br::DynamicArrayState::Static(&[
                    irradiance_precompute_render_rect.make_viewport(0.0..1.0)
                ]),
                br::DynamicArrayState::Static(&[irradiance_precompute_render_rect.clone()]),
            )
            .vertex_processing(
                irradiance_precompute_shader
                    .generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP),
            );
        let irradiance_precompute_pipeline = precompute_pipeline
            .create(
                e.graphics().device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )
            .expect("Failed to create irradiance precompute pipeline");
        precompute_pipeline
            .layout(&prefilter_envmap_pl)
            .viewport_scissors(
                br::DynamicArrayState::Dynamic(1),
                br::DynamicArrayState::Dynamic(1),
            )
            .vertex_processing(
                prefilter_envmap_shader.generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP),
            );
        let envmap_prefilter_pipeline = precompute_pipeline
            .create(
                e.graphics().device().clone(),
                None::<&br::PipelineCacheObject<peridot::DeviceObject>>,
            )
            .expect("Failed to create envmap prefilter pipeline");

        async_std::task::block_on(async {
            let cubemap_precompute_meshes = (0..6)
                .map(|n| StandardMesh {
                    vertex_buffers: vec![
                        RangedBuffer::from_offset_length(&buffer, fillrect_offset, 1),
                        RangedBuffer::from_offset_length(
                            &buffer,
                            cube_ref_positions_offset
                                + (n * std::mem::size_of::<[peridot::math::Vector4F32; 4]>())
                                    as u64,
                            1,
                        ),
                    ],
                    vertex_count: 4,
                })
                .collect::<Vec<_>>();

            e.submit_commands_async(|mut r| {
                tfb.sink_transfer_commands(&mut r);
                tfb.sink_graphics_ready_commands(&mut r);
                tfb2.generate_commands(&mut r);

                let vertex_buffer = RangedBuffer(&buffer, 0..buffer_data_size);
                let temporary_loaded_image =
                    RangedImage::single_color_plane(tmp_loaded_image_view.image());

                let pre_barrier = PipelineBarrier::new()
                    .with_barrier(
                        vertex_buffer
                            .usage_barrier(BufferUsage::HOST_RW, BufferUsage::VERTEX_BUFFER),
                    )
                    .with_barrier(temporary_loaded_image.barrier(
                        br::ImageLayout::Preinitialized,
                        br::ImageLayout::ShaderReadOnlyOpt,
                    ));

                let pipeline = peridot::LayoutedPipeline::combine(
                    &equirect_to_cubemap_pipeline,
                    &precompute_common_pl,
                );
                let descriptor_sets =
                    DescriptorSets(vec![precompute_ds.raw_descriptor(0).unwrap().into()]);

                // multiview拡張とかつかうとbegin_render_pass一回にできるけど面倒なので適当に回す
                let render_pass_begin_commands = equirect_to_cubemap_fbs.iter().map(|fb| {
                    BeginRenderPass::new(
                        &precompute_rp,
                        fb,
                        equirect_to_cubemap_render_rect.clone(),
                    )
                });
                let draw_commands = cubemap_precompute_meshes.iter().map(|m| m.ref_draw(1));
                let renders = render_pass_begin_commands
                    .zip(draw_commands)
                    .map(|(rp, d)| d.between(rp, EndRenderPass))
                    .collect::<Vec<_>>();

                renders
                    .after_of((pipeline, descriptor_sets.bind_graphics()))
                    .after_of(pre_barrier)
                    .execute(&mut r.as_dyn_ref());

                r
            })
            .expect("Failed to submit initialize commands")
            .await
            .expect("Failed to initialize cubemap");

            let irradiance_precompute_task = e
                .submit_commands_async(|mut r| {
                    let pipeline = peridot::LayoutedPipeline::combine(
                        &irradiance_precompute_pipeline,
                        &precompute_common_pl,
                    );
                    let descriptor_sets =
                        DescriptorSets(vec![precompute_ds.raw_descriptor(1).unwrap().into()]);

                    let render_pass_begin_commands = irradiance_precompute_fbs.iter().map(|fb| {
                        BeginRenderPass::new(
                            &precompute_rp,
                            fb,
                            irradiance_precompute_render_rect.clone(),
                        )
                    });
                    let draw_commands = cubemap_precompute_meshes.iter().map(|m| m.ref_draw(1));
                    let renders = render_pass_begin_commands
                        .zip(draw_commands)
                        .map(|(rp, d)| d.between(rp, EndRenderPass))
                        .collect::<Vec<_>>();

                    renders
                        .after_of((pipeline, descriptor_sets.bind_graphics()))
                        .execute(&mut r.as_dyn_ref());

                    r
                })
                .expect("Failed to submit irradiance precompute commands");
            let prefilter_envmap_task = e
                .submit_commands_async(|mut r| {
                    let pipeline = peridot::LayoutedPipeline::combine(
                        &envmap_prefilter_pipeline,
                        &prefilter_envmap_pl,
                    );
                    let descriptor_sets =
                        DescriptorSets(vec![precompute_ds.raw_descriptor(1).unwrap().into()]);

                    let by_cubemap_renders = cubemap_precompute_meshes
                        .iter()
                        .enumerate()
                        .map(|(n, transform_mesh)| {
                            let target_framebuffers = &prefiltered_envmap_fbs[n
                                * PREFILTERED_ENVMAP_MIP_LEVELS as usize
                                ..(n + 1) * PREFILTERED_ENVMAP_MIP_LEVELS as usize];

                            let by_mip_renders = target_framebuffers
                                .iter()
                                .enumerate()
                                .map(|(mip_level, fb)| {
                                    let roughness = mip_level as f32
                                        / (PREFILTERED_ENVMAP_MIP_LEVELS as f32 - 1.0);
                                    let envmap_size = (PREFILTERED_ENVMAP_SIZE as f32
                                        * 0.5f32.powi(mip_level as _))
                                        as u32;
                                    let region = br::vk::VkExtent2D::spread1(envmap_size)
                                        .into_rect(br::vk::VkOffset2D::ZERO);

                                    let rp =
                                        BeginRenderPass::new(&precompute_rp, fb, region.clone());
                                    let set_viewports = ViewportWithScissorRect(
                                        region.make_viewport(0.0..1.0),
                                        region,
                                    );
                                    let set_roughness = PushConstant::for_fragment(0, roughness);
                                    let draw_plane = SimpleDraw(4, 1, 0, 0);

                                    draw_plane
                                        .after_of(set_roughness)
                                        .between(rp.then(set_viewports), EndRenderPass)
                                })
                                .collect::<Vec<_>>();

                            by_mip_renders.after_of(transform_mesh.ref_pre_configure_for_draw())
                        })
                        .collect::<Vec<_>>();

                    by_cubemap_renders
                        .after_of((pipeline, descriptor_sets.bind_graphics()))
                        .execute(&mut r.as_dyn_ref());

                    r
                })
                .expect("Failed to submit prefilter envmap commands");

            let (a, b) = futures_util::join!(irradiance_precompute_task, prefilter_envmap_task);
            a.or(b).expect("Failed to precompute envmaps");
        });
        // keep alive resources while command execution
        drop(tmp_loaded_image_mem);
        drop(buffer);
        drop(prefiltered_envmap_fbs);
        drop(irradiance_precompute_fbs);
        drop(equirect_to_cubemap_fbs);
        drop(tmp_loaded_image_view);

        let descriptors = DescriptorStore::new(
            e.graphics(),
            [
                &const_res.dsl_ub1,
                &const_res.dsl_ub1,
                &const_res.dsl_ub2_f,
                &const_res.dsl_ub1_f,
                &const_res.dsl_utb1,
                &const_res.dsl_utb1,
                &const_res.dsl_utb1,
                &const_res.dsl_ub1,
                &const_res.dsl_ub1,
                &const_res.dsl_ics1_f,
                &const_res.dsl_ics1_f,
            ],
        )
        .expect("Failed to allocate descriptors");
        let mut descriptor_writes = Vec::with_capacity(12);
        descriptor_writes.extend([
            descriptors
                .descriptor(0)
                .unwrap()
                .write(br::DescriptorContents::UniformBuffer(vec![mem
                    .grid_transform_buffer()
                    .into_descriptor_buffer_ref()])),
            descriptors
                .descriptor(1)
                .unwrap()
                .write(br::DescriptorContents::UniformBuffer(vec![mem
                    .object_transform_buffer()
                    .into_descriptor_buffer_ref()])),
        ]);
        descriptor_writes.extend(descriptors.descriptor(2).unwrap().write_multiple([
            br::DescriptorContents::UniformBuffer(vec![
                mem.camera_info_buffer().into_descriptor_buffer_ref(),
            ]),
            br::DescriptorContents::UniformBuffer(
                vec![mem.directional_light_info_buffer().into_descriptor_buffer_ref()],
            ),
        ]));
        descriptor_writes.extend([
            descriptors
                .descriptor(3)
                .unwrap()
                .write(br::DescriptorContents::UniformBuffer(vec![mem
                    .material_buffer()
                    .into_descriptor_buffer_ref()])),
            descriptors
                .descriptor(4)
                .unwrap()
                .write(br::DescriptorContents::UniformTexelBuffer(vec![
                    br::VkHandleRef::new(&mem.ui_transform_buffer_view),
                ])),
            descriptors
                .descriptor(5)
                .unwrap()
                .write(br::DescriptorContents::UniformTexelBuffer(vec![
                    br::VkHandleRef::new(&mem.ui_mask_transform_buffer_view),
                ])),
            descriptors
                .descriptor(6)
                .unwrap()
                .write(br::DescriptorContents::UniformTexelBuffer(vec![
                    br::VkHandleRef::new(ui_dynamic_buffers.transform_buffer_view()),
                ])),
            descriptors
                .descriptor(7)
                .unwrap()
                .write(br::DescriptorContents::UniformBuffer(vec![mem
                    .ui_transform_buffer()
                    .into_descriptor_buffer_ref()])),
            descriptors
                .descriptor(8)
                .unwrap()
                .write(br::DescriptorContents::UniformBuffer(vec![mem
                    .camera_vp_separated_buffer()
                    .into_descriptor_buffer_ref()])),
            descriptors
                .descriptor(9)
                .unwrap()
                .write(br::DescriptorContents::CombinedImageSampler(vec![
                    br::DescriptorImageRef::new(
                        &mem.dwt_ibl_cubemap,
                        br::ImageLayout::ShaderReadOnlyOpt,
                    ),
                ])),
            descriptors.descriptor(10).unwrap().write(
                br::DescriptorContents::CombinedImageSampler(vec![br::DescriptorImageRef::new(
                    &mem.dwt_irradiance_cubemap,
                    br::ImageLayout::ShaderReadOnlyOpt,
                )]),
            ),
        ]);
        e.graphics()
            .device()
            .update_descriptor_sets(&descriptor_writes, &[]);

        let screen_res = ScreenResources::new(e, &mut mem.manager, &const_res);

        let mut render_bundles = (0..6)
            .map(|_| {
                RenderBundle::new(e.graphics(), e.back_buffer_count() as _)
                    .expect("Failed to create render bundle")
            })
            .collect::<Vec<_>>();
        rayon::scope(|s| {
            let (rb0, rb1, rb2, rb3, rb4, rb5) = match &mut render_bundles[..] {
                &mut [ref mut rb0, ref mut rb1, ref mut rb2, ref mut rb3, ref mut rb4, ref mut rb5] => {
                    (rb0, rb1, rb2, rb3, rb4, rb5)
                }
                _ => unreachable!(),
            };

            s.spawn(|_| {
                for n in 0..e.back_buffer_count() {
                    Self::repopulate_ui_mask_render_commands(
                        e,
                        rb0.synchronized(n),
                        &const_res.render_pass,
                        &screen_res,
                        n,
                        &**mem.mem.buffer.object,
                        descriptors.raw_descriptor(5).unwrap(),
                        &mem.ui_mask_render_params,
                    );
                }
            });
            s.spawn(|_| {
                for n in 0..e.back_buffer_count() {
                    Self::repopulate_grid_render_commands(
                        e,
                        rb1.synchronized(n),
                        &const_res.render_pass,
                        &screen_res,
                        n,
                        &mem.mem.buffer.object,
                        &mem.static_offsets,
                        descriptors.raw_descriptor(0).unwrap(),
                    );
                }
            });
            s.spawn(|_| {
                for n in 0..e.back_buffer_count() {
                    Self::repopulate_pbr_object_render_commands(
                        e,
                        rb2.synchronized(n),
                        &const_res.render_pass,
                        &screen_res,
                        n,
                        &mem.mem.buffer.object,
                        &mem.static_offsets,
                        mem.icosphere_vertex_count as _,
                        descriptors.raw_descriptor(1).unwrap(),
                        descriptors.raw_descriptor(2).unwrap(),
                        descriptors.raw_descriptor(3).unwrap(),
                        descriptors.raw_descriptor(10).unwrap(),
                    );
                }
            });
            s.spawn(|_| {
                for n in 0..e.back_buffer_count() {
                    Self::repopulate_static_ui_render_commands(
                        e,
                        rb3.synchronized(n),
                        &const_res.render_pass,
                        &screen_res,
                        n,
                        &mem.ui_render_params,
                        &**mem.mem.buffer.object,
                        &mem.static_offsets,
                        &mem.mutable_offsets,
                        mem.mem.mut_buffer_placement,
                        descriptors.raw_descriptor(7).unwrap(),
                        descriptors.raw_descriptor(4).unwrap(),
                    );
                }
            });
            s.spawn(|_| {
                for n in 0..e.back_buffer_count() {
                    Self::repopulate_dynamic_ui_render_commands(
                        e,
                        rb4.synchronized(n),
                        &const_res.render_pass,
                        &screen_res,
                        n,
                        &ui_dynamic_buffers,
                        descriptors.raw_descriptor(6).unwrap(),
                    );
                }
            });
            s.spawn(|_| {
                for n in 0..e.back_buffer_count() {
                    Self::repopulate_skybox_render_commands(
                        e,
                        rb5.synchronized(n),
                        &const_res.render_pass,
                        &screen_res,
                        n,
                        &mem.mem.buffer.object,
                        &mem.static_offsets,
                        &[
                            descriptors.raw_descriptor(8).unwrap().into(),
                            descriptors.raw_descriptor(9).unwrap().into(),
                        ],
                    )
                }
            });
        });

        let mut command_buffers = peridot::CommandBundle::new(
            e.graphics(),
            peridot::CBSubmissionType::Graphics,
            e.back_buffer_count(),
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
            last_frame_tfb2: peridot::TransferBatch2::new(),
            ph: std::marker::PhantomData,
        }
    }

    fn update(
        &mut self,
        e: &mut peridot::Engine<NL>,
        on_backbuffer_of: u32,
        delta_time: std::time::Duration,
    ) {
        self.last_frame_tfb = peridot::TransferBatch::new();
        self.last_frame_tfb2 = peridot::TransferBatch2::new();

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
                    self.material_data.modify().roughness = self.ui_roughness_slider.value();
                    ui_mesh_dirty = true;
                }
                Some(CapturingComponent::Anisotropic) => {
                    self.ui_anisotropic_slider.update_capturing_input(e);
                    self.material_data.modify().anisotropic = self.ui_anisotropic_slider.value();
                    ui_mesh_dirty = true;
                }
                Some(CapturingComponent::Metallic) => {
                    self.ui_metallic_slider.update_capturing_input(e);
                    self.material_data.modify().metallic = self.ui_metallic_slider.value();
                    ui_mesh_dirty = true;
                }
                Some(CapturingComponent::Reflectance) => {
                    self.ui_reflectance_slider.update_capturing_input(e);
                    self.material_data.modify().reflectance = self.ui_reflectance_slider.value();
                    ui_mesh_dirty = true;
                }
                None => (),
            }
        }

        self.main_camera
            .update(e, delta_time, self.capturing_component);

        if self.main_camera.camera.take_dirty_flag() {
            self.mem.apply_main_camera(
                e,
                &self.main_camera.camera.get().0,
                self.main_camera.camera.get().1,
            );
            self.mem.set_camera_info(
                e,
                RasterizationCameraInfo {
                    pos: peridot::math::Vector4(
                        self.main_camera.camera.get().0.position.0,
                        self.main_camera.camera.get().0.position.1,
                        self.main_camera.camera.get().0.position.2,
                        1.0,
                    ),
                },
            )
        }

        if self.material_data.take_dirty_flag() {
            self.mem.set_material(e, self.material_data.get().clone());
        }

        if ui_mesh_dirty {
            let mut ui_dynamic_texts = peridot_vg::Context::new(e.rendering_precision());
            self.mem.construct_new_ui_fill_rect_vertices(e, |vs| {
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
            let ui_dynamic_buffers = UIRenderingBuffers::new(
                e.graphics(),
                &mut self.mem.manager,
                &ui_dynamic_texts,
                &mut self.last_frame_tfb,
            )
            .expect("Failed to allocate ui dynamic buffers");
            e.graphics().update_descriptor_sets(
                &[self.descriptors.descriptor(6).unwrap().write(
                    br::DescriptorContents::UniformTexelBuffer(vec![br::VkHandleRef::new(
                        ui_dynamic_buffers.transform_buffer_view(),
                    )]),
                )],
                &[],
            );
            self.render_bundles[4]
                .reset()
                .expect("Failed to reset dynamic ui render bundles");
            for n in 0..e.back_buffer_count() {
                Self::repopulate_dynamic_ui_render_commands(
                    e,
                    self.render_bundles[4].synchronized(n),
                    &self.const_res.render_pass,
                    &self.screen_res,
                    n,
                    &ui_dynamic_buffers,
                    self.descriptors.raw_descriptor(6).unwrap(),
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
            .ready_transfer(e.graphics(), &mut self.last_frame_tfb2);
        let update_submission = if self.last_frame_tfb.has_copy_ops()
            || self.last_frame_tfb.has_ready_barrier_ops()
            || self.last_frame_tfb2.has_ops()
        {
            self.update_commands
                .reset()
                .expect("Failed to reset update commands");
            unsafe {
                let mut r = self.update_commands[0]
                    .begin()
                    .expect("Failed to begin recording update commands");
                self.last_frame_tfb.sink_transfer_commands(&mut r);
                self.last_frame_tfb.sink_graphics_ready_commands(&mut r);
                self.last_frame_tfb2.generate_commands(&mut r);
                r.end().expect("Failed to finish update commands");
            }

            Some(br::EmptySubmissionBatch.with_command_buffers(&self.update_commands[..]))
        } else {
            None
        };

        e.do_render(
            on_backbuffer_of,
            update_submission,
            br::EmptySubmissionBatch.with_command_buffers(
                &self.command_buffers[on_backbuffer_of as usize..=on_backbuffer_of as usize],
            ),
        )
        .expect("Failed to present");
        self.mem.dynamic_stg.clear();
    }

    fn discard_back_buffer_resources(&mut self) {
        self.command_buffers
            .reset()
            .expect("Failed to reset screen commands");
        for rb in &mut self.render_bundles {
            rb.reset()
                .expect("Failed to reset individual render bundles");
        }
        self.screen_res.frame_buffers.clear();
    }
    fn on_resize(&mut self, e: &mut peridot::Engine<NL>, new_size: peridot::math::Vector2<usize>) {
        self.screen_res = ScreenResources::new(e, &mut self.mem.manager, &self.const_res);

        rayon::scope(|s| {
            let (rb0, rb1, rb2, rb3, rb4, rb5) = match &mut self.render_bundles[..] {
                &mut [ref mut rb0, ref mut rb1, ref mut rb2, ref mut rb3, ref mut rb4, ref mut rb5] => {
                    (rb0, rb1, rb2, rb3, rb4, rb5)
                }
                _ => unreachable!(),
            };

            s.spawn(|_| {
                for n in 0..e.back_buffer_count() {
                    Self::repopulate_ui_mask_render_commands(
                        e,
                        rb0.synchronized(n),
                        &self.const_res.render_pass,
                        &self.screen_res,
                        n,
                        &self.mem.mem.buffer.object,
                        self.descriptors.raw_descriptor(5).unwrap(),
                        &self.mem.ui_mask_render_params,
                    );
                }
            });
            s.spawn(|_| {
                for n in 0..e.back_buffer_count() {
                    Self::repopulate_grid_render_commands(
                        e,
                        rb1.synchronized(n),
                        &self.const_res.render_pass,
                        &self.screen_res,
                        n,
                        &self.mem.mem.buffer.object,
                        &self.mem.static_offsets,
                        self.descriptors.raw_descriptor(0).unwrap(),
                    );
                }
            });
            s.spawn(|_| {
                for n in 0..e.back_buffer_count() {
                    Self::repopulate_pbr_object_render_commands(
                        e,
                        rb2.synchronized(n),
                        &self.const_res.render_pass,
                        &self.screen_res,
                        n,
                        &self.mem.mem.buffer.object,
                        &self.mem.static_offsets,
                        self.mem.icosphere_vertex_count as _,
                        self.descriptors.raw_descriptor(1).unwrap(),
                        self.descriptors.raw_descriptor(2).unwrap(),
                        self.descriptors.raw_descriptor(3).unwrap(),
                        self.descriptors.raw_descriptor(10).unwrap(),
                    );
                }
            });
            s.spawn(|_| {
                for n in 0..e.back_buffer_count() {
                    Self::repopulate_static_ui_render_commands(
                        e,
                        rb3.synchronized(n),
                        &self.const_res.render_pass,
                        &self.screen_res,
                        n,
                        &self.mem.ui_render_params,
                        &self.mem.mem.buffer.object,
                        &self.mem.static_offsets,
                        &self.mem.mutable_offsets,
                        self.mem.mem.mut_buffer_placement,
                        self.descriptors.raw_descriptor(7).unwrap(),
                        self.descriptors.raw_descriptor(4).unwrap(),
                    );
                }
            });
            s.spawn(|_| {
                for n in 0..e.back_buffer_count() {
                    Self::repopulate_dynamic_ui_render_commands(
                        e,
                        rb4.synchronized(n),
                        &self.const_res.render_pass,
                        &self.screen_res,
                        n,
                        &self.ui_dynamic_buffers,
                        self.descriptors.raw_descriptor(6).unwrap(),
                    );
                }
            });
            s.spawn(|_| {
                for n in 0..e.back_buffer_count() {
                    Self::repopulate_skybox_render_commands(
                        e,
                        rb5.synchronized(n),
                        &self.const_res.render_pass,
                        &self.screen_res,
                        n,
                        &self.mem.mem.buffer.object,
                        &self.mem.static_offsets,
                        &[
                            self.descriptors.raw_descriptor(8).unwrap().into(),
                            self.descriptors.raw_descriptor(9).unwrap().into(),
                        ],
                    )
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
            e,
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
        mut command_buffer: br::SynchronizedCommandBuffer<
            br::CommandPoolObject<peridot::DeviceObject>,
            br::CommandBufferObject<peridot::DeviceObject>,
        >,
        renderpass: &br::RenderPassObject<peridot::DeviceObject>,
        screen_res: &ScreenResources,
        frame_buffer_index: usize,
        device_buffer: &impl br::Buffer<ConcreteDevice = peridot::DeviceObject>,
        transform_buffer_desc: br::DescriptorSet,
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
                transform_buffer_descriptor_set: transform_buffer_desc,
                target_pixels: peridot::math::Vector2(fb.size().width as _, fb.size().height as _),
            },
        );

        rec.end().expect("Failed to finish ui mask render commands");
    }

    fn repopulate_grid_render_commands(
        engine: &peridot::Engine<NL>,
        mut command_buffer: br::SynchronizedCommandBuffer<
            impl br::CommandPool + br::VkHandleMut,
            impl br::CommandBuffer + br::VkHandleMut,
        >,
        renderpass: &br::RenderPassObject<peridot::DeviceObject>,
        screen_res: &ScreenResources,
        frame_buffer_index: usize,
        device_buffer: &impl br::Buffer,
        static_offsets: &StaticBufferOffsets,
        grid_transform_desc: br::DescriptorSet,
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

        let descriptor_sets = DescriptorSets(vec![grid_transform_desc.into()]);
        let mesh = StandardMesh {
            vertex_buffers: vec![RangedBuffer::from_offset_length(
                device_buffer,
                static_offsets.grid,
                1,
            )],
            vertex_count: mesh::GRID_MESH_LINE_COUNT as _,
        };

        let setup = (
            &screen_res.grid_render_pipeline,
            descriptor_sets.bind_graphics(),
        );
        mesh.draw(1)
            .after_of(setup)
            .execute_and_finish(rec.as_dyn_ref())
            .expect("Failed to record grid rendering commands");
    }

    fn repopulate_pbr_object_render_commands(
        engine: &peridot::Engine<NL>,
        mut command_buffer: br::SynchronizedCommandBuffer<
            impl br::CommandPool + br::VkHandleMut,
            impl br::CommandBuffer + br::VkHandleMut,
        >,
        renderpass: &br::RenderPassObject<peridot::DeviceObject>,
        screen_res: &ScreenResources,
        frame_buffer_index: usize,
        device_buffer: &impl br::Buffer,
        static_offsets: &StaticBufferOffsets,
        icosphere_vertex_count: u32,
        object_transform_desc: br::DescriptorSet,
        rasterization_scene_info_desc: br::DescriptorSet,
        material_info_desc: br::DescriptorSet,
        precomputed_map_desc: br::DescriptorSet,
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

        let descriptor_sets = DescriptorSets(vec![
            object_transform_desc.into(),
            rasterization_scene_info_desc.into(),
            material_info_desc.into(),
            precomputed_map_desc.into(),
        ]);
        let mesh = StandardIndexedMesh {
            vertex_buffers: vec![RangedBuffer::from_offset_length(
                device_buffer,
                static_offsets.icosphere_vertices,
                1,
            )],
            index_buffer: RangedBuffer::from_offset_length(
                device_buffer,
                static_offsets.icosphere_indices,
                1,
            ),
            index_type: br::IndexType::U16,
            vertex_count: icosphere_vertex_count,
        };

        let setup = (&screen_res.pbr_pipeline, descriptor_sets.bind_graphics());
        mesh.draw(1)
            .after_of(setup)
            .execute_and_finish(rec.as_dyn_ref())
            .expect("Failed to finish pbr object render commands");
    }

    fn repopulate_static_ui_render_commands(
        engine: &peridot::Engine<NL>,
        mut command_buffer: br::SynchronizedCommandBuffer<
            impl br::CommandPool + br::VkHandleMut,
            impl br::CommandBuffer + br::VkHandleMut,
        >,
        renderpass: &br::RenderPassObject<peridot::DeviceObject>,
        screen_res: &ScreenResources,
        frame_buffer_index: usize,
        render_params: &peridot_vg::RendererParams,
        device_buffer: &impl br::Buffer<ConcreteDevice = peridot::DeviceObject>,
        static_offsets: &StaticBufferOffsets,
        mutable_offsets: &MutableBufferOffsets,
        mut_buffer_placement: u64,
        fill_rect_transform_desc: br::DescriptorSet,
        vg_transform_desc: br::DescriptorSet,
    ) {
        let fb = &screen_res.frame_buffers[frame_buffer_index];
        let mut rec = command_buffer
            .begin_inherit(
                Some((fb, renderpass, ConstResources::RENDER_MAIN_PASS)),
                None,
            )
            .expect("Failed to initiate static ui render bundle recording");

        let descriptor_sets = DescriptorSets(vec![fill_rect_transform_desc.into()]);
        let set_fill_rect_color = PushConstant::for_fragment(0, [1.0f32, 1.0, 1.0, 0.25]);
        let set_border_rect_color = PushConstant::for_fragment(0, [1.0f32, 1.0, 1.0, 1.0]);
        let fill_rect_mesh = StandardIndexedMesh {
            vertex_buffers: vec![RangedBuffer::from_offset_length(
                device_buffer,
                mutable_offsets.ui_fill_rects + mut_buffer_placement,
                1,
            )],
            index_buffer: RangedBuffer::from_offset_length(
                device_buffer,
                static_offsets.ui_fill_rect_indices,
                1,
            ),
            index_type: br::IndexType::U16,
            vertex_count: mesh::UI_FILL_RECT_INDEX_COUNT as _,
        };
        let border_rect_mesh = StandardIndexedMesh {
            vertex_buffers: vec![RangedBuffer::from_offset_length(
                device_buffer,
                mutable_offsets.ui_fill_rects + mut_buffer_placement,
                1,
            )],
            index_buffer: RangedBuffer::from_offset_length(
                device_buffer,
                static_offsets.ui_border_line_indices,
                1,
            ),
            index_type: br::IndexType::U16,
            vertex_count: mesh::UI_FILL_RECT_BORDER_INDEX_COUNT as _,
        };

        let fill_render = fill_rect_mesh.draw(1).after_of((
            &screen_res.ui_fill_rect_pipeline,
            descriptor_sets.bind_graphics(),
            set_fill_rect_color,
        ));
        let border_render = border_rect_mesh
            .draw(1)
            .after_of((&screen_res.ui_border_line_pipeline, set_border_rect_color));

        fill_render
            .then(border_render)
            .execute(&mut rec.as_dyn_ref());

        render_params.default_render_commands(
            engine,
            &mut rec,
            device_buffer,
            peridot_vg::RendererExternalInstances {
                interior_pipeline: &screen_res.vg_interior_pipeline,
                curve_pipeline: &screen_res.vg_curve_pipeline,
                transform_buffer_descriptor_set: vg_transform_desc,
                target_pixels: peridot::math::Vector2(fb.size().width as _, fb.size().height as _),
            },
        );

        rec.end()
            .expect("Failed to finish static ui render commands");
    }

    fn repopulate_dynamic_ui_render_commands(
        engine: &peridot::Engine<NL>,
        mut command_buffer: br::SynchronizedCommandBuffer<
            br::CommandPoolObject<peridot::DeviceObject>,
            br::CommandBufferObject<peridot::DeviceObject>,
        >,
        renderpass: &br::RenderPassObject<peridot::DeviceObject>,
        screen_res: &ScreenResources,
        frame_buffer_index: usize,
        ui_dynamic_buffers: &UIRenderingBuffers,
        transform_desc: br::DescriptorSet,
    ) {
        let fb = &screen_res.frame_buffers[frame_buffer_index];
        let mut rec = command_buffer
            .begin_inherit(
                Some((fb, renderpass, ConstResources::RENDER_MAIN_PASS)),
                None,
            )
            .expect("Failed to initiate static ui render bundle recording");

        ui_dynamic_buffers.populate_render_commands(
            engine,
            &mut rec,
            &screen_res,
            transform_desc,
            peridot::math::Vector2(fb.size().width as _, fb.size().height as _),
        );

        rec.end()
            .expect("Failed to finish dynamic ui render commands");
    }

    fn repopulate_skybox_render_commands(
        engine: &peridot::Engine<NL>,
        mut command_buffer: br::SynchronizedCommandBuffer<
            impl br::CommandPool + br::VkHandleMut,
            impl br::CommandBuffer + br::VkHandleMut,
        >,
        renderpass: &br::RenderPassObject<peridot::DeviceObject>,
        screen_res: &ScreenResources,
        frame_buffer_index: usize,
        device_buffer: &impl br::Buffer,
        static_offsets: &StaticBufferOffsets,
        descriptors: &[br::vk::VkDescriptorSet],
    ) {
        let fb = &screen_res.frame_buffers[frame_buffer_index];
        let mut rc = command_buffer
            .begin_inherit(
                Some((fb, renderpass, ConstResources::RENDER_MAIN_PASS)),
                None,
            )
            .expect("Failed to initiate skybox rendering");

        let mesh = StandardIndexedMesh {
            vertex_buffers: vec![RangedBuffer::from_offset_length(
                device_buffer,
                static_offsets.skybox_cube,
                1,
            )],
            index_buffer: RangedBuffer::from_offset_length(
                device_buffer,
                static_offsets.skybox_cube_indices,
                1,
            ),
            index_type: br::IndexType::U16,
            vertex_count: 36,
        };

        let setup = (
            &screen_res.skybox_render_pipeline,
            BindGraphicsDescriptorSets::new(descriptors),
        );
        mesh.draw(1)
            .after_of(setup)
            .execute_and_finish(rc.as_dyn_ref())
            .expect("Failed to finish skybox render commands");
    }

    fn repopulate_screen_commands(
        engine: &peridot::Engine<NL>,
        render_area: br::vk::VkRect2D,
        command_buffers: &mut peridot::CommandBundle<peridot::DeviceObject>,
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

            let rp = BeginRenderPass::new(&const_res.render_pass, fb, render_area.clone())
                .with_clear_values(vec![
                    br::ClearValue::color_f32([0.25 * 0.25, 0.5 * 0.25, 1.0 * 0.25, 1.0]),
                    br::ClearValue::depth_stencil(1.0, 0),
                ]);
            let pre_stencil_pass_commands = [render_bundles[0].buffers[n].native_ptr()];
            let color_pass_commands = render_bundles[1..]
                .iter()
                .map(|b| b.buffers[n].native_ptr())
                .collect::<Vec<_>>();

            (
                pre_stencil_pass_commands,
                NextSubpass::WITH_COMMAND_BUFFER_EXECUTIONS,
                color_pass_commands,
            )
                .between(rp.non_inline_commands(), EndRenderPass)
                .execute_and_finish(r.as_dyn_ref())
                .expect("Failed to record main commands");
        }
    }
}
