use std::rc::Rc;

use bedrock as br;
use br::{MemoryBound, VkHandle};
use peridot;
use peridot::math::One;
use peridot_vertex_processing_pack::*;

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
pub struct MaterialInfo {
    base_color: peridot::math::Vector4F32,
    roughness: f32,
    anisotropic: f32,
    metallic: f32,
    reflectance: f32,
}

pub struct ConstResources {
    render_pass: br::RenderPass,
    dsl_ub1: br::DescriptorSetLayout,
    dsl_ub1_f: br::DescriptorSetLayout,
    dsl_ub2_f: br::DescriptorSetLayout,
    dp: br::DescriptorPool,
    descriptors: Vec<br::vk::VkDescriptorSet>,
    unlit_colored_shader: PvpShaderModules<'static>,
    unlit_colored_pipeline_layout: Rc<br::PipelineLayout>,
    pbr_shader: PvpShaderModules<'static>,
    pbr_pipeline_layout: Rc<br::PipelineLayout>,
}
impl ConstResources {
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
        .store_op(br::StoreOp::DontCare);
        let main_pass = br::SubpassDescription::new()
            .add_color_output(0, br::ImageLayout::ColorAttachmentOpt, None)
            .depth_stencil(1, br::ImageLayout::DepthStencilAttachmentOpt);
        let passdep = br::vk::VkSubpassDependency {
            srcSubpass: br::vk::VK_SUBPASS_EXTERNAL,
            dstSubpass: 0,
            srcStageMask: br::PipelineStageFlags::TOP_OF_PIPE.0,
            dstStageMask: br::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.0,
            srcAccessMask: 0,
            dstAccessMask: br::AccessFlags::COLOR_ATTACHMENT.write,
            dependencyFlags: br::vk::VK_DEPENDENCY_BY_REGION_BIT,
        };
        let render_pass = br::RenderPassBuilder::new()
            .add_attachments([main_attachment, depth_attachment])
            .add_subpass(main_pass)
            .add_dependency(passdep)
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
        let dp = br::DescriptorPool::new(
            e.graphics_device(),
            4,
            &[br::DescriptorPoolSize(br::DescriptorType::UniformBuffer, 5)],
            false,
        )
        .expect("Failed to create descriptor pool");
        let descriptors = dp
            .alloc(&[&dsl_ub1, &dsl_ub1, &dsl_ub2_f, &dsl_ub1_f])
            .expect("Failed to allocate descriptors");

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

        Self {
            render_pass,
            dsl_ub1,
            dsl_ub1_f,
            dsl_ub2_f,
            dp,
            descriptors,
            unlit_colored_shader,
            unlit_colored_pipeline_layout,
            pbr_shader,
            pbr_pipeline_layout,
        }
    }

    pub fn grid_transform_desc(&self) -> br::vk::VkDescriptorSet {
        self.descriptors[0]
    }

    pub fn object_transform_desc(&self) -> br::vk::VkDescriptorSet {
        self.descriptors[1]
    }

    pub fn rasterization_scene_info_desc(&self) -> br::vk::VkDescriptorSet {
        self.descriptors[2]
    }

    pub fn material_info_desc(&self) -> br::vk::VkDescriptorSet {
        self.descriptors[3]
    }
}

pub struct ScreenResources {
    depth_texture: peridot::Image,
    depth_texture_view: br::ImageView,
    frame_buffers: Vec<br::Framebuffer>,
    grid_render_pipeline: peridot::LayoutedPipeline,
    pbr_pipeline: peridot::LayoutedPipeline,
}
impl ScreenResources {
    pub fn new(
        e: &peridot::Engine<impl peridot::NativeLinker>,
        const_res: &ConstResources,
    ) -> Self {
        let bb0 = e.backbuffer(0).expect("no backbuffers?");

        let depth_image = br::ImageDesc::new(
            AsRef::<br::Extent2D>::as_ref(bb0.size()),
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
        let depth_texture = peridot::Image::bound(depth_image, &Rc::new(depth_mem), 0)
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
                    b.size(),
                    1,
                )
                .expect("Failed to create framebuffer")
            })
            .collect();

        let area = br::vk::VkRect2D {
            offset: br::vk::VkOffset2D { x: 0, y: 0 },
            extent: br::vk::VkExtent2D {
                width: bb0.size().0,
                height: bb0.size().1,
            },
        };
        let viewport = br::Viewport::from_rect_with_depth_range(&area, 0.0..1.0).into_inner();

        let unlit_colored_vps = const_res
            .unlit_colored_shader
            .generate_vps(br::vk::VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
        let mut pb = br::GraphicsPipelineBuilder::new(
            &const_res.unlit_colored_pipeline_layout,
            (&const_res.render_pass, 0),
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

        Self {
            depth_texture,
            depth_texture_view,
            frame_buffers,
            grid_render_pipeline,
            pbr_pipeline,
        }
    }
}

pub struct MutableBufferOffsets {
    grid_mvp: u64,
    object_mvp: u64,
    camera_info: u64,
    directional_light_info: u64,
    material: u64,
}
pub struct StaticBufferOffsets {
    grid: u64,
    icosphere_vertices: u64,
    icosphere_indices: u64,
}
pub struct StaticBufferInitializer<'o> {
    offsets: &'o StaticBufferOffsets,
    icosphere: &'o mesh::UnitIcosphere,
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
}
impl UpdateSets {
    pub fn new() -> Self {
        Self {
            grid_mvp_stg_offset: None,
            object_mvp_stg_offset: None,
            camera_info_stg_offset: None,
            directional_light_info_stg_offset: None,
            material_stg_offset: None,
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
}
impl Memory {
    pub fn new(
        e: &peridot::Engine<impl peridot::NativeLinker>,
        tfb: &mut peridot::TransferBatch,
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
        };

        Self {
            mem: peridot::FixedMemory::new(
                e.graphics(),
                static_bp,
                mutable_bp,
                textures,
                &mut StaticBufferInitializer {
                    offsets: &offsets,
                    icosphere: &icosphere,
                },
                tfb,
            )
            .expect("Failed to initialize fixed memory"),
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
        e: &peridot::Graphics,
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

    pub fn set_camera_info(&mut self, e: &peridot::Graphics, info: RasterizationCameraInfo) {
        self.update_sets.camera_info_stg_offset = Some(self.dynamic_stg.push(e, info));
    }

    pub fn set_directional_light_info(
        &mut self,
        e: &peridot::Graphics,
        info: RasterizationDirectionalLightInfo,
    ) {
        self.update_sets.directional_light_info_stg_offset = Some(self.dynamic_stg.push(e, info));
    }

    pub fn set_material(&mut self, e: &peridot::Graphics, info: MaterialInfo) {
        self.update_sets.material_stg_offset = Some(self.dynamic_stg.push(e, info));
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
}

pub fn range_cast_u64_usize(r: std::ops::Range<u64>) -> std::ops::Range<usize> {
    r.start as _..r.end as _
}

pub struct Game<NL: peridot::NativeLinker> {
    const_res: ConstResources,
    mem: Memory,
    screen_res: ScreenResources,
    command_buffers: peridot::CommandBundle,
    update_commands: peridot::CommandBundle,
    main_camera: peridot::math::Camera,
    aspect: f32,
    dirty_main_camera: bool,
    ph: std::marker::PhantomData<*const NL>,
}
impl<NL: peridot::NativeLinker> Game<NL> {
    pub const NAME: &'static str = "PBR test";
    pub const VERSION: (u32, u32, u32) = (0, 1, 0);
}
impl<NL: peridot::NativeLinker> peridot::FeatureRequests for Game<NL> {}
impl<NL: peridot::NativeLinker> peridot::EngineEvents<NL> for Game<NL> {
    fn init(e: &mut peridot::Engine<NL>) -> Self {
        let bb0 = e.backbuffer(0).expect("no backbuffers?");
        let render_area = br::vk::VkRect2D {
            offset: br::vk::VkOffset2D { x: 0, y: 0 },
            extent: br::vk::VkExtent2D {
                width: bb0.size().0,
                height: bb0.size().1,
            },
        };
        let aspect = bb0.size().0 as f32 / bb0.size().1 as f32;

        let const_res = ConstResources::new(e);
        let mut tfb = peridot::TransferBatch::new();
        let mut mem = Memory::new(e, &mut tfb);
        let mut main_camera = peridot::math::Camera {
            projection: Some(peridot::math::ProjectionMethod::Perspective {
                fov: 60.0f32.to_radians(),
            }),
            depth_range: 0.1..100.0,
            position: peridot::math::Vector3(2.0, 1.0, -5.0),
            rotation: peridot::math::Quaternion::ONE,
        };
        main_camera.look_at(peridot::math::Vector3(0.0, 0.0, 0.0));
        mem.apply_main_camera(e.graphics(), &main_camera, aspect);
        mem.set_camera_info(
            e.graphics(),
            RasterizationCameraInfo {
                pos: peridot::math::Vector4(
                    main_camera.position.0,
                    main_camera.position.1,
                    main_camera.position.2,
                    1.0,
                ),
            },
        );
        mem.set_directional_light_info(
            e.graphics(),
            RasterizationDirectionalLightInfo {
                dir: -peridot::math::Vector4(0.2f32, 0.3, 0.5, 0.0).normalize(),
                intensity: peridot::math::Vector4(2.0, 2.0, 2.0, 1.0),
            },
        );
        mem.set_material(
            e.graphics(),
            MaterialInfo {
                base_color: peridot::math::Vector4(1.0, 0.0, 0.0, 1.0),
                roughness: 0.4,
                anisotropic: 0.0,
                metallic: 0.0,
                reflectance: 0.5,
            },
        );
        mem.ready_transfer(e.graphics(), &mut tfb);
        e.submit_commands(|r| {
            tfb.sink_transfer_commands(r);
            tfb.sink_graphics_ready_commands(r);
        })
        .expect("Failed to initialize resources");

        let mut dub = peridot::DescriptorSetUpdateBatch::new();
        dub.write(
            const_res.grid_transform_desc(),
            0,
            br::DescriptorUpdateInfo::UniformBuffer(vec![(
                mem.mem.buffer.0.native_ptr(),
                range_cast_u64_usize(mem.grid_transform_range()),
            )]),
        );
        dub.write(
            const_res.object_transform_desc(),
            0,
            br::DescriptorUpdateInfo::UniformBuffer(vec![(
                mem.mem.buffer.0.native_ptr(),
                range_cast_u64_usize(mem.object_transform_range()),
            )]),
        );
        dub.write(
            const_res.rasterization_scene_info_desc(),
            0,
            br::DescriptorUpdateInfo::UniformBuffer(vec![(
                mem.mem.buffer.0.native_ptr(),
                range_cast_u64_usize(mem.camera_info_range()),
            )]),
        );
        dub.write(
            const_res.rasterization_scene_info_desc(),
            1,
            br::DescriptorUpdateInfo::UniformBuffer(vec![(
                mem.mem.buffer.0.native_ptr(),
                range_cast_u64_usize(mem.directional_light_info_range()),
            )]),
        );
        dub.write(
            const_res.material_info_desc(),
            0,
            br::DescriptorUpdateInfo::UniformBuffer(vec![(
                mem.mem.buffer.0.native_ptr(),
                range_cast_u64_usize(mem.material_range()),
            )]),
        );
        dub.submit(e.graphics_device());

        let screen_res = ScreenResources::new(e, &const_res);

        let command_buffers = peridot::CommandBundle::new(
            e.graphics(),
            peridot::CBSubmissionType::Graphics,
            e.backbuffer_count(),
        )
        .expect("Failed to alloc command bundle");
        Self::repopulate_screen_commands(
            render_area,
            &command_buffers,
            &const_res,
            &screen_res,
            &mem,
        );
        let update_commands =
            peridot::CommandBundle::new(e.graphics(), peridot::CBSubmissionType::Transfer, 1)
                .expect("Failed to alloc update command buffers");

        Self {
            const_res,
            mem,
            main_camera,
            screen_res,
            command_buffers,
            update_commands,
            dirty_main_camera: false,
            aspect,
            ph: std::marker::PhantomData,
        }
    }

    fn update(
        &mut self,
        e: &peridot::Engine<NL>,
        on_backbuffer_of: u32,
        _delta_time: std::time::Duration,
    ) -> (Option<br::SubmissionBatch>, br::SubmissionBatch) {
        self.mem.dynamic_stg.clear();
        let mut tfb = peridot::TransferBatch::new();

        if self.dirty_main_camera {
            self.mem
                .apply_main_camera(e.graphics(), &self.main_camera, self.aspect);
            self.dirty_main_camera = false;
        }

        self.mem.ready_transfer(e.graphics(), &mut tfb);
        let update_submission = if tfb.has_copy_ops() || tfb.has_ready_barrier_ops() {
            self.update_commands
                .reset()
                .expect("Failed to reset update commands");
            {
                let mut r = self.update_commands[0]
                    .begin()
                    .expect("Failed to begin recording update commands");
                tfb.sink_transfer_commands(&mut r);
                tfb.sink_graphics_ready_commands(&mut r);
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
    fn on_resize(&mut self, e: &peridot::Engine<NL>, new_size: peridot::math::Vector2<usize>) {
        self.screen_res = ScreenResources::new(e, &self.const_res);

        Self::repopulate_screen_commands(
            br::vk::VkRect2D {
                offset: br::vk::VkOffset2D { x: 0, y: 0 },
                extent: br::vk::VkExtent2D {
                    width: new_size.0 as _,
                    height: new_size.1 as _,
                },
            },
            &self.command_buffers,
            &self.const_res,
            &self.screen_res,
            &self.mem,
        );

        self.dirty_main_camera = true;
        self.aspect = new_size.0 as f32 / new_size.1 as f32;
    }
}
impl<NL: peridot::NativeLinker> Game<NL> {
    fn repopulate_screen_commands(
        render_area: br::vk::VkRect2D,
        command_buffers: &peridot::CommandBundle,
        const_res: &ConstResources,
        screen_res: &ScreenResources,
        mem: &Memory,
    ) {
        for (cb, fb) in command_buffers.iter().zip(&screen_res.frame_buffers) {
            let mut r = cb.begin().expect("Failed to begin command recording");
            r.begin_render_pass(
                &const_res.render_pass,
                fb,
                render_area.clone(),
                &[
                    br::ClearValue::color_f32([0.25 * 0.25, 0.5 * 0.25, 1.0 * 0.25, 1.0]),
                    br::ClearValue::depth_stencil(1.0, 0),
                ],
                true,
            );
            screen_res.grid_render_pipeline.bind(&mut r);
            r.bind_graphics_descriptor_sets(0, &[const_res.grid_transform_desc()], &[]);
            r.bind_vertex_buffers(0, &[(&mem.mem.buffer.0, mem.static_offsets.grid as _)]);
            r.draw(mesh::GRID_MESH_LINE_COUNT as _, 1, 0, 0);
            screen_res.pbr_pipeline.bind(&mut r);
            r.bind_graphics_descriptor_sets(
                0,
                &[
                    const_res.object_transform_desc(),
                    const_res.rasterization_scene_info_desc(),
                    const_res.material_info_desc(),
                ],
                &[],
            );
            r.bind_vertex_buffers(
                0,
                &[(
                    &mem.mem.buffer.0,
                    mem.static_offsets.icosphere_vertices as _,
                )],
            );
            r.bind_index_buffer(
                &mem.mem.buffer.0,
                mem.static_offsets.icosphere_indices as _,
                br::IndexType::U16,
            );
            r.draw_indexed(mem.icosphere_vertex_count as _, 1, 0, 0, 0);
            r.end_render_pass();
        }
    }
}
