use bedrock as br;
use br::{MemoryBound, VkHandle};

pub struct DynamicStagingBuffer {
    buffer: peridot::Buffer,
    mapped: Option<std::ptr::NonNull<u8>>,
    require_explicit_flushing: bool,
    cap: u64,
    top: u64,
}
impl DynamicStagingBuffer {
    const DEFAULT_INIT_CAP: u64 = 128;

    pub fn new(e: &peridot::Graphics) -> br::Result<Self> {
        let buffer =
            br::BufferDesc::new(Self::DEFAULT_INIT_CAP as _, br::BufferUsage::TRANSFER_SRC)
                .create(e)?;
        let mreq = buffer.requirements();
        let mtype = e
            .memory_type_manager
            .host_visible_index(mreq.memoryTypeBits, br::MemoryPropertyFlags::HOST_COHERENT)
            .or_else(|| {
                e.memory_type_manager
                    .host_visible_index(mreq.memoryTypeBits, br::MemoryPropertyFlags::EMPTY)
            })
            .expect("no matching memory for staging buffer");
        let memory = br::DeviceMemory::allocate(e, mreq.size as _, mtype.index())?;
        let buffer = peridot::Buffer::bound(buffer, &std::rc::Rc::new(memory), 0)?;

        Ok(Self {
            buffer,
            mapped: None,
            require_explicit_flushing: !mtype.is_host_coherent(),
            cap: Self::DEFAULT_INIT_CAP,
            top: 0,
        })
    }

    pub fn buffer(&self) -> &peridot::Buffer {
        &self.buffer
    }

    pub fn clear(&mut self) {
        self.top = 0;
    }

    fn mapped(&mut self) -> std::ptr::NonNull<u8> {
        match self.mapped {
            Some(p) => p,
            None => {
                let mapped = self.buffer.map(0..self.cap).expect("Failed to map memory");
                let p =
                    unsafe { std::ptr::NonNull::new_unchecked(mapped.get_mut::<u8>(0) as *mut _) };
                self.mapped = Some(p);
                p
            }
        }
    }

    pub fn end_mapped(&mut self, e: &peridot::Graphics) {
        if let Some(_) = self.mapped.take() {
            if self.require_explicit_flushing {
                unsafe {
                    e.flush_mapped_memory_ranges(&[br::vk::VkMappedMemoryRange {
                        memory: self.buffer.memory().native_ptr(),
                        offset: 0,
                        size: self.cap,
                        ..Default::default()
                    }])
                    .expect("Failed to flush memory range");
                }
            }

            unsafe {
                self.buffer.unmap();
            }
        }
    }

    fn resize(&mut self, e: &peridot::Graphics, new_size: u64) -> br::Result<()> {
        self.end_mapped(e);

        let buffer =
            br::BufferDesc::new(new_size as _, br::BufferUsage::TRANSFER_SRC.transfer_dest())
                .create(e)?;
        let mreq = buffer.requirements();
        let mtype = e
            .memory_type_manager
            .host_visible_index(mreq.memoryTypeBits, br::MemoryPropertyFlags::HOST_COHERENT)
            .or_else(|| {
                e.memory_type_manager
                    .host_visible_index(mreq.memoryTypeBits, br::MemoryPropertyFlags::EMPTY)
            })
            .expect("no matching memory for staging buffer");
        let memory = br::DeviceMemory::allocate(e, mreq.size as _, mtype.index())?;
        let buffer = peridot::Buffer::bound(buffer, &std::rc::Rc::new(memory), 0)?;

        e.submit_commands(|r| {
            let buffers_in = &[
                br::BufferMemoryBarrier::new(
                    &buffer,
                    0..new_size,
                    0,
                    br::AccessFlags::TRANSFER.write,
                )
                .into(),
                br::BufferMemoryBarrier::new(
                    &self.buffer,
                    0..self.cap,
                    br::AccessFlags::HOST.write,
                    br::AccessFlags::TRANSFER.read,
                )
                .into(),
            ];
            let buffers_out = &[br::BufferMemoryBarrier::new(
                &buffer,
                0..new_size,
                br::AccessFlags::TRANSFER.write,
                br::AccessFlags::HOST.write,
            )];
            let copy_region = br::vk::VkBufferCopy {
                srcOffset: 0,
                dstOffset: 0,
                size: self.cap,
            };

            r.pipeline_barrier(
                br::PipelineStageFlags::HOST,
                br::PipelineStageFlags::TRANSFER,
                false,
                &[],
                buffers_in,
                &[],
            )
            .copy_buffer(&self.buffer, &buffer, &[copy_region])
            .pipeline_barrier(
                br::PipelineStageFlags::TRANSFER,
                br::PipelineStageFlags::HOST,
                false,
                &[],
                buffers_out,
                &[],
            );
        })?;

        self.cap = new_size;
        self.buffer = buffer;
        self.require_explicit_flushing = !mtype.is_host_coherent();

        Ok(())
    }

    /// returns placement of the value
    pub fn push<T>(&mut self, e: &peridot::Graphics, value: T) -> u64 {
        if self.top + std::mem::size_of::<T>() as u64 > self.cap {
            self.resize(e, self.cap * 2)
                .expect("Failed to resize dynamic staging buffer");
        }

        let p = self.mapped();
        let placement = self.top;
        self.top = placement + std::mem::size_of::<T>() as u64;
        unsafe {
            std::ptr::write(p.as_ptr().add(placement as _) as *mut T, value);
        }

        placement
    }
}
