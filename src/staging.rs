use bedrock as br;
use peridot::mthelper::{make_shared_mutable_ref, DynamicMutabilityProvider, SharedMutableRef};
use peridot_command_object::{
    BufferUsage, GraphicsCommandCombiner, GraphicsCommandSubmission, RangedBuffer,
};
use peridot_memory_manager::{AnyPointer, MemoryManager};

pub struct DynamicStagingBuffer {
    buffer: SharedMutableRef<peridot_memory_manager::Buffer>,
    mapped: core::cell::OnceCell<AnyPointer>,
    cap: u64,
    top: u64,
}
impl DynamicStagingBuffer {
    const DEFAULT_INIT_CAP: u64 = 128;

    pub fn new(e: &peridot::Graphics, memory_manager: &mut MemoryManager) -> br::Result<Self> {
        let buffer = make_shared_mutable_ref(memory_manager.allocate_upload_buffer(
            e,
            br::BufferDesc::new(Self::DEFAULT_INIT_CAP as _, br::BufferUsage::TRANSFER_SRC),
        )?);

        Ok(Self {
            buffer,
            mapped: core::cell::OnceCell::new(),
            cap: Self::DEFAULT_INIT_CAP,
            top: 0,
        })
    }

    pub fn buffer(&self) -> &SharedMutableRef<peridot_memory_manager::Buffer> {
        &self.buffer
    }

    pub fn clear(&mut self) {
        self.top = 0;
    }

    fn mapped(&mut self) -> AnyPointer {
        *self.mapped.get_or_init(|| unsafe {
            self.buffer
                .borrow_mut()
                .map_raw(0..self.cap as _)
                .expect("Failed to map memory")
        })
    }

    pub fn end_mapped(&mut self, e: &peridot::Graphics) {
        if let Some(_) = self.mapped.take() {
            if self.buffer.borrow().requires_explicit_sync() {
                unsafe {
                    self.buffer
                        .borrow_mut()
                        .flush_ranges_raw(&[0..self.cap as _])
                        .expect("Failed to flush memory range");
                }
            }

            unsafe {
                self.buffer.borrow_mut().unmap_raw();
            }
        }
    }

    fn resize(
        &mut self,
        e: &mut peridot::Engine<impl peridot::NativeLinker>,
        memory_manager: &mut MemoryManager,
        new_size: u64,
    ) -> br::Result<()> {
        self.end_mapped(e.graphics());

        let buffer = make_shared_mutable_ref(memory_manager.allocate_upload_buffer(
            e.graphics(),
            br::BufferDesc::new(new_size as _, br::BufferUsage::TRANSFER_SRC.transfer_dest()),
        )?);

        {
            let old_buffer_locked = self.buffer.borrow();
            let new_buffer_locked = buffer.borrow();

            let old_buffer = RangedBuffer(&old_buffer_locked, 0..self.cap);
            let new_buffer = RangedBuffer(&new_buffer_locked, 0..new_size);

            let [new_buffer_in_barrier, new_buffer_out_barrier] =
                new_buffer.clone().usage_barrier3(
                    BufferUsage::UNUSED,
                    BufferUsage::TRANSFER_DST,
                    BufferUsage::HOST_RW,
                );
            let in_barriers = [
                old_buffer
                    .clone()
                    .usage_barrier(BufferUsage::HOST_RW, BufferUsage::TRANSFER_SRC),
                new_buffer_in_barrier,
            ];
            let out_barriers = [new_buffer_out_barrier];

            let copy = old_buffer.mirror_to(new_buffer.subslice(0..self.cap));

            copy.between(in_barriers, out_barriers).submit(e)?;
        }

        self.cap = new_size;
        self.buffer = buffer;

        Ok(())
    }

    /// returns placement of the value
    pub fn push<T>(
        &mut self,
        e: &mut peridot::Engine<impl peridot::NativeLinker>,
        memory_manager: &mut MemoryManager,
        value: T,
    ) -> u64 {
        if self.top + std::mem::size_of::<T>() as u64 > self.cap {
            self.resize(
                e,
                memory_manager,
                (self.cap * 2).max(self.top + std::mem::size_of::<T>() as u64),
            )
            .expect("Failed to resize dynamic staging buffer");
        }

        let p = self.mapped();
        let placement = self.top;
        self.top = placement + std::mem::size_of::<T>() as u64;
        unsafe {
            *p.get_mut_at(placement as _) = value;
        }

        placement
    }

    /// returns first placement of the value
    pub fn push_multiple_values<T: Clone>(
        &mut self,
        e: &mut peridot::Engine<impl peridot::NativeLinker>,
        memory_manager: &mut MemoryManager,
        values: &[T],
    ) -> u64 {
        let size = std::mem::size_of::<T>() * values.len();
        if self.top + size as u64 > self.cap {
            self.resize(
                e,
                memory_manager,
                (self.cap * 2).max(self.top + size as u64),
            )
            .expect("Failed to resize dynamic staging buffer");
        }

        let p = self.mapped();
        let placement = self.top;
        self.top = placement + size as u64;
        unsafe {
            p.clone_slice_to(placement as _, values);
        }

        placement
    }

    /// returns first placement of the value
    pub fn construct_multiple_values_inplace<T>(
        &mut self,
        e: &mut peridot::Engine<impl peridot::NativeLinker>,
        memory_manager: &mut MemoryManager,
        length: usize,
        ctor: impl FnOnce(&mut [T]),
    ) -> u64 {
        let size = std::mem::size_of::<T>() * length;
        if self.top + size as u64 > self.cap {
            self.resize(
                e,
                memory_manager,
                (self.cap * 2).max(self.top + size as u64),
            )
            .expect("Failed to resize dynamic staging buffer");
        }

        let p = self.mapped();
        let placement = self.top;
        self.top = placement + size as u64;
        unsafe {
            ctor(p.slice_mut(placement as _, length));
        }

        placement
    }
}
