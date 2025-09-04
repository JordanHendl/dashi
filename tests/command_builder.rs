use dashi::gfx::cmd::{CommandBuffer, CommandBuilder, Recording, PipelineBound};

fn assert_impl<T: CommandBuilder>() {}

#[test]
fn command_buffer_states_impl_command_builder() {
    assert_impl::<CommandBuffer<Recording>>();
    assert_impl::<CommandBuffer<PipelineBound>>();
}
