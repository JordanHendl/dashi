#![cfg(feature = "vulkan")]

mod common;

use common::ValidationContext;
use dashi::{execution::CommandRing, ContextInfo, QueueType, SubmitInfo};

#[test]
#[serial_test::serial]
fn command_ring_completion_remains_visible_to_multiple_observers() {
    let mut context = ValidationContext::headless(&ContextInfo::default()).unwrap();
    let mut ring =
        CommandRing::new(&mut context, "fence observers", 2, QueueType::Graphics).unwrap();
    for _ in 0..8 {
        ring.record(|_| {}).unwrap();
        let fence = ring.submit_with_fence(&SubmitInfo::default()).unwrap();
        context.wait_fence(fence).unwrap();
        assert!(context.poll_fence(fence).unwrap());
        assert!(context.poll_fence(fence).unwrap());
        context.wait_fence(fence).unwrap();
    }
    ring.wait_all().unwrap();
    drop(ring);
}
