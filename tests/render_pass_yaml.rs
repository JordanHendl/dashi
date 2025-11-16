use dashi::gpu::vulkan::*;
use serial_test::serial;

const TEST_RENDERPASS_YAML: &str = r#"
debug_name: "test.render_pass"
viewport:
  area:    { x: 0.0, y: 0.0, w: 4.0, h: 4.0 }
  scissor: { x: 0,   y: 0,   w: 4,    h: 4    }
  min_depth: 0.0
  max_depth: 1.0
subpasses:
  - color_attachments:
      - debug_name: "gbuffer.albedo"
        format: RGBA8
        samples: S1
        load_op: Clear
        store_op: Store
        stencil_load_op: DontCare
        stencil_store_op: DontCare
        clear_value:
          kind: color
          value: [0.2, 0.3, 0.4, 1.0]
    depth_stencil_attachment:
      debug_name: "gbuffer.depth"
      format: D24S8
      samples: S1
      load_op: Clear
      store_op: DontCare
      stencil_load_op: Clear
      stencil_store_op: DontCare
      clear_value:
        kind: depth_stencil
        depth: 1.0
        stencil: 0
  - color_attachments:
      - debug_name: "swapchain.present"
        format: BGRA8
        samples: S1
        load_op: Clear
        store_op: Store
        stencil_load_op: DontCare
        stencil_store_op: DontCare
"#;

#[test]
#[serial]
fn yaml_render_pass_produces_expected_targets() {
    let mut ctx = Context::headless(&Default::default()).unwrap();
    let rp = ctx
        .make_render_pass_from_yaml(TEST_RENDERPASS_YAML)
        .expect("render pass from yaml");

    assert_eq!(rp.subpasses.len(), 2);

    let first = &rp.subpasses[0];
    let first_color = first.color_attachments[0]
        .as_ref()
        .expect("first subpass color attachment");
    assert_eq!(first_color.name.as_deref(), Some("gbuffer.albedo"));
    assert_eq!(first_color.description.format, Format::RGBA8);
    assert_eq!(
        first_color.clear_value,
        Some(ClearValue::Color([0.2, 0.3, 0.4, 1.0]))
    );

    let depth = first
        .depth_attachment
        .as_ref()
        .expect("first subpass depth attachment");
    assert_eq!(depth.name.as_deref(), Some("gbuffer.depth"));
    assert_eq!(depth.description.format, Format::D24S8);
    assert_eq!(
        depth.clear_value,
        Some(ClearValue::DepthStencil {
            depth: 1.0,
            stencil: 0,
        })
    );

    let second = &rp.subpasses[1];
    let present = second.color_attachments[0]
        .as_ref()
        .expect("second subpass color attachment");
    assert_eq!(present.name.as_deref(), Some("swapchain.present"));
    assert_eq!(present.description.format, Format::BGRA8);
    assert!(present.clear_value.is_none());

    ctx.destroy_render_pass(rp.render_pass);
    ctx.destroy();
}

#[test]
#[serial]
fn yaml_render_pass_allows_overriding_attachment_views() {
    let mut ctx = Context::headless(&Default::default()).unwrap();
    let mut rp = ctx
        .make_render_pass_from_yaml(TEST_RENDERPASS_YAML)
        .expect("render pass from yaml");

    let override_img = ctx
        .make_image(&ImageInfo {
            debug_name: "override",
            dim: [4, 4, 1],
            layers: 1,
            format: Format::BGRA8,
            mip_levels: 1,
            initial_data: None,
            ..Default::default()
        })
        .expect("override image");

    let override_view = ImageView {
        img: override_img,
        range: Default::default(),
        aspect: AspectMask::Color,
    };

    {
        let subpass = rp.subpasses.get_mut(1).expect("second subpass available");
        let target = subpass
            .find_color_attachment_mut("swapchain.present")
            .expect("present attachment to exist");
        target.view = override_view;
    }

    let views = rp.subpasses[1].color_views();
    assert_eq!(views[0].expect("view set").img, override_img);

    ctx.destroy_image(override_img);
    ctx.destroy_render_pass(rp.render_pass);
    ctx.destroy();
}
