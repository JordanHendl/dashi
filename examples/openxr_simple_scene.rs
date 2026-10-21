#[cfg(not(feature = "dashi-openxr"))]
fn main() {
    eprintln!("Enable --features dashi-openxr to run this example");
}

#[cfg(feature = "dashi-openxr")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use dashi::*;
    use openxr as xr;
    let bootstrap = XrGraphics::load("dashi stereo clear")?;
    let mut ctx = Context::new(&ContextInfo {
        xr: Some(bootstrap.clone()),
        ..Default::default()
    })?;
    let (session, mut waiter, mut stream) = ctx.create_xr_session()?;
    let configurations = bootstrap.instance.enumerate_view_configuration_views(
        bootstrap.system,
        xr::ViewConfigurationType::PRIMARY_STEREO,
    )?;
    let width = configurations
        .iter()
        .map(|v| v.recommended_image_rect_width)
        .max()
        .ok_or("no views")?;
    let height = configurations
        .iter()
        .map(|v| v.recommended_image_rect_height)
        .max()
        .ok_or("no views")?;
    let formats = session.enumerate_swapchain_formats()?;
    let (format, raw_format) = [(Format::RGBA8, 43), (Format::BGRA8, 50)]
        .into_iter()
        .find(|(_, raw)| formats.contains(raw))
        .ok_or("runtime has no supported sRGB format")?;
    let mut swapchain = session.create_swapchain(&xr::SwapchainCreateInfo {
        create_flags: xr::SwapchainCreateFlags::EMPTY,
        usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
            | xr::SwapchainUsageFlags::TRANSFER_SRC
            | xr::SwapchainUsageFlags::TRANSFER_DST
            | xr::SwapchainUsageFlags::SAMPLED,
        format: raw_format,
        sample_count: 1,
        width,
        height,
        face_count: 1,
        array_size: configurations.len() as u32,
        mip_count: 1,
    })?;
    let images = swapchain
        .enumerate_images()?
        .into_iter()
        .map(|raw| unsafe {
            ctx.import_xr_image(
                raw,
                &ImageInfo {
                    debug_name: "OpenXR swapchain",
                    dim: [width, height, 1],
                    layers: configurations.len() as u32,
                    format,
                    storage: false,
                    ..Default::default()
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let space =
        session.create_reference_space(xr::ReferenceSpaceType::LOCAL, xr::Posef::IDENTITY)?;
    let viewport = Viewport {
        area: FRect2D {
            w: width as f32,
            h: height as f32,
            ..Default::default()
        },
        scissor: Rect2D {
            w: width,
            h: height,
            ..Default::default()
        },
        ..Default::default()
    };
    let pass = ctx.make_render_pass(&RenderPassInfo {
        debug_name: "XR clear",
        viewport,
        subpasses: &[SubpassDescription {
            color_attachments: &[AttachmentDescription {
                format,
                ..Default::default()
            }],
            depth_stencil_attachment: None,
            subpass_dependencies: &[],
        }],
    })?;
    let mut events = xr::EventDataBuffer::new();
    let mut running = false;
    let mut frames = 0;
    'main: loop {
        while let Some(event) = bootstrap.instance.poll_event(&mut events)? {
            match event {
                xr::Event::SessionStateChanged(event) => match event.state() {
                    xr::SessionState::READY => {
                        session.begin(xr::ViewConfigurationType::PRIMARY_STEREO)?;
                        running = true;
                    }
                    xr::SessionState::STOPPING => {
                        session.end()?;
                        running = false;
                    }
                    xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => break 'main,
                    _ => {}
                },
                xr::Event::InstanceLossPending(_) => break 'main,
                _ => {}
            }
        }
        if !running {
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }
        let frame = waiter.wait()?;
        stream.begin()?;
        let (flags, views) = session.locate_views(
            xr::ViewConfigurationType::PRIMARY_STEREO,
            frame.predicted_display_time,
            &space,
        )?;
        if !frame.should_render
            || !flags.contains(
                xr::ViewStateFlags::ORIENTATION_VALID | xr::ViewStateFlags::POSITION_VALID,
            )
        {
            stream.end(
                frame.predicted_display_time,
                xr::EnvironmentBlendMode::OPAQUE,
                &[],
            )?;
            continue;
        }
        let index = swapchain.acquire_image()?;
        swapchain.wait_image(xr::Duration::INFINITE)?;
        let mut queue = ctx.begin_command_queue(QueueType::Graphics, "XR clear", false)?;
        let mut commands = CommandStream::new().begin();
        for eye in 0..views.len() {
            let mut colors = [None; 8];
            colors[0] = Some(ImageView {
                img: images[index as usize],
                range: SubresourceRange::new(0, 1, eye as u32, 1),
                ..Default::default()
            });
            let mut clears = [None; 8];
            clears[0] = Some(ClearValue::Color([
                0.02,
                0.05 + eye as f32 * 0.03,
                0.1,
                1.0,
            ]));
            commands = commands
                .begin_render_pass(&driver::command::BeginRenderPass {
                    render_pass: pass,
                    viewport,
                    color_attachments: colors,
                    clear_values: clears,
                    ..Default::default()
                })
                .stop_drawing();
        }
        commands.end().append(&mut queue)?;
        let fence = ctx.submit_command_queue(&mut queue, &Default::default())?;
        ctx.wait_fence(fence)?;
        ctx.destroy_command_queue(queue);
        swapchain.release_image()?;
        let projection_views = views
            .iter()
            .enumerate()
            .map(|(eye, view)| {
                xr::CompositionLayerProjectionView::new()
                    .pose(view.pose)
                    .fov(view.fov)
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&swapchain)
                            .image_array_index(eye as u32)
                            .image_rect(xr::Rect2Di {
                                offset: xr::Offset2Di { x: 0, y: 0 },
                                extent: xr::Extent2Di {
                                    width: width as i32,
                                    height: height as i32,
                                },
                            }),
                    )
            })
            .collect::<Vec<_>>();
        stream.end(
            frame.predicted_display_time,
            xr::EnvironmentBlendMode::OPAQUE,
            &[&xr::CompositionLayerProjection::new()
                .space(&space)
                .views(&projection_views)],
        )?;
        frames += 1;
        if frames == 300 {
            session.request_exit()?;
        }
    }
    ctx.destroy_render_pass(pass);
    for image in images {
        ctx.destroy_image(image);
    }
    drop(space);
    drop(swapchain);
    drop(stream);
    drop(waiter);
    drop(session);
    ctx.destroy();
    Ok(())
}
