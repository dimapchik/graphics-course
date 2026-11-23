#include "App.hpp"

#include <etna/Assert.hpp>
#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/Profiling.hpp>
#include <etna/RenderTargetStates.hpp>

#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "shaders/metering_config.h"


namespace
{
constexpr vk::Format HDR_FORMAT = vk::Format::eB10G11R11UfloatPack32;
constexpr const char* SCENE_PROGRAM = "tonemapping_scene";
constexpr const char* TONEMAP_PROGRAM = "tonemapping_post";
constexpr const char* RANGE_PROGRAM = "tonemapping_range";
constexpr const char* HISTOGRAM_PROGRAM = "tonemapping_histogram";
constexpr const char* EXPOSURE_PROGRAM = "tonemapping_exposure";

// The std430 storage buffer holds a float exposure, two uint extrema and the histogram.
constexpr vk::DeviceSize MINIMUM_OFFSET = sizeof(float);
constexpr vk::DeviceSize MAXIMUM_OFFSET = MINIMUM_OFFSET + sizeof(uint32_t);
constexpr vk::DeviceSize METERING_SIZE = MAXIMUM_OFFSET + (1 + HISTOGRAM_BINS) * sizeof(uint32_t);

etna::GraphicsPipeline fullscreen_pipeline(const char* program, vk::Format format)
{
  etna::GraphicsPipeline::CreateInfo info;
  info.depthConfig.depthTestEnable = vk::False;
  info.depthConfig.depthWriteEnable = vk::False;
  info.fragmentShaderOutput.colorAttachmentFormats = {format};
  return etna::get_context().getPipelineManager().createGraphicsPipeline(program, info);
}

bool is_srgb(vk::Format format)
{
  return format == vk::Format::eB8G8R8A8Srgb || format == vk::Format::eR8G8B8A8Srgb ||
    format == vk::Format::eA8B8G8R8SrgbPack32 || format == vk::Format::eR8G8B8Srgb ||
    format == vk::Format::eB8G8R8Srgb;
}

glm::uvec2 drawable_size(GLFWwindow* window)
{
  if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) == GLFW_TRUE)
    return {};

  glm::ivec2 size{};
  glfwGetFramebufferSize(window, &size.x, &size.y);
  return glm::uvec2{size};
}
} // namespace


App::App()
{
  const auto glfwExtensions = windowing.getRequiredVulkanInstanceExtensions();
  const std::vector<const char*> instanceExtensions{glfwExtensions.begin(), glfwExtensions.end()};
  const std::vector<const char*> deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  etna::initialize(etna::InitParams{
    .applicationName = "Tone mapping",
    .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
    .instanceExtensions = instanceExtensions,
    .deviceExtensions = deviceExtensions,
    // The HDR image and exposure history are reused by each frame.
    .numFramesInFlight = 1,
  });

  auto& ctx = etna::get_context();
  constexpr auto REQUIRED_HDR_FEATURES =
    vk::FormatFeatureFlagBits::eColorAttachment | vk::FormatFeatureFlagBits::eSampledImage;
  const auto hdrFeatures =
    ctx.getPhysicalDevice().getFormatProperties(HDR_FORMAT).optimalTilingFeatures;
  ETNA_VERIFYF(
    (hdrFeatures & REQUIRED_HDR_FEATURES) == REQUIRED_HDR_FEATURES,
    "The device does not support B10G11R11_UFLOAT_PACK32 as a sampled color attachment.");

  commandManager = ctx.createPerFrameCmdMgr();
  osWindow = windowing.createWindow(OsWindow::CreateInfo{.resolution = resolution});
  vkWindow = ctx.createWindow(etna::Window::CreateInfo{
    .surface = osWindow->createVkSurface(ctx.getInstance()),
  });

  etna::create_program(
    SCENE_PROGRAM,
    {TONEMAPPING_SHADERS_ROOT "fullscreen.vert.spv", TONEMAPPING_SHADERS_ROOT "scene.frag.spv"});
  etna::create_program(
    TONEMAP_PROGRAM,
    {TONEMAPPING_SHADERS_ROOT "fullscreen.vert.spv", TONEMAPPING_SHADERS_ROOT "tonemap.frag.spv"});
  etna::create_program(RANGE_PROGRAM, {TONEMAPPING_SHADERS_ROOT "luminance_range.comp.spv"});
  etna::create_program(HISTOGRAM_PROGRAM, {TONEMAPPING_SHADERS_ROOT "histogram.comp.spv"});
  etna::create_program(EXPOSURE_PROGRAM, {TONEMAPPING_SHADERS_ROOT "exposure.comp.spv"});

  auto& pipelineManager = ctx.getPipelineManager();
  scenePipeline = fullscreen_pipeline(SCENE_PROGRAM, HDR_FORMAT);
  rangePipeline = pipelineManager.createComputePipeline(RANGE_PROGRAM, {});
  histogramPipeline = pipelineManager.createComputePipeline(HISTOGRAM_PROGRAM, {});
  exposurePipeline = pipelineManager.createComputePipeline(EXPOSURE_PROGRAM, {});

  hdrSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eNearest,
    .mipmapMode = vk::SamplerMipmapMode::eNearest,
    .name = "hdr_sampler",
  });
  metering = ctx.createBuffer(etna::Buffer::CreateInfo{
    .size = METERING_SIZE,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
    .name = "exposure_and_histogram",
  });

  updateTitle();
}

App::~App()
{
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}

void App::recreateSwapchain(glm::uvec2 requested_resolution)
{
  auto& ctx = etna::get_context();
  ETNA_CHECK_VK_RESULT(ctx.getDevice().waitIdle());

  auto [width, height] = vkWindow->recreateSwapchain(etna::Window::DesiredProperties{
    .resolution = {requested_resolution.x, requested_resolution.y},
    .vsync = true,
    .autoGamma = false, // Prefer an sRGB attachment for hardware encoding.
    .numFramesInFlight = static_cast<uint32_t>(commandManager->getCmdBufferCount()),
  });
  resolution = {width, height};

  const auto format = vkWindow->getCurrentFormat();
  const bool unorm = format == vk::Format::eB8G8R8A8Unorm || format == vk::Format::eR8G8B8A8Unorm ||
    format == vk::Format::eA8B8G8R8UnormPack32 || format == vk::Format::eR8G8B8Unorm ||
    format == vk::Format::eB8G8R8Unorm;
  ETNA_VERIFYF(
    is_srgb(format) || unorm,
    "Tone mapping requires an 8-bit RGB swapchain, but Etna selected {}.",
    vk::to_string(format));

  hdrImage = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{width, height, 1},
    .name = "hdr_scene",
    .format = HDR_FORMAT,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });
  tonemapPipeline = fullscreen_pipeline(TONEMAP_PROGRAM, format);
}

void App::updateTitle()
{
  std::string title = useTonemapping ? "Tone mapping on" : "Direct output (clipping)";
  title += useSharedMemory ? " | shared memory" : " | global atomics";
  glfwSetWindowTitle(osWindow->native(), title.c_str());
}

void App::run()
{
  double previousTime = windowing.getTime();
  while (!osWindow->isBeingClosed())
  {
    windowing.poll();
    const auto& kb = osWindow->keyboard;
    if (osWindow->isBeingClosed() || kb[KeyboardKey::kEscape] == ButtonState::Rising)
      break;

    const auto framebufferSize = drawable_size(osWindow->native());
    if (framebufferSize.x == 0 || framebufferSize.y == 0)
    {
      glfwWaitEventsTimeout(0.1);
      previousTime = windowing.getTime();
      continue;
    }
    if (!hdrImage.get())
      recreateSwapchain(framebufferSize);

    const double now = windowing.getTime();
    const float dt = static_cast<float>(now - previousTime);
    previousTime = now;

    cameraYaw += dt *
      (static_cast<float>(is_held_down(kb[KeyboardKey::kRight])) -
       static_cast<float>(is_held_down(kb[KeyboardKey::kLeft])));
    if (kb[KeyboardKey::kL] == ButtonState::Rising)
      localLightOn = !localLightOn;
    if (kb[KeyboardKey::kT] == ButtonState::Rising)
    {
      useTonemapping = !useTonemapping;
      updateTitle();
    }
    if (kb[KeyboardKey::kM] == ButtonState::Rising)
    {
      useSharedMemory = !useSharedMemory;
      updateTitle();
    }

    drawFrame(dt);
    FrameMark;
  }
}

void App::bindMetering(
  vk::CommandBuffer cmd_buf,
  vk::PipelineBindPoint bind_point,
  const etna::PipelineBase& pipeline,
  const char* program) const
{
  auto shader = etna::get_shader_program(program);
  auto set = etna::create_descriptor_set(
    shader.getDescriptorLayoutId(0),
    cmd_buf,
    {etna::Binding{
       0, hdrImage.genBinding(hdrSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
     etna::Binding{1, metering.genBinding()}},
    BarrierBehavior::eSuppressBarriers);

  cmd_buf.bindPipeline(bind_point, pipeline.getVkPipeline());
  cmd_buf.bindDescriptorSets(bind_point, pipeline.getVkPipelineLayout(), 0, {set.getVkSet()}, {});
}

void App::measureExposure(vk::CommandBuffer cmd_buf, float dt)
{
  etna::set_state(
    cmd_buf,
    hdrImage.get(),
    vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eFragmentShader,
    vk::AccessFlagBits2::eShaderSampledRead,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    vk::ImageAspectFlagBits::eColor);
  etna::set_state(
    cmd_buf,
    metering.get(),
    vk::PipelineStageFlagBits2::eTransfer,
    vk::AccessFlagBits2::eTransferWrite);
  etna::flush_barriers(cmd_buf);
  // Reset this frame's extrema and bins while keeping the previous exposure.
  cmd_buf.fillBuffer(
    metering.get(), MINIMUM_OFFSET, sizeof(uint32_t), std::numeric_limits<uint32_t>::max());
  cmd_buf.fillBuffer(metering.get(), MAXIMUM_OFFSET, vk::WholeSize, 0);

  const auto prepareCompute = [&]() {
    etna::set_state(
      cmd_buf,
      metering.get(),
      vk::PipelineStageFlagBits2::eComputeShader,
      vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
      ForceSetState::eTrue);
    etna::flush_barriers(cmd_buf);
  };
  const auto dispatchTiles = [&](const etna::ComputePipeline& pipeline, const char* program) {
    prepareCompute();
    bindMetering(cmd_buf, vk::PipelineBindPoint::eCompute, pipeline, program);
    const uint32_t shared = useSharedMemory ? 1u : 0u;
    cmd_buf.pushConstants(
      pipeline.getVkPipelineLayout(),
      vk::ShaderStageFlagBits::eCompute,
      0,
      sizeof(shared),
      &shared);
    cmd_buf.dispatch(
      (resolution.x + METERING_TILE_WIDTH - 1) / METERING_TILE_WIDTH,
      (resolution.y + METERING_TILE_HEIGHT - 1) / METERING_TILE_HEIGHT,
      1);
  };

  {
    ETNA_PROFILE_GPU(cmd_buf, luminanceRange);
    dispatchTiles(rangePipeline, RANGE_PROGRAM);
  }
  {
    ETNA_PROFILE_GPU(cmd_buf, luminanceHistogram);
    dispatchTiles(histogramPipeline, HISTOGRAM_PROGRAM);
  }

  prepareCompute();
  bindMetering(cmd_buf, vk::PipelineBindPoint::eCompute, exposurePipeline, EXPOSURE_PROGRAM);
  const float adaptation = exposureReady ? 1.0f - std::exp(-dt / 0.6f) : 1.0f;
  cmd_buf.pushConstants(
    exposurePipeline.getVkPipelineLayout(),
    vk::ShaderStageFlagBits::eCompute,
    0,
    sizeof(adaptation),
    &adaptation);
  cmd_buf.dispatch(1, 1, 1);
  exposureReady = true;

  etna::set_state(
    cmd_buf,
    metering.get(),
    vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eComputeShader,
    vk::AccessFlagBits2::eShaderStorageRead);
}

void App::drawFrame(float dt)
{
  auto cmd = commandManager->acquireNext();
  etna::begin_frame();
  auto nextImage = vkWindow->acquireNext();

  if (nextImage)
  {
    auto [backbuffer, view, available, ready] = *nextImage;
    const vk::Rect2D area{{0, 0}, {resolution.x, resolution.y}};

    ETNA_CHECK_VK_RESULT(cmd.begin(vk::CommandBufferBeginInfo{}));
    {
      etna::RenderTargetState targets(
        cmd,
        area,
        {{.image = hdrImage.get(),
          .view = hdrImage.getView({}),
          .loadOp = vk::AttachmentLoadOp::eDontCare}},
        {});
      const float sceneParams[] = {
        static_cast<float>(resolution.x),
        static_cast<float>(resolution.y),
        cameraYaw,
        localLightOn ? 1.0f : 0.0f,
      };
      cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, scenePipeline.getVkPipeline());
      cmd.pushConstants(
        scenePipeline.getVkPipelineLayout(),
        vk::ShaderStageFlagBits::eFragment,
        0,
        sizeof(sceneParams),
        sceneParams);
      cmd.draw(3, 1, 0, 0);
    }

    measureExposure(cmd, dt);

    {
      etna::RenderTargetState targets(
        cmd,
        area,
        {{.image = backbuffer, .view = view, .loadOp = vk::AttachmentLoadOp::eDontCare}},
        {});
      bindMetering(cmd, vk::PipelineBindPoint::eGraphics, tonemapPipeline, TONEMAP_PROGRAM);
      const uint32_t postParams[] = {
        useTonemapping ? 1u : 0u,
        is_srgb(vkWindow->getCurrentFormat()) ? 0u : 1u,
      };
      cmd.pushConstants(
        tonemapPipeline.getVkPipelineLayout(),
        vk::ShaderStageFlagBits::eFragment,
        0,
        sizeof(postParams),
        postParams);
      cmd.draw(3, 1, 0, 0);
    }

    etna::set_state(
      cmd,
      backbuffer,
      vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      {},
      vk::ImageLayout::ePresentSrcKHR,
      vk::ImageAspectFlagBits::eColor);
    etna::flush_barriers(cmd);
    ETNA_READ_BACK_GPU_PROFILING(cmd);
    ETNA_CHECK_VK_RESULT(cmd.end());

    auto renderingDone = commandManager->submit(std::move(cmd), available, std::move(ready));
    if (!vkWindow->present(renderingDone, view))
      nextImage.reset();
  }

  etna::end_frame();

  if (!nextImage)
  {
    const auto requestedResolution = drawable_size(osWindow->native());
    if (requestedResolution.x != 0 && requestedResolution.y != 0)
      recreateSwapchain(requestedResolution);
  }
}
