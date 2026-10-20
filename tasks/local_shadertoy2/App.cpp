#include "App.hpp"

#include <etna/Etna.hpp>
#include <etna/GlobalContext.hpp>
#include <etna/PipelineManager.hpp>
#include <etna/RenderTargetStates.hpp>

#include <cstddef>
#include <cstdint>


namespace
{
constexpr vk::Format materialFormat = vk::Format::eR16G16B16A16Sfloat;

// Matches the fragment shaders' push constants (vec2, float, uint).
struct Parameters
{
  float resolution[2];
  float time;
  uint32_t targetIsSrgb;
};
static_assert(offsetof(Parameters, time) == 8);
static_assert(offsetof(Parameters, targetIsSrgb) == 12);
static_assert(sizeof(Parameters) == 16);

bool isSrgb(vk::Format format)
{
  return format == vk::Format::eB8G8R8A8Srgb || format == vk::Format::eR8G8B8A8Srgb ||
    format == vk::Format::eA8B8G8R8SrgbPack32;
}
} // namespace


App::App()
{
  auto glfwInstExts = windowing.getRequiredVulkanInstanceExtensions();
  std::vector<const char*> instanceExtensions{glfwInstExts.begin(), glfwInstExts.end()};
  std::vector<const char*> deviceExtensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  etna::initialize(etna::InitParams{
    .applicationName = "Local Shadertoy 2",
    .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
    .instanceExtensions = instanceExtensions,
    .deviceExtensions = deviceExtensions,
    .physicalDeviceIndexOverride = {},
    // Both passes share their images; wait for the previous frame before reusing them.
    .numFramesInFlight = 1,
  });

  auto& ctx = etna::get_context();
  commandManager = ctx.createPerFrameCmdMgr();
  osWindow = windowing.createWindow(OsWindow::CreateInfo{.resolution = resolution});
  vkWindow = ctx.createWindow(etna::Window::CreateInfo{
    .surface = osWindow->createVkSurface(ctx.getInstance()),
  });

  etna::create_program(
    "shadertoy2_procedural",
    {LOCAL_SHADERTOY2_SHADERS_ROOT "fullscreen.vert.spv",
     LOCAL_SHADERTOY2_SHADERS_ROOT "procedural.frag.spv"});
  etna::create_program(
    "shadertoy2_scene",
    {LOCAL_SHADERTOY2_SHADERS_ROOT "fullscreen.vert.spv",
     LOCAL_SHADERTOY2_SHADERS_ROOT "toy.frag.spv"});

  materialSampler = etna::Sampler(etna::Sampler::CreateInfo{
    .filter = vk::Filter::eLinear,
    .addressMode = vk::SamplerAddressMode::eRepeat,
    .name = "shadertoy2_material_sampler",
    .maxLod = 0.0f,
  });

  recreateSwapchain(resolution);
  startTime = windowing.getTime();
}

App::~App()
{
  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}

void App::recreateSwapchain(glm::uvec2 requested_resolution)
{
  auto& ctx = etna::get_context();
  ETNA_CHECK_VK_RESULT(ctx.getDevice().waitIdle());

  auto [w, h] = vkWindow->recreateSwapchain(etna::Window::DesiredProperties{
    .resolution = {requested_resolution.x, requested_resolution.y},
    .vsync = useVsync,
    .numFramesInFlight = static_cast<uint32_t>(commandManager->getCmdBufferCount()),
  });
  resolution = {w, h};

  // Floating-point linear attachments preserve the height derivatives used for bump mapping.
  albedoHeightImage = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{w, h, 1},
    .name = "shadertoy2_albedo_height",
    .format = materialFormat,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });
  roughnessMetalnessImage = ctx.createImage(etna::Image::CreateInfo{
    .extent = vk::Extent3D{w, h, 1},
    .name = "shadertoy2_roughness_metalness",
    .format = materialFormat,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
  });

  etna::GraphicsPipeline::CreateInfo proceduralInfo;
  proceduralInfo.depthConfig.depthTestEnable = vk::False;
  proceduralInfo.depthConfig.depthWriteEnable = vk::False;
  proceduralInfo.fragmentShaderOutput.colorAttachmentFormats = {materialFormat, materialFormat};
  // Etna defaults to one blending attachment. MRT needs one entry per fragment output.
  proceduralInfo.blendingConfig.attachments = {
    proceduralInfo.blendingConfig.attachments.front(),
    proceduralInfo.blendingConfig.attachments.front(),
  };

  auto& pipelineManager = ctx.getPipelineManager();
  proceduralPipeline = {};
  proceduralPipeline =
    pipelineManager.createGraphicsPipeline("shadertoy2_procedural", proceduralInfo);

  etna::GraphicsPipeline::CreateInfo sceneInfo;
  sceneInfo.depthConfig.depthTestEnable = vk::False;
  sceneInfo.depthConfig.depthWriteEnable = vk::False;
  sceneInfo.fragmentShaderOutput.colorAttachmentFormats = {vkWindow->getCurrentFormat()};
  // The surface format may change when the swapchain is restored.
  scenePipeline = {};
  scenePipeline = pipelineManager.createGraphicsPipeline("shadertoy2_scene", sceneInfo);
}

void App::run()
{
  while (!osWindow->isBeingClosed())
  {
    windowing.poll();
    if (!osWindow->isBeingClosed())
      drawFrame();
  }

  ETNA_CHECK_VK_RESULT(etna::get_context().getDevice().waitIdle());
}

void App::drawFrame()
{
  auto currentCmdBuf = commandManager->acquireNext();
  etna::begin_frame();
  auto nextSwapchainImage = vkWindow->acquireNext();

  if (nextSwapchainImage)
  {
    auto [backbuffer, backbufferView, backbufferAvailableSem, backbufferReadyForPresentSem] =
      *nextSwapchainImage;

    Parameters params{
      {static_cast<float>(resolution.x), static_cast<float>(resolution.y)},
      static_cast<float>(windowing.getTime() - startTime),
      0,
    };
    const vk::Rect2D renderArea{{0, 0}, {resolution.x, resolution.y}};

    ETNA_CHECK_VK_RESULT(currentCmdBuf.begin(vk::CommandBufferBeginInfo{}));

    // Pass 1: the independent Shadertoy buffers A and B become two MRT outputs.
    // RenderTargetState transitions both images to color attachments, flushes the
    // barriers, sets viewport/scissor and begins dynamic rendering.
    {
      etna::RenderTargetState renderTargets(
        currentCmdBuf,
        renderArea,
        {{.image = albedoHeightImage.get(),
          .view = albedoHeightImage.getView({}),
          .loadOp = vk::AttachmentLoadOp::eDontCare},
         {.image = roughnessMetalnessImage.get(),
          .view = roughnessMetalnessImage.getView({}),
          .loadOp = vk::AttachmentLoadOp::eDontCare}},
        {});

      currentCmdBuf.bindPipeline(
        vk::PipelineBindPoint::eGraphics, proceduralPipeline.getVkPipeline());
      currentCmdBuf.pushConstants(
        proceduralPipeline.getVkPipelineLayout(),
        vk::ShaderStageFlagBits::eFragment,
        0,
        sizeof(params),
        &params);
      currentCmdBuf.draw(3, 1, 0, 0);
    } // End rendering before transitioning the outputs to sampled images.

    // Etna 1.15 uses BY_REGION for its image barriers. Triplanar sampling reads
    // arbitrary texels, so also synchronize the whole framebuffer between passes.
    const vk::MemoryBarrier2 materialBarrier{
      .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
      .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
      .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
    };
    currentCmdBuf.pipelineBarrier2(vk::DependencyInfo{
      .dependencyFlags = {},
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &materialBarrier,
    });

    auto shaderInfo = etna::get_shader_program("shadertoy2_scene");
    auto descriptorSet = etna::create_descriptor_set(
      shaderInfo.getDescriptorLayoutId(0),
      currentCmdBuf,
      {etna::Binding{
         0,
         albedoHeightImage.genBinding(
           materialSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
       etna::Binding{
         1,
         roughnessMetalnessImage.genBinding(
           materialSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)}});

    // Pass 2: descriptor creation schedules color-write -> fragment-sampled-read
    // barriers; RenderTargetState flushes them before starting the scene pass.
    {
      etna::RenderTargetState renderTargets(
        currentCmdBuf,
        renderArea,
        {{.image = backbuffer, .view = backbufferView, .loadOp = vk::AttachmentLoadOp::eDontCare}},
        {});

      params.targetIsSrgb = isSrgb(vkWindow->getCurrentFormat()) ? 1u : 0u;
      currentCmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, scenePipeline.getVkPipeline());
      currentCmdBuf.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        scenePipeline.getVkPipelineLayout(),
        0,
        {descriptorSet.getVkSet()},
        {});
      currentCmdBuf.pushConstants(
        scenePipeline.getVkPipelineLayout(),
        vk::ShaderStageFlagBits::eFragment,
        0,
        sizeof(params),
        &params);
      currentCmdBuf.draw(3, 1, 0, 0);
    }

    etna::set_state(
      currentCmdBuf,
      backbuffer,
      vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      {},
      vk::ImageLayout::ePresentSrcKHR,
      vk::ImageAspectFlagBits::eColor);
    etna::flush_barriers(currentCmdBuf);
    ETNA_CHECK_VK_RESULT(currentCmdBuf.end());

    auto renderingDone = commandManager->submit(
      std::move(currentCmdBuf),
      std::move(backbufferAvailableSem),
      std::move(backbufferReadyForPresentSem));
    if (!vkWindow->present(std::move(renderingDone), backbufferView))
      nextSwapchainImage = std::nullopt;
  }

  etna::end_frame();

  if (!nextSwapchainImage)
  {
    const auto requestedResolution = osWindow->getResolution();
    if (requestedResolution.x != 0 && requestedResolution.y != 0)
      recreateSwapchain(requestedResolution);
  }
}
