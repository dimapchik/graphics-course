#pragma once

#include <etna/Buffer.hpp>
#include <etna/ComputePipeline.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/Image.hpp>
#include <etna/PerFrameCmdMgr.hpp>
#include <etna/Sampler.hpp>
#include <etna/Window.hpp>

#include "wsi/OsWindowingManager.hpp"


class App
{
public:
  App();
  ~App();

  void run();

private:
  void recreateSwapchain(glm::uvec2 requested_resolution);
  void updateTitle();
  void drawFrame(float dt);
  void measureExposure(vk::CommandBuffer cmd_buf, float dt);
  void bindMetering(
    vk::CommandBuffer cmd_buf,
    vk::PipelineBindPoint bind_point,
    const etna::PipelineBase& pipeline,
    const char* program) const;

  OsWindowingManager windowing;
  std::unique_ptr<OsWindow> osWindow;
  glm::uvec2 resolution{1280, 720};

  std::unique_ptr<etna::Window> vkWindow;
  std::unique_ptr<etna::PerFrameCmdMgr> commandManager;
  etna::GraphicsPipeline scenePipeline;
  etna::GraphicsPipeline tonemapPipeline;
  etna::ComputePipeline rangePipeline;
  etna::ComputePipeline histogramPipeline;
  etna::ComputePipeline exposurePipeline;
  etna::Image hdrImage;
  etna::Sampler hdrSampler;
  etna::Buffer metering;

  float cameraYaw = 0.0f;
  bool brightLight = true;
  bool useTonemapping = true;
  bool useSharedMemory = true;
  bool exposureReady = false;
};
