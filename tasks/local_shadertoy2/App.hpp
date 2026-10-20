#pragma once

#include <etna/Window.hpp>
#include <etna/PerFrameCmdMgr.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/Image.hpp>
#include <etna/Sampler.hpp>

#include "wsi/OsWindowingManager.hpp"


class App
{
public:
  App();
  ~App();

  void run();

private:
  void recreateSwapchain(glm::uvec2 requested_resolution);
  void drawFrame();

  OsWindowingManager windowing;
  std::unique_ptr<OsWindow> osWindow;
  glm::uvec2 resolution{1280, 720};
  bool useVsync = true;

  std::unique_ptr<etna::Window> vkWindow;
  std::unique_ptr<etna::PerFrameCmdMgr> commandManager;
  etna::GraphicsPipeline proceduralPipeline;
  etna::GraphicsPipeline scenePipeline;
  etna::Image albedoHeightImage;
  etna::Image roughnessMetalnessImage;
  etna::Sampler materialSampler;
  double startTime = 0.0;
};
