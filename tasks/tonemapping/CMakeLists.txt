add_executable(tonemapping
  main.cpp
  App.cpp
)

target_link_libraries(tonemapping
  PRIVATE glfw etna glm::glm wsi)

target_add_shaders(tonemapping
  shaders/fullscreen.vert
  shaders/scene.frag
  shaders/luminance_range.comp
  shaders/histogram.comp
  shaders/exposure.comp
  shaders/tonemap.frag
)
