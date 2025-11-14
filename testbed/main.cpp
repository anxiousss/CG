#include "veekay/input.hpp"
#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>

#define _USE_MATH_DEFINES
#include <math.h>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

#include <lodepng.h>

namespace {

    constexpr uint32_t max_models = 1024;
    constexpr uint32_t max_point_lights = 16;
    constexpr uint32_t max_spot_lights = 16;

    struct Vertex {
        veekay::vec3 position;
        veekay::vec3 normal;
        veekay::vec2 uv;
    };

    struct SceneUniforms {
        veekay::mat4 view_projection;
        veekay::vec3 view_position; float _pad0;
        veekay::vec3 ambient_light_intensity; float _pad1;
        veekay::vec3 sun_light_direction; float _pad2;
        veekay::vec3 sun_light_color; float _pad3;

        uint32_t point_lights_count;
        uint32_t spot_lights_count;
        float _pad4; float _pad5;
    };

    struct ModelUniforms {
        veekay::mat4 model;
        veekay::vec3 albedo_color; float _pad0;
        veekay::vec3 specular_color; float _pad1;
        float shininess; float _pad2;  float _pad3;  float _pad4;
    };

    struct PointLight {
        veekay::vec3 position;
        float radius;
        veekay::vec3 color; float _pad0;
    };

    struct SpotLight {
        veekay::vec3 position;
        float radius;
        veekay::vec3 direction;
        float angle; // Косинус угла
        veekay::vec3 color; float _pad0;
    };

    struct Mesh {
        veekay::graphics::Buffer* vertex_buffer;
        veekay::graphics::Buffer* index_buffer;
        uint32_t indices;
    };

    struct Transform {
        veekay::vec3 position = {};
        veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
        veekay::vec3 rotation = {};

        veekay::mat4 matrix() const;
    };

    struct Model {
        Mesh mesh;
        Transform transform;
        veekay::vec3 albedo_color;
        veekay::vec3 specular_color;
        float shininess;
    };

    struct Camera {
        constexpr static float default_fov = 60.0f;
        constexpr static float default_near_plane = 0.01f;
        constexpr static float default_far_plane = 100.0f;

        veekay::vec3 position = {};
        veekay::vec3 rotation = {-45.0f, 0.0f, 0.0f};
        veekay::vec3 scale = {1.0f, 1.0f, 1.0f};

        float fov = default_fov;
        float near_plane = default_near_plane;
        float far_plane = default_far_plane;

        veekay::mat4 view() const;
        veekay::mat4 inverse_view() const;
        veekay::mat4 view_projection(float aspect_ratio) const;
    };

// NOTE: Scene objects
    inline namespace {
        Camera camera{
                .position = {0.0f, -2.5f, -3.0f}
        };

        std::vector<Model> models;
        std::vector<PointLight> point_lights;
        std::vector<SpotLight> spot_lights;
    }

// NOTE: Vulkan objects
    inline namespace {
        VkShaderModule vertex_shader_module;
        VkShaderModule fragment_shader_module;

        VkDescriptorPool descriptor_pool;
        VkDescriptorSetLayout descriptor_set_layout;
        VkDescriptorSet descriptor_set;

        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;

        veekay::graphics::Buffer* scene_uniforms_buffer;
        veekay::graphics::Buffer* model_uniforms_buffer;

        veekay::graphics::Buffer* point_lights_buffer;
        veekay::graphics::Buffer* spot_lights_buffer;

        Mesh plane_mesh;
        Mesh cube_mesh;

        veekay::graphics::Texture* missing_texture;
        VkSampler missing_texture_sampler;
    }

    float toRadians(float degrees) {
        return degrees * float(M_PI) / 180.0f;
    }

    veekay::mat4 Transform::matrix() const {
        auto t = veekay::mat4::translation(position) *
                 veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z)) *
                 veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y)) *
                 veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x)) *
                 veekay::mat4::scaling(scale);
        return t;
    }

    veekay::mat4 Camera::view() const {
        auto t = veekay::mat4::translation(-position);
        auto r = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x)) *
                 veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y)) *
                 veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z));
        r = veekay::mat4::transpose(r);
        return t * r;
    }

    veekay::mat4 Camera::inverse_view() const {
        auto t = veekay::mat4::inverse_translation(position) *
                 veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, toRadians(rotation.x)) *
                 veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, toRadians(rotation.y)) *
                 veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, toRadians(rotation.z));
        return t;
    }

    veekay::mat4 Camera::view_projection(float aspect_ratio) const {
        auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
        return view() * projection;
    }

    VkShaderModule loadShaderModule(const char* path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        size_t size = file.tellg();
        std::vector<uint32_t> buffer(size / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(buffer.data()), size);
        file.close();

        VkShaderModuleCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = size,
                .pCode = buffer.data(),
        };

        VkShaderModule result;
        if (vkCreateShaderModule(veekay::app.vk_device, &
                info, nullptr, &result) != VK_SUCCESS) {
            return nullptr;
        }

        return result;
    }

    void initialize(VkCommandBuffer cmd) {
        VkDevice& device = veekay::app.vk_device;
        VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

        { // NOTE: Build graphics pipeline
            vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
            if (!vertex_shader_module) {
                std::cerr << "Failed to load Vulkan vertex shader from file\n";
                veekay::app.running = false;
                return;
            }

            fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
            if (!fragment_shader_module) {
                std::cerr << "Failed to load Vulkan fragment shader from file\n";
                veekay::app.running = false;
                return;
            }

            VkPipelineShaderStageCreateInfo stage_infos[2];

            stage_infos[0] = VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vertex_shader_module,
                    .pName = "main",
            };

            stage_infos[1] = VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = fragment_shader_module,
                    .pName = "main",
            };

            VkVertexInputBindingDescription buffer_binding{
                    .binding = 0,
                    .stride = sizeof(Vertex),
                    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };

            VkVertexInputAttributeDescription attributes[] = {
                    {
                            .location = 0,
                            .binding = 0,
                            .format = VK_FORMAT_R32G32B32_SFLOAT,
                            .offset = offsetof(Vertex, position),
                    },
                    {
                            .location = 1,
                            .binding = 0,
                            .format = VK_FORMAT_R32G32B32_SFLOAT,
                            .offset = offsetof(Vertex, normal),
                    },
                    {
                            .location = 2,
                            .binding = 0,
                            .format = VK_FORMAT_R32G32_SFLOAT,
                            .offset = offsetof(Vertex, uv),
                    },
            };

            VkPipelineVertexInputStateCreateInfo input_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    .vertexBindingDescriptionCount = 1,
                    .pVertexBindingDescriptions = &buffer_binding,
                    .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
                    .pVertexAttributeDescriptions = attributes,
            };

            VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };

            VkPipelineRasterizationStateCreateInfo raster_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                    .polygonMode = VK_POLYGON_MODE_FILL,
                    .cullMode = VK_CULL_MODE_BACK_BIT,
                    .frontFace = VK_FRONT_FACE_CLOCKWISE,
                    .lineWidth = 1.0f,
            };

            VkPipelineMultisampleStateCreateInfo sample_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                    .sampleShadingEnable = false,
                    .minSampleShading = 1.0f,
            };

            VkViewport viewport{
                    .x = 0.0f,
                    .y = 0.0f,
                    .width = static_cast<float>(veekay::app.window_width),
                    .height = static_cast<float>(veekay::app.window_height),
                    .minDepth = 0.0f,
                    .maxDepth = 1.0f,
            };

            VkRect2D scissor{
                    .offset = {0, 0},
                    .extent = {veekay::app.window_width, veekay::app.window_height},
            };

            VkPipelineViewportStateCreateInfo viewport_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                    .viewportCount = 1,
                    .pViewports = &viewport,
                    .scissorCount = 1,
                    .pScissors = &scissor,
            };

            VkPipelineDepthStencilStateCreateInfo depth_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                    .depthTestEnable = true,
                    .depthWriteEnable = true,
                    .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };

            VkPipelineColorBlendAttachmentState attachment_info{
                    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                      VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT,
            };

            VkPipelineColorBlendStateCreateInfo blend_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                    .logicOpEnable = false,
                    .logicOp = VK_LOGIC_OP_COPY,
                    .attachmentCount = 1,
                    .pAttachments = &attachment_info
            };

            {
                VkDescriptorPoolSize pools[] = {
                        {
                                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                .descriptorCount = 8,
                        },
                        {
                                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                .descriptorCount = 8,
                        },
                        {
                                .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                .descriptorCount = 8,
                        },
                        {
                                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                .descriptorCount = 8
                        },
                };

                VkDescriptorPoolCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                        .maxSets = 1,
                        .poolSizeCount = sizeof(pools) / sizeof(pools[0]),
                        .pPoolSizes = pools,
                };

                if (vkCreateDescriptorPool(device, &info, nullptr,
                                           &descriptor_pool) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor pool\n";
                    veekay::app.running = false;
                    return;
                }
            }

            {
                VkDescriptorSetLayoutBinding bindings[] = {
                        {
                                .binding = 0,
                                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                .descriptorCount = 1,
                                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                        },
                        {
                                .binding = 1,
                                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                .descriptorCount = 1,
                                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                        },
                        {
                                .binding = 2,
                                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                .descriptorCount = 1,
                                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                        },
                        {
                                .binding = 3,
                                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                .descriptorCount = 1,
                                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                        }
                };

                VkDescriptorSetLayoutCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
                        .pBindings = bindings,
                };

                if (vkCreateDescriptorSetLayout(device, &info, nullptr,
                                                &descriptor_set_layout) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor set layout\n";
                    veekay::app.running = false;
                    return;
                }
            }

            {
                VkDescriptorSetAllocateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                        .descriptorPool = descriptor_pool,
                        .descriptorSetCount = 1,
                        .pSetLayouts = &descriptor_set_layout,
                };

                if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor set\n";
                    veekay::app.running = false;
                    return;
                }
            }

            VkPipelineLayoutCreateInfo layout_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    .setLayoutCount = 1,
                    .pSetLayouts = &descriptor_set_layout,
            };

            if (vkCreatePipelineLayout(device, &layout_info,
                                       nullptr, &pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline layout\n";
                veekay::app.running = false;
                return;
            }

            VkGraphicsPipelineCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    .stageCount = 2,
                    .pStages = stage_infos,
                    .pVertexInputState = &input_state_info,
                    .pInputAssemblyState = &assembly_state_info,
                    .pViewportState = &viewport_info,
                    .pRasterizationState = &raster_info,
                    .pMultisampleState = &sample_info,
                    .pDepthStencilState = &depth_info,
                    .pColorBlendState = &blend_info,
                    .layout = pipeline_layout,
                    .renderPass = veekay::app.vk_render_pass,
            };

            if (vkCreateGraphicsPipelines(device, nullptr,
                                          1, &info, nullptr, &pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline\n";
                veekay::app.running = false;
                return;
            }
        }

        scene_uniforms_buffer = new veekay::graphics::Buffer(
                sizeof(SceneUniforms),
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        model_uniforms_buffer = new veekay::graphics::Buffer(
                max_models * sizeof(ModelUniforms),
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        point_lights_buffer = new veekay::graphics::Buffer(
                max_point_lights * sizeof(PointLight),
                nullptr,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        spot_lights_buffer = new veekay::graphics::Buffer(
                max_spot_lights * sizeof(SpotLight),
                nullptr,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        {
            VkSamplerCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            };

            if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan texture sampler\n";
                veekay::app.running = false;
                return;
            }

            uint32_t pixels[] = {
                    0xff000000, 0xffff00ff,
                    0xffff00ff, 0xff000000,
            };

            missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
                                                            VK_FORMAT_B8G8R8A8_UNORM,
                                                            pixels);
        }

        {
            VkDescriptorBufferInfo buffer_infos[] = {
                    {
                            .buffer = scene_uniforms_buffer->buffer,
                            .offset = 0,
                            .range = sizeof(SceneUniforms),
                    },
                    {
                            .buffer = model_uniforms_buffer->buffer,
                            .offset = 0,
                            .range = sizeof(ModelUniforms),
                    },
                    {
                            .buffer = point_lights_buffer->buffer,
                            .range = max_point_lights * sizeof(PointLight)
                    },
                    {
                            .buffer = spot_lights_buffer->buffer,
                            .range = max_spot_lights * sizeof(SpotLight)
                    }
            };

            VkWriteDescriptorSet write_infos[] = {
                    {
                            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = descriptor_set,
                            .dstBinding = 0,
                            .dstArrayElement = 0,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            .pBufferInfo = &buffer_infos[0],
                    },
                    {
                            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = descriptor_set,
                            .dstBinding = 1,
                            .dstArrayElement = 0,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                            .pBufferInfo = &buffer_infos[1],
                    },
                    {
                            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = descriptor_set,
                            .dstBinding = 2,
                            .dstArrayElement = 0,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .pBufferInfo = &buffer_infos[2],
                    },
                    {
                            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = descriptor_set,
                            .dstBinding = 3,
                            .dstArrayElement = 0,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .pBufferInfo = &buffer_infos[3],
                    }
            };

            vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
                                   write_infos, 0, nullptr);
        }

        // NOTE: Plane mesh initialization
        {
            std::vector<Vertex> vertices = {
                    {{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                    {{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                    {{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
                    {{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
            };

            std::vector<uint32_t> indices = {
                    0, 1, 2, 2, 3, 0
            };

            plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
                    vertices.size() * sizeof(Vertex), vertices.data(),
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

            plane_mesh.index_buffer = new veekay::graphics::Buffer(
                    indices.size() * sizeof(uint32_t), indices.data(),
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

            plane_mesh.indices = uint32_t(indices.size());
        }

        // NOTE: Cube mesh initialization
        {
            std::vector<Vertex> vertices = {
                    {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
                    {{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
                    {{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
                    {{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

                    {{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                    {{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                    {{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                    {{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

                    {{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
                    {{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
                    {{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
                    {{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

                    {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                    {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                    {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                    {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

                    {{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                    {{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                    {{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
                    {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

                    {{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                    {{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                    {{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                    {{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            };

            std::vector<uint32_t> indices = {
                    0, 1, 2, 2, 3, 0,
                    4, 5, 6, 6, 7, 4,
                    8, 9, 10, 10, 11, 8,
                    12, 13, 14, 14, 15, 12,
                    16, 17, 18, 18, 19, 16,
                    20, 21, 22, 22, 23, 20,
            };

            cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
                    vertices.size() * sizeof(Vertex), vertices.data(),
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

            cube_mesh.index_buffer = new veekay::graphics::Buffer(
                    indices.size() * sizeof(uint32_t), indices.data(),
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

            cube_mesh.indices = uint32_t(indices.size());
        }

        // NOTE: Добавляем точечные источники света
        // Точечный источник 1: Парящий над центром
        point_lights.emplace_back(PointLight{
                .position = {0.0f, 2.0f, 0.0f},
                .radius = 8.0f,
                .color = {1.0f, 0.8f, 0.2f}, // Теплый желтый свет
        });

        // Точечный источник 2: Возле красного куба
        point_lights.emplace_back(PointLight{
                .position = {-2.5f, 1.0f, -1.5f},
                .radius = 6.0f,
                .color = {1.0f, 0.3f, 0.3f}, // Красноватый свет
        });

        // Точечный источник 3: Возле зеленого куба
        point_lights.emplace_back(PointLight{
                .position = {2.0f, 1.5f, -0.5f},
                .radius = 7.0f,
                .color = {0.3f, 1.0f, 0.3f}, // Зеленоватый свет
        });

        // Точечный источник 4: Возле синего куба
        point_lights.emplace_back(PointLight{
                .position = {0.5f, 0.8f, 1.5f},
                .radius = 5.0f,
                .color = {0.3f, 0.3f, 1.0f}, // Синеватый свет
        });

        // Точечный источник 5: В углу сцены
        point_lights.emplace_back(PointLight{
                .position = {-4.0f, 1.2f, 3.0f},
                .radius = 10.0f,
                .color = {0.8f, 0.5f, 1.0f}, // Пурпурный свет
        });

        // NOTE: Добавляем прожекторы с разными цветами и направлениями
        // Прожектор 1: Сверху-справа, светит вниз на красный куб
        spot_lights.emplace_back(SpotLight{
                .position = {3.0f, 3.0f, 0.0f},
                .radius = 15.0f,
                .direction = {-0.7f, -0.7f, 0.0f}, // Направлен вниз и немного влево
                .angle = cosf(toRadians(25.0f)), // Узкий луч
                .color = {1.0f, 0.2f, 0.2f}, // Красноватый свет
        });

        // Прожектор 2: Слева-сзади, светит на зеленый куб
        spot_lights.emplace_back(SpotLight{
                .position = {-3.0f, 2.0f, 3.0f},
                .radius = 12.0f,
                .direction = {0.5f, -0.3f, -0.8f}, // Направлен вперед и вправо
                .angle = cosf(toRadians(35.0f)), // Средний луч
                .color = {0.2f, 1.0f, 0.2f}, // Зеленоватый свет
        });

        // Прожектор 3: Спереди-сверху, светит на синий куб
        spot_lights.emplace_back(SpotLight{
                .position = {0.0f, 4.0f, 4.0f},
                .radius = 18.0f,
                .direction = {0.0f, -0.5f, -1.0f}, // Направлен прямо вниз и вперед
                .angle = cosf(toRadians(45.0f)), // Широкий луч
                .color = {0.2f, 0.2f, 1.0f}, // Синеватый свет
        });

        // Прожектор 4: Снизу-спереди, светит вверх на плоскость
        spot_lights.emplace_back(SpotLight{
                .position = {0.0f, -1.0f, 2.0f},
                .radius = 10.0f,
                .direction = {0.0f, 1.0f, -0.5f}, // Направлен вверх и немного назад
                .angle = cosf(toRadians(40.0f)),
                .color = {1.0f, 1.0f, 0.5f}, // Желтоватый свет
        });

        // NOTE: Add models to scene
        models.emplace_back(Model{
                .mesh = plane_mesh,
                .transform = Transform{},
                .albedo_color = veekay::vec3{0.3f, 0.3f, 0.3f}, // Сделаем плоскость светлее для лучшей видимости
                .specular_color = {0.5f, 0.5f, 0.5f},
                .shininess = 32.0f,
        });

        models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{ .position = {-2.0f, -0.5f, -1.5f} },
                .albedo_color = {1.0f, 0.0f, 0.0f},
                .specular_color = {1.0f, 1.0f, 1.0f},
                .shininess = 64.0f
        });

        models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{ .position = {1.5f, -0.5f, -0.5f} },
                .albedo_color = {0.0f, 1.0f, 0.0f},
                .specular_color = {1.0f, 1.0f, 1.0f},
                .shininess = 64.0f
        });

        models.emplace_back(Model{
                .mesh = cube_mesh,
                .transform = Transform{ .position = {0.0f, -0.5f, 1.0f} },
                .albedo_color = {0.0f, 0.0f, 1.0f},
                .specular_color = {1.0f, 1.0f, 1.0f},
                .shininess = 64.0f
        });
    }

    void shutdown() {
        VkDevice& device = veekay::app.vk_device;

        vkDestroySampler(device, missing_texture_sampler, nullptr);
        delete missing_texture;

        delete cube_mesh.index_buffer;
        delete cube_mesh.vertex_buffer;

        delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;

        delete model_uniforms_buffer;
        delete scene_uniforms_buffer;

        delete point_lights_buffer;
        delete spot_lights_buffer;

        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    void update(double time) {
        ImGui::Begin("Controls:");
        ImGui::InputFloat3("Camera Rotation", reinterpret_cast<float*>(&(camera.rotation)));
        ImGui::InputFloat3("Camera Position", reinterpret_cast<float*>(&(camera.position)));

        // Добавляем UI для управления точечными источниками
        ImGui::Separator();
        ImGui::Text("Point Lights:");
        for (size_t i = 0; i < point_lights.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            ImGui::Text("Point Light %d", static_cast<int>(i));
            ImGui::InputFloat3("Position", reinterpret_cast<float*>(&point_lights[i].position));
            ImGui::InputFloat("Radius", &point_lights[i].radius);
            ImGui::ColorEdit3("Color", reinterpret_cast<float*>(&point_lights[i].color));
            ImGui::PopID();
        }

        // Добавляем UI для управления прожекторами
        ImGui::Separator();
        ImGui::Text("Spot Lights:");
        for (size_t i = 0; i < spot_lights.size(); ++i) {
            ImGui::PushID(static_cast<int>(i + point_lights.size()));
            ImGui::Text("Spot Light %d", static_cast<int>(i));
            ImGui::InputFloat3("Position", reinterpret_cast<float*>(&spot_lights[i].position));
            ImGui::InputFloat("Radius", &spot_lights[i].radius);
            ImGui::InputFloat3("Direction", reinterpret_cast<float*>(&spot_lights[i].direction));
            float angle_degrees = acosf(spot_lights[i].angle) * 180.0f / float(M_PI);
            if (ImGui::InputFloat("Angle (degrees)", &angle_degrees)) {
                spot_lights[i].angle = cosf(toRadians(angle_degrees));
            }
            ImGui::ColorEdit3("Color", reinterpret_cast<float*>(&spot_lights[i].color));
            ImGui::PopID();
        }

        ImGui::End();

        if (!ImGui::IsWindowHovered()) {
            using namespace veekay::input;

            if (mouse::isButtonDown(mouse::Button::left)) {
                auto move_delta = mouse::cursorDelta();

                // Обновляем вращение камеры с помощью мыши
                camera.rotation.y += move_delta.x * 0.1f;
                camera.rotation.x += move_delta.y * 0.1f;

                auto view = camera.view();

                // Вычисляем правильные направления из матрицы вида
                veekay::vec3 right = {view.elements[0][0], view.elements[1][0], view.elements[2][0]};
                veekay::vec3 up = {view.elements[0][1], view.elements[1][1], view.elements[2][1]};
                veekay::vec3 front = {view.elements[0][2], view.elements[1][2], view.elements[2][2]};

                if (keyboard::isKeyDown(keyboard::Key::w))
                    camera.position += front * 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::s))
                    camera.position -= front * 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::d))
                    camera.position += right * 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::a))
                    camera.position -= right * 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::q))
                    camera.position += up * 0.1f;

                if (keyboard::isKeyDown(keyboard::Key::z))
                    camera.position -= up * 0.1f;
            }
        }

        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
        SceneUniforms scene_uniforms{
                .view_projection = camera.view_projection(aspect_ratio),
                .view_position = camera.position,
                .ambient_light_intensity = {0.05f, 0.05f, 0.05f}, // Еще меньше окружающего света
                .sun_light_direction = {0.0f, 1.0f, -1.0f},
                .sun_light_color = {0.2f, 0.2f, 0.2f}, // Уменьшаем солнечный свет чтобы лучше видеть точечные источники
                .point_lights_count = static_cast<uint32_t>(point_lights.size()),
                .spot_lights_count = static_cast<uint32_t>(spot_lights.size())
        };

        std::vector<ModelUniforms> model_uniforms(models.size());
        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model& model = models[i];
            ModelUniforms& uniforms = model_uniforms[i];

            uniforms.model = model.transform.matrix();
            uniforms.shininess = model.shininess;
            uniforms.specular_color = model.specular_color;
            uniforms.albedo_color = model.albedo_color;
        }

        *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;
        std::copy(model_uniforms.begin(),
                  model_uniforms.end(),
                  static_cast<ModelUniforms*>(model_uniforms_buffer->mapped_region));

        // Копируем данные точечных источников в буфер
        if (!point_lights.empty()) {
            std::copy(point_lights.begin(),
                      point_lights.end(),
                      static_cast<PointLight*>(point_lights_buffer->mapped_region));
        }

        // Копируем данные прожекторов в буфер
        if (!spot_lights.empty()) {
            std::copy(spot_lights.begin(),
                      spot_lights.end(),
                      static_cast<SpotLight*>(spot_lights_buffer->mapped_region));
        }
    }

    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0);

        { // NOTE: Start recording rendering commands
            VkCommandBufferBeginInfo info{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };

            vkBeginCommandBuffer(cmd, &info);
        }

        { // NOTE: Use current swapchain framebuffer and clear it
            VkClearValue clear_color{.color = {{0.05f, 0.05f, 0.05f, 1.0f}}}; // Темнее фон для лучшего контраста
            VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

            VkClearValue clear_values[] = {clear_color, clear_depth};

            VkRenderPassBeginInfo info{
                    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                    .renderPass = veekay::app.vk_render_pass,
                    .framebuffer = framebuffer,
                    .renderArea = {
                            .extent = {
                                    veekay::app.window_width,
                                    veekay::app.window_height
                            },
                    },
                    .clearValueCount = 2,
                    .pClearValues = clear_values,
            };

            vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkDeviceSize zero_offset = 0;

        VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
        VkBuffer current_index_buffer = VK_NULL_HANDLE;

        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model& model = models[i];
            const Mesh& mesh = model.mesh;

            if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
                current_vertex_buffer = mesh.vertex_buffer->buffer;
                vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
            }

            if (current_index_buffer != mesh.index_buffer->buffer) {
                current_index_buffer = mesh.index_buffer->buffer;
                vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
            }

            uint32_t offset = i * sizeof(ModelUniforms);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &descriptor_set, 1, &offset);

            vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

} // namespace

int main() {
    return veekay::run({
                               .init = initialize,
                               .shutdown = shutdown,
                               .update = update,
                               .render = render,
                       });
}