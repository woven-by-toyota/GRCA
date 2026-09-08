
// --- Common pipeline creation helpers ---
#include <vector>
#include <string>
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <set>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "../include/grca_common.h" // for GrcaHit, Tri3, etc.
#include <GLFW/glfw3.h>

// Vertex struct for point cloud (xyz + dist)
struct Vertex {
    float x, y, z, dist;
};

// FPS-style flythrough camera: eye position + yaw/pitch orientation.
// RMB hold = look around, WASD/QE = move, scroll = speed adjust, F = fit to hits.
struct OrbitCamera {
    float eye[3]   = {0, 0, 10};
    float yaw      = 0.0f;    // radians, around world Y
    float pitch    = 0.0f;    // radians, around camera X, clamped ±1.5
    float distance = 10.0f;   // fly speed scale (also used by fitCameraToHits)
    float lastX    = 0.0f, lastY = 0.0f;
    void forward(float& fx, float& fy, float& fz) const {
        fx = -sinf(yaw) * cosf(pitch);
        fy =  sinf(pitch);
        fz = -cosf(yaw) * cosf(pitch);
    }
    void right(float& rx, float& ry, float& rz) const {
        rx = cosf(yaw); ry = 0.0f; rz = -sinf(yaw);
    }
};

// Queue family indices for Vulkan
struct QueueFamilyIndices {
    uint32_t graphicsFamily = 0;
    uint32_t presentFamily = 0;
    bool graphicsFound = false;
    bool presentFound = false;
};

// VulkanContext struct (minimal, for context)
struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue presentQueue = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkExtent2D extent = {0,0};
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    VkPipeline meshWirePipeline = VK_NULL_HANDLE;
    VkPipeline meshSolidPipeline = VK_NULL_HANDLE;
    VkPipeline conePipeline = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::vector<VkImageView> imageViews;
    std::vector<VkFramebuffer> framebuffers;
    std::vector<VkCommandBuffer> commandBuffers;
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
    void* mappedVertexData = nullptr;
    VkBuffer meshWireBuffer = VK_NULL_HANDLE;
    VkDeviceMemory meshWireMemory = VK_NULL_HANDLE;
    void* mappedMeshWireData = nullptr;
    uint32_t meshWireVertexCount = 0;
    uint32_t staticWireVertexCount = 0;
    std::vector<float> staticWireData;
    VkBuffer meshSolidBuffer = VK_NULL_HANDLE;
    VkDeviceMemory meshSolidMemory = VK_NULL_HANDLE;
    void* mappedMeshSolidData = nullptr;
    VkDeviceSize meshSolidCapacity = 0;
    uint32_t meshSolidVertexCount = 0;
    uint32_t staticSolidVertexCount = 0;
    std::vector<float> staticSolidData;
    VkImage depthImage = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory = VK_NULL_HANDLE;
    VkImageView depthImageView = VK_NULL_HANDLE;
    VkBuffer aabbWireBuffer = VK_NULL_HANDLE;
    VkDeviceMemory aabbWireMemory = VK_NULL_HANDLE;
    void* mappedAabbWireData = nullptr;
    VkDeviceSize aabbWireCapacity = 0;
    uint32_t aabbWireVertexCount = 0;
    VkBuffer lidarSolidBuffer = VK_NULL_HANDLE;
    VkDeviceMemory lidarSolidMemory = VK_NULL_HANDLE;
    void* mappedLidarSolidData = nullptr;
    VkDeviceSize lidarSolidCapacity = 0;
    uint32_t lidarSolidVertexCount = 0;
    VkBuffer lidarConeBuffer = VK_NULL_HANDLE;
    VkDeviceMemory lidarConeMemory = VK_NULL_HANDLE;
    void* mappedLidarConeData = nullptr;
    VkDeviceSize lidarConeCapacity = 0;
    uint32_t lidarConeVertexCount = 0;
    VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
    VkSemaphore renderFinishedSemaphore = VK_NULL_HANDLE;
    VkFence inFlightFence = VK_NULL_HANDLE;
    bool initialized = false;
    VkDeviceSize vertexBufferCapacity = 0;
    VkDeviceSize meshWireCapacity = 0;
    VkSurfaceFormatKHR surfFormat = {};
    QueueFamilyIndices queueIndices = {};
};

// Helper function declarations
std::vector<uint32_t> readFile(const std::string& filename);
VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& code);
void initVulkanContext(VulkanContext& ctx, GLFWwindow* window, const char* title);
void destroyVulkanContext(VulkanContext& ctx);

inline QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
    QueueFamilyIndices indices;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());
    for (uint32_t i = 0; i < count; i++) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
            indices.graphicsFound  = true;
        }
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if (presentSupport) { indices.presentFamily = i; indices.presentFound = true; }
        if (indices.graphicsFound && indices.presentFound) break;
    }
    return indices;
}

// Helper to create a pipeline layout with a single push constant range
inline VkPipelineLayout createPipelineLayout(VkDevice device, uint32_t pushConstantSize) {
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = pushConstantSize;
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    VkPipelineLayout layout;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &layout) != VK_SUCCESS)
        throw std::runtime_error("failed to create pipeline layout!");
    return layout;
}

// Helper to create a graphics pipeline for point cloud rendering
inline VkPipeline createPointCloudPipeline(
    VkDevice device,
    VkExtent2D /*extent*/,
    VkPipelineLayout pipelineLayout,
    VkRenderPass renderPass,
    const std::vector<uint32_t>& vertCode,
    const std::vector<uint32_t>& fragCode
) {
    VkShaderModule vertModule = createShaderModule(device, vertCode);
    VkShaderModule fragModule = createShaderModule(device, fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";
    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertStage, fragStage};

    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(Vertex);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attrDescs[2] = {};
    attrDescs[0].binding = 0; attrDescs[0].location = 0; attrDescs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrDescs[0].offset = 0;
    attrDescs[1].binding = 0; attrDescs[1].location = 1; attrDescs[1].format = VK_FORMAT_R32_SFLOAT;        attrDescs[1].offset = 12;
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = 2;
    vertexInputInfo.pVertexAttributeDescriptions = attrDescs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stageCount = 2;
    pipelineCreateInfo.pStages = shaderStages;
    pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pRasterizationState = &rasterizer;
    pipelineCreateInfo.pMultisampleState = &multisampling;
    pipelineCreateInfo.pColorBlendState = &colorBlending;
    pipelineCreateInfo.pDepthStencilState = &depthStencil;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.layout = pipelineLayout;
    pipelineCreateInfo.renderPass = renderPass;
    pipelineCreateInfo.subpass = 0;

    VkPipeline pipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics pipeline!");

    vkDestroyShaderModule(device, vertModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);
    return pipeline;
}

// Helper to create a mesh wireframe pipeline
inline VkPipeline createMeshWirePipeline(
    VkDevice device,
    VkExtent2D /*extent*/,
    VkPipelineLayout pipelineLayout,
    VkRenderPass renderPass,
    const std::vector<uint32_t>& vertCode,
    const std::vector<uint32_t>& fragCode
) {
    VkShaderModule vertModule = createShaderModule(device, vertCode);
    VkShaderModule fragModule = createShaderModule(device, fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";
    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertStage, fragStage};

    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = 6 * sizeof(float); // xyz + rgb
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attrDescs[2] = {};
    attrDescs[0].binding = 0; attrDescs[0].location = 0;
    attrDescs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrDescs[0].offset = 0;
    attrDescs[1].binding = 0; attrDescs[1].location = 1;
    attrDescs[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrDescs[1].offset = 3 * sizeof(float);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = 2;
    vertexInputInfo.pVertexAttributeDescriptions = attrDescs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                       VK_DYNAMIC_STATE_LINE_WIDTH };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 3;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = -2.0f;
    rasterizer.depthBiasSlopeFactor = -2.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stageCount = 2;
    pipelineCreateInfo.pStages = shaderStages;
    pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pRasterizationState = &rasterizer;
    pipelineCreateInfo.pMultisampleState = &multisampling;
    pipelineCreateInfo.pColorBlendState = &colorBlending;
    pipelineCreateInfo.pDepthStencilState = &depthStencil;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.layout = pipelineLayout;
    pipelineCreateInfo.renderPass = renderPass;
    pipelineCreateInfo.subpass = 0;

    VkPipeline pipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create mesh wire pipeline!");

    vkDestroyShaderModule(device, vertModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);
    return pipeline;
}

// Helper to create a mesh solid (filled triangles) pipeline
inline VkPipeline createMeshSolidPipeline(
    VkDevice device,
    VkExtent2D /*extent*/,
    VkPipelineLayout pipelineLayout,
    VkRenderPass renderPass,
    const std::vector<uint32_t>& vertCode,
    const std::vector<uint32_t>& fragCode
) {
    VkShaderModule vertModule = createShaderModule(device, vertCode);
    VkShaderModule fragModule = createShaderModule(device, fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName = "main";
    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName = "main";
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertStage, fragStage};

    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = 6 * sizeof(float); // xyz + rgb
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attrDescs[2] = {};
    attrDescs[0].binding = 0; attrDescs[0].location = 0;
    attrDescs[0].format = VK_FORMAT_R32G32B32_SFLOAT; attrDescs[0].offset = 0;
    attrDescs[1].binding = 0; attrDescs[1].location = 1;
    attrDescs[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrDescs[1].offset = 3 * sizeof(float);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = 2;
    vertexInputInfo.pVertexAttributeDescriptions = attrDescs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stageCount = 2;
    pipelineCreateInfo.pStages = shaderStages;
    pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pRasterizationState = &rasterizer;
    pipelineCreateInfo.pMultisampleState = &multisampling;
    pipelineCreateInfo.pColorBlendState = &colorBlending;
    pipelineCreateInfo.pDepthStencilState = &depthStencil;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.layout = pipelineLayout;
    pipelineCreateInfo.renderPass = renderPass;
    pipelineCreateInfo.subpass = 0;

    VkPipeline pipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create mesh solid pipeline!");

    vkDestroyShaderModule(device, vertModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);
    return pipeline;
}


// Transparent cone pipeline: filled triangles, alpha blending, depth test but no depth write.
inline VkPipeline createConePipeline(
    VkDevice device,
    VkExtent2D /*extent*/,
    VkPipelineLayout pipelineLayout,
    VkRenderPass renderPass,
    const std::vector<uint32_t>& vertCode,
    const std::vector<uint32_t>& fragCode
) {
    VkShaderModule vertModule = createShaderModule(device, vertCode);
    VkShaderModule fragModule = createShaderModule(device, fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule; vertStage.pName = "main";
    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule; fragStage.pName = "main";
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertStage, fragStage};

    VkVertexInputBindingDescription bindingDesc{};
    bindingDesc.binding = 0;
    bindingDesc.stride = 6 * sizeof(float);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    VkVertexInputAttributeDescription attrDescs[2] = {};
    attrDescs[0].binding=0; attrDescs[0].location=0;
    attrDescs[0].format=VK_FORMAT_R32G32B32_SFLOAT; attrDescs[0].offset=0;
    attrDescs[1].binding=0; attrDescs[1].location=1;
    attrDescs[1].format=VK_FORMAT_R32G32B32_SFLOAT; attrDescs[1].offset=3*sizeof(float);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = 2;
    vertexInputInfo.pVertexAttributeDescriptions = attrDescs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1; viewportState.scissorCount = 1;

    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Alpha blending: out = src.a * src.rgb + (1 - src.a) * dst.rgb
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_FALSE; // don't write depth for transparent surfaces
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stageCount = 2;
    pipelineCreateInfo.pStages = shaderStages;
    pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pRasterizationState = &rasterizer;
    pipelineCreateInfo.pMultisampleState = &multisampling;
    pipelineCreateInfo.pColorBlendState = &colorBlending;
    pipelineCreateInfo.pDepthStencilState = &depthStencil;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.layout = pipelineLayout;
    pipelineCreateInfo.renderPass = renderPass;
    pipelineCreateInfo.subpass = 0;

    VkPipeline pipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create cone pipeline!");

    vkDestroyShaderModule(device, vertModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);
    return pipeline;
}

// fitCameraToHits: Unity-style F — frame valid hits without changing orientation.
// Computes the bounding sphere of valid hits, then positions the eye behind the
// centre along the current forward direction so the sphere fits the 45° FOV.
inline void fitCameraToHits(OrbitCamera& camera, const std::vector<GrcaHit>& hits,
                             float range_max_hint = 1e9f) {
    if (hits.empty()) return;

    bool found = false;
    float minx = 0, maxx = 0, miny = 0, maxy = 0, minz = 0, maxz = 0;
    for (const auto& h : hits) {
        if (h.dist <= 0.0f || h.dist >= range_max_hint * 0.9999f) continue;
        if (!found) {
            minx = maxx = h.hx;
            miny = maxy = h.hy;
            minz = maxz = h.hz;
            found = true;
        } else {
            if (h.hx < minx) minx = h.hx;
            if (h.hx > maxx) maxx = h.hx;
            if (h.hy < miny) miny = h.hy;
            if (h.hy > maxy) maxy = h.hy;
            if (h.hz < minz) minz = h.hz;
            if (h.hz > maxz) maxz = h.hz;
        }
    }
    if (!found) return;

    float cx = 0.5f*(minx+maxx), cy = 0.5f*(miny+maxy), cz = 0.5f*(minz+maxz);
    // Bounding sphere radius
    float dx = (maxx-minx)*0.5f, dy = (maxy-miny)*0.5f, dz = (maxz-minz)*0.5f;
    float radius = sqrtf(dx*dx + dy*dy + dz*dz) + 0.5f; // small padding
    // Distance to fit sphere in 45-deg vertical FOV: radius / tan(22.5 deg)
    float fitDist = radius / 0.41421356f; // tan(π/8)
    camera.distance = fitDist;
    // Keep current yaw/pitch; move eye back along the forward direction from centre
    float fx, fy, fz; camera.forward(fx, fy, fz);
    camera.eye[0] = cx - fx * fitDist;
    camera.eye[1] = cy - fy * fitDist;
    camera.eye[2] = cz - fz * fitDist;
}
inline void initWindow(GLFWwindow*& window, const char* title = "Vulkan Viewer",
                       int width = 1280, int height = 720) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
}

inline void cleanup(GLFWwindow* window) {
    glfwDestroyWindow(window);
    glfwTerminate();
}

// -----------------------------------------------------------------------
// Vulkan initialisation helpers
// -----------------------------------------------------------------------
inline void createVulkanInstance(VkApplicationInfo& appInfo, VkInstance& instance) {
    uint32_t     glfwExtCount = 0;
    const char** glfwExts     = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    VkInstanceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo        = &appInfo;
    ci.enabledExtensionCount   = glfwExtCount;
    ci.ppEnabledExtensionNames = glfwExts;
    ci.enabledLayerCount       = 0;
    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("failed to create Vulkan instance");
}

inline void createVulkanSurface(VkInstance instance, GLFWwindow* window, VkSurfaceKHR& surface) {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("failed to create window surface");
}

inline void pickPhysicalDevice(VkInstance instance, VkSurfaceKHR surface,
                                VkPhysicalDevice& physicalDevice, QueueFamilyIndices& indices) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) throw std::runtime_error("failed to find GPUs with Vulkan support");
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    for (const auto& dev : devices) {
        indices = findQueueFamilies(dev, surface);
        if (indices.graphicsFound && indices.presentFound) { physicalDevice = dev; break; }
    }
    if (physicalDevice == VK_NULL_HANDLE) throw std::runtime_error("failed to find suitable GPU");
}

inline void createLogicalDevice(VkPhysicalDevice physicalDevice, QueueFamilyIndices indices,
                                 VkDevice& device, VkQueue& graphicsQueue, VkQueue& presentQueue) {
    std::set<uint32_t> uniqueQueues = {indices.graphicsFamily, indices.presentFamily};
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    float queuePriority = 1.0f;
    for (uint32_t qf : uniqueQueues) {
        VkDeviceQueueCreateInfo qci{};
        qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = qf;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(qci);
    }
    VkPhysicalDeviceFeatures features{};
    features.largePoints = VK_TRUE;  // enables gl_PointSize > 1 in vertex shaders
    features.wideLines   = VK_TRUE;  // enables lineWidth > 1 via vkCmdSetLineWidth
    const char* ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    VkDeviceCreateInfo ci{};
    ci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.queueCreateInfoCount    = (uint32_t)queueCreateInfos.size();
    ci.pQueueCreateInfos       = queueCreateInfos.data();
    ci.enabledExtensionCount   = 1;
    ci.ppEnabledExtensionNames = &ext;
    ci.pEnabledFeatures        = &features;
    if (vkCreateDevice(physicalDevice, &ci, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("failed to create logical device");
    vkGetDeviceQueue(device, indices.graphicsFamily, 0, &graphicsQueue);
    vkGetDeviceQueue(device, indices.presentFamily,  0, &presentQueue);
}

inline void createSwapchain(VkDevice device, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
                             QueueFamilyIndices indices, VkExtent2D& extent,
                             VkSwapchainKHR& swapchain, VkSurfaceFormatKHR& surfFormat,
                             uint32_t& imageCount) {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &caps);

    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &fmtCount, formats.data());
    surfFormat = formats[0];
    for (const auto& f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            { surfFormat = f; break; }

    extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) { extent.width = 1280; extent.height = 720; }

    imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) imageCount = caps.maxImageCount;

    uint32_t qfis[] = {indices.graphicsFamily, indices.presentFamily};
    VkSwapchainCreateInfoKHR ci{};
    ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface          = surface;
    ci.minImageCount    = imageCount;
    ci.imageFormat      = surfFormat.format;
    ci.imageColorSpace  = surfFormat.colorSpace;
    ci.imageExtent      = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (indices.graphicsFamily != indices.presentFamily) {
        ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices   = qfis;
    } else {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    ci.preTransform   = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode    = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped        = VK_TRUE;
    if (vkCreateSwapchainKHR(device, &ci, nullptr, &swapchain) != VK_SUCCESS)
        throw std::runtime_error("failed to create swapchain");
}

inline void createImageViews(VkDevice device, VkSwapchainKHR swapchain,
                              VkSurfaceFormatKHR surfFormat, std::vector<VkImageView>& imageViews) {
    uint32_t n = 0;
    vkGetSwapchainImagesKHR(device, swapchain, &n, nullptr);
    std::vector<VkImage> images(n);
    vkGetSwapchainImagesKHR(device, swapchain, &n, images.data());
    imageViews.clear();
    for (VkImage img : images) {
        VkImageViewCreateInfo vi{};
        vi.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image                           = img;
        vi.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        vi.format                          = surfFormat.format;
        vi.components                      = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                              VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        vi.subresourceRange                = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView view;
        if (vkCreateImageView(device, &vi, nullptr, &view) != VK_SUCCESS)
            throw std::runtime_error("failed to create image view");
        imageViews.push_back(view);
    }
}

inline void createRenderPass(VkDevice device, VkSurfaceFormatKHR surfFormat, VkRenderPass& renderPass) {
    VkAttachmentDescription attachments[2] = {};
    // Color attachment
    attachments[0].format         = surfFormat.format;
    attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    // Depth attachment
    attachments[1].format         = VK_FORMAT_D32_SFLOAT;
    attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sp{};
    sp.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp.colorAttachmentCount    = 1;
    sp.pColorAttachments       = &colorRef;
    sp.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpi{};
    rpi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpi.attachmentCount = 2;
    rpi.pAttachments    = attachments;
    rpi.subpassCount    = 1;
    rpi.pSubpasses      = &sp;
    rpi.dependencyCount = 1;
    rpi.pDependencies   = &dep;
    if (vkCreateRenderPass(device, &rpi, nullptr, &renderPass) != VK_SUCCESS)
        throw std::runtime_error("failed to create render pass");
}

inline void createDepthResources(VkDevice device, VkPhysicalDevice physicalDevice,
                                 VkExtent2D extent, VkImage& depthImage,
                                 VkDeviceMemory& depthMemory, VkImageView& depthImageView) {
    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_D32_SFLOAT;
    ici.extent        = {extent.width, extent.height, 1};
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (vkCreateImage(device, &ici, nullptr, &depthImage) != VK_SUCCESS)
        throw std::runtime_error("failed to create depth image");

    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(device, depthImage, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mp);
    uint32_t memIdx = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((mr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            { memIdx = i; break; }
    }
    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = mr.size;
    mai.memoryTypeIndex = memIdx;
    if (vkAllocateMemory(device, &mai, nullptr, &depthMemory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate depth image memory");
    vkBindImageMemory(device, depthImage, depthMemory, 0);

    VkImageViewCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image                           = depthImage;
    vi.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    vi.format                          = VK_FORMAT_D32_SFLOAT;
    vi.subresourceRange                = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(device, &vi, nullptr, &depthImageView) != VK_SUCCESS)
        throw std::runtime_error("failed to create depth image view");
}

inline void destroyDepthResources(VkDevice device, VkImage& depthImage,
                                  VkDeviceMemory& depthMemory, VkImageView& depthImageView) {
    if (depthImageView != VK_NULL_HANDLE) { vkDestroyImageView(device, depthImageView, nullptr); depthImageView = VK_NULL_HANDLE; }
    if (depthImage != VK_NULL_HANDLE) { vkDestroyImage(device, depthImage, nullptr); depthImage = VK_NULL_HANDLE; }
    if (depthMemory != VK_NULL_HANDLE) { vkFreeMemory(device, depthMemory, nullptr); depthMemory = VK_NULL_HANDLE; }
}

inline void createFramebuffers(VkDevice device, VkRenderPass renderPass,
                                std::vector<VkImageView>& imageViews, VkExtent2D extent,
                                std::vector<VkFramebuffer>& framebuffers,
                                VkImageView depthImageView = VK_NULL_HANDLE) {
    framebuffers.clear();
    for (VkImageView view : imageViews) {
        VkImageView att[2] = {view, depthImageView};
        VkFramebufferCreateInfo fi{};
        fi.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass      = renderPass;
        fi.attachmentCount = (depthImageView != VK_NULL_HANDLE) ? 2u : 1u;
        fi.pAttachments    = att;
        fi.width           = extent.width;
        fi.height          = extent.height;
        fi.layers          = 1;
        VkFramebuffer fb;
        if (vkCreateFramebuffer(device, &fi, nullptr, &fb) != VK_SUCCESS)
            throw std::runtime_error("failed to create framebuffer");
        framebuffers.push_back(fb);
    }
}

inline void createCommandPool(VkDevice device, uint32_t graphicsFamily, VkCommandPool& commandPool) {
    VkCommandPoolCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = graphicsFamily;
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device, &ci, nullptr, &commandPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create command pool");
}

inline void allocateCommandBuffers(VkDevice device, VkCommandPool commandPool, size_t count,
                                   std::vector<VkCommandBuffer>& commandBuffers) {
    commandBuffers.resize(count);
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = (uint32_t)count;
    if (vkAllocateCommandBuffers(device, &ai, commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate command buffers");
}

// Recreate the swapchain, image views, framebuffers, and command buffers after a resize.
// Pipelines do not need to be recreated because they use dynamic viewport/scissor state.
// Safe to call from the GLFW framebuffer-size callback (main thread, inside glfwPollEvents).
inline void rebuildSwapchain(VulkanContext& ctx) {
    vkDeviceWaitIdle(ctx.device);

    vkFreeCommandBuffers(ctx.device, ctx.commandPool,
                         (uint32_t)ctx.commandBuffers.size(), ctx.commandBuffers.data());
    for (auto fb : ctx.framebuffers) vkDestroyFramebuffer(ctx.device, fb, nullptr);
    ctx.framebuffers.clear();
    for (auto iv : ctx.imageViews) vkDestroyImageView(ctx.device, iv, nullptr);
    ctx.imageViews.clear();
    destroyDepthResources(ctx.device, ctx.depthImage, ctx.depthMemory, ctx.depthImageView);
    vkDestroySwapchainKHR(ctx.device, ctx.swapchain, nullptr);

    VkSurfaceFormatKHR dummy;
    uint32_t imageCount;
    createSwapchain(ctx.device, ctx.physicalDevice, ctx.surface, ctx.queueIndices,
                    ctx.extent, ctx.swapchain, dummy, imageCount);
    createImageViews(ctx.device, ctx.swapchain, ctx.surfFormat, ctx.imageViews);
    createDepthResources(ctx.device, ctx.physicalDevice, ctx.extent,
                         ctx.depthImage, ctx.depthMemory, ctx.depthImageView);
    createFramebuffers(ctx.device, ctx.renderPass, ctx.imageViews, ctx.extent,
                       ctx.framebuffers, ctx.depthImageView);
    allocateCommandBuffers(ctx.device, ctx.commandPool, ctx.framebuffers.size(), ctx.commandBuffers);
}

// Creates a CPU-visible/coherent vertex buffer of the given byte size, maps it,
// and returns the persistent mapped pointer in mappedData.
inline void allocateMappedBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
                                  VkDeviceSize size, VkBuffer& buffer,
                                  VkDeviceMemory& memory, void*& mappedData) {
    VkBufferCreateInfo bi{};
    bi.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size        = size;
    bi.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bi, nullptr, &buffer) != VK_SUCCESS)
        throw std::runtime_error("failed to create buffer");

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(device, buffer, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mp);
    uint32_t memIdx = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((mr.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            { memIdx = i; break; }
    }
    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = mr.size;
    mai.memoryTypeIndex = memIdx;
    if (vkAllocateMemory(device, &mai, nullptr, &memory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate buffer memory");
    vkBindBufferMemory(device, buffer, memory, 0);
    vkMapMemory(device, memory, 0, size, 0, &mappedData);
}

// Creates and binds CPU-visible/coherent memory for a vertex buffer (unmapped).
// Vertex layout assumed: { float x, y, z, w } = 4 floats per vertex.
inline void createVertexBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
                                size_t vertexCount, VkBuffer& vertexBuffer,
                                VkDeviceMemory& vertexMemory) {
    void* unused = nullptr;
    allocateMappedBuffer(device, physicalDevice, vertexCount * sizeof(float) * 4,
                         vertexBuffer, vertexMemory, unused);
    vkUnmapMemory(device, vertexMemory);
}

// -----------------------------------------------------------------------
// Matrix helpers (column-major, Vulkan convention)
// -----------------------------------------------------------------------
inline void matMul(const float A[16], const float B[16], float C[16]) {
    for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++) {
            C[col*4 + row] = 0.0f;
            for (int k = 0; k < 4; k++)
                C[col*4 + row] += A[k*4 + row] * B[col*4 + k];
        }
}

inline void makePerspective(float fovy, float aspect, float znear, float zfar, float m[16]) {
    memset(m, 0, 16*sizeof(float));
    float f = 1.0f / tanf(fovy * 0.5f);
    m[0]  =  f / aspect;
    m[5]  = -f;                                  // flip Y for Vulkan NDC
    m[10] =  zfar / (znear - zfar);
    m[11] = -1.0f;
    m[14] = (znear * zfar) / (znear - zfar);
}

inline void makeLookAt(const float eye[3], const float center[3], const float up[3], float m[16]) {
    float f[3] = { center[0]-eye[0], center[1]-eye[1], center[2]-eye[2] };
    float flen = sqrtf(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
    for (int i = 0; i < 3; i++) f[i] /= flen;

    float s[3] = { f[1]*up[2]-f[2]*up[1], f[2]*up[0]-f[0]*up[2], f[0]*up[1]-f[1]*up[0] };
    float slen = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
    for (int i = 0; i < 3; i++) s[i] /= slen;

    float u[3] = { s[1]*f[2]-s[2]*f[1], s[2]*f[0]-s[0]*f[2], s[0]*f[1]-s[1]*f[0] };

    // Column-major: m[col*4 + row]. Rows are s, u, -f.
    m[0]=s[0]; m[1]=u[0]; m[2]=-f[0]; m[3]=0.0f;
    m[4]=s[1]; m[5]=u[1]; m[6]=-f[1]; m[7]=0.0f;
    m[8]=s[2]; m[9]=u[2]; m[10]=-f[2]; m[11]=0.0f;
    m[12]=-(s[0]*eye[0]+s[1]*eye[1]+s[2]*eye[2]);
    m[13]=-(u[0]*eye[0]+u[1]*eye[1]+u[2]*eye[2]);
    m[14]= (f[0]*eye[0]+f[1]*eye[1]+f[2]*eye[2]);
    m[15]=1.0f;
}
