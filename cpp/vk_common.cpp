#include "vk_common.h"
#include <fstream>
#include <vector>
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <cstring>

std::vector<uint32_t> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open file: " + filename);
    size_t fileSize = (size_t)file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();
    return buffer;
}

VkShaderModule createShaderModule(VkDevice device, const std::vector<uint32_t>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size() * sizeof(uint32_t);
    createInfo.pCode = code.data();
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        throw std::runtime_error("failed to create shader module!");
    return shaderModule;
}

void initVulkanContext(VulkanContext& ctx, GLFWwindow* window, const char* title) {
    VkApplicationInfo appInfo{};
    appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName   = title;
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName        = "No Engine";
    appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_0;

    createVulkanInstance(appInfo, ctx.instance);
    createVulkanSurface(ctx.instance, window, ctx.surface);

    QueueFamilyIndices indices;
    pickPhysicalDevice(ctx.instance, ctx.surface, ctx.physicalDevice, indices);
    createLogicalDevice(ctx.physicalDevice, indices, ctx.device, ctx.graphicsQueue, ctx.presentQueue);
    ctx.queueIndices = indices;

    VkSurfaceFormatKHR surfFormat;
    uint32_t imageCount;
    createSwapchain(ctx.device, ctx.physicalDevice, ctx.surface, indices,
                    ctx.extent, ctx.swapchain, surfFormat, imageCount);
    ctx.surfFormat = surfFormat;
    createImageViews(ctx.device, ctx.swapchain, surfFormat, ctx.imageViews);
    createRenderPass(ctx.device, surfFormat, ctx.renderPass);
    createFramebuffers(ctx.device, ctx.renderPass, ctx.imageViews, ctx.extent, ctx.framebuffers);
    createCommandPool(ctx.device, indices.graphicsFamily, ctx.commandPool);
    allocateCommandBuffers(ctx.device, ctx.commandPool, ctx.framebuffers.size(), ctx.commandBuffers);

    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    if (vkCreateSemaphore(ctx.device, &sci, nullptr, &ctx.imageAvailableSemaphore) != VK_SUCCESS ||
        vkCreateSemaphore(ctx.device, &sci, nullptr, &ctx.renderFinishedSemaphore) != VK_SUCCESS)
        throw std::runtime_error("failed to create semaphores");

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(ctx.device, &fci, nullptr, &ctx.inFlightFence) != VK_SUCCESS)
        throw std::runtime_error("failed to create fence");

    ctx.initialized = true;
}

void destroyVulkanContext(VulkanContext& ctx) {
    if (ctx.device == VK_NULL_HANDLE) return;

    vkDestroyFence(ctx.device, ctx.inFlightFence, nullptr);
    vkDestroySemaphore(ctx.device, ctx.renderFinishedSemaphore, nullptr);
    vkDestroySemaphore(ctx.device, ctx.imageAvailableSemaphore, nullptr);
    vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr);
    for (auto fb : ctx.framebuffers) vkDestroyFramebuffer(ctx.device, fb, nullptr);
    ctx.framebuffers.clear();
    vkDestroyRenderPass(ctx.device, ctx.renderPass, nullptr);
    for (auto iv : ctx.imageViews) vkDestroyImageView(ctx.device, iv, nullptr);
    ctx.imageViews.clear();
    vkDestroySwapchainKHR(ctx.device, ctx.swapchain, nullptr);
    vkDestroyDevice(ctx.device, nullptr);
    vkDestroySurfaceKHR(ctx.instance, ctx.surface, nullptr);
    vkDestroyInstance(ctx.instance, nullptr);
    ctx.initialized = false;
}
