#include <fstream>
#include <sstream>
#include <string>
#include "../include/grca_common.h"
#include "vk_common.h"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstring>
#include <thread>


int main(int argc, char** argv) {
        // Print thread info
        unsigned int nThreads = std::thread::hardware_concurrency();
        std::cout << "[csv_viewer] Hardware concurrency (threads): " << nThreads << std::endl;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <csv_file>\n";
        return 1;
    }
    std::string csv_path = argv[1];

    std::vector<GrcaHit> hits;
    {
        std::ifstream infile(csv_path);
        if (!infile) { std::cerr << "Failed to open CSV: " << csv_path << "\n"; return 1; }
        std::string line;
        std::getline(infile, line);
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            std::string val;
            GrcaHit hit;
            std::getline(iss, val, ','); hit.dx   = std::stof(val);
            std::getline(iss, val, ','); hit.dy   = std::stof(val);
            std::getline(iss, val, ','); hit.dz   = std::stof(val);
            std::getline(iss, val, ','); hit.dist = std::stof(val);
            std::getline(iss, val, ','); hit.hx   = std::stof(val);
            std::getline(iss, val, ','); hit.hy   = std::stof(val);
            std::getline(iss, val, ','); hit.hz   = std::stof(val);
            if (hit.dist < 1e9f) hits.push_back(hit);
        }
    }
    std::cout << "Loaded " << hits.size() << " hitpoints from " << csv_path << "\n";

    float minDist = 1e30f, maxDist = -1e30f;
    for (const auto& h : hits) {
        if (h.dist > 0.0f && h.dist < minDist) minDist = h.dist;
        if (h.dist > maxDist) maxDist = h.dist;
    }
    if (minDist > maxDist) { minDist = 0.0f; maxDist = 1.0f; }

    struct CsvViewerState : OrbitCamera {
        bool refitRequested = false;
        bool mouseLocked    = false;
        bool skipFirst      = true;
        const std::vector<GrcaHit>* hits = nullptr;
    };

    CsvViewerState state;
    state.hits = &hits;
    fitCameraToHits(state, hits);

    VulkanContext vk;
    GLFWwindow* window = nullptr;
    initWindow(window, "Vulkan CSV Viewer");

    glfwSetWindowUserPointer(window, &state);

    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* win, int w, int h) {
        if (w == 0 || h == 0) return;
        // vk is not accessible here; resize handled next frame
    });

    glfwSetCursorPosCallback(window, [](GLFWwindow* win, double xpos, double ypos) {
        CsvViewerState* s = (CsvViewerState*)glfwGetWindowUserPointer(win);
        if (!s->mouseLocked) return;
        if (s->skipFirst) {
            s->skipFirst = false;
            s->lastX = float(xpos); s->lastY = float(ypos);
            return;
        }
        float dx = float(xpos - s->lastX), dy = float(ypos - s->lastY);
        s->lastX = float(xpos); s->lastY = float(ypos);
        s->yaw   -= dx * 0.003f;
        s->pitch -= dy * 0.003f;
        if (s->pitch < -1.5f) s->pitch = -1.5f;
        if (s->pitch >  1.5f) s->pitch =  1.5f;
    });

    glfwSetScrollCallback(window, [](GLFWwindow* win, double, double yoff) {
        CsvViewerState* s = (CsvViewerState*)glfwGetWindowUserPointer(win);
        s->distance *= powf(1.1f, float(yoff));
        if (s->distance < 0.1f) s->distance = 0.1f;
    });

    glfwSetKeyCallback(window, [](GLFWwindow* win, int key, int, int action, int) {
        if (action != GLFW_PRESS) return;
        CsvViewerState* s = (CsvViewerState*)glfwGetWindowUserPointer(win);
        switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(win, GLFW_TRUE); break;
        case GLFW_KEY_F:
        case GLFW_KEY_KP_0:   s->refitRequested = true; break;
        case GLFW_KEY_LEFT_CONTROL:
        case GLFW_KEY_RIGHT_CONTROL:
            s->mouseLocked = !s->mouseLocked;
            s->skipFirst   = true;
            glfwSetInputMode(win, GLFW_CURSOR,
                s->mouseLocked ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
            break;
        }
    });

    initVulkanContext(vk, window, "CSV Viewer");

    std::vector<Vertex> vertices;
    vertices.reserve(hits.size());
    for (const auto& h : hits) vertices.push_back({h.hx, h.hy, h.hz, h.dist});

    VkBuffer       vertexBuffer;
    VkDeviceMemory vertexMemory;
    createVertexBuffer(vk.device, vk.physicalDevice, vertices.size(), vertexBuffer, vertexMemory);
    {
        VkDeviceSize bufSize = vertices.size() * sizeof(Vertex);
        void* data;
        vkMapMemory(vk.device, vertexMemory, 0, bufSize, 0, &data);
        memcpy(data, vertices.data(), (size_t)bufSize);
        vkUnmapMemory(vk.device, vertexMemory);
    }

    std::vector<uint32_t> vertCode = readFile("build/hit_point.vert.spv");
    std::vector<uint32_t> fragCode = readFile("build/hit_point.frag.spv");
    VkPipelineLayout pipelineLayout = createPipelineLayout(vk.device, sizeof(float) * 18);
    VkPipeline graphicsPipeline = createPointCloudPipeline(
        vk.device, vk.extent, pipelineLayout, vk.renderPass,
        vertCode, fragCode
    );

    static double flyLastTime = 0.0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (vk.extent.width == 0 || vk.extent.height == 0) continue;

        if (state.refitRequested && !hits.empty()) {
            fitCameraToHits(state, hits);
            state.refitRequested = false;
        }

        // WASD/QE movement — active when mouse locked (Ctrl toggles)
        {
            double now = glfwGetTime();
            float dt = float(now - flyLastTime);
            flyLastTime = now;
            if (state.mouseLocked && dt > 0.0f && dt < 0.5f) {
                bool fast = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)  == GLFW_PRESS ||
                            glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
                float speed = state.distance * (fast ? 1.5f : 0.3f) * dt;
                float fx, fy, fz; state.forward(fx, fy, fz);
                float rx, ry, rz; state.right(rx, ry, rz);
                if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) { state.eye[0] += fx*speed; state.eye[1] += fy*speed; state.eye[2] += fz*speed; }
                if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) { state.eye[0] -= fx*speed; state.eye[1] -= fy*speed; state.eye[2] -= fz*speed; }
                if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) { state.eye[0] += rx*speed; state.eye[1] += ry*speed; state.eye[2] += rz*speed; }
                if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) { state.eye[0] -= rx*speed; state.eye[1] -= ry*speed; state.eye[2] -= rz*speed; }
                if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) { state.eye[1] += speed; }
                if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) { state.eye[1] -= speed; }
            }
        }

        float fx, fy, fz; state.forward(fx, fy, fz);
        float target[3] = { state.eye[0]+fx, state.eye[1]+fy, state.eye[2]+fz };
        float up[3] = {0, 1, 0};
        if (fabsf(state.pitch) > 1.4f) { up[0] = 0; up[1] = 0; up[2] = (state.pitch > 0) ? -1.0f : 1.0f; }

        float view[16], proj[16], mvp[16];
        makeLookAt(state.eye, target, up, view);
        makePerspective(0.785398f /*45 deg*/, (float)vk.extent.width / (float)vk.extent.height,
                        0.1f, 100000.0f, proj);
        matMul(proj, view, mvp);

        float pushData[18];
        memcpy(pushData, mvp, 16 * sizeof(float));
        pushData[16] = minDist;
        pushData[17] = maxDist;

        vkWaitForFences(vk.device, 1, &vk.inFlightFence, VK_TRUE, UINT64_MAX);
        vkResetFences(vk.device, 1, &vk.inFlightFence);

        uint32_t imageIndex;
        vkAcquireNextImageKHR(vk.device, vk.swapchain, UINT64_MAX, vk.imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

        VkCommandBuffer cmd = vk.commandBuffers[imageIndex];
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &bi);

        VkClearValue clearColor = {{{0.05f, 0.05f, 0.1f, 1.0f}}};
        VkRenderPassBeginInfo rpbi{};
        rpbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpbi.renderPass      = vk.renderPass;
        rpbi.framebuffer     = vk.framebuffers[imageIndex];
        rpbi.renderArea      = {{0,0}, vk.extent};
        rpbi.clearValueCount = 1;
        rpbi.pClearValues    = &clearColor;

        vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{};
        viewport.x = 0.0f; viewport.y = 0.0f;
        viewport.width  = (float)vk.extent.width;
        viewport.height = (float)vk.extent.height;
        viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
        VkRect2D scissor{{0, 0}, vk.extent};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &offset);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(pushData), pushData);
        vkCmdDraw(cmd, (uint32_t)hits.size(), 1, 0, 0);
        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);

        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo si{};
        si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount   = 1;
        si.pWaitSemaphores      = &vk.imageAvailableSemaphore;
        si.pWaitDstStageMask    = &waitStage;
        si.commandBufferCount   = 1;
        si.pCommandBuffers      = &cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores    = &vk.renderFinishedSemaphore;
        vkQueueSubmit(vk.graphicsQueue, 1, &si, vk.inFlightFence);

        VkPresentInfoKHR pi{};
        pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores    = &vk.renderFinishedSemaphore;
        pi.swapchainCount     = 1;
        pi.pSwapchains        = &vk.swapchain;
        pi.pImageIndices      = &imageIndex;
        vkQueuePresentKHR(vk.presentQueue, &pi);
    }

    vkDeviceWaitIdle(vk.device);

    vkDestroyPipeline(vk.device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(vk.device, pipelineLayout, nullptr);
    vkDestroyBuffer(vk.device, vertexBuffer, nullptr);
    vkFreeMemory(vk.device, vertexMemory, nullptr);
    destroyVulkanContext(vk);
    cleanup(window);
    return 0;
}
