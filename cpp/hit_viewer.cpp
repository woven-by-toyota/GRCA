#include "hit_viewer.h"
#include "vk_common.h"
#include "grca_common.h"
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

struct VisState : OrbitCamera {
	GLFWwindow* window    = nullptr;
	bool fitPending       = true;
	bool refitRequested   = false;
	bool captureRequested = false;
	bool cycleLidarRequested = false;
	bool mouseLocked      = false;
	bool skipFirst        = true;
	bool lidarFollowMode  = false;
	bool sceneBoundsValid = false;
	float sceneBounds[6]  = {}; // xmin,ymin,zmin,xmax,ymax,zmax
	int  activeLidarIndex = -1;
	int  visMode          = 1;
	int  savedVisMode     = 1;
	bool meshVisible      = true;
	bool dynamicVisible   = true;
	bool conesVisible     = false;
	std::vector<VisLidar> lidars;
};

static VulkanContext g_vk;

static bool normalize3(float& x, float& y, float& z) {
	float len = sqrtf(x*x + y*y + z*z);
	if (len < 1e-6f) return false;
	x /= len; y /= len; z /= len;
	return true;
}

static void setCameraFromLidar(VisState* vs, const VisLidar& lidar) {
	if (!vs) return;

	float fx = lidar.fwd[0], fy = lidar.fwd[1], fz = lidar.fwd[2];
	if (!normalize3(fx, fy, fz)) {
		fx = 0.0f; fy = 0.0f; fz = 1.0f;
	}

	float ux = lidar.up[0], uy = lidar.up[1], uz = lidar.up[2];
	if (!normalize3(ux, uy, uz)) {
		ux = 0.0f; uy = 1.0f; uz = 0.0f;
	}

	// Match lidar forward; place camera 0.5 up and 0.5 back so the lidar stays in view.
	vs->eye[0] = lidar.pos[0] + 0.5f * ux - 0.5f * fx;
	vs->eye[1] = lidar.pos[1] + 0.5f * uy - 0.5f * fy;
	vs->eye[2] = lidar.pos[2] + 0.5f * uz - 0.5f * fz;
	vs->pitch = asinf(std::max(-1.0f, std::min(1.0f, fy)));
	vs->yaw   = atan2f(-fx, -fz);
	vs->fitPending = false;
}

static void fitCameraToSceneBounds(VisState* vs) {
	if (!vs || !vs->sceneBoundsValid) return;

	const float minx = vs->sceneBounds[0], miny = vs->sceneBounds[1], minz = vs->sceneBounds[2];
	const float maxx = vs->sceneBounds[3], maxy = vs->sceneBounds[4], maxz = vs->sceneBounds[5];
	const float cx = 0.5f * (minx + maxx), cy = 0.5f * (miny + maxy), cz = 0.5f * (minz + maxz);
	const float dx = 0.5f * (maxx - minx), dy = 0.5f * (maxy - miny), dz = 0.5f * (maxz - minz);

	float radius = sqrtf(dx*dx + dy*dy + dz*dz) + 0.5f;
	if (radius < 1.0f) radius = 1.0f;

	// Use a consistent elevated oblique startup angle for easier scene framing.
	vs->yaw = 0.785398f;    // 45 deg around Y
	vs->pitch = -0.523599f; // -30 deg tilt (look down)

	// Fit bounding sphere into default 45-degree vertical FOV.
	const float fitDist = radius / 0.41421356f;
	vs->distance = fitDist;
	float fx, fy, fz;
	vs->forward(fx, fy, fz);
	vs->eye[0] = cx - fx * fitDist;
	vs->eye[1] = cy - fy * fitDist;
	vs->eye[2] = cz - fz * fitDist;
}

static void ensureVertexBuffer(size_t vertexCount) {
	VkDeviceSize needed = vertexCount * sizeof(Vertex);
	if (needed <= g_vk.vertexBufferCapacity) return;

	if (g_vk.vertexBuffer != VK_NULL_HANDLE) {
		vkDeviceWaitIdle(g_vk.device);
		vkUnmapMemory(g_vk.device, g_vk.vertexMemory);
		g_vk.mappedVertexData = nullptr;
		vkDestroyBuffer(g_vk.device, g_vk.vertexBuffer, nullptr);
		vkFreeMemory(g_vk.device, g_vk.vertexMemory, nullptr);
		g_vk.vertexBuffer = VK_NULL_HANDLE;
		g_vk.vertexMemory = VK_NULL_HANDLE;
	}

	VkDeviceSize newCapacity = std::max(needed + needed / 2, VkDeviceSize(4096));
	allocateMappedBuffer(g_vk.device, g_vk.physicalDevice, newCapacity,
	                     g_vk.vertexBuffer, g_vk.vertexMemory, g_vk.mappedVertexData);
	g_vk.vertexBufferCapacity = newCapacity;
}

static bool ensureMeshWireBuffer(size_t vertexCount) {
	VkDeviceSize needed = vertexCount * 6 * sizeof(float);
	if (needed <= g_vk.meshWireCapacity) return false;

	if (g_vk.meshWireBuffer != VK_NULL_HANDLE) {
		vkDeviceWaitIdle(g_vk.device);
		vkUnmapMemory(g_vk.device, g_vk.meshWireMemory);
		g_vk.mappedMeshWireData = nullptr;
		vkDestroyBuffer(g_vk.device, g_vk.meshWireBuffer, nullptr);
		vkFreeMemory(g_vk.device, g_vk.meshWireMemory, nullptr);
		g_vk.meshWireBuffer = VK_NULL_HANDLE;
		g_vk.meshWireMemory = VK_NULL_HANDLE;
	}

	VkDeviceSize newCapacity = std::max(needed + needed / 2, VkDeviceSize(4096));
	allocateMappedBuffer(g_vk.device, g_vk.physicalDevice, newCapacity,
	                     g_vk.meshWireBuffer, g_vk.meshWireMemory, g_vk.mappedMeshWireData);
	g_vk.meshWireCapacity = newCapacity;
	return true;
}

static bool ensureMeshSolidBuffer(size_t vertexCount) {
	VkDeviceSize needed = vertexCount * 6 * sizeof(float);
	if (needed <= g_vk.meshSolidCapacity) return false;

	if (g_vk.meshSolidBuffer != VK_NULL_HANDLE) {
		vkDeviceWaitIdle(g_vk.device);
		vkUnmapMemory(g_vk.device, g_vk.meshSolidMemory);
		g_vk.mappedMeshSolidData = nullptr;
		vkDestroyBuffer(g_vk.device, g_vk.meshSolidBuffer, nullptr);
		vkFreeMemory(g_vk.device, g_vk.meshSolidMemory, nullptr);
		g_vk.meshSolidBuffer = VK_NULL_HANDLE;
		g_vk.meshSolidMemory = VK_NULL_HANDLE;
	}

	VkDeviceSize newCapacity = std::max(needed + needed / 2, VkDeviceSize(4096));
	allocateMappedBuffer(g_vk.device, g_vk.physicalDevice, newCapacity,
	                     g_vk.meshSolidBuffer, g_vk.meshSolidMemory, g_vk.mappedMeshSolidData);
	g_vk.meshSolidCapacity = newCapacity;
	return true;
}

static void ensureLidarSolidBuffer(size_t vertexCount) {
	VkDeviceSize needed = vertexCount * 6 * sizeof(float);
	if (needed <= g_vk.lidarSolidCapacity) return;

	if (g_vk.lidarSolidBuffer != VK_NULL_HANDLE) {
		vkDeviceWaitIdle(g_vk.device);
		vkUnmapMemory(g_vk.device, g_vk.lidarSolidMemory);
		g_vk.mappedLidarSolidData = nullptr;
		vkDestroyBuffer(g_vk.device, g_vk.lidarSolidBuffer, nullptr);
		vkFreeMemory(g_vk.device, g_vk.lidarSolidMemory, nullptr);
		g_vk.lidarSolidBuffer = VK_NULL_HANDLE;
		g_vk.lidarSolidMemory = VK_NULL_HANDLE;
	}

	VkDeviceSize newCapacity = std::max(needed + needed / 2, VkDeviceSize(4096));
	allocateMappedBuffer(g_vk.device, g_vk.physicalDevice, newCapacity,
	                     g_vk.lidarSolidBuffer, g_vk.lidarSolidMemory, g_vk.mappedLidarSolidData);
	g_vk.lidarSolidCapacity = newCapacity;
}

static void ensureLidarConeBuffer(size_t vertexCount) {
	VkDeviceSize needed = vertexCount * 6 * sizeof(float);
	if (needed <= g_vk.lidarConeCapacity) return;

	if (g_vk.lidarConeBuffer != VK_NULL_HANDLE) {
		vkDeviceWaitIdle(g_vk.device);
		vkUnmapMemory(g_vk.device, g_vk.lidarConeMemory);
		g_vk.mappedLidarConeData = nullptr;
		vkDestroyBuffer(g_vk.device, g_vk.lidarConeBuffer, nullptr);
		vkFreeMemory(g_vk.device, g_vk.lidarConeMemory, nullptr);
		g_vk.lidarConeBuffer = VK_NULL_HANDLE;
		g_vk.lidarConeMemory = VK_NULL_HANDLE;
	}

	VkDeviceSize newCapacity = std::max(needed + needed / 2, VkDeviceSize(4096));
	allocateMappedBuffer(g_vk.device, g_vk.physicalDevice, newCapacity,
	                     g_vk.lidarConeBuffer, g_vk.lidarConeMemory, g_vk.mappedLidarConeData);
	g_vk.lidarConeCapacity = newCapacity;
}

static void ensureAabbWireBuffer(size_t vertexCount) {
	VkDeviceSize needed = vertexCount * 6 * sizeof(float);
	if (needed <= g_vk.aabbWireCapacity) return;

	if (g_vk.aabbWireBuffer != VK_NULL_HANDLE) {
		vkDeviceWaitIdle(g_vk.device);
		vkUnmapMemory(g_vk.device, g_vk.aabbWireMemory);
		g_vk.mappedAabbWireData = nullptr;
		vkDestroyBuffer(g_vk.device, g_vk.aabbWireBuffer, nullptr);
		vkFreeMemory(g_vk.device, g_vk.aabbWireMemory, nullptr);
		g_vk.aabbWireBuffer = VK_NULL_HANDLE;
		g_vk.aabbWireMemory = VK_NULL_HANDLE;
	}

	VkDeviceSize newCapacity = std::max(needed + needed / 2, VkDeviceSize(4096));
	allocateMappedBuffer(g_vk.device, g_vk.physicalDevice, newCapacity,
	                     g_vk.aabbWireBuffer, g_vk.aabbWireMemory, g_vk.mappedAabbWireData);
	g_vk.aabbWireCapacity = newCapacity;
}


VisState* vis_init(const RunConfig& cfg) {
	auto* vs = new VisState();
	vs->visMode = cfg.mesh_vis_on; // 0 = hitpoints only, 1 = wireframe, 2 = solid + wireframe
	vs->savedVisMode = (cfg.mesh_vis_on > 0) ? cfg.mesh_vis_on : 1;
	vs->meshVisible = (cfg.mesh_vis_on > 0);
	vs->dynamicVisible = (cfg.mesh_vis_on > 0);
	vs->conesVisible = cfg.cone_vis_on;
	initWindow(vs->window, "Hit Vulkan Viewer", 1000, 1000);
	g_vk.initialized = false;

	glfwSetWindowUserPointer(vs->window, vs);
	glfwSetCursorPosCallback(vs->window, [](GLFWwindow* win, double xpos, double ypos) {
		VisState* s = (VisState*)glfwGetWindowUserPointer(win);
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
	glfwSetScrollCallback(vs->window, [](GLFWwindow* win, double, double yoff) {
		VisState* s = (VisState*)glfwGetWindowUserPointer(win);
		s->distance *= powf(1.1f, float(yoff));
		if (s->distance < 0.1f) s->distance = 0.1f;
	});
	glfwSetKeyCallback(vs->window, [](GLFWwindow* win, int key, int, int action, int) {
		if (action != GLFW_PRESS) return;
		VisState* s = (VisState*)glfwGetWindowUserPointer(win);
		switch (key) {
		case GLFW_KEY_T:
			s->cycleLidarRequested = true;
			break;
		case GLFW_KEY_P:
			s->lidarFollowMode = false;
			break;
		case GLFW_KEY_M:
			if (s->meshVisible || s->dynamicVisible) {
				s->savedVisMode = s->visMode;
				s->meshVisible = false;
				s->dynamicVisible = false;
				s->visMode = 0;
			} else {
				s->meshVisible = true;
				s->dynamicVisible = true;
				s->visMode = std::max(1, s->savedVisMode);
			}
			break;
		case GLFW_KEY_B:
			s->dynamicVisible = !s->dynamicVisible;
			if (s->dynamicVisible)
				s->visMode = std::max(1, s->savedVisMode);
			else if (!s->meshVisible)
				s->visMode = 0;
			break;
		case GLFW_KEY_C:
			s->conesVisible = !s->conesVisible;
			break;
		case GLFW_KEY_O:      s->captureRequested = true; break;
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

	glfwSetFramebufferSizeCallback(vs->window, [](GLFWwindow*, int w, int h) {
		if (w == 0 || h == 0 || !g_vk.initialized) return;
		rebuildSwapchain(g_vk);
	});

	initVulkanContext(g_vk, vs->window, "Hit Vulkan Viewer");

	// Create depth resources and rebuild framebuffers with depth
	createDepthResources(g_vk.device, g_vk.physicalDevice, g_vk.extent,
	                     g_vk.depthImage, g_vk.depthMemory, g_vk.depthImageView);
	for (auto fb : g_vk.framebuffers) vkDestroyFramebuffer(g_vk.device, fb, nullptr);
	g_vk.framebuffers.clear();
	createFramebuffers(g_vk.device, g_vk.renderPass, g_vk.imageViews, g_vk.extent,
	                   g_vk.framebuffers, g_vk.depthImageView);

	// Create pipeline layout
	g_vk.pipelineLayout = createPipelineLayout(g_vk.device, sizeof(float) * 18);

	// Create point cloud pipeline
	auto vertCode = readFile("build/hit_point.vert.spv");
	auto fragCode = readFile("build/hit_point.frag.spv");
	g_vk.graphicsPipeline = createPointCloudPipeline(
		g_vk.device, g_vk.extent, g_vk.pipelineLayout, g_vk.renderPass, vertCode, fragCode);

	// Create mesh wireframe pipeline
	auto mwVertCode = readFile("build/mesh_wire.vert.spv");
	auto mwFragCode = readFile("build/mesh_wire.frag.spv");
	g_vk.meshWirePipeline = createMeshWirePipeline(
		g_vk.device, g_vk.extent, g_vk.pipelineLayout, g_vk.renderPass, mwVertCode, mwFragCode);

	// Create mesh solid (filled triangles) pipeline — same shaders, TRIANGLE_LIST topology
	g_vk.meshSolidPipeline = createMeshSolidPipeline(
		g_vk.device, g_vk.extent, g_vk.pipelineLayout, g_vk.renderPass, mwVertCode, mwFragCode);

	// Create transparent cone pipeline
	auto coneFragCode = readFile("build/cone.frag.spv");
	g_vk.conePipeline = createConePipeline(
		g_vk.device, g_vk.extent, g_vk.pipelineLayout, g_vk.renderPass, mwVertCode, coneFragCode);

	g_vk.initialized = true;
	return vs;
}

void vis_set_camera(VisState* vs, const float eye[3], const float forward[3], const float up[3]) {
	if (!vs) return;
	vs->eye[0] = eye[0]; vs->eye[1] = eye[1]; vs->eye[2] = eye[2];
	if (forward) {
		// Derive yaw and pitch from the forward vector
		vs->pitch = asinf(std::max(-1.0f, std::min(1.0f, forward[1])));
		vs->yaw   = atan2f(-forward[0], -forward[2]);
	}
	// up is not used separately — the camera derives up from yaw/pitch
	(void)up;
	vs->fitPending = false; // don't auto-fit over the explicit position
}

void vis_update(VisState* vs, const std::vector<GrcaHit>& hits, const RunConfig& cfg, int frame, float sim_time) {
	if (!vs || !vs->window || !g_vk.initialized) return;
	if (glfwWindowShouldClose(vs->window)) return;
	if (g_vk.extent.width == 0 || g_vk.extent.height == 0) { glfwPollEvents(); return; }

	glfwPollEvents();

	if (vs->cycleLidarRequested) {
		vs->cycleLidarRequested = false;
		if (!vs->lidars.empty()) {
			vs->activeLidarIndex = (vs->activeLidarIndex + 1) % (int)vs->lidars.size();
			vs->lidarFollowMode = true;
			setCameraFromLidar(vs, vs->lidars[(size_t)vs->activeLidarIndex]);
		}
	}

	if (vs->lidarFollowMode) {
		if (vs->lidars.empty()) {
			vs->lidarFollowMode = false;
			vs->activeLidarIndex = -1;
		} else {
			if (vs->activeLidarIndex < 0 || vs->activeLidarIndex >= (int)vs->lidars.size())
				vs->activeLidarIndex = 0;
			setCameraFromLidar(vs, vs->lidars[(size_t)vs->activeLidarIndex]);
		}
	}

	float rangeMaxCfg = (!cfg.robots.empty() && !cfg.robots[0].lidars.empty())
	                    ? cfg.robots[0].lidars[0].range_max : 1e9f;
	// Startup framing: fit to scene bounds (AABB upload), not transient hit points.
	if (vs->fitPending && vs->sceneBoundsValid) {
		fitCameraToSceneBounds(vs);
		vs->fitPending = false;
	}
	// F key: focus on valid hit bounding box (Unity-style)
	if (vs->refitRequested && !hits.empty()) {
		fitCameraToHits(*vs, hits, rangeMaxCfg);
		vs->refitRequested = false;
	}

	// Update window title with live stats + camera position
	float fx2, fy2, fz2; vs->forward(fx2, fy2, fz2);
	char title[256];
	snprintf(title, sizeof(title),
	         "Hit Viewer  frame %d  t=%.2fs  hits=%zu  | cam pos=(%.2f,%.2f,%.2f)  fwd=(%.2f,%.2f,%.2f)  mode=%s",
	         frame, sim_time, hits.size(),
	         vs->eye[0], vs->eye[1], vs->eye[2],
	         fx2, fy2, fz2,
	         vs->lidarFollowMode ? "lidar-follow" : "free-roam");
	glfwSetWindowTitle(vs->window, title);

	// Print camera pose to stderr once per second
	{
		static double lastPrint = -1.0;
		double now = glfwGetTime();
		if (now - lastPrint >= 1.0) {
			lastPrint = now;
			fprintf(stderr, "[cam] pos=(%.3f, %.3f, %.3f)  fwd=(%.3f, %.3f, %.3f)  yaw=%.3f  pitch=%.3f\n",
			        vs->eye[0], vs->eye[1], vs->eye[2],
			        fx2, fy2, fz2,
			        vs->yaw, vs->pitch);
		}
	}

	// Upload hit positions/distances to vertex buffer; compute distance range for colour ramp.
	// Misses (dist == range_max) are encoded as dist=-1 so the shader renders them black.
	float minDist = 1e30f, maxDist = -1e30f;
	if (!hits.empty()) {
		ensureVertexBuffer(hits.size());
		auto* verts = (Vertex*)g_vk.mappedVertexData;
		for (size_t i = 0; i < hits.size(); i++) {
			bool isMiss = (hits[i].dist <= 0.0f || hits[i].dist >= rangeMaxCfg * 0.9999f);
			float d = isMiss ? -1.0f : hits[i].dist;
			verts[i] = {hits[i].hx, hits[i].hy, hits[i].hz, d};
			if (!isMiss && d > 0.0f && d < minDist) minDist = d;
			if (!isMiss && d > maxDist) maxDist = d;
		}
		if (minDist > maxDist) { minDist = 0.0f; maxDist = 1.0f; }
	}

	// WASD/QE movement — active when mouse is locked (Ctrl toggles)
	{
		static double lastTime = 0.0;
		double now = glfwGetTime();
		float dt = float(now - lastTime);
		lastTime = now;
		if (dt > 5.0f) dt = 5.0f; // clamp after long pauses, but don't suppress movement
		if (vs->mouseLocked && !vs->lidarFollowMode && dt > 0.0f) {
			bool fast = glfwGetKey(vs->window, GLFW_KEY_LEFT_SHIFT)  == GLFW_PRESS ||
			            glfwGetKey(vs->window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
			float speed = vs->distance * (fast ? 1.5f : 0.3f) * dt;
			float fx, fy, fz; vs->forward(fx, fy, fz);
			float rx, ry, rz; vs->right(rx, ry, rz);
			if (glfwGetKey(vs->window, GLFW_KEY_W) == GLFW_PRESS) { vs->eye[0] += fx*speed; vs->eye[1] += fy*speed; vs->eye[2] += fz*speed; }
			if (glfwGetKey(vs->window, GLFW_KEY_S) == GLFW_PRESS) { vs->eye[0] -= fx*speed; vs->eye[1] -= fy*speed; vs->eye[2] -= fz*speed; }
			if (glfwGetKey(vs->window, GLFW_KEY_D) == GLFW_PRESS) { vs->eye[0] += rx*speed; vs->eye[1] += ry*speed; vs->eye[2] += rz*speed; }
			if (glfwGetKey(vs->window, GLFW_KEY_A) == GLFW_PRESS) { vs->eye[0] -= rx*speed; vs->eye[1] -= ry*speed; vs->eye[2] -= rz*speed; }
			if (glfwGetKey(vs->window, GLFW_KEY_E) == GLFW_PRESS) { vs->eye[1] += speed; }
			if (glfwGetKey(vs->window, GLFW_KEY_Q) == GLFW_PRESS) { vs->eye[1] -= speed; }
		}
	}

	// Build view matrix from eye position + forward direction
	float fx, fy, fz; vs->forward(fx, fy, fz);
	float target[3] = { vs->eye[0] + fx, vs->eye[1] + fy, vs->eye[2] + fz };
	float up[3] = {0, 1, 0};
	if (fabsf(vs->pitch) > 1.4f) { up[0] = 0; up[1] = 0; up[2] = (vs->pitch > 0) ? -1.0f : 1.0f; }
	float view[16], proj[16], mvp[16];
	makeLookAt(vs->eye, target, up, view);
	const float fovY = vs->lidarFollowMode ? 1.919862f : 0.785398f; // 110 deg in follow mode, 45 deg otherwise
	makePerspective(fovY, (float)g_vk.extent.width / (float)g_vk.extent.height, 0.1f, 100000.0f, proj);
	matMul(proj, view, mvp);

	float pushData[18];
	memcpy(pushData, mvp, 16 * sizeof(float));
	pushData[16] = minDist;
	pushData[17] = maxDist;

	// Frame synchronization
	vkWaitForFences(g_vk.device, 1, &g_vk.inFlightFence, VK_TRUE, UINT64_MAX);
	vkResetFences(g_vk.device, 1, &g_vk.inFlightFence);

	uint32_t imageIndex;
	vkAcquireNextImageKHR(g_vk.device, g_vk.swapchain, UINT64_MAX, g_vk.imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

	// Re-record command buffer for this frame
	VkCommandBuffer cmd = g_vk.commandBuffers[imageIndex];
	vkResetCommandBuffer(cmd, 0);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	vkBeginCommandBuffer(cmd, &beginInfo);

	VkClearValue clearValues[2] = {};
	clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
	clearValues[1].depthStencil = {1.0f, 0};
	VkRenderPassBeginInfo rpbi{};
	rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	rpbi.renderPass = g_vk.renderPass;
	rpbi.framebuffer = g_vk.framebuffers[imageIndex];
	rpbi.renderArea.offset = {0, 0};
	rpbi.renderArea.extent = g_vk.extent;
	rpbi.clearValueCount = 2;
	rpbi.pClearValues = clearValues;

	vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

	VkViewport viewport{};
	viewport.x = 0.0f; viewport.y = 0.0f;
	viewport.width  = (float)g_vk.extent.width;
	viewport.height = (float)g_vk.extent.height;
	viewport.minDepth = 0.0f; viewport.maxDepth = 1.0f;
	VkRect2D scissor{{0, 0}, g_vk.extent};
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	const uint32_t staticSolidVerts = g_vk.staticSolidVertexCount;
	const uint32_t dynamicSolidVerts =
	    (g_vk.meshSolidVertexCount > g_vk.staticSolidVertexCount)
	        ? (g_vk.meshSolidVertexCount - g_vk.staticSolidVertexCount)
	        : 0;
	const uint32_t staticWireVerts = g_vk.staticWireVertexCount;
	const uint32_t dynamicWireVerts =
	    (g_vk.meshWireVertexCount > g_vk.staticWireVertexCount)
	        ? (g_vk.meshWireVertexCount - g_vk.staticWireVertexCount)
	        : 0;

	// mesh_vis_on=2: solid mesh with material colors
	if (vs->visMode >= 2 && g_vk.meshSolidBuffer != VK_NULL_HANDLE) {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_vk.meshSolidPipeline);
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &g_vk.meshSolidBuffer, &offset);
		vkCmdPushConstants(cmd, g_vk.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);
		if (vs->meshVisible && staticSolidVerts > 0)
			vkCmdDraw(cmd, staticSolidVerts, 1, 0, 0);
		if (vs->dynamicVisible && dynamicSolidVerts > 0)
			vkCmdDraw(cmd, dynamicSolidVerts, 1, staticSolidVerts, 0);
	}

	// lidar cylinders always visible
	if (g_vk.lidarSolidVertexCount > 0 && g_vk.lidarSolidBuffer != VK_NULL_HANDLE) {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_vk.meshSolidPipeline);
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &g_vk.lidarSolidBuffer, &offset);
		vkCmdPushConstants(cmd, g_vk.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);
		vkCmdDraw(cmd, g_vk.lidarSolidVertexCount, 1, 0, 0);
	}

	// lidar channel cones — translucent filled surface
	if (vs->conesVisible && g_vk.lidarConeVertexCount > 0 && g_vk.lidarConeBuffer != VK_NULL_HANDLE) {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_vk.conePipeline);
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &g_vk.lidarConeBuffer, &offset);
		vkCmdPushConstants(cmd, g_vk.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);
		vkCmdDraw(cmd, g_vk.lidarConeVertexCount, 1, 0, 0);
	}

	// mesh_vis_on>=1: wireframe
	if (vs->visMode >= 1 && g_vk.meshWireBuffer != VK_NULL_HANDLE) {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_vk.meshWirePipeline);
		vkCmdSetLineWidth(cmd, 1.0f);
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &g_vk.meshWireBuffer, &offset);
		vkCmdPushConstants(cmd, g_vk.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);
		if (vs->meshVisible && staticWireVerts > 0)
			vkCmdDraw(cmd, staticWireVerts, 1, 0, 0);
		if (vs->dynamicVisible && dynamicWireVerts > 0)
			vkCmdDraw(cmd, dynamicWireVerts, 1, staticWireVerts, 0);
	}

	// Draw hit points
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_vk.graphicsPipeline);
	if (!hits.empty()) {
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &g_vk.vertexBuffer, &offset);
		vkCmdPushConstants(cmd, g_vk.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);
		vkCmdDraw(cmd, (uint32_t)hits.size(), 1, 0, 0);
	}

	// AABB wireframes last (always visible, on top of everything)
	if (g_vk.aabbWireVertexCount > 0 && g_vk.aabbWireBuffer != VK_NULL_HANDLE) {
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_vk.meshWirePipeline);
		vkCmdSetLineWidth(cmd, 3.0f);
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &g_vk.aabbWireBuffer, &offset);
		vkCmdPushConstants(cmd, g_vk.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);
		vkCmdDraw(cmd, g_vk.aabbWireVertexCount, 1, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);

	VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = &g_vk.imageAvailableSemaphore;
	submitInfo.pWaitDstStageMask = waitStages;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmd;
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = &g_vk.renderFinishedSemaphore;

	vkQueueSubmit(g_vk.graphicsQueue, 1, &submitInfo, g_vk.inFlightFence);

	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &g_vk.renderFinishedSemaphore;
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &g_vk.swapchain;
	presentInfo.pImageIndices = &imageIndex;

	vkQueuePresentKHR(g_vk.presentQueue, &presentInfo);
}

bool vis_running(VisState* vs) {
	return vs && vs->window && !glfwWindowShouldClose(vs->window);
}

bool vis_capture_requested(VisState* vs) {
	if (!vs) return false;
	bool result = vs->captureRequested;
	vs->captureRequested = false;
	return result;
}

// Write wire+solid vertices for one mesh into pre-sized buffers.
// Returns pointer advances: wireDst += count*36, solidDst += count*18.
static void uploadMeshVertices(float*& wireDst, float*& solidDst, const VisMesh& m) {
	for (size_t t = 0; t < m.count; ++t) {
		float r, g, b;
		if (m.tri_colors) {
			r = m.tri_colors[t*3+0]; g = m.tri_colors[t*3+1]; b = m.tri_colors[t*3+2];
		} else {
			r = m.color[0]; g = m.color[1]; b = m.color[2];
		}
		const float wr = r*0.3f, wg = g*0.3f, wb = b*0.3f;
		const auto& tri = m.tris[t];
		solidDst[ 0]=tri.v[0][0]; solidDst[ 1]=tri.v[0][1]; solidDst[ 2]=tri.v[0][2]; solidDst[ 3]=r;  solidDst[ 4]=g;  solidDst[ 5]=b;
		solidDst[ 6]=tri.v[1][0]; solidDst[ 7]=tri.v[1][1]; solidDst[ 8]=tri.v[1][2]; solidDst[ 9]=r;  solidDst[10]=g;  solidDst[11]=b;
		solidDst[12]=tri.v[2][0]; solidDst[13]=tri.v[2][1]; solidDst[14]=tri.v[2][2]; solidDst[15]=r;  solidDst[16]=g;  solidDst[17]=b;
		solidDst += 18;
		wireDst[ 0]=tri.v[0][0]; wireDst[ 1]=tri.v[0][1]; wireDst[ 2]=tri.v[0][2]; wireDst[ 3]=wr; wireDst[ 4]=wg; wireDst[ 5]=wb;
		wireDst[ 6]=tri.v[1][0]; wireDst[ 7]=tri.v[1][1]; wireDst[ 8]=tri.v[1][2]; wireDst[ 9]=wr; wireDst[10]=wg; wireDst[11]=wb;
		wireDst[12]=tri.v[1][0]; wireDst[13]=tri.v[1][1]; wireDst[14]=tri.v[1][2]; wireDst[15]=wr; wireDst[16]=wg; wireDst[17]=wb;
		wireDst[18]=tri.v[2][0]; wireDst[19]=tri.v[2][1]; wireDst[20]=tri.v[2][2]; wireDst[21]=wr; wireDst[22]=wg; wireDst[23]=wb;
		wireDst[24]=tri.v[2][0]; wireDst[25]=tri.v[2][1]; wireDst[26]=tri.v[2][2]; wireDst[27]=wr; wireDst[28]=wg; wireDst[29]=wb;
		wireDst[30]=tri.v[0][0]; wireDst[31]=tri.v[0][1]; wireDst[32]=tri.v[0][2]; wireDst[33]=wr; wireDst[34]=wg; wireDst[35]=wb;
		wireDst += 36;
	}
}

void vis_set_meshes(VisState* vs, const std::vector<VisMesh>& meshes) {
	(void)vs;
	if (!g_vk.initialized) return;

	size_t totalWireVerts = 0, totalSolidVerts = 0;
	for (const auto& m : meshes) { totalWireVerts += m.count * 6; totalSolidVerts += m.count * 3; }
	if (totalWireVerts == 0) { g_vk.meshWireVertexCount = 0; g_vk.meshSolidVertexCount = 0; return; }

	ensureMeshWireBuffer(totalWireVerts);
	ensureMeshSolidBuffer(totalSolidVerts);
	float* wireDst  = (float*)g_vk.mappedMeshWireData;
	float* solidDst = (float*)g_vk.mappedMeshSolidData;
	for (const auto& m : meshes) uploadMeshVertices(wireDst, solidDst, m);
	g_vk.meshWireVertexCount  = (uint32_t)totalWireVerts;
	g_vk.meshSolidVertexCount = (uint32_t)totalSolidVerts;
}

void vis_set_static_meshes(VisState* vs, const std::vector<VisMesh>& meshes) {
	(void)vs;
	if (!g_vk.initialized) return;

	size_t staticWireVerts = 0;
	size_t staticSolidVerts = 0;
	for (const auto& m : meshes) {
		staticWireVerts += m.count * 6;
		staticSolidVerts += m.count * 3;
	}
	ensureMeshWireBuffer(staticWireVerts);
	ensureMeshSolidBuffer(staticSolidVerts);

	float* wireDst  = (float*)g_vk.mappedMeshWireData;
	float* solidDst = (float*)g_vk.mappedMeshSolidData;
	for (const auto& m : meshes) uploadMeshVertices(wireDst, solidDst, m);
	g_vk.staticWireVertexCount  = (uint32_t)staticWireVerts;
	g_vk.meshWireVertexCount    = (uint32_t)staticWireVerts;
	g_vk.staticSolidVertexCount = (uint32_t)staticSolidVerts;
	g_vk.meshSolidVertexCount   = (uint32_t)staticSolidVerts;

	// Save CPU-side copies so we can re-upload if the buffer grows later.
	float* wireSrc  = (float*)g_vk.mappedMeshWireData;
	g_vk.staticWireData.assign(wireSrc, wireSrc + staticWireVerts * 6);
	float* solidSrc = (float*)g_vk.mappedMeshSolidData;
	g_vk.staticSolidData.assign(solidSrc, solidSrc + staticSolidVerts * 6);
}

void vis_update_dynamic_meshes(VisState* vs, const std::vector<VisMesh>& meshes) {
	(void)vs;
	if (!g_vk.initialized) return;

	size_t dynWireVerts = 0, dynSolidVerts = 0;
	for (const auto& m : meshes) { dynWireVerts += m.count * 6; dynSolidVerts += m.count * 3; }

	size_t totalWireVerts  = g_vk.staticWireVertexCount  + dynWireVerts;
	size_t totalSolidVerts = g_vk.staticSolidVertexCount + dynSolidVerts;
	bool wireReallocated  = ensureMeshWireBuffer(totalWireVerts);
	bool solidReallocated = ensureMeshSolidBuffer(totalSolidVerts);

	if (wireReallocated && !g_vk.staticWireData.empty())
		memcpy(g_vk.mappedMeshWireData, g_vk.staticWireData.data(),
		       g_vk.staticWireData.size() * sizeof(float));
	if (solidReallocated && !g_vk.staticSolidData.empty())
		memcpy(g_vk.mappedMeshSolidData, g_vk.staticSolidData.data(),
		       g_vk.staticSolidData.size() * sizeof(float));

	float* wireDst  = (float*)g_vk.mappedMeshWireData  + g_vk.staticWireVertexCount  * 6;
	float* solidDst = (float*)g_vk.mappedMeshSolidData + g_vk.staticSolidVertexCount * 6;
	for (const auto& m : meshes) uploadMeshVertices(wireDst, solidDst, m);
	g_vk.meshWireVertexCount  = (uint32_t)totalWireVerts;
	g_vk.meshSolidVertexCount = (uint32_t)totalSolidVerts;
}

// Write 24 line-list vertices (12 edges × 2) for one AABB box into dst.
// Each vertex: xyz + rgb (6 floats). dst is advanced by 24*6 floats.
static void uploadAABBWireVertices(float*& dst, const VisAABB& a) {
	const float xn = a.aabb[0], yn = a.aabb[1], zn = a.aabb[2];
	const float xx = a.aabb[3], yx = a.aabb[4], zx = a.aabb[5];
	const float r = a.color[0], g = a.color[1], b = a.color[2];
	// 8 corner positions
	const float cx[8][3] = {
		{xn,yn,zn},{xx,yn,zn},{xx,yx,zn},{xn,yx,zn},
		{xn,yn,zx},{xx,yn,zx},{xx,yx,zx},{xn,yx,zx}
	};
	// 12 edges as pairs of corner indices
	const int edges[12][2] = {
		{0,1},{1,2},{2,3},{3,0}, // bottom face
		{4,5},{5,6},{6,7},{7,4}, // top face
		{0,4},{1,5},{2,6},{3,7}  // verticals
	};
	for (const auto& e : edges) {
		for (int v = 0; v < 2; ++v) {
			dst[0]=cx[e[v]][0]; dst[1]=cx[e[v]][1]; dst[2]=cx[e[v]][2];
			dst[3]=r; dst[4]=g; dst[5]=b;
			dst += 6;
		}
	}
}

// Emit solid cylinder triangles into dst (xyz+rgb, 6 floats/vertex).
// Sides: SEGS quads = 2*SEGS tris = 6*SEGS floats*6; caps: 2*SEGS tris.
// Total: 4*SEGS triangles = 12*SEGS vertices.
static void uploadCylinder(float*& dst, const VisLidar& L) {
	const int SEGS = 16;
	const float r = L.radius, h = L.height;
	const float R = L.color[0], G = L.color[1], B = L.color[2];

	// Build an orthonormal basis: axis = fwd, two perpendicular vectors
	float ax = L.fwd[0], ay = L.fwd[1], az = L.fwd[2];
	// Use up to derive right = up × fwd, then recompute a clean perp
	float ux = L.up[0], uy = L.up[1], uz = L.up[2];
	// right = fwd × up  (then normalise)
	float rx = ay*uz - az*uy, ry = az*ux - ax*uz, rz = ax*uy - ay*ux;
	float rlen = sqrtf(rx*rx + ry*ry + rz*rz);
	if (rlen < 1e-6f) { rx=1; ry=0; rz=0; } else { rx/=rlen; ry/=rlen; rz/=rlen; }
	// recompute clean up = fwd × right
	float cx2 = ay*rz - az*ry, cy2 = az*rx - ax*rz, cz2 = ax*ry - ay*rx;

	// Base centre and top centre
	const float bx = L.pos[0],          by = L.pos[1],          bz = L.pos[2];
	const float tx = bx + ax*h,          ty = by + ay*h,          tz = bz + az*h;

	// Precompute ring vertices at base (b_) and top (t_)
	float bvx[SEGS], bvy[SEGS], bvz[SEGS];
	float tvx[SEGS], tvy[SEGS], tvz[SEGS];
	for (int s = 0; s < SEGS; ++s) {
		float a = (float)s / (float)SEGS * 6.28318530f;
		float ca = cosf(a), sa = sinf(a);
		float px = rx*ca + cx2*sa, py = ry*ca + cy2*sa, pz = rz*ca + cz2*sa;
		bvx[s] = bx + px*r;  bvy[s] = by + py*r;  bvz[s] = bz + pz*r;
		tvx[s] = tx + px*r;  tvy[s] = ty + py*r;  tvz[s] = tz + pz*r;
	}

	auto emit = [&](float vx, float vy, float vz) {
		dst[0]=vx; dst[1]=vy; dst[2]=vz; dst[3]=R; dst[4]=G; dst[5]=B; dst+=6;
	};

	for (int s = 0; s < SEGS; ++s) {
		int n = (s + 1) % SEGS;
		// Side quad (2 tris)
		emit(bvx[s],bvy[s],bvz[s]); emit(tvx[s],tvy[s],tvz[s]); emit(bvx[n],bvy[n],bvz[n]);
		emit(tvx[s],tvy[s],tvz[s]); emit(tvx[n],tvy[n],tvz[n]); emit(bvx[n],bvy[n],bvz[n]);
		// Bottom cap (fan from base centre)
		emit(bx,by,bz); emit(bvx[n],bvy[n],bvz[n]); emit(bvx[s],bvy[s],bvz[s]);
		// Top cap (fan from top centre)
		emit(tx,ty,tz); emit(tvx[s],tvy[s],tvz[s]); emit(tvx[n],tvy[n],tvz[n]);
	}
}

void vis_set_lidars(VisState* vs, const std::vector<VisLidar>& lidars) {
	if (vs) {
		vs->lidars = lidars;
		if (vs->lidars.empty()) {
			vs->activeLidarIndex = -1;
			vs->lidarFollowMode = false;
		} else if (vs->activeLidarIndex >= (int)vs->lidars.size()) {
			vs->activeLidarIndex = 0;
		}
	}
	if (!g_vk.initialized) return;
	const int SEGS = 16;
	const size_t vertsPerLidar = (size_t)SEGS * 4 * 3; // 4 tris per seg, 3 verts each
	const size_t totalVerts = lidars.size() * vertsPerLidar;
	if (totalVerts == 0) { g_vk.lidarSolidVertexCount = 0; return; }
	ensureLidarSolidBuffer(totalVerts);
	float* dst = (float*)g_vk.mappedLidarSolidData;
	for (const auto& L : lidars) uploadCylinder(dst, L);
	g_vk.lidarSolidVertexCount = (uint32_t)totalVerts;
}

void vis_set_lidar_cones(VisState* vs, const std::vector<VisLidar>& lidars) {
	(void)vs;
	if (!g_vk.initialized) return;
	const int SEGS = 32;

	// Solid cone per channel: SEGS triangles (apex + 2 rim pts) = 3*SEGS verts
	size_t totalVerts = 0;
	for (const auto& L : lidars)
		totalVerts += (size_t)L.vnum * SEGS * 3;
	if (totalVerts == 0) { g_vk.lidarConeVertexCount = 0; return; }

	ensureLidarConeBuffer(totalVerts);
	float* dst = (float*)g_vk.mappedLidarConeData;

	for (const auto& L : lidars) {
		if (L.vnum <= 0) continue;

		float ax = L.fwd[0], ay = L.fwd[1], az = L.fwd[2];
		float ux = L.up[0],  uy = L.up[1],  uz = L.up[2];
		float r1x = ay*uz - az*uy, r1y = az*ux - ax*uz, r1z = ax*uy - ay*ux;
		float rlen = sqrtf(r1x*r1x + r1y*r1y + r1z*r1z);
		if (rlen < 1e-6f) { r1x=1; r1y=0; r1z=0; } else { r1x/=rlen; r1y/=rlen; r1z/=rlen; }
		float r2x = ay*r1z - az*r1y, r2y = az*r1x - ax*r1z, r2z = ax*r1y - ay*r1x;

		const float R = L.color[0], G = L.color[1], B = L.color[2];
		const float px = L.pos[0], py = L.pos[1], pz = L.pos[2];

		auto emit = [&](float vx, float vy, float vz) {
			dst[0]=vx; dst[1]=vy; dst[2]=vz; dst[3]=R; dst[4]=G; dst[5]=B; dst+=6;
		};

		float vStep = (L.vnum > 1) ? (L.vmax - L.vmin) / (float)(L.vnum - 1) : 0.0f;

		for (int ch = 0; ch < L.vnum; ++ch) {
			float vAngle = L.vmin + ch * vStep;
			float sinV   = sinf(vAngle), cosV = cosf(vAngle);
			const float visRange = 1.0f;
			float hOff = visRange * sinV;
			float rimR = fabsf(visRange * cosV);
			float cx = px + ax*hOff, cy = py + ay*hOff, cz = pz + az*hOff;

			for (int s = 0; s < SEGS; ++s) {
				float a0 = (float)s       / (float)SEGS * 6.28318530f;
				float a1 = (float)(s + 1) / (float)SEGS * 6.28318530f;
				float rim0x = cx + (r1x*cosf(a0) + r2x*sinf(a0))*rimR;
				float rim0y = cy + (r1y*cosf(a0) + r2y*sinf(a0))*rimR;
				float rim0z = cz + (r1z*cosf(a0) + r2z*sinf(a0))*rimR;
				float rim1x = cx + (r1x*cosf(a1) + r2x*sinf(a1))*rimR;
				float rim1y = cy + (r1y*cosf(a1) + r2y*sinf(a1))*rimR;
				float rim1z = cz + (r1z*cosf(a1) + r2z*sinf(a1))*rimR;
				// Solid triangle: apex → rim0 → rim1
				emit(px, py, pz); emit(rim0x, rim0y, rim0z); emit(rim1x, rim1y, rim1z);
			}
		}
	}

	g_vk.lidarConeVertexCount = (uint32_t)totalVerts;
}

void vis_set_aabbs(VisState* vs, const std::vector<VisAABB>& aabbs) {
	if (vs) {
		if (aabbs.empty()) {
			vs->sceneBoundsValid = false;
		} else {
			float minx = aabbs[0].aabb[0], miny = aabbs[0].aabb[1], minz = aabbs[0].aabb[2];
			float maxx = aabbs[0].aabb[3], maxy = aabbs[0].aabb[4], maxz = aabbs[0].aabb[5];
			for (const auto& a : aabbs) {
				if (a.aabb[0] < minx) minx = a.aabb[0];
				if (a.aabb[1] < miny) miny = a.aabb[1];
				if (a.aabb[2] < minz) minz = a.aabb[2];
				if (a.aabb[3] > maxx) maxx = a.aabb[3];
				if (a.aabb[4] > maxy) maxy = a.aabb[4];
				if (a.aabb[5] > maxz) maxz = a.aabb[5];
			}
			vs->sceneBounds[0] = minx; vs->sceneBounds[1] = miny; vs->sceneBounds[2] = minz;
			vs->sceneBounds[3] = maxx; vs->sceneBounds[4] = maxy; vs->sceneBounds[5] = maxz;
			vs->sceneBoundsValid = true;
		}
	}
	if (!g_vk.initialized) return;
	const size_t totalVerts = aabbs.size() * 24;
	if (totalVerts == 0) { g_vk.aabbWireVertexCount = 0; return; }
	ensureAabbWireBuffer(totalVerts);
	float* dst = (float*)g_vk.mappedAabbWireData;
	for (const auto& a : aabbs) uploadAABBWireVertices(dst, a);
	g_vk.aabbWireVertexCount = (uint32_t)totalVerts;
}

void vis_update_aabbs(VisState* vs, const std::vector<VisAABB>& aabbs) {
	vis_set_aabbs(vs, aabbs);
}

void vis_destroy(VisState* vs) {
	if (!g_vk.initialized) return;
	vkDeviceWaitIdle(g_vk.device);
	if (g_vk.vertexBuffer != VK_NULL_HANDLE) {
		vkUnmapMemory(g_vk.device, g_vk.vertexMemory);
		vkDestroyBuffer(g_vk.device, g_vk.vertexBuffer, nullptr);
		vkFreeMemory(g_vk.device, g_vk.vertexMemory, nullptr);
	}
	if (g_vk.meshWireBuffer != VK_NULL_HANDLE) {
		vkUnmapMemory(g_vk.device, g_vk.meshWireMemory);
		vkDestroyBuffer(g_vk.device, g_vk.meshWireBuffer, nullptr);
		vkFreeMemory(g_vk.device, g_vk.meshWireMemory, nullptr);
	}
	if (g_vk.meshSolidBuffer != VK_NULL_HANDLE) {
		vkUnmapMemory(g_vk.device, g_vk.meshSolidMemory);
		vkDestroyBuffer(g_vk.device, g_vk.meshSolidBuffer, nullptr);
		vkFreeMemory(g_vk.device, g_vk.meshSolidMemory, nullptr);
	}
	if (g_vk.aabbWireBuffer != VK_NULL_HANDLE) {
		vkUnmapMemory(g_vk.device, g_vk.aabbWireMemory);
		vkDestroyBuffer(g_vk.device, g_vk.aabbWireBuffer, nullptr);
		vkFreeMemory(g_vk.device, g_vk.aabbWireMemory, nullptr);
	}
	if (g_vk.lidarSolidBuffer != VK_NULL_HANDLE) {
		vkUnmapMemory(g_vk.device, g_vk.lidarSolidMemory);
		vkDestroyBuffer(g_vk.device, g_vk.lidarSolidBuffer, nullptr);
		vkFreeMemory(g_vk.device, g_vk.lidarSolidMemory, nullptr);
	}
	if (g_vk.lidarConeBuffer != VK_NULL_HANDLE) {
		vkUnmapMemory(g_vk.device, g_vk.lidarConeMemory);
		vkDestroyBuffer(g_vk.device, g_vk.lidarConeBuffer, nullptr);
		vkFreeMemory(g_vk.device, g_vk.lidarConeMemory, nullptr);
	}
	destroyDepthResources(g_vk.device, g_vk.depthImage, g_vk.depthMemory, g_vk.depthImageView);
	vkDestroyPipeline(g_vk.device, g_vk.conePipeline, nullptr);
	vkDestroyPipeline(g_vk.device, g_vk.meshSolidPipeline, nullptr);
	vkDestroyPipeline(g_vk.device, g_vk.meshWirePipeline, nullptr);
	vkDestroyPipeline(g_vk.device, g_vk.graphicsPipeline, nullptr);
	vkDestroyPipelineLayout(g_vk.device, g_vk.pipelineLayout, nullptr);
	destroyVulkanContext(g_vk);
	if (vs && vs->window) cleanup(vs->window);
	delete vs;
}
