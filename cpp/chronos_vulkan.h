#pragma once

#include "chronos_matmul.h"

#ifdef CHRONOS_USE_VULKAN
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <map>
#include <memory>
#include <iostream>

struct VulkanBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
};

// A singleton class to manage the Vulkan device, queues, and pipelines.
class VulkanManager {
public:
    static VulkanManager& getInstance() {
        static VulkanManager instance;
        return instance;
    }

    // Delete copy and assignment operators
    VulkanManager(const VulkanManager&) = delete;
    void operator=(const VulkanManager&) = delete;

    void execute(
        const void* A_data, size_t A_size,
        const void* B_data, size_t B_size,
        void* Y_data, size_t Y_size,
        ssize_t N, ssize_t K, ssize_t M,
        const std::string& qtype
    );

private:
    VulkanManager();
    ~VulkanManager();
    
    void initVulkan();
    void cleanup();
    
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    VulkanBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    std::vector<char> readShaderFile(const std::string& filename);
    VkShaderModule createShaderModule(const std::string& qtype);

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    uint32_t computeQueueFamilyIndex;

    std::map<std::string, VkShaderModule> shaderModules;
    std::map<std::string, VkPipeline> computePipelines;
    std::map<std::string, VkPipelineLayout> pipelineLayouts;
    std::map<std::string, VkDescriptorSetLayout> descriptorSetLayouts;
};

#endif // CHRONOS_USE_VULKAN
