#include "hierarchos_vulkan.h"

#ifdef HIERARCHOS_USE_VULKAN

#include <vector>
#include <fstream>
#include <map>
#include <cstring>
#include <filesystem>
#include <cmath> // For std::ceil

// Simple helper for error checking
#define VK_CHECK(x)                                                 \
    do {                                                            \
        VkResult err = x;                                           \
        if (err) {                                                  \
            throw std::runtime_error("Vulkan Error: " + std::to_string(err)); \
        }                                                           \
    } while (0)

// Push constants for shader
struct PushConstantData {
    int K;
    int M;
    int N;
};

// ---- Main Entry Point ----
void matmul_quantized_vulkan(const void* B_quantized, const float* A, float* Y,
                             ssize_t N, ssize_t K, ssize_t M, const std::string& qtype) {
    VulkanManager& vk = VulkanManager::getInstance();
    
    size_t A_size = (size_t)N * K * sizeof(float);

    size_t B_size = 0;
    if (qtype == "INT4") B_size = (size_t)M * (K / Q_BLOCK_SIZE_INT4) * sizeof(block_int4);
    else if (qtype == "Q4_0") B_size = (size_t)M * (K / Q_BLOCK_SIZE_Q4_0) * sizeof(block_q4_0);
    else if (qtype == "Q8_0") B_size = (size_t)M * (K / Q_BLOCK_SIZE_Q8_0) * sizeof(block_q8_0);
    else if (qtype == "Q2_K") B_size = (size_t)M * (K / QK_K) * sizeof(block_q2_k);
    else throw std::runtime_error("Unsupported qtype for Vulkan matmul: " + qtype);

    size_t Y_size = (size_t)N * M * sizeof(float);

    vk.execute(A, A_size, B_quantized, B_size, Y, Y_size, N, K, M, qtype);
}

// ---- VulkanManager Implementation ----

VulkanManager::VulkanManager() {
    initVulkan();
}

VulkanManager::~VulkanManager() {
    cleanup();
}

void VulkanManager::initVulkan() {
    // 1. Create Instance
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "hierarchos Matmul";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));

    // 2. Select Physical Device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    
    // Prefer discrete GPU, but any will do
    for (const auto& d : devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(d, &properties);
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physicalDevice = d;
            break;
        }
    }
    if (physicalDevice == VK_NULL_HANDLE) {
        physicalDevice = devices[0];
    }

    // 3. Find Compute Queue Family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    bool found = false;
    for (uint32_t i = 0; i < queueFamilies.size(); ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeQueueFamilyIndex = i;
            found = true;
            break;
        }
    }
    if (!found) {
        throw std::runtime_error("Failed to find a compute queue family!");
    }

    // 4. Create Logical Device
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures deviceFeatures{};
    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
    VK_CHECK(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);

    // 5. Create Command Pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));
}

void VulkanManager::cleanup() {
    if (device) {
        vkDeviceWaitIdle(device);
        for (auto const& [key, val] : computePipelines) vkDestroyPipeline(device, val, nullptr);
        for (auto const& [key, val] : pipelineLayouts) vkDestroyPipelineLayout(device, val, nullptr);
        for (auto const& [key, val] : descriptorSetLayouts) vkDestroyDescriptorSetLayout(device, val, nullptr);
        for (auto const& [key, val] : shaderModules) vkDestroyShaderModule(device, val, nullptr);
        if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
    }
    if (instance) vkDestroyInstance(instance, nullptr);
}

uint32_t VulkanManager::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

VulkanBuffer VulkanManager::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
    VulkanBuffer buffer;
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer.buffer));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer.buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
    VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &buffer.memory));

    vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0);
    return buffer;
}

void VulkanManager::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

std::vector<char> VulkanManager::readShaderFile(const std::string& qtype) {
    std::string filename;
    if (qtype == "INT4") filename = "shaders/INT4.spv";
    else if (qtype == "Q4_0") filename = "shaders/Q4_0.spv";
    else if (qtype == "Q8_0") filename = "shaders/Q8_0.spv";
    else if (qtype == "Q2_K") filename = "shaders/Q2_K.spv";
    else throw std::runtime_error("Invalid qtype for shader path");
    
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open shader file: " + filename);
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

VkShaderModule VulkanManager::createShaderModule(const std::string& qtype) {
    if (shaderModules.count(qtype)) {
        return shaderModules[qtype];
    }
    
    std::vector<char> code = readShaderFile(qtype);
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));
    shaderModules[qtype] = shaderModule;
    return shaderModule;
}

void VulkanManager::execute(
    const void* A_data, size_t A_size,
    const void* B_data, size_t B_size,
    void* Y_data, size_t Y_size,
    ssize_t N, ssize_t K, ssize_t M,
    const std::string& qtype
) {
    // 1. Create Buffers (Staging and Device)
    VulkanBuffer stagingBufferA = createBuffer(A_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer stagingBufferB = createBuffer(B_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VulkanBuffer stagingBufferY = createBuffer(Y_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VulkanBuffer deviceBufferA = createBuffer(A_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VulkanBuffer deviceBufferB = createBuffer(B_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VulkanBuffer deviceBufferY = createBuffer(Y_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // 2. Copy data to staging buffers
    void* data;
    vkMapMemory(device, stagingBufferA.memory, 0, A_size, 0, &data);
    memcpy(data, A_data, A_size);
    vkUnmapMemory(device, stagingBufferA.memory);

    vkMapMemory(device, stagingBufferB.memory, 0, B_size, 0, &data);
    memcpy(data, B_data, B_size);
    vkUnmapMemory(device, stagingBufferB.memory);

    // 3. Copy from staging to device buffers
    copyBuffer(stagingBufferA.buffer, deviceBufferA.buffer, A_size);
    copyBuffer(stagingBufferB.buffer, deviceBufferB.buffer, B_size);
    
    // 4. Create Pipeline if it doesn't exist
    if (computePipelines.find(qtype) == computePipelines.end()) {
        VkDescriptorSetLayoutBinding bindings[3];
        for(int i=0; i<3; ++i) {
            bindings[i] = {};
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 3;
        layoutInfo.pBindings = bindings;
        
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayouts[qtype]));

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(PushConstantData);
        
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayouts[qtype];
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

        VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayouts[qtype]));

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = pipelineLayouts[qtype];
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = createShaderModule(qtype);
        pipelineInfo.stage.pName = "main";

        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipelines[qtype]));

    }
    
    // 5. Descriptor Sets
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    VkDescriptorPool descriptorPool;
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));

    VkDescriptorSetLayout setLayout = descriptorSetLayouts[qtype];
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &setLayout;

    VkDescriptorSet descriptorSet;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
    
    VkDescriptorBufferInfo bufferInfos[3];
    bufferInfos[0] = {};
    bufferInfos[0].buffer = deviceBufferA.buffer;
    bufferInfos[0].range = VK_WHOLE_SIZE;
    bufferInfos[1] = {};
    bufferInfos[1].buffer = deviceBufferB.buffer;
    bufferInfos[1].range = VK_WHOLE_SIZE;
    bufferInfos[2] = {};
    bufferInfos[2].buffer = deviceBufferY.buffer;
    bufferInfos[2].range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet descriptorWrites[3];
    for(int i=0; i<3; ++i) {
        descriptorWrites[i] = {};
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = descriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }
    vkUpdateDescriptorSets(device, 3, descriptorWrites, 0, nullptr);

    // 6. Record and Submit Command Buffer
    VkCommandBufferAllocateInfo cmdBufAllocInfo{};
    cmdBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocInfo.commandPool = commandPool;
    cmdBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    VK_CHECK(vkAllocateCommandBuffers(device, &cmdBufAllocInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
    
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelines[qtype]);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayouts[qtype], 0, 1, &descriptorSet, 0, nullptr);

    PushConstantData pc = { (int)K, (int)M, (int)N };
    vkCmdPushConstants(commandBuffer, pipelineLayouts[qtype], VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantData), &pc);

    // Dispatch compute shaders. Workgroup size is 16x16 in the shaders.
    vkCmdDispatch(commandBuffer, (uint32_t)std::ceil(N / 16.0), (uint32_t)std::ceil(M / 16.0), 1);
    
    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    VkFence fence;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence));
    
    VK_CHECK(vkQueueSubmit(computeQueue, 1, &submitInfo, fence));
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    // 7. Copy result back
    copyBuffer(deviceBufferY.buffer, stagingBufferY.buffer, Y_size);
    
    // 8. Map staging buffer and copy to output array
    vkMapMemory(device, stagingBufferY.memory, 0, Y_size, 0, &data);
    memcpy(Y_data, data, Y_size);
    vkUnmapMemory(device, stagingBufferY.memory);

    // 9. Cleanup
    vkDestroyFence(device, fence, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    
    vkDestroyBuffer(device, deviceBufferA.buffer, nullptr);
    vkFreeMemory(device, deviceBufferA.memory, nullptr);
    vkDestroyBuffer(device, deviceBufferB.buffer, nullptr);
    vkFreeMemory(device, deviceBufferB.memory, nullptr);
    vkDestroyBuffer(device, deviceBufferY.buffer, nullptr);
    vkFreeMemory(device, deviceBufferY.memory, nullptr);
    
    vkDestroyBuffer(device, stagingBufferA.buffer, nullptr);
    vkFreeMemory(device, stagingBufferA.memory, nullptr);
    vkDestroyBuffer(device, stagingBufferB.buffer, nullptr);
    vkFreeMemory(device, stagingBufferB.memory, nullptr);
    vkDestroyBuffer(device, stagingBufferY.buffer, nullptr);
    vkFreeMemory(device, stagingBufferY.memory, nullptr);
}

#endif // HIERARCHOS_USE_VULKAN
