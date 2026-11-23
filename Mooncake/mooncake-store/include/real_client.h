#pragma once

#include <atomic>
#include <boost/lockfree/queue.hpp>
#include <csignal>
#include <memory>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pyclient.h"
#include "client_service.h"
#include "client_buffer.hpp"
#include "mutex.h"
#include "utils.h"
#include "rpc_types.h"
#include <ylt/coro_http/coro_http_server.hpp>
#include <ylt/coro_rpc/coro_rpc_server.hpp>

namespace mooncake {

class RealClient;

// Global resource tracker to handle cleanup on abnormal termination
class ResourceTracker {
   public:
    // Get the singleton instance
    static ResourceTracker &getInstance();

    // Register a DistributedObjectStore instance for cleanup
    void registerInstance(const std::shared_ptr<PyClient> &instance);

   private:
    ResourceTracker();
    ~ResourceTracker();

    // Prevent copying
    ResourceTracker(const ResourceTracker &) = delete;
    ResourceTracker &operator=(const ResourceTracker &) = delete;

    // Cleanup all registered resources
    void cleanupAllResources();

    // Signal handler function
    static void signalHandler(int signal);

    // Exit handler function
    static void exitHandler();

    Mutex mutex_;
    std::vector<std::weak_ptr<PyClient>> instances_ GUARDED_BY(mutex_);

    // Ensure cleanup runs at most once
    std::atomic<bool> cleaned_{false};

    // Dedicated signal handling thread
    void startSignalThread();
    std::once_flag signal_once_{};
    std::jthread signal_thread_{};  // joins on destruction
};

class RealClient : public PyClient {
   public:
    RealClient();
    ~RealClient();

    // Factory to create shared instances and auto-register to ResourceTracker
    static std::shared_ptr<RealClient> create();

    int setup_real(
        const std::string &local_hostname, const std::string &metadata_server,
        size_t global_segment_size = 1024 * 1024 * 16,
        size_t local_buffer_size = 1024 * 1024 * 16,
        const std::string &protocol = "tcp",
        const std::string &rdma_devices = "",
        const std::string &master_server_addr = "127.0.0.1:50051",
        const std::shared_ptr<TransferEngine> &transfer_engine = nullptr,
        const std::string &ipc_socket_path = "",
        bool enable_ssd_offload = false,
        const std::string &ssd_offload_path = "");

    int setup_dummy(size_t mem_pool_size, size_t local_buffer_size,
                    const std::string &server_address,
                    const std::string &ipc_socket_path) {
        // Real client does not support dummy setup
        return -1;
    };

    int initAll(const std::string &protocol, const std::string &device_name,
                size_t mount_segment_size = 1024 * 1024 * 16);  // Default 16MB

    uint64_t alloc_from_mem_pool(size_t size) { return 0; };

    int put(const std::string &key, std::span<const char> value,
            const ReplicateConfig &config = ReplicateConfig{});

    int register_buffer(void *buffer, size_t size);

    int unregister_buffer(void *buffer);

    /**
     * @brief Get object data directly into a pre-allocated buffer
     * @param key Key of the object to get
     * @param buffer Pointer to the pre-allocated buffer (must be registered
     * with register_buffer)
     * @param size Size of the buffer
     * @return Number of bytes read on success, negative value on error
     * @note The buffer address must be previously registered with
     * register_buffer() for zero-copy operations
     */
    int64_t get_into(const std::string &key, void *buffer, size_t size);

    std::vector<std::vector<std::vector<int64_t>>> get_into_ranges(
        const std::vector<void *> &buffers,
        const std::vector<std::vector<std::string>> &all_keys,
        const std::vector<std::vector<std::vector<size_t>>> &all_dst_offsets,
        const std::vector<std::vector<std::vector<size_t>>> &all_src_offsets,
        const std::vector<std::vector<std::vector<size_t>>> &all_sizes)
        override;

    /**
     * @brief Get object data directly into pre-allocated buffers for multiple
     * keys (batch version)
     * @param keys Vector of keys of the objects to get
     * @param buffers Vector of pointers to the pre-allocated buffers
     * @param sizes Vector of sizes of the buffers
     * @return Vector of 64-bit integers, where each element is the number of
     * bytes read on success, or a negative value on error
     * @note The buffer addresses must be previously registered with
     * register_buffer() for zero-copy operations
     */
    std::vector<int64_t> batch_get_into(const std::vector<std::string> &keys,
                                        const std::vector<void *> &buffers,
                                        const std::vector<size_t> &sizes);

    /**
     * @brief Get object data directly into pre-allocated buffers for multiple
     * keys
     * @param keys Vector of keys of the objects to get
     * @param all_buffers Vector of vectors of pointers to the pre-allocated
     * buffers
     * @param all_sizes Vector of vectors of sizes of the buffers
     * @return Vector of integers, where each element is the number of bytes
     * read on success, or a negative value on error
     * @note The buffer addresses must be previously registered with
     * register_buffer() for zero-copy operations
     */
    std::vector<int> batch_get_into_multi_buffers(
        const std::vector<std::string> &keys,
        const std::vector<std::vector<void *>> &all_buffers,
        const std::vector<std::vector<size_t>> &all_sizes,
        bool prefer_same_node);

    /**
     * @brief Put object data directly from a pre-allocated buffer
     * @param key Key of the object to put
     * @param buffer Pointer to the buffer containing data (must be registered
     * with register_buffer)
     * @param size Size of the data to put
     * @return 0 on success, negative value on error
     * @note The buffer address must be previously registered with
     * register_buffer() for zero-copy operations
     */
    int put_from(const std::string &key, void *buffer, size_t size,
                 const ReplicateConfig &config = ReplicateConfig{});

    /**
     * @brief Put object data directly from pre-allocated buffers for multiple
     * keys(metadata version, better not be directly used in Python)
     * @param keys Vector of keys of the objects to put
     * @param buffers Vector of pointers to the pre-allocated buffers
     * @param metadata_buffers Vector of pointers to the pre-allocated metadata
     * buffers
     * @param size Number of sizes of the buffers
     * @param metadata_size Number of sizes of the metadata buffers
     * @param config Replication configuration
     * @return Vector of integers, where each element is 0 on success, or a
     * negative value on error
     * @note The buffer addresses must be previously registered with
     * register_buffer() for zero-copy operations
     */
    int put_from_with_metadata(
        const std::string &key, void *buffer, void *metadata_buffer,
        size_t size, size_t metadata_size,
        const ReplicateConfig &config = ReplicateConfig{});

    /**
     * @brief Put object data directly from pre-allocated buffers for multiple
     * keys (batch version)
     * @param keys Vector of keys of the objects to put
     * @param buffers Vector of pointers to the pre-allocated buffers
     * @param sizes Vector of sizes of the buffers
     * @param config Replication configuration
     * @return Vector of integers, where each element is 0 on success, or a
     * negative value on error
     * @note The buffer addresses must be previously registered with
     * register_buffer() for zero-copy operations
     */

    std::vector<int> batch_put_from(
        const std::vector<std::string> &keys,
        const std::vector<void *> &buffers, const std::vector<size_t> &sizes,
        const ReplicateConfig &config = ReplicateConfig{});

    /**
     * @brief Put object data directly from multiple pre-allocated buffers for
     * multiple keys (batch version)
     * @param keys Vector of keys of the objects to put
     * @param all_buffers Vector of vectors of pointers to the multiple
     * pre-allocated buffers
     * @param all_sizes Vector of vectors of sizes of the multiple buffers
     * @param config Replication configuration
     * @return Vector of integers, where each element is 0 on success, or a
     * negative value on error
     * @note The buffer addresses must be previously registered with
     * register_buffer() for zero-copy operations
     */
    std::vector<int> batch_put_from_multi_buffers(
        const std::vector<std::string> &keys,
        const std::vector<std::vector<void *>> &all_buffers,
        const std::vector<std::vector<size_t>> &all_sizes,
        const ReplicateConfig &config = ReplicateConfig{});

    int put_parts(const std::string &key,
                  std::vector<std::span<const char>> values,
                  const ReplicateConfig &config = ReplicateConfig{});

    int put_batch(const std::vector<std::string> &keys,
                  const std::vector<std::span<const char>> &values,
                  const ReplicateConfig &config = ReplicateConfig{});

    int upsert(const std::string &key, std::span<const char> value,
               const ReplicateConfig &config = ReplicateConfig{});

    int upsert_from(const std::string &key, void *buffer, size_t size,
                    const ReplicateConfig &config = ReplicateConfig{});

    std::vector<int> batch_upsert_from(
        const std::vector<std::string> &keys,
        const std::vector<void *> &buffers, const std::vector<size_t> &sizes,
        const ReplicateConfig &config = ReplicateConfig{});

    int upsert_parts(const std::string &key,
                     std::vector<std::span<const char>> values,
                     const ReplicateConfig &config = ReplicateConfig{});

    int upsert_batch(const std::vector<std::string> &keys,
                     const std::vector<std::span<const char>> &values,
                     const ReplicateConfig &config = ReplicateConfig{});

    [[nodiscard]] std::string get_hostname() const;

    /**
     * @brief Get a buffer containing the data for a key
     * @param key Key to get data for
     * @return std::shared_ptr<BufferHandle> Buffer containing the data, or
     * nullptr if error
     */
    std::shared_ptr<BufferHandle> get_buffer(const std::string &key);

    /**
     * @brief Get buffers containing the data for multiple keys (batch version)
     * @param keys Vector of keys to get data for
     * @return Vector of std::shared_ptr<BufferHandle> buffers containing the
     * data, or nullptr for each key if error
     */
    std::vector<std::shared_ptr<BufferHandle>> batch_get_buffer(
        const std::vector<std::string> &keys);

    int remove(const std::string &key, bool force = false);

    long removeByRegex(const std::string &str, bool force = false);

    long removeAll(bool force = false);

    std::vector<int> batchRemove(const std::vector<std::string> &keys,
                                 bool force = false);

    int tearDownAll();

    int health_check() override;

    /**
     * @brief Check if an object exists
     * @param key Key to check
     * @return 1 if exists, 0 if not exists, -1 if error
     */
    int isExist(const std::string &key);

    /**
     * @brief Check if multiple objects exist
     * @param keys Vector of keys to check
     * @return Vector of existence results: 1 if exists, 0 if not exists, -1 if
     * error
     */
    std::vector<int> batchIsExist(const std::vector<std::string> &keys);

    /**
     * @brief Get the size of an object
     * @param key Key of the object
     * @return Size of the object in bytes, or -1 if error or object doesn't
     * exist
     */
    int64_t getSize(const std::string &key);

    /**
     * @brief Create a copy task to replicate an object's data to target
     * segments
     * @param key Object key
     * @param targets Target segments
     * @return tl::expected<UUID, ErrorCode> Task ID on success, ErrorCode on
     * failure
     */
    tl::expected<UUID, ErrorCode> create_copy_task(
        const std::string &key, const std::vector<std::string> &targets);

    /**
     * @brief Create a move task to move an object's replica from source segment
     * to target segment
     * @param key Object key
     * @param source Source segment
     * @param target Target segment
     * @return tl::expected<UUID, ErrorCode> Task ID on success, ErrorCode on
     * failure
     */
    tl::expected<UUID, ErrorCode> create_move_task(const std::string &key,
                                                   const std::string &source,
                                                   const std::string &target);

    /**
     * @brief Query a task by task id
     * @param task_id Task ID to query
     * @return tl::expected<QueryTaskResponse, ErrorCode> Task basic info
     * on success, ErrorCode on failure
     */
    tl::expected<QueryTaskResponse, ErrorCode> query_task(const UUID &task_id);

    // Hot cache acquire: returns (offset, size) within the shm region.
    // Increments ref_count to prevent eviction.
    tl::expected<std::tuple<uint64_t, size_t>, ErrorCode> acquire_hot_cache(
        const std::string &key);

    // Hot cache release: decrements ref_count after dummy is done reading.
    tl::expected<void, ErrorCode> release_hot_cache(const std::string &key);

    // Batch hot cache acquire: returns vector of (offset, size).
    // SIZE_MAX offset indicates cache miss for that key.
    std::vector<tl::expected<std::tuple<uint64_t, size_t>, ErrorCode>>
    batch_acquire_hot_cache(const std::vector<std::string> &keys);

    // Batch hot cache release.
    tl::expected<void, ErrorCode> batch_release_hot_cache(
        const std::vector<std::string> &keys);

    // Allocator-backed buffer acquire: allocates + fills, keeps handle alive.
    // Returns (dummy_addr, size).
    tl::expected<std::tuple<uint64_t, size_t>, ErrorCode> acquire_buffer_dummy(
        const std::string &key, const UUID &client_id);

    // Allocator-backed buffer release: frees the handle held by acquire.
    tl::expected<void, ErrorCode> release_buffer_dummy(uint64_t dummy_addr,
                                                       const UUID &client_id);

    // Batch allocator-backed buffer acquire.
    // Returns vector of (dummy_addr, size) per key.
    std::vector<tl::expected<std::tuple<uint64_t, size_t>, ErrorCode>>
    batch_acquire_buffer_dummy(const std::vector<std::string> &keys,
                               const UUID &client_id);

    tl::expected<void, ErrorCode> put_dummy_helper(
        const std::string &key, std::span<const char> value,
        const ReplicateConfig &config, const UUID &client_id);

    tl::expected<void, ErrorCode> upsert_dummy_helper(
        const std::string &key, std::span<const char> value,
        const ReplicateConfig &config, const UUID &client_id);

    tl::expected<void, ErrorCode> upsert_parts_dummy_helper(
        const std::string &key, std::vector<std::span<const char>> values,
        const ReplicateConfig &config, const UUID &client_id);

    tl::expected<void, ErrorCode> upsert_from_dummy_helper(
        const std::string &key, uint64_t dummy_buffer, size_t size,
        const ReplicateConfig &config, const UUID &client_id);

    std::vector<tl::expected<void, ErrorCode>> batch_upsert_from_dummy_helper(
        const std::vector<std::string> &keys,
        const std::vector<uint64_t> &dummy_buffers,
        const std::vector<size_t> &sizes, const ReplicateConfig &config,
        const UUID &client_id);

    tl::expected<void, ErrorCode> upsert_batch_dummy_helper(
        const std::vector<std::string> &keys,
        const std::vector<std::span<const char>> &values,
        const ReplicateConfig &config, const UUID &client_id);

    tl::expected<void, ErrorCode> put_batch_dummy_helper(
        const std::vector<std::string> &keys,
        const std::vector<std::span<const char>> &values,
        const ReplicateConfig &config, const UUID &client_id);

    tl::expected<void, ErrorCode> put_parts_dummy_helper(
        const std::string &key, std::vector<std::span<const char>> values,
        const ReplicateConfig &config, const UUID &client_id);

    std::vector<tl::expected<int64_t, ErrorCode>> batch_get_into_dummy_helper(
        const std::vector<std::string> &keys,
        const std::vector<uint64_t> &buffers, const std::vector<size_t> &sizes,
        int32_t device_id, const UUID &client_id);

    std::vector<tl::expected<void, ErrorCode>> batch_put_from_dummy_helper(
        const std::vector<std::string> &keys,
        const std::vector<uint64_t> &dummy_buffers,
        const std::vector<size_t> &sizes, const ReplicateConfig &config,
        int32_t device_id, const UUID &client_id);

    std::vector<tl::expected<void, ErrorCode>>
    batch_put_from_multi_buffers_dummy_helper(
        const std::vector<std::string> &keys,
        const std::vector<std::vector<uint64_t>> &dummy_all_buffers,
        const std::vector<std::vector<size_t>> &all_sizes,
        const ReplicateConfig &config, int32_t device_id,
        const UUID &client_id);

    std::vector<tl::expected<int64_t, ErrorCode>>
    batch_get_into_multi_buffers_dummy_helper(
        const std::vector<std::string> &keys,
        const std::vector<std::vector<uint64_t>> &dummy_all_buffers,
        const std::vector<std::vector<size_t>> &all_sizes,
        bool prefer_alloc_in_same_node, int32_t device_id,
        const UUID &client_id);

    tl::expected<int64_t, ErrorCode> get_into_range_shm_helper(
        const std::string &key, uint64_t buffer, size_t dst_offset,
        size_t src_offset, size_t size, const UUID &client_id);

    std::vector<std::vector<std::vector<tl::expected<int64_t, ErrorCode>>>>
    get_into_ranges_shm_helper(
        const std::vector<uint64_t> &dummy_buffers,
        const std::vector<std::vector<std::string>> &all_keys,
        const std::vector<std::vector<std::vector<size_t>>> &all_dst_offsets,
        const std::vector<std::vector<std::vector<size_t>>> &all_src_offsets,
        const std::vector<std::vector<std::vector<size_t>>> &all_sizes,
        int32_t device_id, const UUID &client_id);

    // Share mem management for dummy client
    // Modified: map_shm_internal now takes fd instead of just name
    tl::expected<void, ErrorCode> map_shm_internal(int fd,
                                                   uint64_t shm_base_addr,
                                                   size_t shm_size,
                                                   bool is_local_buffer,
                                                   const UUID &client_id);

    tl::expected<void, ErrorCode> unmap_shm_internal(const UUID &client_id);

    tl::expected<void, ErrorCode> ascend_shm_internal(
        uint64_t dummy_base_addr, size_t vmm_size, bool is_local_buffer,
        const std::string &shareable_handle_bytes, int32_t device_id,
        const UUID &client_id);

    tl::expected<void, ErrorCode> ascend_ipc_shm_internal(
        uint64_t dummy_base_addr, size_t mem_size, bool is_local_buffer,
        const std::string &ipc_key_bytes, int32_t device_id,
        const UUID &client_id);

    tl::expected<void, ErrorCode> ascend_unmap_shm_internal(
        const UUID &client_id);

    tl::expected<void, ErrorCode> unregister_shm_buffer_internal(
        uint64_t dummy_base_addr, const UUID &client_id);

    // Internal versions that return tl::expected
    tl::expected<void, ErrorCode> service_ready_internal() { return {}; }

    tl::expected<void, ErrorCode> setup_internal(
        const std::string &local_hostname, const std::string &metadata_server,
        size_t global_segment_size = 1024 * 1024 * 16,
        size_t local_buffer_size = 1024 * 1024 * 16,
        const std::string &protocol = "tcp",
        const std::string &rdma_devices = "",
        const std::string &master_server_addr = "127.0.0.1:50051",
        const std::shared_ptr<TransferEngine> &transfer_engine = nullptr,
        const std::string &ipc_socket_path = "", int local_rpc_port = 50052,
        bool enable_ssd_offload = false, bool start_offload_rpc_server = false,
        const std::string &ssd_offload_path = "");

    // Overload that accepts a configuration dictionary
    tl::expected<void, ErrorCode> setup_internal(const ConfigDict &config);

    tl::expected<void, ErrorCode> initAll_internal(
        const std::string &protocol, const std::string &device_name,
        size_t mount_segment_size = 1024 * 1024 * 16);

    tl::expected<void, ErrorCode> unregister_buffer_internal(void *buffer);

    tl::expected<void, ErrorCode> put_internal(
        const std::string &key, std::span<const char> value,
        const ReplicateConfig &config = ReplicateConfig{},
        const std::shared_ptr<ClientBufferAllocator> &client_buffer_allocator =
            nullptr);

    tl::expected<void, ErrorCode> register_buffer_internal(void *buffer,
                                                           size_t size);

    struct RangedReadMetadata {
        QueryResult query_result;
        Replica::Descriptor replica;
        uint64_t total_size;
    };

    tl::expected<RangedReadMetadata, ErrorCode> resolve_ranged_read_metadata(
        const std::string &key);

    tl::expected<int64_t, ErrorCode> execute_ranged_read(
        const std::string &key, void *buffer, size_t dst_offset,
        size_t src_offset, size_t size, const RangedReadMetadata &metadata,
        bool size_is_buffer_capacity = false);

    tl::expected<int64_t, ErrorCode> get_into_range_internal(
        const std::string &key, void *buffer, size_t dst_offset,
        size_t src_offset, size_t size, bool size_is_buffer_capacity = false);

    std::vector<std::vector<std::vector<tl::expected<int64_t, ErrorCode>>>>
    get_into_ranges_internal(
        const std::vector<void *> &buffers,
        const std::vector<std::vector<std::string>> &all_keys,
        const std::vector<std::vector<std::vector<size_t>>> &all_dst_offsets,
        const std::vector<std::vector<std::vector<size_t>>> &all_src_offsets,
        const std::vector<std::vector<std::vector<size_t>>> &all_sizes,
        const std::vector<size_t> *buffer_capacities = nullptr,
        std::vector<std::vector<std::vector<tl::expected<int64_t, ErrorCode>>>>
            *prepared_results = nullptr,
        const std::vector<std::vector<std::vector<bool>>> *valid_fragments =
            nullptr);

    std::vector<tl::expected<int64_t, ErrorCode>> batch_get_into_internal(
        const std::vector<std::string> &keys,
        const std::vector<void *> &buffers, const std::vector<size_t> &sizes);

    std::vector<tl::expected<int64_t, ErrorCode>>
    batch_get_into_multi_buffers_internal(
        const std::vector<std::string> &keys,
        const std::vector<std::vector<void *>> &all_buffers,
        const std::vector<std::vector<size_t>> &all_sizes,
        bool prefer_same_node);

    tl::expected<void, ErrorCode> put_from_internal(
        const std::string &key, void *buffer, size_t size,
        const ReplicateConfig &config = ReplicateConfig{});

    tl::expected<void, ErrorCode> upsert_internal(
        const std::string &key, std::span<const char> value,
        const ReplicateConfig &config = ReplicateConfig{},
        std::shared_ptr<ClientBufferAllocator> client_buffer_allocator =
            nullptr);

    tl::expected<void, ErrorCode> upsert_from_internal(
        const std::string &key, void *buffer, size_t size,
        const ReplicateConfig &config = ReplicateConfig{});

    std::vector<tl::expected<void, ErrorCode>> batch_upsert_from_internal(
        const std::vector<std::string> &keys,
        const std::vector<void *> &buffers, const std::vector<size_t> &sizes,
        const ReplicateConfig &config = ReplicateConfig{});

    tl::expected<void, ErrorCode> upsert_parts_internal(
        const std::string &key, std::vector<std::span<const char>> values,
        const ReplicateConfig &config = ReplicateConfig{},
        std::shared_ptr<ClientBufferAllocator> client_buffer_allocator =
            nullptr);

    tl::expected<void, ErrorCode> upsert_batch_internal(
        const std::vector<std::string> &keys,
        const std::vector<std::span<const char>> &values,
        const ReplicateConfig &config = ReplicateConfig{},
        std::shared_ptr<ClientBufferAllocator> client_buffer_allocator =
            nullptr);

    std::vector<tl::expected<void, ErrorCode>> batch_put_from_internal(
        const std::vector<std::string> &keys,
        const std::vector<void *> &buffers, const std::vector<size_t> &sizes,
        const ReplicateConfig &config = ReplicateConfig{});

    std::vector<tl::expected<void, ErrorCode>>
    batch_put_from_multi_buffers_internal(
        const std::vector<std::string> &keys,
        const std::vector<std::vector<void *>> &all_buffers,
        const std::vector<std::vector<size_t>> &all_sizes,
        const ReplicateConfig &config = ReplicateConfig{});

    tl::expected<void, ErrorCode> put_parts_internal(
        const std::string &key,
        const std::vector<std::span<const char>> &values,
        const ReplicateConfig &config = ReplicateConfig{},
        const std::shared_ptr<ClientBufferAllocator> &client_buffer_allocator =
            nullptr);

    tl::expected<void, ErrorCode> put_batch_internal(
        const std::vector<std::string> &keys,
        const std::vector<std::span<const char>> &values,
        const ReplicateConfig &config = ReplicateConfig{},
        const std::shared_ptr<ClientBufferAllocator> &client_buffer_allocator =
            nullptr);

    tl::expected<void, ErrorCode> remove_internal(const std::string &key,
                                                  bool force = false);

    tl::expected<long, ErrorCode> removeByRegex_internal(const std::string &str,
                                                         bool force = false);

    tl::expected<int64_t, ErrorCode> removeAll_internal(bool force = false);

    std::vector<tl::expected<void, ErrorCode>> batchRemove_internal(
        const std::vector<std::string> &keys, bool force = false);

    tl::expected<void, ErrorCode> tearDownAll_internal();

    tl::expected<bool, ErrorCode> isExist_internal(const std::string &key);

    std::vector<tl::expected<bool, ErrorCode>> batchIsExist_internal(
        const std::vector<std::string> &keys);

    tl::expected<int64_t, ErrorCode> getSize_internal(const std::string &key);

    std::shared_ptr<BufferHandle> get_buffer_internal(
        const std::string &key,
        const std::shared_ptr<ClientBufferAllocator> &client_buffer_allocator =
            nullptr);

    std::vector<std::shared_ptr<BufferHandle>> batch_get_buffer_internal(
        const std::vector<std::string> &keys,
        const std::shared_ptr<ClientBufferAllocator> &client_buffer_allocator =
            nullptr);

    std::map<std::string, std::vector<Replica::Descriptor>>
    batch_get_replica_desc(const std::vector<std::string> &keys);
    std::vector<Replica::Descriptor> get_replica_desc(const std::string &key);

    /**
     * @brief Maps the shared-memory segments of the given same-host peers into
     * this process, so later loads can address peer KV directly.
     *
     * Intended to be called once during start-up, after every peer has mounted
     * its segments. Establishing the mapping eagerly keeps shm_open/mmap and
     * the CUDA host registration off the per-request path.
     *
     * @param peer_endpoints Peer segment names (not shm names); the master
     *        resolves them to shared-memory segments.
     * @return Number of segments newly mapped, or -1 on failure.
     */
    int preload_shared_memory_segments(
        const std::vector<std::string> &peer_endpoints);

    /**
     * @brief KVPilot: where each key sits in this client's local SSD files,
     * as (data file path, value offset, value size); empty path if absent or
     * SSD offload is off. See FileStorage::BatchLocate.
     */
    std::vector<std::tuple<std::string, int64_t, int64_t>> batch_locate_local_ssd(
        const std::vector<std::string> &keys);

    /**
     * @brief KVPilot exclusive placement: drop keys whose chunk left SSD from
     * this client's SSD, freeing buckets left empty. See
     * FileStorage::BatchRemoveKeys.
     * @return Number of keys this client held, or -1 without SSD offload.
     */
    int64_t batch_remove_local_disk(const std::vector<std::string> &keys);

    /**
     * @brief KVPilot exclusive placement: write values from host memory (the
     * caller's own staging buffer, not the segment) to this client's SSD as
     * one bucket and register them with the master. See
     * FileStorage::OffloadHostBuffers.
     * @return Number of keys written, or -1 on failure.
     */
    int64_t batch_offload_local(const std::vector<std::string> &keys,
                                const std::vector<uintptr_t> &ptrs,
                                const std::vector<size_t> &sizes);

    /**
     * @brief KVPilot: set chunk groups' (the chunk-hash field of LMCache keys)
     * tier and penalty (ms) on the master.
     * @return Number of groups updated, or -1 on failure.
     */
    int64_t batch_set_group_placement(const std::vector<std::string> &groups,
                                      const std::vector<int32_t> &tiers,
                                      const std::vector<double> &penalties_ms);

    /**
     * @brief KVPilot: drop refcnt pins from GetReplicaList / batch_get_replica_desc
     * under priority eviction. Call when zero-copy views are done.
     * @return Number of keys unpinned, or -1 on failure.
     */
    int64_t batch_unpin_memory_replicas(const std::vector<std::string> &keys);

    /**
     * @brief KVPilot zero-copy write-back: PutStart for keys whose memory
     * replica the caller fills itself. Per key, the host address of the new
     * replica in this client's own segment, or 0 when the key already has a
     * memory replica, allocation failed, or the replica landed outside this
     * client's segment (those are revoked here). Follow with batch_put_end
     * once the buffers are written, or batch_put_revoke.
     */
    std::vector<uint64_t> batch_put_start_local(
        const std::vector<std::string> &keys,
        const std::vector<uint64_t> &sizes, const ReplicateConfig &config);
    /** Per key, 0 or the error code. */
    std::vector<int> batch_put_end(const std::vector<std::string> &keys);
    std::vector<int> batch_put_revoke(const std::vector<std::string> &keys);

    std::vector<std::string> batch_replica_clear(
        const std::vector<std::string> &keys,
        const std::string &segment_name = "") override;

    tl::expected<PingResponse, ErrorCode> ping(const UUID &client_id);

    tl::expected<BatchGetOffloadObjectResponse, ErrorCode>
    batch_get_offload_object(const std::vector<std::string> &keys,
                             const std::vector<int64_t> &sizes);

    /**
     * @brief Releases buffer associated with a specific batch_id.
     * Called by remote client after transfer completion.
     * @param batch_id The unique identifier of the batch to release
     * @return true if batch was found and released, false otherwise
     */
    bool release_offload_buffer(uint64_t batch_id);

    /**
     * @brief Retrieves multiple stored objects from a remote service.
     * @param target_rpc_service_addr Address of the remote RPC service (e.g.,
     "ip:port").

     */
    tl::expected<void, ErrorCode> batch_get_into_offload_object_internal(
        const std::string &target_rpc_service_addr,
        std::unordered_map<std::string, Slice> &objects);

    std::unique_ptr<AutoPortBinder> port_binder_ = nullptr;

    struct SegmentDeleter {
        void operator()(void *ptr) {
            if (ptr) {
                free(ptr);
            }
        }
    };

    struct HugepageSegmentDeleter {
        size_t size = 0;
        void operator()(void *ptr) const {
            if (ptr && size > 0) {
                free_buffer_mmap_memory(ptr, size);
            }
        }
    };

    struct AscendSegmentDeleter {
        std::string protocol = "ascend";
        void operator()(void *ptr) {
            if (ptr) {
                free_memory(protocol.c_str(), ptr);
            }
        }
    };

    // Deleter for anonymous (memfd) shared segments owned by this process.
    // Unregisters the CUDA mapping, unmaps the range, and closes the memfd,
    // whose memory is freed once every peer has unmapped it too.
    struct SharedSegmentDeleter {
        size_t size = 0;
        int fd = -1;
        bool cuda_host_registered = false;
        void operator()(void *ptr) const;
    };

    std::vector<std::unique_ptr<void, HugepageSegmentDeleter>>
        hugepage_segment_ptrs_;
    std::vector<std::unique_ptr<void, SegmentDeleter>> segment_ptrs_;
    std::vector<std::unique_ptr<void, AscendSegmentDeleter>>
        ascend_segment_ptrs_;
    std::vector<std::unique_ptr<void, SharedSegmentDeleter>>
        shared_segment_ptrs_;

    // A peer's shared-memory segment mapped into this process.
    struct MappedSharedSegment {
        void *base = nullptr;
        size_t size = 0;
        // Segment base in the owner's address space, used to turn an owner
        // buffer address into an offset.
        uintptr_t owner_base = 0;
        bool cuda_host_registered = false;
    };

    // Identifies a peer segment: the owner's segment name plus the index the
    // owner assigned it.
    struct PeerSegmentKey {
        std::string owner;
        uint64_t index = 0;
        bool operator==(const PeerSegmentKey &other) const {
            return index == other.index && owner == other.owner;
        }
    };
    struct PeerSegmentKeyHash {
        size_t operator()(const PeerSegmentKey &k) const {
            return std::hash<std::string>{}(k.owner) ^
                   (std::hash<uint64_t>{}(k.index) << 1);
        }
    };

    // Mapped at most once per peer segment, so shm fd retrieval and the CUDA
    // registration happen only on the preload path.
    std::unordered_map<PeerSegmentKey, MappedSharedSegment, PeerSegmentKeyHash>
        mapped_shared_segments_;

    // memfds of the segments this process owns, keyed by segment index, so
    // the IPC server can hand them to same-host peers.
    std::unordered_map<uint64_t, int> owned_shared_segment_fds_;

    /**
     * @brief Serves shm fd requests from same-host peers on a Unix socket named
     * after this worker's segment name. Started only when shm segments
     * are enabled.
     */
    int start_shm_fd_server();
    void stop_shm_fd_server();
    void shm_fd_server_loop();
    void handle_shm_fd_request(int client_sock);

    /**
     * @brief Requests one peer segment's memfd over its owner's socket and maps
     * it into this process.
     */
    tl::expected<void, ErrorCode> map_peer_shared_segment(
        const Segment &segment);

    /**
     * @brief Translates a buffer address in a peer's segment into an address
     * valid in this process. Mapping only happens during preload, so this never
     * blocks on IPC.
     * @param desc Buffer descriptor as reported by the master
     * @return Address usable in this process, or an error.
     */
    tl::expected<uintptr_t, ErrorCode> resolve_shared_memory_address(
        const AllocatedBuffer::Descriptor &desc);

    /**
     * @brief Rewrites preloaded peer shared-memory replicas in place so their
     * addresses are valid in this process and their endpoint reads as local.
     * Replicas we did not preload -- notably those owned by another host -- are
     * left untouched so the caller can still fetch them over the network.
     */
    void rewrite_shared_memory_replicas(
        std::vector<Replica::Descriptor> &replicas);

    // Unregisters and unmaps every peer segment mapped by this client.
    void clear_mapped_shared_segments();
    std::string protocol;
    std::string device_name;
    std::string local_hostname;
    std::string local_rpc_addr;
    std::unique_ptr<coro_rpc::coro_rpc_server> offload_rpc_server_;
    int offload_rpc_port_ = 0;
    bool use_hugepage_ = false;
    // KVPilot: allocate global segments as POSIX shared memory so same-host
    // peers can map them (always on).
    bool use_shm_segments_ = true;
    // Distinguishes the segments this process owns.
    uint64_t shared_segment_index_ = 0;
    // Socket + thread serving shm fds to same-host peers.
    int shm_fd_server_sock_ = -1;
    std::jthread shm_fd_server_thread_{};
    std::atomic<bool> shm_fd_server_running_{false};

    struct MappedShm {
        std::string shm_name;
        // Offset = real_base - dummy_base
        uintptr_t shm_addr_offset = 0;
        void *shm_buffer = nullptr;
        size_t shm_size = 0;
        uintptr_t dummy_base_addr = 0;
        bool is_ascend = false;
        bool is_ipc = false;
        // Ascend physical device id from dummy (dummy-real RPC).
        int32_t device_id = -1;
        uint64_t vmm_handle = 0;
        std::string ipc_key_data;
    };

    struct ShmContext {
        // List of all mapped shared memory for this client
        std::vector<MappedShm> mapped_shms;
        std::shared_ptr<ClientBufferAllocator> client_buffer_allocator =
            nullptr;
        // Buffers held by dummy via acquire_buffer; keyed by dummy address
        std::unordered_map<uint64_t, std::shared_ptr<BufferHandle>>
            active_handles;
    };
    mutable std::shared_mutex dummy_client_mutex_;
    std::unordered_map<UUID, ShmContext, boost::hash<UUID>> shm_contexts_;

    mutable std::shared_mutex registered_buffer_mutex_;
    std::unordered_map<void *, size_t> registered_buffer_sizes_;

    // Dummy VA -> real VA using mapped_shms; last_hit_shm caches locality.
    bool map_dummy_range_in_shm(const MappedShm &shm, uint64_t dummy_addr,
                                size_t offset, size_t size,
                                void *&out_real) const;

    bool map_dummy_buffer_to_real(const ShmContext &shm_ctx,
                                  uint64_t dummy_addr, size_t buf_size,
                                  const MappedShm *&last_hit_shm,
                                  void *&out_real) const;

    bool map_dummy_buffer_range_to_real(const ShmContext &shm_ctx,
                                        uint64_t dummy_addr, size_t dst_offset,
                                        size_t size, void *&out_real) const;

    tl::expected<std::vector<void *>, ErrorCode> map_dummy_addrs_to_real_ptrs(
        const ShmContext &context, const std::vector<uint64_t> &dummy_addrs,
        const std::vector<size_t> &sizes, const UUID &client_id) const;

    tl::expected<std::vector<std::vector<void *>>, ErrorCode>
    map_dummy_nested_addrs_to_real_ptrs(
        const ShmContext &context,
        const std::vector<std::vector<uint64_t>> &dummy_all_buffers,
        const std::vector<std::vector<size_t>> &all_sizes,
        const std::vector<std::string> &keys, const UUID &client_id) const;

    // Ensure cleanup executes at most once across multiple entry points
    std::atomic<bool> closed_{false};

    // Dummy Client manage related members
    void dummy_client_monitor_func();
    int start_dummy_client_monitor();
    std::thread dummy_client_monitor_thread_;
    std::atomic<bool> dummy_client_monitor_running_{false};
    static constexpr uint64_t kDummyClientMonitorSleepMs =
        1000;  // 1000 ms sleep between client monitor checks
    // boost lockfree queue requires trivial assignment operator
    struct PodUUID {
        uint64_t first;
        uint64_t second;
    };
    static constexpr size_t kDummyClientPingQueueSize =
        128 * 1024;  // Size of the client ping queue
    boost::lockfree::queue<PodUUID> dummy_client_ping_queue_{
        kDummyClientPingQueueSize};
    const int64_t dummy_client_live_ttl_sec_ = DEFAULT_CLIENT_LIVE_TTL_SEC;
    int64_t view_version_ = 0;

    // IPC Server members for receiving FD from Dummy Clients
    std::string ipc_socket_path_;
    std::jthread ipc_thread_;
    std::atomic<bool> ipc_running_{false};
    int start_ipc_server();
    int stop_ipc_server();
    void ipc_server_func();
    // Embedded HTTP server for health-check / metrics
    std::unique_ptr<coro_http::coro_http_server> http_server_;
    int start_http_server();
    void stop_http_server();

    void handle_ipc_shm_register(int client_sock);
    void handle_ipc_shm_fd_request(int client_sock);

    void teardown_ascend_shm_buffer(MappedShm &shm);
    tl::expected<void, ErrorCode> setup_ascend_internal(
        size_t local_buffer_size);
};

}  // namespace mooncake
