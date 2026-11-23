// SPDX-License-Identifier: Apache-2.0
// Indexed binary min-heap over dense ids [0, capacity), ordered by (key, seq).
//
// Backs the penalty-ordered HBM free block queue: every op the block pool runs
// per block (push on free, remove on prefix hit, update on verdict, pop on
// allocation) is O(log n) with no tombstones, and the batch calls cross the
// Python boundary once per batch instead of once per block. `seq` is assigned
// at push (or given by the caller, push_seq) and kept across updates; equal
// keys pop in seq order.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

class IndexedHeap {
 public:
  explicit IndexedHeap(int64_t capacity)
      : pos_(capacity, -1), key_(capacity, 0.0), seq_(capacity, 0) {
    if (capacity < 0) throw std::invalid_argument("capacity must be >= 0");
    heap_.reserve(static_cast<size_t>(capacity));
  }

  int64_t size() const { return static_cast<int64_t>(heap_.size()); }

  bool contains(int64_t id) const {
    check(id);
    return pos_[id] >= 0;
  }

  double key(int64_t id) const {
    check(id);
    if (pos_[id] < 0) throw std::out_of_range("id not in heap");
    return key_[id];
  }

  void push(int64_t id, double key) { push_seq(id, key, next_seq_++); }

  void push_seq(int64_t id, double key, int64_t seq) {
    check(id);
    if (pos_[id] >= 0) throw std::runtime_error("id already in heap");
    key_[id] = key;
    seq_[id] = seq;
    heap_.push_back(id);
    pos_[id] = static_cast<int64_t>(heap_.size()) - 1;
    sift_up(pos_[id]);
  }

  void push_many(const std::vector<int64_t>& ids,
                 const std::vector<double>& keys) {
    if (ids.size() != keys.size())
      throw std::invalid_argument("ids and keys differ in length");
    for (size_t i = 0; i < ids.size(); ++i) push(ids[i], keys[i]);
  }

  // Like push_many, but equal keys order by the caller's seqs.
  void push_many_seq(const std::vector<int64_t>& ids,
                     const std::vector<double>& keys,
                     const std::vector<int64_t>& seqs) {
    if (ids.size() != keys.size() || ids.size() != seqs.size())
      throw std::invalid_argument("ids, keys and seqs differ in length");
    for (size_t i = 0; i < ids.size(); ++i) push_seq(ids[i], keys[i], seqs[i]);
  }

  bool remove(int64_t id) {
    check(id);
    int64_t i = pos_[id];
    if (i < 0) return false;
    erase_at(i);
    return true;
  }

  bool update(int64_t id, double key) {
    check(id);
    int64_t i = pos_[id];
    if (i < 0) return false;
    double old = key_[id];
    key_[id] = key;
    if (key < old)
      sift_up(i);
    else if (key > old)
      sift_down(i);
    return true;
  }

  int64_t update_many(const std::vector<int64_t>& ids,
                      const std::vector<double>& keys) {
    if (ids.size() != keys.size())
      throw std::invalid_argument("ids and keys differ in length");
    int64_t n = 0;
    for (size_t i = 0; i < ids.size(); ++i) n += update(ids[i], keys[i]);
    return n;
  }

  int64_t pop() {
    if (heap_.empty()) throw std::out_of_range("pop from empty heap");
    int64_t id = heap_[0];
    erase_at(0);
    return id;
  }

  std::vector<int64_t> pop_n(int64_t n) {
    if (n < 0 || n > size()) throw std::out_of_range("pop_n beyond heap size");
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) out.push_back(pop());
    return out;
  }

  std::vector<int64_t> ids() const { return heap_; }

  void clear() {
    for (int64_t id : heap_) pos_[id] = -1;
    heap_.clear();
  }

 private:
  void check(int64_t id) const {
    if (id < 0 || id >= static_cast<int64_t>(pos_.size()))
      throw std::out_of_range("id out of range");
  }

  bool less(int64_t a, int64_t b) const {
    return key_[a] < key_[b] || (key_[a] == key_[b] && seq_[a] < seq_[b]);
  }

  void place(int64_t i, int64_t id) {
    heap_[i] = id;
    pos_[id] = i;
  }

  void sift_up(int64_t i) {
    int64_t id = heap_[i];
    while (i > 0) {
      int64_t parent = (i - 1) / 2;
      if (!less(id, heap_[parent])) break;
      place(i, heap_[parent]);
      i = parent;
    }
    place(i, id);
  }

  void sift_down(int64_t i) {
    int64_t n = size();
    int64_t id = heap_[i];
    while (true) {
      int64_t child = 2 * i + 1;
      if (child >= n) break;
      if (child + 1 < n && less(heap_[child + 1], heap_[child])) ++child;
      if (!less(heap_[child], id)) break;
      place(i, heap_[child]);
      i = child;
    }
    place(i, id);
  }

  void erase_at(int64_t i) {
    int64_t id = heap_[i];
    int64_t last = heap_.back();
    heap_.pop_back();
    pos_[id] = -1;
    if (i < size()) {
      place(i, last);
      sift_down(i);
      sift_up(pos_[last]);
    }
  }

  std::vector<int64_t> heap_;
  std::vector<int64_t> pos_;
  std::vector<double> key_;
  std::vector<int64_t> seq_;
  int64_t next_seq_ = 0;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<IndexedHeap>(m, "IndexedHeap")
      .def(py::init<int64_t>())
      .def("size", &IndexedHeap::size)
      .def("contains", &IndexedHeap::contains)
      .def("key", &IndexedHeap::key)
      .def("push", &IndexedHeap::push)
      .def("push_many", &IndexedHeap::push_many)
      .def("push_seq", &IndexedHeap::push_seq)
      .def("push_many_seq", &IndexedHeap::push_many_seq)
      .def("remove", &IndexedHeap::remove)
      .def("update", &IndexedHeap::update)
      .def("update_many", &IndexedHeap::update_many)
      .def("pop", &IndexedHeap::pop)
      .def("pop_n", &IndexedHeap::pop_n)
      .def("ids", &IndexedHeap::ids)
      .def("clear", &IndexedHeap::clear);
}
