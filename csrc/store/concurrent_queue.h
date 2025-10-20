#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T> class ConcurrentQueue {
public:
  ConcurrentQueue() = default;
  ConcurrentQueue(const ConcurrentQueue &) = delete;
  ConcurrentQueue &operator=(const ConcurrentQueue &) = delete;

  void enqueue(T item);
  T dequeue();
  bool isEmpty();

private:
  std::queue<T> queue_;
  std::mutex mtx_;
  std::condition_variable cond_;
};

template <typename T> void ConcurrentQueue<T>::enqueue(T item) {
  std::unique_lock<std::mutex> lock(mtx_);
  queue_.push(std::move(item));
  lock.unlock(); // explicitly unlock before notifying to minimize the waiting
                 // time of the notified thread
  cond_.notify_one();
}

template <typename T> T ConcurrentQueue<T>::dequeue() {
  std::unique_lock<std::mutex> lock(mtx_);
  while (queue_.empty()) {
    cond_.wait(lock); // release lock and wait to be notified
  }
  T item = std::move(queue_.front());
  queue_.pop();
  return item;
}

template <typename T> bool ConcurrentQueue<T>::isEmpty() {
  std::unique_lock<std::mutex> lock(mtx_);
  return queue_.empty();
}