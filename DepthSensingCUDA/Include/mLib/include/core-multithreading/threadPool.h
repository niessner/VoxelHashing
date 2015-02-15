#ifndef CORE_MULTITHREADING_THREADPOOL_H_
#define CORE_MULTITHREADING_THREADPOOL_H_

namespace ml
{

class ThreadPool
{
public:
    ~ThreadPool();

    void init(UINT threadCount);
    void init(UINT threadCount, const std::vector<ThreadLocalStorage*> &threadLocalStorage);
    void runTasks(TaskList<WorkerThreadTask*> &tasks, bool useConsole = true);

private:
    std::vector<WorkerThread> m_threads;
};

}  // namespace ml

#endif  // CORE_MULTITHREADING_THREADPOOL_H_
