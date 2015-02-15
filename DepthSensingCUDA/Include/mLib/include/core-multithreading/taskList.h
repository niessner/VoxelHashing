
#ifndef CORE_MULTITHREADING_TASKLIST_H_
#define CORE_MULTITHREADING_TASKLIST_H_

namespace ml
{

template <class T> class TaskList
{
public:
    void insert(const T &task)
    {
        m_mutex.lock();
        m_tasks.pushBack(task);
        m_mutex.unlock();
    }

    bool done()
    {
        m_mutex.lock();
        bool result = (m_tasks.size() == 0);
        m_mutex.unlock();
        return result;
    }

    UINT64 tasksLeft()
    {
        m_mutex.lock();
        UINT64 result = m_tasks.size();
        m_mutex.unlock();
        return result;
    }

    bool getNextTask(T &nextTask)
    {
        m_mutex.lock();
        if(m_tasks.size() == 0)
        {
            m_mutex.unlock();
            return false;
        }

        nextTask = m_tasks.back();
        m_tasks.pop_back();
        m_mutex.unlock();
        return true;
    }

private:
    std::mutex m_mutex;
    std::vector<T> m_tasks;
};

}  // namespace ml

#endif  // CORE_MULTITHREADING_TASKLIST_H_
