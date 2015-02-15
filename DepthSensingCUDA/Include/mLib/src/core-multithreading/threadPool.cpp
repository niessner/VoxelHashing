
namespace ml
{

void ThreadPool::init(UINT threadCount)
{
	m_threads.resize(threadCount);
	for(UINT threadIndex = 0; threadIndex < threadCount; threadIndex++)
		m_threads[threadIndex].init(threadIndex, nullptr);
}

void ThreadPool::init(UINT threadCount, const std::vector<ThreadLocalStorage*> &threadLocalStorage)
{
	m_threads.resize(threadCount);
	for(UINT threadIndex = 0; threadIndex < threadCount; threadIndex++)
		m_threads[threadIndex].init(threadIndex, threadLocalStorage[threadIndex]);
}

void ThreadPool::runTasks(TaskList<WorkerThreadTask*> &tasks, bool useConsole)
{
	Console::log(std::string("running ") + std::to_string(tasks.tasksLeft()) + " tasks");

	for(UINT threadIndex = 0; threadIndex < m_threads.size(); threadIndex++)
		m_threads[threadIndex].processTasks(tasks);

	UINT consoleDelay = 0;

	bool allThreadsCompleted = false;
	while(!allThreadsCompleted)
	{
		UINT activeThreadCount = 0;
		for(UINT threadIndex = 0; threadIndex < m_threads.size(); threadIndex++)
			if(!m_threads[threadIndex].done())
				activeThreadCount++;

		if(activeThreadCount == 0)
			allThreadsCompleted = true;

		if(consoleDelay == 0)
		{
			if(useConsole) Console::log(std::string("tasks left: ") + std::to_string(tasks.tasksLeft() + activeThreadCount));
			consoleDelay = 40;
		}
		else
		{
			consoleDelay--;
		}
		std::this_thread::sleep_for( std::chrono::milliseconds(25) );
	}
	if(useConsole) Console::log("all tasks completed");
}

}  // namespace ml
