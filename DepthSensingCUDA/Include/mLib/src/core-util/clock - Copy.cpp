namespace ml {

#ifdef _WIN32
Clock::Clock()
{
	LARGE_INTEGER ticksPerSecond;
	QueryPerformanceFrequency( &ticksPerSecond );
	m_ticksPerSecond = ticksPerSecond.QuadPart;
	start();
}

void Clock::start()
{
	LARGE_INTEGER time;
	QueryPerformanceCounter( &time );
	m_startTime = time.QuadPart;
}

double Clock::elapsed()
{
	LARGE_INTEGER time;
	QueryPerformanceCounter( &time );
	return double(time.QuadPart - m_startTime) / double(m_ticksPerSecond);
}
#endif  // _WIN32

#ifdef LINUX
Clock::Clock()
{

}

void Clock::start()
{
	struct timeval timevalue;
	gettimeofday(&timevalue, NULL);
	m_startTime = (UINT64)timevalue.tv_sec * 1000000ULL + (UINT64)timevalue.tv_usec;
}

double Clock::elapsed()
{
	struct timeval timevalue;
	gettimeofday(&timevalue, NULL);
	UINT64 endtime = (UINT64)timevalue.tv_sec * 1000000ULL + (UINT64)timevalue.tv_usec;
	return double(endtime - m_startTime) / double(1000000);
}
#endif  // LINUX

}  // namespace ml
