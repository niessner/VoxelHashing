#pragma once

#ifndef CORE_UTIL_TIMER_H_
#define CORE_UTIL_TIMER_H_

namespace ml {

class Timer {
public:
	Timer(bool _start = true) {
		m_bRunning = false;
		if (_start) start();
	}

	void start() {
		m_bRunning = true;
		m_Start = getTime();
	}
	
	void stop() {
		m_bRunning = false;
		m_Stop = getTime();
	}

	//! returns the elapsed time in seconds
	double getElapsedTime() {
		if (m_bRunning) {
			return getTime() - m_Start;
		} else {
			return m_Stop - m_Start;
		}
	}

	double getElapsedTimeMS() {
		return getElapsedTime() * 1000.0;
	}

	void printElapsedTimeMS() {
		printf("%.2f ms\n", getElapsedTimeMS());
	}

	//! returns the time in seconds
	static double getTime();
private:
	bool m_bRunning;

	double m_Start;
	double m_Stop;
};


class ComponentTimer
{
public:
	ComponentTimer(const std::string &prompt)
	{
		m_prompt = prompt;
		m_terminated = false;
		Console::log("start " + prompt);
	}
	~ComponentTimer()
	{
		if(!m_terminated) end();
	}
	void end()
	{
		m_terminated = true;
		Console::log("end " + m_prompt + ", " + std::to_string(m_clock.getElapsedTime()) + "s");
	}

private:
	std::string m_prompt;
	Timer m_clock;
	bool m_terminated;
};



class FrameTimer
{
public:
	FrameTimer()
	{
		m_secondsPerFrame = 1.0 / 60.0;
	}
	void frame()
	{
		double elapsed = m_clock.getElapsedTime();
		m_clock.start();
		m_secondsPerFrame = elapsed * 0.2 + m_secondsPerFrame * 0.8;
	}
	float framesPerSecond()
	{
		return 1.0f / (float)m_secondsPerFrame;
	}
	float secondsPerFrame()
	{
		return (float)m_secondsPerFrame;
	}

private:
	Timer m_clock;
	double m_secondsPerFrame;
};

} // namespace ml


#endif // CORE_UTIL_TIMER_H_