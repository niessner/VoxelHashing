#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>  

template<class T>
class FunctionValue
{
	public:

		FunctionValue(T nonLinearError) : m_nonLinearError(nonLinearError)
		{
		}

		void setTimeStamp(unsigned int timeStamp)
		{
			m_timeStamp = timeStamp;
		}

		unsigned int getTimeStamp()
		{
			return m_timeStamp;
		}

		T getNonLinearError()
		{
			return m_nonLinearError;
		}

	private:

		unsigned int m_timeStamp;
		T			 m_nonLinearError;
};

template<class T>
class ConvergenceAnalysis
{
	public:

		ConvergenceAnalysis()
		{
			reset();
		}

		~ConvergenceAnalysis()
		{
		}

		void reset()
		{
			m_CurrentTimeStamp = 0;
			m_samples.clear();
		}

		void addSample(const FunctionValue<T>& sample)
		{
			m_samples.push_back(sample);
			m_samples[m_samples.size() - 1].setTimeStamp(m_CurrentTimeStamp++);
		}

		//! saves the analysis to a graph file and also resets all entries
		void saveGraph(const std::string& filename)
		{
			std::ofstream ofs(filename, std::ofstream::out);

			for(unsigned int i = 0; i<m_samples.size(); i++) ofs << m_samples[i].getTimeStamp() << "\t";
			ofs << std::endl;

			for(unsigned int i = 0; i<m_samples.size(); i++) ofs << m_samples[i].getNonLinearError() << "\t";
			ofs << std::endl;

			ofs.close();

			reset();
		}

	private:

		unsigned int m_CurrentTimeStamp;
		std::vector<FunctionValue<T> > m_samples;
};
