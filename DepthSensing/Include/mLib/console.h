
#ifndef CORE_BASE_CONSOLE_H_
#define CORE_BASE_CONSOLE_H_

namespace ml
{

//
// see http://stackoverflow.com/questions/1760726/how-can-i-compose-output-streams-so-output-goes-multiple-places-at-once
//

class MultiStream: public std::ostream
{
public: 
	MultiStream(std::ostream &baseStream) : std::ostream(NULL)
	{
		std::ostream::rdbuf(&m_multiBuffer);
		addStream(baseStream);
	}

	void addStream(std::ostream &out)
	{
		out.flush();
		m_multiBuffer.addBuffer(out.rdbuf());
	}

	UINT64 streamCount() const
	{
		return m_multiBuffer.m_buffers.size();
	}

private:
	struct MultiBuffer: public std::streambuf
	{
		void addBuffer(std::streambuf* buf)
		{
			m_buffers.push_back(buf);
		}

		virtual int overflow(int c)
		{
			for(std::streambuf* &s : m_buffers) s->sputc(c);
			return c;
		}

		std::vector<std::streambuf*> m_buffers;

	};
	MultiBuffer m_multiBuffer;
};

class Console
{
public:
	static void openLogFile(const char *filename)
	{
		if(s_stream.streamCount() == 1)
		{
			s_logFile.open(filename);
			s_stream.addStream(s_logFile);
		}
	}

	static std::ostream& log()
	{
		return s_stream;
	}

	//static void log(const std::string &s)
	//{
	//	s_stream << s << std::endl;
	//}

	static void log(const std::string &s);

private:
	static MultiStream s_stream;
	static std::ofstream s_logFile;
};

}  // namespace ml

#endif  // CORE_BASE_CONSOLE_H_
