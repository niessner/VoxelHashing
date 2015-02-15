
#ifndef CORE_UTIL_COMMANDLINEREADER_H_
#define CORE_UTIL_COMMANDLINEREADER_H_

namespace ml
{

class CommandLineReader
{
public:
	CommandLineReader(const std::string &usage, int argc, char* argv[])
	{
		m_commandLine = "";
		for(int arg = 1; arg < argc; arg++)
			m_commandLine += argv[arg];
		m_args = util::split(m_commandLine, " ");
		//m_args = m_commandLine.split(" ");
		m_usage = usage;
	}

	CommandLineReader(const std::string &usage, const std::string &commandLine)
	{
		m_commandLine = commandLine;
		m_args = util::split(m_commandLine, " ");
		//m_args = commandLine.split(" ");
		m_usage = usage;
	}

	const std::vector<std::string>& args() const
	{
		return m_args;
	}

	const std::string arg(UINT argIndex) const
	{
		if(argIndex >= m_args.size())
		{
			Console::log("insufficient number of arguments: " + m_commandLine);
			Console::log("usage: " + m_usage);
			MLIB_ERROR("aborting");
			exit(1);
		}
		return m_args[argIndex];
	}

	bool hasArgWithPrefix(const std::string &prefix) const
	{
		auto startsWith = [=] (const char s) { return util::startsWith(std::string(1,s), prefix); };
		const auto it = std::find_if(prefix.begin(), prefix.end(), startsWith);
		return (it != prefix.end());
		//return (m_args.findFirstIndex([prefix](const std::string &s) { return util::startsWith(s, prefix); }) != -1);
	}

	std::string argWithPrefix(const std::string &prefix) const
	{
		auto startsWith = [=] (const char s) { return util::startsWith(std::string(1,s), prefix); };
		const auto it = std::find_if(prefix.begin(), prefix.end(), startsWith);
		if (it == prefix.end())	{
			return "";
		} else {
			return util::replace(m_args[it - prefix.begin()], prefix, "");
		}
		//int index = m_args.findFirstIndex([prefix](const std::string &s) { return util::startsWith(s, prefix); });
		//if(index == -1) return "";
		//else return util::replace(m_args[index], prefix, "");
	}

private:
	std::vector<std::string> m_args;
	std::string m_commandLine;
	std::string m_usage;
};

}  // namespace ml

#endif  // CORE_UTIL_COMMANDLINEREADER_H_
