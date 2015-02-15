
#ifdef _WIN32

namespace ml {

void UIConnection::init(const std::string &executableFile, const std::string &pipeBaseName)
{
    if (executableFile.size() > 0 && util::runCommand(executableFile, "", false) != 0)
	{
		Console::log("Failed to launch UI");
		return;
	}
	m_readFromUIPipe.createPipe(pipeBaseName + "ReadFromUI", false);
	m_writeToUIPipe.connectToLocalPipe(pipeBaseName + "WriteToUI");
    Console::log("UI pipes created");
}

void UIConnection::readMessages()
{
	while(m_readFromUIPipe.messagePresent())
	{
		std::vector<BYTE> message;
		m_readFromUIPipe.readMessage(message);
		message.push_back(0);
		std::string s = std::string((const char *)&message[0]);
		if(s.size() > 0)
		{
			m_messages.push_back(s);
		}
	}
}

void UIConnection::sendMessage(const std::string &message)
{
	m_writeToUIPipe.sendMessage(message);
}

}  // namespace ml

#endif  // _WIN32
