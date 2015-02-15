#include <string>

namespace ml
{


void warningFunctionMLIB(const std::string &description)
{
	Console::log() << description << std::endl;
    //DEBUG_BREAK;
}

void errorFunctionMLIB(const std::string &description)
{
	Console::log() << description << std::endl;
	DEBUG_BREAK;
}

void assertFunctionMLIB(bool statement, const std::string &description)
{
	if(!statement)
	{
		Console::log() << description << std::endl;
		DEBUG_BREAK;
	}
}

}  // namespace ml