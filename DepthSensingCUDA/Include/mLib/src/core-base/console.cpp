
namespace ml
{

MultiStream Console::s_stream(std::cout);
std::ofstream Console::s_logFile;

void Console::log(const std::string &s)
{
	s_stream << s << std::endl;
}

}  //namespace ml