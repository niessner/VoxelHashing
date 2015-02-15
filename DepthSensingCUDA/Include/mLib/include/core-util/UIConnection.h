#ifndef CORE_UTIL_UICONNECTION_H_
#define CORE_UTIL_UICONNECTION_H_

#ifdef _WIN32

#include <string>

namespace ml {

class UIConnection
{
 public:
    void init(const std::string &executableFile, const std::string &pipeBaseName);
    void readMessages();
    void sendMessage(const std::string &message);

    inline std::vector<std::string>& messages()
    {
        return m_messages;
    }
 private:
    std::vector<std::string> m_messages;
    Pipe m_writeToUIPipe;
    Pipe m_readFromUIPipe;
};

}  // namespace ml

#endif  // _WIN32

#endif  // CORE_UTIL_UICONNECTION_H_
