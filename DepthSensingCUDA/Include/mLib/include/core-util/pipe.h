
#ifndef CORE_UTIL_PIPE_H_
#define CORE_UTIL_PIPE_H_

namespace ml
{

class Pipe
{
public:
    Pipe();
    ~Pipe();
    
    //
    // Connection
    //
    void closePipe();
    void createPipe(const std::string &pipeName, bool block);
    void connectToLocalPipe(const std::string &pipeName);
    void connectToPipe(const std::string &pipeName);

    //
    // Messaging
    //
    bool messagePresent();
    bool readMessage(std::vector<BYTE> &message);
    void sendMessage(const BYTE *message, UINT messageLength);
    void sendMessage(const std::vector<BYTE> &message);
	void sendMessage(const std::string &message);

    //
    // Query
    //
    UINT activeInstances();
    std::string userName();
    bool valid();
    
private:

#ifdef _WIN32
    HANDLE m_handle;
#endif
};

}  // namespace ml

#endif  // CORE_UTIL_PIPE_H_
