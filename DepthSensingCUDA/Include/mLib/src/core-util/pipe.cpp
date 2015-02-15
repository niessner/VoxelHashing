
#ifdef _WIN32

#include <AccCtrl.h>
#include <Aclapi.h>

namespace ml {

Pipe::Pipe()
{
    m_handle = nullptr;
}

Pipe::~Pipe()
{
    closePipe();
}

void Pipe::closePipe()
{
    if(m_handle != nullptr)
    {
        FlushFileBuffers(m_handle);
        DisconnectNamedPipe(m_handle);
        CloseHandle(m_handle);
        m_handle = nullptr;
    }
}

void Pipe::createPipe(const std::string &pipeName, bool block)
{
    //Console::log() << "creating pipe " << pipeName << std::endl;

    closePipe();
    const UINT PipeBufferSize = 100000;

    DWORD dwRes;
    PSID pEveryoneSID = nullptr, pAdminSID = nullptr;
    PACL pACL = nullptr;
    PSECURITY_DESCRIPTOR pSD = nullptr;
    EXPLICIT_ACCESS ea[1];
    SID_IDENTIFIER_AUTHORITY SIDAuthWorld = SECURITY_WORLD_SID_AUTHORITY;
    SID_IDENTIFIER_AUTHORITY SIDAuthNT = SECURITY_NT_AUTHORITY;
    SECURITY_ATTRIBUTES attributes;
    HKEY hkSub = nullptr;

    // Create a well-known SID for the Everyone group.
    BOOL success = AllocateAndInitializeSid(&SIDAuthWorld, 1,
        SECURITY_WORLD_RID,
        0, 0, 0, 0, 0, 0, 0,
        &pEveryoneSID);
    MLIB_ASSERT_STR(success != FALSE, "AllocateAndInitializeSid failed in Pipe::CreatePipe");

    // Initialize an EXPLICIT_ACCESS structure for an ACE.
    // The ACE will allow Everyone read access to the key.
    ZeroMemory(&ea, sizeof(EXPLICIT_ACCESS));
    ea[0].grfAccessPermissions = FILE_ALL_ACCESS;
    ea[0].grfAccessMode = SET_ACCESS;
    ea[0].grfInheritance= NO_INHERITANCE;
    ea[0].Trustee.TrusteeForm = TRUSTEE_IS_SID;
    ea[0].Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
    ea[0].Trustee.ptstrName  = (LPTSTR) pEveryoneSID;

    // Create a new ACL that contains the new ACEs.
    dwRes = SetEntriesInAcl(1, ea, nullptr, &pACL);
    MLIB_ASSERT_STR(dwRes == ERROR_SUCCESS, "SetEntriesInAcl failed in Pipe::CreatePipe");

    // Initialize a security descriptor.  
    pSD = (PSECURITY_DESCRIPTOR) LocalAlloc(LPTR, SECURITY_DESCRIPTOR_MIN_LENGTH);
    MLIB_ASSERT_STR(pSD != nullptr, "LocalAlloc failed in Pipe::CreatePipe");

    success = InitializeSecurityDescriptor(pSD, SECURITY_DESCRIPTOR_REVISION);
    MLIB_ASSERT_STR(success != FALSE, "InitializeSecurityDescriptor failed in Pipe::CreatePipe");

    // Add the ACL to the security descriptor. 
    success = SetSecurityDescriptorDacl(pSD, 
        TRUE,     // bDaclPresent flag
        pACL, 
        FALSE);
    MLIB_ASSERT_STR(success != FALSE, "SetSecurityDescriptorDacl failed in Pipe::CreatePipe");

    // Initialize a security attributes structure.
    attributes.nLength = sizeof(SECURITY_ATTRIBUTES);
    attributes.lpSecurityDescriptor = pSD;
    attributes.bInheritHandle = FALSE;

    std::string fullPipeName = std::string("\\\\.\\pipe\\") + pipeName;
    m_handle = CreateNamedPipeA( 
        fullPipeName.c_str(),		// pipe name
        PIPE_ACCESS_DUPLEX,         // read/write access
        PIPE_TYPE_MESSAGE |         // message type pipe 
        PIPE_READMODE_MESSAGE |     // message-read mode 
        PIPE_WAIT,                  // blocking mode 
        PIPE_UNLIMITED_INSTANCES,   // max. instances  
        PipeBufferSize,             // output buffer size 
        PipeBufferSize,             // input buffer size 
        NMPWAIT_USE_DEFAULT_WAIT,   // client time-out 
        &attributes);               // default security attribute
    MLIB_ASSERT_STR(m_handle != INVALID_HANDLE_VALUE, "CreateNamedPipe failed in Pipe::CreatePipe");

    //
    // Block until a connection comes in
    //

    if(block)
    {
        Console::log("Pipe created, waiting for connection");
        BOOL Connected = (ConnectNamedPipe(m_handle, nullptr) != 0);
        MLIB_ASSERT_STR(Connected != FALSE, "ConnectNamedPipe failed in Pipe::CreatePipe");
        Console::log("Connected");
    }
    else
    {
        //cout << "Not blocking for connection to complete" << endl;
    }
}

void Pipe::connectToLocalPipe(const std::string &pipeName)
{
    connectToPipe(std::string("\\\\.\\pipe\\") + pipeName);
}

void Pipe::connectToPipe(const std::string &pipeName)
{
    //Console::log("Connecting to " + pipeName);
    closePipe();
    bool done = false;
    while(!done)
    {
        m_handle = CreateFileA( 
            pipeName.c_str(),             // pipe name 
            GENERIC_READ |                // read and write access 
            GENERIC_WRITE, 
            0,                            // no sharing 
            nullptr,                         // default security attributes
            OPEN_EXISTING,                // opens existing pipe 
            0,                            // default attributes 
            nullptr);                        // no template file
        if(m_handle != INVALID_HANDLE_VALUE)
        {
            done = true;
        }
        Sleep(100);
    }
    //cout << "Connected" << endl;

    //DWORD mode = PIPE_READMODE_MESSAGE;
    DWORD mode = PIPE_READMODE_BYTE;
    BOOL success = SetNamedPipeHandleState( 
        m_handle,  // pipe handle 
        &mode,    // new pipe mode 
        nullptr,     // don't set maximum bytes 
        nullptr);    // don't set maximum time 
    MLIB_ASSERT_STR(success != FALSE, "SetNamedPipeHandleState failed in Pipe::ConnectToPipe");
}

bool Pipe::messagePresent()
{
    MLIB_ASSERT_STR(m_handle != nullptr, "Pipe invalid in Pipe::MessagePresent");
    DWORD BytesReady  = 0;
    DWORD BytesLeft   = 0;
    BOOL success = PeekNamedPipe(
        m_handle,
        nullptr,
        0,
        nullptr,
        &BytesReady,
        &BytesLeft);
    //MLIB_ASSERT_STR(success != FALSE, "PeekNamedPipe failed in Pipe::MessagePresent");
    return (BytesReady > 0);
}

bool Pipe::readMessage(std::vector<BYTE> &Message)
{
    MLIB_ASSERT_STR(m_handle != nullptr, "Pipe invalid in Pipe::ReadMessage");
    DWORD BytesReady  = 0;
    BOOL success = PeekNamedPipe(
        m_handle,
        nullptr,
        0,
        nullptr,
        &BytesReady,
        nullptr);
    MLIB_ASSERT_STR(success != FALSE, "PeekNamedPipe failed in Pipe::ReadMessage");
    Message.resize(BytesReady);
    if(BytesReady == 0)
    {
        return false;
    }

    DWORD BytesRead;
    success = ReadFile( 
        m_handle,                // handle to pipe 
        &Message[0],            // buffer to receive data 
        (DWORD)Message.size(),  // size of buffer 
        &BytesRead,             // number of bytes read 
        nullptr);                  // not overlapped I/O 
    MLIB_ASSERT_STR(success != FALSE && BytesRead > 0, "ReadFile failed in Pipe::ReadMessage");
    return true;
}

void Pipe::sendMessage(const std::vector<BYTE> &Message)
{
    sendMessage(&Message[0], (UINT)Message.size());
}

void Pipe::sendMessage(const std::string &message)
{
    sendMessage((const BYTE *)message.c_str(), (UINT)message.size());

    std::string endLine;
    endLine.push_back('\n');
    sendMessage((const BYTE *)endLine.c_str(), 1);
}

void Pipe::sendMessage(const BYTE *Message, UINT MessageLength)
{
    if(Message == nullptr || MessageLength == 0) return;
    MLIB_ASSERT_STR(m_handle != nullptr, "Pipe invalid in Pipe::SendMessage");

    DWORD BytesWritten;
    BOOL success = WriteFile( 
        m_handle,               // pipe handle
        Message,               // message
        MessageLength,         // message length
        &BytesWritten,         // bytes written
        nullptr);                 // not overlapped
    MLIB_ASSERT_STR(success != FALSE, "WriteFile failed in Pipe::ReadMessage");
    MLIB_ASSERT_STR(BytesWritten == MessageLength, "WriteFile failed to send entire message in Pipe::ReadMessage");
}

UINT Pipe::activeInstances()
{
    MLIB_ASSERT_STR(m_handle != nullptr, "Pipe invalid in Pipe::ActiveInstances");
    DWORD Instances;
    BOOL success = GetNamedPipeHandleState(
        m_handle,
        nullptr,
        &Instances,
        nullptr,
        nullptr,
        nullptr,
        0);
    MLIB_ASSERT_STR(success != FALSE, "GetNamedPipeHandleState failed in Pipe::ActiveInstances");
    return Instances;
}

std::string Pipe::userName()
{
    MLIB_ASSERT_STR(m_handle != nullptr, "Pipe invalid in Pipe::UserName");
    char buffer[512];
    BOOL success = GetNamedPipeHandleStateA(
        m_handle,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        buffer,
        512);
    MLIB_ASSERT_STR(success != FALSE, "GetNamedPipeHandleState failed in Pipe::UserName");
    return std::string(buffer);
}

bool Pipe::valid()
{
    return (m_handle != nullptr);
}

}  // namespace ml

#endif  // _WIN32
