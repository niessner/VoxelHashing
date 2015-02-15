#pragma once


#undef UNICODE

#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>

// Need to link with Ws2_32.lib
#pragma comment (lib, "Ws2_32.lib")
// #pragma comment (lib, "Mswsock.lib")



class NetworkServer
{
public:
	NetworkServer() {
		m_clientSocket = INVALID_SOCKET;
		m_bIsOpen = false;
	}
	~NetworkServer() {

	}

	//! opens a network socket; blocks the thread until a connection is found
	bool open(unsigned int port, std::string& client) {

		if (m_bIsOpen) throw MLIB_EXCEPTION("server already open");

		WSADATA wsaData;
		int iResult;

		SOCKET ListenSocket = INVALID_SOCKET;

		struct addrinfo *result = NULL;
		struct addrinfo hints;


		// Initialize Winsock
		iResult = WSAStartup(MAKEWORD(2,2), &wsaData);
		if (iResult != 0) {
			printf("WSAStartup failed with error: %d\n", iResult);
			return false;
		}

		ZeroMemory(&hints, sizeof(hints));
		hints.ai_family = AF_INET;
		hints.ai_socktype = SOCK_STREAM;
		hints.ai_protocol = IPPROTO_TCP;
		hints.ai_flags = AI_PASSIVE;

		// Resolve the server address and port
		iResult = getaddrinfo(NULL, std::to_string(port).c_str(), &hints, &result);
		if ( iResult != 0 ) {
			printf("getaddrinfo failed with error: %d\n", iResult);
			WSACleanup();
			return false;
		}

		// Create a SOCKET for connecting to server
		ListenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
		if (ListenSocket == INVALID_SOCKET) {
			printf("socket failed with error: %ld\n", WSAGetLastError());
			freeaddrinfo(result);
			WSACleanup();
			return false;
		}

		// Setup the TCP listening socket
		iResult = bind( ListenSocket, result->ai_addr, (int)result->ai_addrlen);
		if (iResult == SOCKET_ERROR) {
			printf("bind failed with error: %d\n", WSAGetLastError());
			freeaddrinfo(result);
			closesocket(ListenSocket);
			WSACleanup();
			return false;
		}

		freeaddrinfo(result);

		iResult = listen(ListenSocket, SOMAXCONN);
		if (iResult == SOCKET_ERROR) {
			printf("listen failed with error: %d\n", WSAGetLastError());
			closesocket(ListenSocket);
			WSACleanup();
			return false;
		}

		// Accept a client socket
		struct sockaddr_in addr;
		socklen_t addrLen = sizeof(addr);
		m_clientSocket = accept(ListenSocket, (sockaddr*)&addr, &addrLen);
		if (m_clientSocket == INVALID_SOCKET) {
			printf("accept failed with error: %d\n", WSAGetLastError());
			closesocket(ListenSocket);
			WSACleanup();
			return false;
		}
		client = std::string(inet_ntoa(addr.sin_addr));

		// No longer need server socket (we only handle a single stream)
		closesocket(ListenSocket);

		m_bIsOpen = true;

		return true;
	}


	//! returns 0 if the connection was closed, the byte length, or -1 upon failure (non-blocking)
	int receiveData(BYTE* data, unsigned int bufferLen) {
		int iResult = recv(m_clientSocket, (char*)data, bufferLen, 0);
		return iResult;
	}

	//! returns 0 if the connection was closed, the byte length, or -1 upon failure (blocking function)
	int receiveDataBlocking(BYTE* data, unsigned int byteSize) {
		
		unsigned int bytesReceived = 0;

		while (bytesReceived < byteSize) {
			int size = receiveData(data + bytesReceived, byteSize - bytesReceived);
			if (size == -1)	return size;
			bytesReceived += size;
		}
		return bytesReceived;
	}

	//! returns the number of send bytes if successfull; otherwise -1 (non-blocking function)
	int sendData(const BYTE* data, unsigned int byteSize) {
		// Echo the buffer back to the sender
		int iSendResult = send( m_clientSocket, (char*)data, byteSize, 0 );
		if (iSendResult == SOCKET_ERROR) {
			printf("send failed with error: %d\n", WSAGetLastError());
			closesocket(m_clientSocket);
			WSACleanup();
			return SOCKET_ERROR;
		}
		return iSendResult;
	}

	//! blocks until all data is sent; returns true upon success; false upon failure
	int sendDataBlocking(const BYTE* data, unsigned int byteSize) {
		int sentBytes = 0;
		while (sentBytes != byteSize) {
			int iResult = sendData(data + sentBytes, byteSize - sentBytes);
			if (iResult == SOCKET_ERROR)	return false;
			sentBytes += iResult;
		} 
		return sentBytes;
	}

	void close() {

		if (m_bIsOpen) {
			// shutdown the connection since we're done
			int iResult = shutdown(m_clientSocket, SD_BOTH);
			if (iResult == SOCKET_ERROR) {
				printf("shutdown failed with error: %d\n", WSAGetLastError());
				closesocket(m_clientSocket);
				WSACleanup();
				//return false;
			}

			//printf("recv failed with error: %d\n", WSAGetLastError());
			closesocket(m_clientSocket);
			WSACleanup();
			m_bIsOpen = false;
		}
	}

private:
	bool	m_bIsOpen;
	SOCKET	m_clientSocket;

};

