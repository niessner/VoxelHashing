
#ifndef APPLICATION_BASE_WINDOWWIN32_H_
#define APPLICATION_BASE_WINDOWWIN32_H_

#include <windows.h>

namespace ml {

class ApplicationWin32;
class WindowWin32 {
public:
	WindowWin32(ApplicationWin32 &parent) :
		m_parent(parent)
	{
		m_className = L"uninitialized";
		m_handle = nullptr;
		ZeroMemory(&m_class, sizeof(m_class));
	}
	~WindowWin32();

	void init(HINSTANCE instance, int width, int height, const std::string &name);
	void destroy();
	void resize(UINT newWidth, UINT newHeight);
	void rename(const std::string &name);
	
	UINT width() const;
	UINT height() const;

	HWND handle() const
	{
		return m_handle;
	}
	ApplicationWin32& parent()
	{
		return m_parent;
	}

private:
	std::wstring     m_className;
	ApplicationWin32 &m_parent;
	WNDCLASSW m_class;
	HWND      m_handle;
};

}  // namespace ml

#endif  // APPLICATION_BASE_WINDOWWIN32_H_