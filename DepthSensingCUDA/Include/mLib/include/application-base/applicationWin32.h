
#ifndef APPLICATION_BASE_APPLICATIONWIN32_H_
#define APPLICATION_BASE_APPLICATIONWIN32_H_

namespace ml {

enum MouseButtonType
{
	MouseButtonLeft,
	MouseButtonRight,
	MouseButtonCount
};

struct MouseState
{
	MouseState()
	{
		buttons[MouseButtonLeft] = false;
		buttons[MouseButtonRight] = false;
	}
	ml::vec2i pos;
	bool buttons[MouseButtonCount];
};

struct InputState
{
public:
	InputState()
	{
		for(bool &b : keys) b = false;
	}

	static const UINT keyCount = 256;
	bool keys[keyCount];
	
	MouseState mouse;
	MouseState prevMouse;
};

struct ApplicationData
{
	ApplicationData(WindowWin32 *window_ptr, GraphicsDevice *graphics_ptr, InputState *input_ptr) :
		window(*window_ptr),
		graphics(*graphics_ptr),
		input(*input_ptr)
	{

	}
	WindowWin32 &window;
	GraphicsDevice &graphics;
	InputState &input;
};

class ApplicationCallback
{
public:
	virtual void init(ApplicationData &app) = 0;
	virtual void render(ApplicationData &app) = 0;
	virtual void keyDown(ApplicationData &app, UINT key) = 0;
	virtual void keyPressed(ApplicationData &app, UINT key) = 0;
	virtual void mouseDown(ApplicationData &app, MouseButtonType button) = 0;
	virtual void mouseMove(ApplicationData &app) = 0;
	virtual void mouseWheel(ApplicationData &app, int wheelDelta) = 0;
	virtual void resize(ApplicationData &app) = 0;
};

class ApplicationWin32
{
public:
	ApplicationWin32(HINSTANCE instance, UINT windowWidth, UINT windowHeight, const std::string &name, GraphicsDeviceType graphicsType, ApplicationCallback &callback);
	~ApplicationWin32();

	void messageLoop();

	inline ApplicationData& data()
	{
		return *m_data;
	}
	inline ApplicationCallback& callback()
	{
		return m_callback;
	}
	inline bool initialized()
	{
		return m_initialized;
	}

private:
	//
	// m_data is just a view to encapsulate all externally-visible application
	// components. THe actual data is owned by m_window, m_device, etc.
	//
	bool m_initialized;
	ApplicationData *m_data;
	
	WindowWin32 m_window;
	InputState m_input;
	GraphicsDevice *m_graphics;

	HINSTANCE m_instance;

	ApplicationCallback &m_callback;
};

}  // namespace ml

#endif  // APPLICATION_BASE_APPLICATIONWIN32_H_