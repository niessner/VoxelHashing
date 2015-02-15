
ml::ApplicationWin32::ApplicationWin32(HINSTANCE instance, UINT windowWidth, UINT windowHeight, const std::string &name, GraphicsDeviceType graphicsType, ApplicationCallback &callback) :
	m_callback(callback),
	m_window(*this)
{
	m_initialized = false;
	m_window.init(instance, windowWidth, windowHeight, name);

	switch(graphicsType)
	{
	case GraphicsDeviceTypeD3D11:
		m_graphics = new D3D11GraphicsDevice();
		break;
	default:
		MLIB_ERROR("invalid graphics device type");
	}
	m_graphics->init(m_window);

	m_data = new ApplicationData(&m_window, m_graphics, &m_input);

	m_callback.init(*m_data);
}

ml::ApplicationWin32::~ApplicationWin32()
{
	delete m_graphics;
	delete m_data;
}

void ml::ApplicationWin32::messageLoop()
{
	bool messageReceived;
	MSG msg;
	msg.message = WM_NULL;
	PeekMessage( &msg, nullptr, 0U, 0U, PM_NOREMOVE );

	m_initialized = true;

	while( WM_QUIT != msg.message )
	{
		// Use PeekMessage() so we can use idle time to render the scene. 
		messageReceived = ( PeekMessage( &msg, nullptr, 0U, 0U, PM_REMOVE ) != 0 );

		if( messageReceived )
		{
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		else
		{
			for(int i = 0; i < m_input.keyCount; i++)
				if(m_input.keys[i]) m_callback.keyPressed(*m_data, i);

			m_graphics->renderBeginFrame();
			m_callback.render(*m_data);
			m_graphics->renderEndFrame();
		}
	}
}
