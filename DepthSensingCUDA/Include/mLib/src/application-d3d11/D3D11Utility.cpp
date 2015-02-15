
ID3DBlob* ml::D3D11Utility::CompileShader(const std::string &filename, const std::string &entryPoint, const std::string &shaderModel)
{
	DWORD shaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
	// Set the D3DCOMPILE_DEBUG flag to embed debug information in the shaders.
	// Setting this flag improves the shader debugging experience, but still allows 
	// the shaders to be optimized and to run exactly the way they will run in 
	// the release configuration of this program.
	shaderFlags |= D3DCOMPILE_DEBUG;
#endif

	ID3DBlob* blob = nullptr;
	ID3DBlob* errorBlob = nullptr;
    MLIB_ASSERT_STR(util::fileExists(filename), "File not found: " + filename);
#ifdef UNICODE
	std::wstring s(filename.begin(), filename.end());
#else
	std::string s(filename.begin(), filename.end());
#endif
	HRESULT hr = D3DX11CompileFromFile( s.c_str(), nullptr, nullptr, entryPoint.c_str(), shaderModel.c_str(), 
		shaderFlags, 0, nullptr, &blob, &errorBlob, nullptr );
	if( FAILED(hr) )
	{
        std::string errorBlobText;
		if( errorBlob != nullptr )
		{
            errorBlobText = (char *)errorBlob->GetBufferPointer();
			Console::log() << "Shader compilation failed for " << filename << std::endl
                           << errorBlobText << std::endl;
		}
		MLIB_ERROR("Shader compilation failed for " + filename);
	}
	if( errorBlob ) errorBlob->Release();

	return blob;
}