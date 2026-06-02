@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
if not exist "x64\Debug" mkdir "x64\Debug"
cl.exe /EHsc /std:c++17 /W3 /DNDEBUG /DCV_IGNORE_DEBUG_BUILD_GUARD /DNOMINMAX /I"C:\Users\siqian\source\repos\SURF\opencv-build\build\install\include" %* /Fe"x64\Debug\SurfDemo.exe" /Fo"x64\Debug\\" /link /LIBPATH:"C:\Users\siqian\source\repos\SURF\opencv-build\build\install\x64\vc17\lib" opencv_world4110.lib /SUBSYSTEM:CONSOLE
