@echo off
setlocal

set "OPENCV_CUDA=C:\Users\siqian\source\repos\cuda_all_moudle\opencv_install"
set "BIN=%OPENCV_CUDA%\x64\vc17\bin"
set "LIB=%OPENCV_CUDA%\x64\vc17\lib"
set "INC=%OPENCV_CUDA%\include"
set "OUTDIR=x64\Debug"

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

if not exist "%OUTDIR%" mkdir "%OUTDIR%"
xcopy /D /Y "%BIN%\*.dll" "%OUTDIR%\" >nul 2>&1

set "SRCNAME=%~n1"
cl.exe /EHsc /std:c++17 /W3 /DNDEBUG /DCV_IGNORE_DEBUG_BUILD_GUARD /DNOMINMAX /I"%INC%" %* /Fe"%OUTDIR%\%SRCNAME%.exe" /Fo"%OUTDIR%\\" /link /LIBPATH:"%LIB%" opencv_core4110.lib opencv_imgproc4110.lib opencv_imgcodecs4110.lib opencv_highgui4110.lib opencv_cudastereo4110.lib opencv_cudafilters4110.lib opencv_cudawarping4110.lib opencv_calib3d4110.lib opencv_features2d4110.lib opencv_flann4110.lib /SUBSYSTEM:CONSOLE

endlocal
