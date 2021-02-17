if (Test-Path "./.catkin_workspace") {
    & "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
    set ChocolateyInstall="c:\opt\chocolatey"
    "c:\opt\ros\noetic\x64\setup.bat"
}