## Prebuilt ANGLE binary for UWP (x86, x64, arm64)

Based on ANGLE master branch, current version is [44722daa5a7ed85f607c8449cf4dbf4724aece9c](https://chromium.googlesource.com/angle/angle/+/44722daa5a7ed85f607c8449cf4dbf4724aece9c) (chromium 3999).

### Compiling instructions

Using the following configuration:

```
# Set build arguments here. See `gn buildargs`.
target_os = "winuwp"
target_cpu = "x86" # x86, x64, arm64. Note that arm is not supported!
is_clang = false
is_debug = false
dcheck_always_on = false
```

Based on instructions from here [DevSetup](https://chromium.googlesource.com/angle/angle/+/HEAD/doc/DevSetup.md)

```
gn args out/Release
autoninja -C out/Release libGLESv2 libEGL
... repeat for all architectures (x86, x64, arm64) ...
```


