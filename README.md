<div align="center">
  <article style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
      <img src="https://user-images.githubusercontent.com/71769312/233330675-dd8df43b-232c-4efd-8128-9063c2f4e9cb.svg"/>
      <h1 style="width: 100%; text-align: center;">ToyCV</h1>
  </article>
</div>

ToyCV是一个无第三方库依赖的C++图像处理玩具库（除了拿来玩没啥用），作为我在学习C++、CMake和CUDA等等工具（后续想到啥都搞上，比如pybind11等）的过程中的练习项目。 这个LOGO也来自学习[Inkscape](https://gitlab.com/inkscape/inkscape)的过程中自己画的。

## 🤡编译使用

1. 克隆项目：

``` shell
git clone https://github.com/geoyee/ToyCV.git
```

- Windows

目前的在Visual Studio已经支持了CMake，可以在Visual Studio加载项目，对根目录的`CMakeLists.txt`右键生成和安装即可，也可使用CMake安装。

需要注意，如果使用的Visual Studio版本附带的CMake版本在3.20.3以下，需要设置`CMakeSettings.json`中的`cmakeExecutable`为新版CMake的路径，具体原因见[VS2019使用cmake构建cuda应用报错Couldn't find CUDA library root](https://blog.csdn.net/qq_39798423/article/details/130495878?spm=1001.2014.3001.5502)。

- Linux

使用CMake进行安装，目前只在WSL2下进行过测试，过程如下：

``` shell
cd ToyCV
mkdir build && cd build  # 采用外部构建
cmake -DCMAKE_INSTALL_PREFIX=<安装路径> ..
make && make install
```

## 🤡TODO

- [x] 基本图像类
- [x] 读取24位彩色bmp图像
- [x] 使用cuda将rgb图像转为gray图像
- [ ] ...

## 🤡参考

- [BMP files](http://paulbourke.net/dataformats/bmp/)
- [cuda 如何将二维数组传入显存](https://www.zhihu.com/question/450735975)

