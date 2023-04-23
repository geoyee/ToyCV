<div align="center">
  <article style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
      <img src="https://user-images.githubusercontent.com/71769312/233330675-dd8df43b-232c-4efd-8128-9063c2f4e9cb.svg"/>
      <h1 style="width: 100%; text-align: center;">ToyCV</h1>
  </article>
</div>

ToyCV是一个无第三方库依赖的C++图像处理玩具库（除了拿来玩没啥用），作为我在学习C++和CMake等工具的过程中的练习项目。 这个自我感觉还不错的LOGO也来自学习[Inkscape](https://gitlab.com/inkscape/inkscape)的过程中自己画的。

## 😋编译使用

1. 克隆项目：

``` shell
git clone https://github.com/geoyee/ToyCV.git
```

- Windows

目前的在Visual Studio已经支持了CMake，可以在Visual Studio加载项目，对根目录的`CMakeLists.txt`右键生成和安装即可，默认安装在`./out/install`下。

- Linux

可以使用CMake进行安装，目前只在WSL2下进行过测试，过程如下：

``` shell
cd ToyCV
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=<安装路径> ..
make && make install
```

## 🤔 参考

- [纯C++实现24位bmp格式图片的读取和修饰](https://blog.csdn.net/m0_62505136/article/details/121153508)

