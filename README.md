<div align="center">
  <article style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
      <img src="https://user-images.githubusercontent.com/71769312/233330675-dd8df43b-232c-4efd-8128-9063c2f4e9cb.svg"/>
      <h1 style="width: 100%; text-align: center;">ToyCV</h1>
  </article>
</div>

ToyCV是一个无第三方库依赖的C++图像处理玩具库（除了拿来玩没啥用），作为我在学习C++和CMake等工具的过程中的练习项目。 这个自我感觉还不错的LOGO也来自学习[Inkscape](https://gitlab.com/inkscape/inkscape)的过程中自己画的。

## 🤡编译使用

1. 克隆项目：

``` shell
git clone https://github.com/geoyee/ToyCV.git
```

- Windows

目前的在Visual Studio已经支持了CMake，可以在Visual Studio加载项目，对根目录的`CMakeLists.txt`右键生成和安装即可，也可使用CMake安装。

- Linux

使用CMake进行安装，目前只在WSL2下进行过测试，过程如下：

``` shell
cd ToyCV
mkdir build && cd build  # 采用外部构建
cmake -DCMAKE_INSTALL_PREFIX=<安装路径> ..
make && make install
```

## 🤡TODO

- [x] 完成基本的图像类`Im`
- [ ] 完成`bmp`的读取和保存
- [ ] 完成简单的图像处理

## 🤡参考

- [BMP files](http://paulbourke.net/dataformats/bmp/)

