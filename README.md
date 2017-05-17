# WebGL 2.0 Ray Tracer

WebGL 2.0 Ray Tracer for dynamic scenes

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Requirements

WebGL 2.0 enabled browser and a good GPU like AMD RX 480. The program has been tested on the following:

```
CPU: Intel Pentium G3258 (4.0 GHz overclock)
GPU: AMD RX 480 (8 GB)
RAM: 8 GB DDR3         
Web browser: Mozilla Firefox, Google Chrome
OS: Windows 10, Ubuntu 14.04
```

### Installing

Just obtain the files and place them in some directory.

### Running

The project consists of seven checkpoints and a k-d Tree advanced checkpoint representing the various stages of its evolution, and the final project:

```
1. Setting the Scene
2. Raytracing Framework
3. Basic Shading
4. Procedural Texturing
5. Reflection
6. Transmission
7. Tone Reproduction

8. Advanced: Spatial Data Structure (k-d Tree)

9. Final Project
```

Open any of the corresponding HTML files with your browser to run the ray tracing program:

```
1-setting_the_scene.html
2-raytracing_framework.html
3-basic_shading.html
4-procedural_texturing.html
5-reflection.html
6-transmission.html
7-tone_reproduction.html

8-kd_tree.html

project.html
```

The scene can be modified by changing the generateScene() method if present.

## Built With

* [Cloud9](https://c9.io/) - Online IDE
* [Three.js](https://threejs.org/) - For Vector and Matrix math

## Author

* **Pratith Kanagaraj**

## Acknowledgments

* Framework inspired by https://github.com/evanw/webgl-path-tracing
* Triangle-AABB intersection from “Fast 3D Triangle-Box Overlap Testing” by Tomas Akenine-Moller [http://www.cs.lth.se/home/Tomas_Akenine_Moller/code/]
* kd-pushdown traversal algorithm from "Review: Kd-tree Traversal Algorithms for Ray Tracing"  by M. Hapala and V. Havran