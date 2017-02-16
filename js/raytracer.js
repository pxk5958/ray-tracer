/* global THREE */

var gl = null;
var canvas = document.getElementById('canvas');
var camera, ui, WIDTH = 800, HEIGHT = 600, nextId = 0;



var renderVertexSource = `
attribute vec3 vertex;
varying vec2 texCoord;

void main() {
    texCoord = vertex.xy * 0.5 + 0.5;
    gl_Position = vec4(vertex, 1.0);
}
`;

var renderFragmentSource = `
precision highp float;

varying vec2 texCoord;
uniform sampler2D texture;

void main() {
    gl_FragColor = texture2D(texture, texCoord);
}
`;

var tracerVertexSource = `
attribute vec3 vertex;
uniform vec3 ray00, ray01, ray10, ray11;
varying vec3 primaryRayDir;

void main() {
    vec2 fraction = vertex.xy * 0.5 + 0.5;
    primaryRayDir = mix(mix(ray00, ray01, fraction.y), mix(ray10, ray11, fraction.y), fraction.x);;
    gl_Position = vec4(vertex, 1.0);
}
`;

function generateTracerFragmentSource(objects) {
    return `
precision highp float;

const int MAX_RECURSION = 1;
const float EPSILON = 0.0001;
const float INFINITY = 10000.0;

uniform vec3 cameraPos;
varying vec3 primaryRayDir;
uniform float time;
uniform float textureWeight;
uniform sampler2D texture;
`
+ concat(objects, function(o){ return o.getGlobalCode(); }) +
`
float intersectSphere(vec3 rayOrigin, vec3 rayDir, vec3 sphereCenter, float sphereRadius) {
	vec3 temp = rayOrigin - sphereCenter;
    float a = dot(rayDir, rayDir);
    float b = 2.0 * dot(temp, rayDir);
    float c = dot(temp, temp) - sphereRadius * sphereRadius;
    float disc = b*b - 4.0*a*c;
    
    if (disc > 0.0) {
        float e = sqrt(disc);
        float denom = 2.0*a;
        
        float t = (-b - e) / denom;  // smaller root
        if (t > 0.0) {
            return t;
        }
        
        t = (-b + e) / denom;  // larger root
        if (t > 0.0) {
            return t;
        }
    }
    
    return INFINITY;
}

float intersectPlane(vec3 rayOrigin, vec3 rayDir, vec3 planePoint, vec3 planeNormal) {
    float t = (dot(planePoint - rayOrigin, planeNormal)) / dot(rayDir, planeNormal);
	
	if(t > 0.0){
		return t;
	}
	
	return INFINITY;
}

float intersectCube(vec3 rayOrigin, vec3 rayDir, vec3 cubeMin, vec3 cubeMax) {
    vec3 tMin = (cubeMin - rayOrigin) / rayDir;
    vec3 tMax = (cubeMax - rayOrigin) / rayDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    
    if(tNear > 0.0 && tNear < tFar) {
        return tNear;
    }
    
    return INFINITY;
}

vec3 rayTrace(vec3 rayOrigin, vec3 rayDir) {
    vec3 colorMask = vec3(1.0);
    vec3 accumulatedColor = vec3(0.0);
    
    for (int i = 0; i < MAX_RECURSION; i++) {
        // find all intersections
`
+ concat(objects, function(o){ return o.getIntersectCode(); }) +
`
    
        // find closest intersection
        float t = INFINITY;
        vec3 hitColor = vec3(0.0);
`
+ concat(objects, function(o){ return o.getClosestIntersectCode(); }) +
`
        // intersection point
        vec3 hitPoint = rayOrigin + rayDir * t;
        
        // calculate color
        if (t < INFINITY) {
            accumulatedColor += colorMask * hitColor;
        }
        
        // next ray origin
        rayOrigin = hitPoint;
    }
    
    return accumulatedColor;
}

void main() {
    vec3 texture = texture2D(texture, vec2(gl_FragCoord.x / ` + WIDTH + `, gl_FragCoord.y / ` + HEIGHT + `)).rgb;
    gl_FragColor = vec4(mix(rayTrace(cameraPos, primaryRayDir), texture, textureWeight), 1.0);
}
    `;
}



/**
 * Utility functions
 */
 
function concat(objects, functionPtr) {
    var result = '';
    for (var i = 0; i < objects.length; i++) {
        result += functionPtr(objects[i]);
    }
    return result;
}

function getPrimaryRay(invViewProjectionMatrix, x, y) {
    var projectedPoint = new THREE.Vector4(x, y, 0, 1);
    projectedPoint.applyMatrix4(invViewProjectionMatrix);
    projectedPoint.divideScalar(projectedPoint.w);
    var unprojectedPoint = new THREE.Vector3(projectedPoint.x, projectedPoint.y, projectedPoint.z);
    return unprojectedPoint.sub(camera.position);
}

function setUniforms(program, uniforms) {
    for(var name in uniforms) {
        var value = uniforms[name];
        var location = gl.getUniformLocation(program, name);
        if(location == null) continue;
        if(value instanceof THREE.Vector3) {
            gl.uniform3fv(location, new Float32Array([value.x, value.y, value.z]));
        } else if(value instanceof THREE.Color) {
            gl.uniform3fv(location, new Float32Array([value.r, value.g, value.b]));
        } else if(value instanceof THREE.Matrix4) {
            gl.uniformMatrix4fv(location, false, new Float32Array(value.toArray()));
        } else {
            gl.uniform1f(location, value);
        }
    }
}

function compileSource(source, type) {
    var shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if(!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw 'compile error: ' + gl.getShaderInfoLog(shader);
    }
    return shader;
}

function compileShader(vertexSource, fragmentSource) {
    var shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, compileSource(vertexSource, gl.VERTEX_SHADER));
    gl.attachShader(shaderProgram, compileSource(fragmentSource, gl.FRAGMENT_SHADER));
    gl.linkProgram(shaderProgram);
    if(!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        throw 'link error: ' + gl.getProgramInfoLog(shaderProgram);
    }
    return shaderProgram;
}



/**
 * class Ray
 */

class Ray {
    constructor(origin, dir) {
        this.origin = origin;
        this.dir = dir;
    }
}



/**
 * Abstract class Primitive
 */

class Primitive {
    constructor() {
        if (this.constructor === Primitive) {
            throw new Error("Primitive: cannot instantiate abstract class.");
        }
        
        this.color = new THREE.Color();
    }
    
    setColor(color) {
        this.color = color;
    }
    
    getGlobalCode() {
        throw new Error("Primitive: cannot call abstract method 'getGlobalCode'");
    }
    
    getClosestIntersectCode() {
        return `
if(` + this.intersectStr + ` < t) {
    t = ` + this.intersectStr + `;
    hitColor = ` + this.colorStr + `;
}
        `;
    }
    
    getIntersectCode() {
        throw new Error("Primitive: cannot call abstract method 'getIntersectCode'");
    }
    
    setUniforms(renderer) {
        renderer.uniforms[this.colorStr] = this.color;
    }
    
    intersect(ray) {
        throw new Error("Primitive: cannot call abstract method 'intersect'");
    }
}


/**
 * class Plane
 */

class Plane extends Primitive {
    constructor(point, normal, id) {
        super();
        this.point = point;
        this.normal = normal;
        this.normal.normalize();
        this.pointStr = 'planePoint' + id;
        this.normalStr = 'planeNormal' + id;
        this.intersectStr = 'tPlane' + id;
        this.colorStr = 'planeColor' + id;
    }

    getGlobalCode() {
        return `
uniform vec3 ` + this.pointStr + `;
uniform vec3 ` + this.normalStr + `;
uniform vec3 ` + this.colorStr + `;
        `;
    }

    getIntersectCode() {
        return `
float ` + this.intersectStr + ` = intersectPlane(rayOrigin, rayDir, ` + this.pointStr + `, ` + this.normalStr + `);
        `;
    }
    
    setUniforms(renderer) {
        super.setUniforms(renderer);
        renderer.uniforms[this.pointStr] = this.point;
        renderer.uniforms[this.normalStr] = this.normal;
    }

    intersect(ray) {
        var t = (this.point.clone().sub(ray.origin)).dot(this.normal) 
                / (ray.dir.clone().dot(this.normal));
        
        if (t > 0) {
            return t;
        }
        
        return Number.MAX_VALUE;
    }
}



/**
 * class Sphere
 */
 
class Sphere extends Primitive {
    constructor(center, radius, id) {
        super();
        this.center = center;
        this.radius = radius;
        this.centerStr = 'sphereCenter' + id;
        this.radiusStr = 'sphereRadius' + id;
        this.intersectStr = 'tSphere' + id;
        this.colorStr = 'sphereColor' + id;
    }

    getGlobalCode() {
        return `
uniform vec3 ` + this.centerStr + `;
uniform float ` + this.radiusStr + `;
uniform vec3 ` + this.colorStr + `;
        `;
    }

    getIntersectCode() {
        return `
float ` + this.intersectStr + ` = intersectSphere(rayOrigin, rayDir, ` + this.centerStr + `, ` + this.radiusStr + `);
        `;
    }
    
    setUniforms(renderer) {
        super.setUniforms(renderer);
        renderer.uniforms[this.centerStr] = this.center;
        renderer.uniforms[this.radiusStr] = this.radius;
    }

    intersect(ray) {
        var temp = ray.origin.clone().sub(this.center);
        var a = ray.dir.clone().dot(ray.dir);
        var b = 2*temp.clone().dot(ray.dir);
        var c = temp.clone().dot(temp) - this.radius*this.radius;
        var disc = b*b - 4*a*c;
        
        if (disc > 0) {
            var e = Math.sqrt(disc);
            var denom = 2*a;
            var t = (-b - e) / denom;  // smaller root
            
            if (t > 0) {
                return t;
            }
            
            t = (-b + e) / denom;  // larger root
            
            if (t > 0) {
                return t;
            }
        }
        
        return Number.MAX_VALUE;
    }
}


/**
 * class Cube
 */

class Cube extends Primitive { 
    constructor(minCorner, maxCorner, id) {
        super();
        this.minCorner = minCorner;
        this.maxCorner = maxCorner;
        this.minStr = 'cubeMin' + id;
        this.maxStr = 'cubeMax' + id;
        this.intersectStr = 'tCube' + id;
        this.colorStr = 'cubeColor' + id;
    }
    
    getGlobalCode() {
        return `
uniform vec3 ` + this.minStr + `;
uniform vec3 ` + this.maxStr + `;
uniform vec3 ` + this.colorStr + `;
        `;
    }

    getIntersectCode() {
        return `
float ` + this.intersectStr + ` = intersectCube(rayOrigin, rayDir, ` + this.minStr + `, ` + this.maxStr + `);
        `;
    }
    
    setUniforms(renderer) {
        super.setUniforms(renderer);
        renderer.uniforms[this.minStr] = this.minCorner;
        renderer.uniforms[this.maxStr] = this.maxCorner;
    }

    intersect(ray) {
        var tMin = this.minCorner.clone().sub(ray.origin).divide(ray.dir);
        var tMax = this.maxCorner.clone().sub(ray.origin).divide(ray.dir);
        var t1 = tMin.clone().min(tMax);
        var t2 = tMin.clone().max(tMax);
        var tNear = Math.max(t1.x, t1.y, t1.z);
        var tFar = Math.min(t2.x, t2.y, t2.z);
        if (tNear > 0 && tNear < tFar) {
            return tNear;
        }
        
        return Number.MAX_VALUE;
    }
}


/**
 * class RayTracer
 */

class RayTracer {
    constructor() {
        var vertices = [
            -1, -1,
            -1,  1,
             1, -1,
             1,  1
        ];
        
        // create vertex buffer
        this.vertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
        
        // create frame buffer
        this.frameBuffer = gl.createFramebuffer();
        
        // create textures to render to
        var type = /*gl.getExtension('OES_texture_float') ? gl.FLOAT :*/ gl.UNSIGNED_BYTE;
        this.textures = [];
        for (var i = 0; i < 2; i++) {
            this.textures.push(gl.createTexture());
            gl.bindTexture(gl.TEXTURE_2D, this.textures[i]);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, WIDTH, HEIGHT, 0, gl.RGB, type, null);
        }
        gl.bindTexture(gl.TEXTURE_2D, null);
        
        // create render shader
        this.renderProgram = compileShader(renderVertexSource, renderFragmentSource);
        this.renderVertexAttribute = gl.getAttribLocation(this.renderProgram, 'vertex');
        gl.enableVertexAttribArray(this.renderVertexAttribute);
        
        // initialize objects and ray tracing shader
        this.objects = [];
        this.sampleCount = 0;
        this.tracerProgram = null;
    }

    setObjects(objects) {
        this.uniforms = {};
        this.sampleCount = 0;
        this.objects = objects;
        
        // create ray tracing shader
        if (this.tracerProgram != null) {
            gl.deleteProgram(this.tracerProgram);
        }
        var tracerFragmentSource = generateTracerFragmentSource(objects);
        this.tracerProgram = compileShader(tracerVertexSource, tracerFragmentSource);
        this.tracerVertexAttribute = gl.getAttribLocation(this.tracerProgram, 'vertex');
        gl.enableVertexAttribArray(this.tracerVertexAttribute);
    }
    
    update(invViewProjectionMatrix, time) {
        // calculate uniforms
        for (var i = 0; i < this.objects.length; i++) {
            this.objects[i].setUniforms(this);
        }
        this.uniforms.cameraPos = camera.position;
        this.uniforms.ray00 = getPrimaryRay(invViewProjectionMatrix, -1, -1);
        this.uniforms.ray01 = getPrimaryRay(invViewProjectionMatrix, -1, +1);
        this.uniforms.ray10 = getPrimaryRay(invViewProjectionMatrix, +1, -1);
        this.uniforms.ray11 = getPrimaryRay(invViewProjectionMatrix, +1, +1);
        this.uniforms.time = time;
        this.uniforms.textureWeight = this.sampleCount / (this.sampleCount + 1);
        
        gl.useProgram(this.tracerProgram);
        
        // set uniforms
        setUniforms(this.tracerProgram, this.uniforms);
        
        // render to texture
        gl.bindTexture(gl.TEXTURE_2D, this.textures[0]);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.frameBuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.textures[1], 0);
        gl.vertexAttribPointer(this.tracerVertexAttribute, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        
        // ping pong textures
        this.textures.reverse();
        this.sampleCount++;
    }

    render() {
        gl.useProgram(this.renderProgram);
        gl.bindTexture(gl.TEXTURE_2D, this.textures[0]);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.vertexAttribPointer(this.renderVertexAttribute, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
}


/**
 * class Renderer
 */
 
class Renderer {
    constructor() {
        this.objects = [];
        this.rayTracer = new RayTracer();
    }
    
    setObjects(objects) {
        this.objects = objects;
        this.rayTracer.setObjects(objects);
    }
    
    update(viewProjection, time) {
        var inverse = new THREE.Matrix4().getInverse(viewProjection);
        this.viewProjection = viewProjection;
        this.rayTracer.update(inverse, time);
    }
    
    render() {
        this.rayTracer.render();
    }
}



/**
 * class UI
 */
 
class UI {
    constructor() {
        camera = new THREE.PerspectiveCamera( 45, WIDTH / HEIGHT, 1, 1000);
        camera.lookAt(new THREE.Vector3(0, 0, -270));
        this.renderer = new Renderer();
    }
    
    setObjects(objects) {
      this.objects = objects;
      this.renderer.setObjects(this.objects);
    }
    
    update(time) {
        this.projection = camera.projectionMatrix;
        this.view = camera.matrixWorldInverse;
        this.viewProjection = new THREE.Matrix4().multiplyMatrices(this.projection, this.view);
        this.renderer.update(this.viewProjection, time);
    }
    
    render() {
        this.renderer.render();
    }
}


initWebGL();

function initWebGL() {
    try { 
        gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl'); 
    } catch(e) {
    }
    
    if (gl) {
        WIDTH = canvas.clientWidth.toFixed(1);
        HEIGHT = canvas.clientHeight.toFixed(1);
        ui = new UI();
        ui.setObjects(generateScene());
        var start = new Date();
        setInterval(function(){ tick((new Date() - start) * 0.001); }, 1000 / 60);
    } else {
        alert('Your browser does not support WebGL.');
    }
}

function tick(time) {
    ui.update(time);
    ui.render();
}

function generateScene() {
    var objects = [];
    var sphere1 = new Sphere(new THREE.Vector3(2, 5, -240), 50, nextId++);
    sphere1.setColor(new THREE.Color(1, 0, 0));
    var sphere2 = new Sphere(new THREE.Vector3(-65, -30, -300), 40, nextId++);
    sphere2.setColor(new THREE.Color(0, 0, 1));
    var floor = new Plane(new THREE.Vector3(-85, -80, -300), new THREE.Vector3(0, 1, 0), nextId++);
    floor.setColor(new THREE.Color(0, 1, 0));
    objects.push(sphere1);
    objects.push(sphere2);
    objects.push(floor);
    return objects;
}

// utility functions
/*
function loadSource(script) {
    var code = "";
    var currentChild = script.firstChild;
    while (currentChild) {
        if (currentChild.nodeType === currentChild.TEXT_NODE) {
            code += currentChild.textContent;
        }
        currentChild = currentChild.nextSibling;
    }
    return code;
}
*/