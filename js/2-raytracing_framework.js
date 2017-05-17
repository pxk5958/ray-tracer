/**
 * WebGL Ray Tracer
 *
 * @author Pratith Kanagaraj <pxk5958@rit.edu>, 2017
 */

/* global THREE */

var raytracer2 = function() {
	
////////////////////////////////////////////////////////////////////////////////
// Namespace constants
////////////////////////////////////////////////////////////////////////////////

const ROTATE = 15;
const ZOOM = 10;
const PAN = 6;
const VIEW_ANGLE = 45;
const NEAR = 0.1;
const FAR = 10000;


////////////////////////////////////////////////////////////////////////////////
// Namespace variables
////////////////////////////////////////////////////////////////////////////////

var gl = null;
var canvas = document.getElementById('canvas');
var camera, ui, WIDTH = 800, HEIGHT = 600, nextId = 0;


////////////////////////////////////////////////////////////////////////////////
// Shaders
////////////////////////////////////////////////////////////////////////////////

/**
 * Vertex shader for Ray tracing 
 */
 
var tracerVertexSource = `
attribute vec3 vertex;
uniform vec3 ray00, ray01, ray10, ray11;
varying vec3 primaryRayDir;

void main() {
    vec2 fraction = vertex.xy * 0.5 + 0.5;
    primaryRayDir = mix(mix(ray00, ray01, fraction.y), mix(ray10, ray11, fraction.y), fraction.x);
    gl_Position = vec4(vertex, 1.0);
}
`;

/**
 * Generates Fragment shader for Ray tracing based on the objects in the scene
 */

function generateTracerFragmentSource(objects) {
    return `
precision highp float;

const int MAX_RECURSION = 1;
const float EPSILON = 0.0001;
const float INFINITY = 100000.0;
const float PI = 3.1415926535897932384;

uniform vec3 cameraPos;
varying vec3 primaryRayDir;
uniform float time;
`
+ concat(objects, function(o){ return o.getGlobalCode(); }) +
`
float intersectSphere(vec3 rayOrigin, vec3 rayDir, float sphereRadius, 
    mat4 sphereModel, mat4 sphereInvModel) {
    
    vec3 localRayOrigin = vec3(sphereInvModel * vec4(rayOrigin, 1.0));
    vec3 localRayDir = vec3(sphereInvModel * vec4(rayDir, 0.0));

    float a = dot(localRayDir, localRayDir);
    float b = 2.0 * dot(localRayOrigin, localRayDir);
    float c = dot(localRayOrigin, localRayOrigin) - sphereRadius * sphereRadius;
    float disc = b*b - 4.0*a*c;
    
    if (disc > 0.0) {
        float e = sqrt(disc);
        float denom = 2.0*a;
        
        float t = (-b - e) / denom;  // smaller root
        if (t > 0.0) {
            //vec3 tPoint = localRayOrigin + localRayDir * t;
            //tPoint = vec3(sphereModel * vec4(tPoint, 1.0));
            //return distance(rayOrigin, tPoint);
            return t;
        }
        
        t = (-b + e) / denom;  // larger root
        if (t > 0.0) {
            //vec3 tPoint = localRayOrigin + localRayDir * t;
            //tPoint = vec3(sphereModel * vec4(tPoint, 1.0));
            //return distance(rayOrigin, tPoint);
            return t;
        }
    }
    
    return INFINITY;
}

float intersectPlane(vec3 rayOrigin, vec3 rayDir, mat4 planeModel, mat4 planeInvModel) {
    vec3 localRayOrigin = vec3(planeInvModel * vec4(rayOrigin, 1.0));
    vec3 localRayDir = vec3(planeInvModel * vec4(rayDir, 0.0));
    
    vec3 planeNormal = vec3(0, 1, 0);
    float t = (dot(-localRayOrigin, planeNormal)) / dot(localRayDir, planeNormal);
	
	if(t > 0.0){
		return t;
	}
	
	return INFINITY;
}

float intersectRect(vec3 rayOrigin, vec3 rayDir, float rectA, float rectB, 
    mat4 rectModel, mat4 rectInvModel) {
    
    vec3 localRayOrigin = vec3(rectInvModel * vec4(rayOrigin, 1.0));
    vec3 localRayDir = vec3(rectInvModel * vec4(rayDir, 0.0));
    
    vec3 rectAVec = vec3(rectA, 0, 0), rectBVec = vec3(0, 0, rectB);
    vec3 rectNormal = cross(rectAVec, rectBVec);
    
    float t = (dot(-localRayOrigin, rectNormal)) / dot(localRayDir, rectNormal);
	
	if(t > 0.0){
	    vec3 d = localRayOrigin + localRayDir * t;
	    
	    float ddota = dot(d, rectAVec);
	    float ddotb = dot(d, rectBVec);
	    if (ddota > 0.0 && ddota < rectA*rectA && ddotb > 0.0 && ddotb < rectB*rectB) {
	        return t;
	    }
	}
	
	return INFINITY;
}

float intersectCube(vec3 rayOrigin, vec3 rayDir, vec3 cubeMin, vec3 cubeMax, 
    mat4 cubeModel, mat4 cubeInvModel) {
    
    vec3 localRayOrigin = vec3(cubeInvModel * vec4(rayOrigin, 1.0));
    vec3 localRayDir = vec3(cubeInvModel * vec4(rayDir, 0.0));
    
    vec3 tMin = (cubeMin - localRayOrigin) / localRayDir;
    vec3 tMax = (cubeMax - localRayOrigin) / localRayDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    
    if(tNear > 0.0 && tNear < tFar) {
        return tNear;
    }
    
    return INFINITY;
}

bool isZero(float x) {
    return (x > -1E-9 && x < 1E-9);
}

float cbrt(float x) {
    return (x > 0.0 ? pow(x, 1.0/3.0) : (x < 0.0 ? -pow(-x, 1.0/3.0) : 0.0));
}

int solveQuadric(float c[3], inout float s[2]) {
    float p, q, D;

    // normal form: x^2 + px + q = 0

    p = c[1] / (2.0 * c[2]);
    q = c[0] / c[2];

    D = p * p - q;

    if (isZero(D)) {
    	s[0] = -p;
    	return 1;
    } else if (D > 0.0) {
    	float sqrt_D = sqrt(D);
    
    	s[0] =  sqrt_D - p;
    	s[1] = -sqrt_D - p;
    	return 2;
    }
    
    // else if (D < 0.0)
    return 0;
}

int solveCubic(float c[4], inout float s[3]) {
    int    num;
    float  sub;
    float  A, B, C;
    float  sq_A, p, q;
    float  cb_p, D;

    // normal form: x^3 + Ax^2 + Bx + C = 0
    A = c[ 2 ] / c[ 3 ];
    B = c[ 1 ] / c[ 3 ];
    C = c[ 0 ] / c[ 3 ];

    //  substitute x = y - A/3 to eliminate quadric term: x^3 +px + q = 0
    sq_A = A * A;
    p = 1.0/3.0 * (- 1.0/3.0 * sq_A + B);
    q = 1.0/2.0 * (2.0/27.0 * A * sq_A - 1.0/3.0 * A * B + C);

    // use Cardano's formula
    cb_p = p * p * p;
    D = q * q + cb_p;

    if (isZero(D)) {
		if (isZero(q)) { 
		    // one triple solution
		    s[ 0 ] = 0.0;
		    num = 1;
    	} else { 
    	    // one single and one double solution
    	    float u = cbrt(-q);
    	    s[ 0 ] = 2.0 * u;
    	    s[ 1 ] = - u;
    	    num = 2;
	    }
    } else if (D < 0.0) { 
        // Casus irreducibilis: three real solutions
		float phi = 1.0/3.0 * acos(-q / sqrt(-cb_p));
		float t = 2.0 * sqrt(-p);

		s[ 0 ] =   t * cos(phi);
		s[ 1 ] = - t * cos(phi + PI / 3.0);
		s[ 2 ] = - t * cos(phi - PI / 3.0);
		num = 3;
    } else { 
        // one real solution
		float sqrt_D = sqrt(D);
		float u = cbrt(sqrt_D - q);
		float v = - cbrt(sqrt_D + q);

		s[ 0 ] = u + v;
		num = 1;
    }

    // resubstitute
    sub = 1.0/3.0 * A;
    for (int i = 0; i < 3; ++i) {
        if (i >= num) break;
	    s[ i ] -= sub;
    }

    return num;
}

int solveQuartic(float c[5], inout float s[4]) {
    float  coeffs[4];
    float  z, u, v, sub;
    float  A, B, C, D;
    float  sq_A, p, q, r;
    int    num;

    // normal form: x^4 + Ax^3 + Bx^2 + Cx + D = 0
    A = c[ 3 ] / c[ 4 ];
    B = c[ 2 ] / c[ 4 ];
    C = c[ 1 ] / c[ 4 ];
    D = c[ 0 ] / c[ 4 ];

    // substitute x = y - A/4 to eliminate cubic term: x^4 + px^2 + qx + r = 0
    sq_A = A * A;
    p = - 3.0/8.0 * sq_A + B;
    q = 1.0/8.0 * sq_A * A - 1.0/2.0 * A * B + C;
    r = - 3.0/256.0*sq_A*sq_A + 1.0/16.0*sq_A*B - 1.0/4.0*A*C + D;

    if (isZero(r)) {
		// no absolute term: y(y^3 + py + q) = 0
		float cubeS[3];
		
		coeffs[ 0 ] = q;
		coeffs[ 1 ] = p;
		coeffs[ 2 ] = 0.0;
		coeffs[ 3 ] = 1.0;
		
		cubeS[ 0 ] = s[0];
        cubeS[ 1 ] = s[1];
        cubeS[ 2 ] = s[2];

		num = solveCubic(coeffs, cubeS);
		
		s[ 0 ] = cubeS[ 0 ];
		s[ 1 ] = cubeS[ 1 ];
		s[ 2 ] = cubeS[ 2 ];

        // s[ num++ ] = 0.0
        for (int i = 0; i < 4; i++) {
            if (i == num) {
                s[ i ] = 0.0;
                num++;
                break;
            }
        }
    } else {
		// solve the resolvent cubic ...
		float cubeS[3];
		
		coeffs[ 0 ] = 1.0/2.0 * r * p - 1.0/8.0 * q * q;
		coeffs[ 1 ] = - r;
		coeffs[ 2 ] = - 1.0/2.0 * p;
		coeffs[ 3 ] = 1.0;
		
        cubeS[ 0 ] = s[0];
        cubeS[ 1 ] = s[1];
        cubeS[ 2 ] = s[2];
        
		solveCubic(coeffs, cubeS);
		
		s[ 0 ] = cubeS[ 0 ];
		s[ 1 ] = cubeS[ 1 ];
		s[ 2 ] = cubeS[ 2 ];

		// ... and take the one real solution ...
		z = s[ 0 ];

		// ... to build two quadric equations
		u = z * z - r;
		v = 2.0 * z - p;

		if (isZero(u))
		    u = 0.0;
		else if (u > 0.0)
		    u = sqrt(u);
		else
		    return 0;

		if (isZero(v))
		    v = 0.0;
		else if (v > 0.0)
		    v = sqrt(v);
		else
		    return 0;

        float quadCoeffs[3];
        float quadS[2];
        
		quadCoeffs[ 0 ] = z - u;
		quadCoeffs[ 1 ] = q < 0.0 ? -v : v;
		quadCoeffs[ 2 ] = 1.0;

        quadS[ 0 ] = s[ 0 ];
        quadS[ 1 ] = s[ 1 ];
        
		num = solveQuadric(quadCoeffs, quadS);
		
		s[ 0 ] = quadS[ 0 ];
		s[ 1 ] = quadS[ 1 ];

		quadCoeffs[ 0 ] = z + u;
		quadCoeffs[ 1 ] = q < 0.0 ? v : -v;
		quadCoeffs[ 2 ] = 1.0;

        for (int i = 0; i <= 2; i++) {
            if (i == num) {
                quadS[ 0 ] = s[ i ];
                quadS[ 1 ] = s[ i+1 ];
                break;
            }
        }
        
		int num2 = solveQuadric(quadCoeffs, quadS);
		
		for (int i = 0; i <= 2; i++) {
            if (i == num) {
                s[ i ] = quadS[ 0 ];
		        s[ i+1 ] = quadS[ 1 ];
                break;
            }
        }
		
		num += num2;
	}

    // resubstitute
    sub = 1.0/4.0 * A;
    for (int i = 0; i < 4; ++i) {
        if (i >= num) break;
		s[ i ] -= sub;
    }

    return num;
}

float intersectTorus(vec3 rayOrigin, vec3 rayDir, float a, float b, 
    mat4 torusModel, mat4 torusInvModel) {
    
    vec3 localRayOrigin = vec3(torusInvModel * vec4(rayOrigin, 1.0));
    vec3 localRayDir = vec3(torusInvModel * vec4(rayDir, 0.0));
    
    float x1 = localRayOrigin.x, y1 = localRayOrigin.y, z1 = localRayOrigin.z,
    d1 = localRayDir.x, d2 = localRayDir.y, d3 = localRayDir.z;
    
    float coeffs[5];    // coefficient array for the quartic equation
    float roots[4];    // solution array for the quartic equation
    
    // define the coefficients of the quartic equation
	float sum_d_sqrd 	= d1 * d1 + d2 * d2 + d3 * d3;
	float e			    = x1 * x1 + y1 * y1 + z1 * z1 - a * a - b * b;
	float f			    = x1 * d1 + y1 * d2 + z1 * d3;
	float four_a_sqrd	= 4.0 * a * a;
	
	coeffs[0] = e * e - four_a_sqrd * (b * b - y1 * y1); 	// constant term
	coeffs[1] = 4.0 * f * e + 2.0 * four_a_sqrd * y1 * d2;
	coeffs[2] = 2.0 * sum_d_sqrd * e + 4.0 * f * f + four_a_sqrd * d2 * d2;
	coeffs[3] = 4.0 * sum_d_sqrd * f;
	coeffs[4] = sum_d_sqrd * sum_d_sqrd;  					// coefficient of t^4
	
	// find roots of the quartic equation
	int num_real_roots = solveQuartic(coeffs, roots);
    
    if (num_real_roots > 0) {
        bool intersected = false;
        float t = INFINITY;
        
        // find the smallest root greater than 0, if any
    	// the roots array is not sorted
    	for (int j = 0; j < 4; j++)  {
    	    if (j >= num_real_roots) break;
    		if (roots[j] > 0.0) {
    			intersected = true;
    			if (roots[j] < t) {
    				t = roots[j];
    			}
    		}
        }
        
        if (intersected == true) {
            return t;
        }
    }
    
    return INFINITY;
}

float intersectCylinder(vec3 rayOrigin, vec3 rayDir, 
    float cylBottom, float cylTop, float cylRadius, 
    mat4 cylModel, mat4 cylInvModel) {
    
    vec3 localRayOrigin = vec3(cylInvModel * vec4(rayOrigin, 1.0));
    vec3 localRayDir = vec3(cylInvModel * vec4(rayDir, 0.0));
    
    float ox = localRayOrigin.x, oy = localRayOrigin.y, oz = localRayOrigin.z,
        dx = localRayDir.x, dy = localRayDir.y, dz = localRayDir.z;
        
    float a = dx * dx + dz * dz;
    float b = 2.0 * (ox * dx + oz * dz);
    float c = ox * ox + oz * oz - cylRadius * cylRadius;
    float disc = b * b - 4.0 * a * c;
    
    if (disc > 0.0) {
        float e = sqrt(disc);
        float denom = 2.0*a;
        
        float t0 = (-b - e) / denom;  // smaller root
        float t1 = (-b + e) / denom;  // larger root
        if (t0 > t1) {
            // swap t0 and t1
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        
        float y0 = oy + t0 * dy;
        float y1 = oy + t1 * dy;
        
        if (y0 < cylBottom) {
        	if (y1 >= cylBottom) {
        		// hit the bottom cap
        		float th = t0 + (t1-t0) * (y0 - cylBottom) / (y0-y1);
        		if (th > 0.0) {
        		    // normal: vec3(0, -1, 0)
        		    return th;
        		}
        	}
        } else if (y0 >= cylBottom && y0 <= cylTop) {
        	// hit the cylinder part
        	if (t0 > 0.0) {
        	    return t0;
        	}
        } else if (y0 > cylTop) {
        	if (y1 <= cylTop) {
        		// hit the top cap
        		float th = t0 + (t1-t0) * (y0 - cylTop) / (y0-y1);
        		if (th > 0.0) {
        		    // normal: vec3(0, 1, 0)
        		    return th;
        		}
        	}
        }
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
    gl_FragColor = vec4(rayTrace(cameraPos, primaryRayDir), 1.0);
}
    `;
}


////////////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////////////

/**
 * Concatenates strings generated by given functions
 */

function concat(objects, functionPtr) {
    var result = '';
    for (var i = 0; i < objects.length; i++) {
        result += functionPtr(objects[i]);
    }
    return result;
}

/**
 * Gets world space direction for primary ray from the camera
 */

function getPrimaryRay(x, y) {
    var vector = new THREE.Vector3( x, y, 0.5 ).unproject( camera );
    var dir = vector.sub(camera.position).normalize();
    return dir;
}

/**
 * Set uniforms in GLSL shader program
 */
 
function setUniforms(program, uniforms) {
    for(var name in uniforms) {
        var value = uniforms[name];
        var location = gl.getUniformLocation(program, name);
        if(location == null) continue;
        if(value instanceof THREE.Vector2) {
            //console.log(name + ": " + "[" + value.x + ", " + value.y + "]");
            gl.uniform2fv(location, new Float32Array([value.x, value.y]));
        } else if(value instanceof THREE.Vector3) {
            //console.log(name + ": " + "[" + value.x + ", " + value.y + ", " + value.z + "]");
            gl.uniform3fv(location, new Float32Array([value.x, value.y, value.z]));
        } else if(value instanceof THREE.Color) {
            //console.log(name + ": " + "[" + value.r + ", " + value.g + ", " + value.b + "]");
            gl.uniform3fv(location, new Float32Array([value.r, value.g, value.b]));
        } else if(value instanceof THREE.Matrix4) {
            //console.log(name + ": " + value.toArray());
            gl.uniformMatrix4fv(location, false, new Float32Array(value.toArray()));
        } else {
            //console.log(name + ": " + value);
            gl.uniform1f(location, value);
        }
    }
}

/**
 * Compiles GLSL shader source, creates and returns the shader
 */
 
function compileSource(source, type) {
    var shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if(!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw 'compile error: ' + gl.getShaderInfoLog(shader);
    }
    return shader;
}

/**
 * Compiles GLSL vertex and fragment shaders, creates and returns shader program
 */
 
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


////////////////////////////////////////////////////////////////////////////////
// class Ray used for selecting objects
////////////////////////////////////////////////////////////////////////////////

class Ray {
    constructor(origin, dir) {
        this.origin = origin;
        this.dir = dir;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Abstract class for geometric primitives
////////////////////////////////////////////////////////////////////////////////

class Primitive {
    constructor(id) {
        if (this.constructor === Primitive) {
            throw new Error("Primitive: cannot instantiate abstract class.");
        }
        
        this.modelTransform = new THREE.Matrix4().identity();
        this.invModelTransform = new THREE.Matrix4().identity();
        this.color = new THREE.Color();
        
        this.intersectStr = 't' + id;
        this.colorStr = 'color' + id;
        this.modelStr = 'model' + id;
        this.invModelStr = 'invModel' + id;
    }
    
    
    // Sets color of the primitive
    
    setColor(color) {
        this.color = color;
    }
    
    getGlobalCode() {
        return `
uniform vec3 ` + this.colorStr + `;
uniform mat4 ` + this.modelStr + `;
uniform mat4 ` + this.invModelStr + `;
        `;
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
        renderer.uniforms[this.modelStr] = this.modelTransform;
        renderer.uniforms[this.invModelStr] = this.invModelTransform;
    }
    
    intersect(ray) {
        throw new Error("Primitive: cannot call abstract method 'intersect'");
    }
    
    translate(translation) {
        this.modelTransform = new THREE.Matrix4().makeTranslation(translation.x, translation.y, translation.z).multiply(this.modelTransform);
        this.invModelTransform = new THREE.Matrix4().getInverse(this.modelTransform, true);
    }
    
    rotate(rotationEuler) {
        this.modelTransform = new THREE.Matrix4().makeRotationFromEuler(rotationEuler).multiply(this.modelTransform);
        this.invModelTransform = new THREE.Matrix4().getInverse(this.modelTransform, true);
    }
    
    scale(scale) {
        this.modelTransform = new THREE.Matrix4().makeScale(scale.x, scale.y, scale.z).multiply(this.modelTransform);
        this.invModelTransform = new THREE.Matrix4().getInverse(this.modelTransform, true);
    }
}


////////////////////////////////////////////////////////////////////////////////
// class Plane in point-normal form with point at origin 
// and normal pointing upwards
////////////////////////////////////////////////////////////////////////////////

class Plane extends Primitive {
    constructor(id) {
        super(id);
    }

    getIntersectCode() {
        return `
float ` + this.intersectStr + ` = intersectPlane(rayOrigin, rayDir, ` 
+ this.modelStr  + `, ` + this.invModelStr + `);
        `;
    }

    intersect(ray) {
        // TODO: perform intersection test for object selection
        
        return Number.MAX_VALUE;
    }

    scale(scale) {
        throw new Error("Plane: cannot call abstract method 'scale'");
    }
}


////////////////////////////////////////////////////////////////////////////////
// class Rectangle with left bottom corner at origin, left side of length
// 'a', bottom side of length 'b' and normal pointing upwards
////////////////////////////////////////////////////////////////////////////////

class Rectangle extends Primitive {
    constructor(id, a, b) {
        super(id);
        this.a = a;
        this.b = b;
        this.aStr = 'rectA' + id;
        this.bStr = 'rectB' + id;
    }

    getGlobalCode() {
        return super.getGlobalCode() + `
uniform float ` + this.aStr + `;
uniform float ` + this.bStr + `;
        `;
    }

    getIntersectCode() {
        return `
float ` + this.intersectStr + ` = intersectRect(rayOrigin, rayDir, ` 
+ this.aStr + `, ` + this.bStr + `, ` 
+ this.modelStr  + `, ` + this.invModelStr + `);
        `;
    }
    
    setUniforms(renderer) {
        super.setUniforms(renderer);
        renderer.uniforms[this.aStr] = this.a;
        renderer.uniforms[this.bStr] = this.b;
    }

    intersect(ray) {
        // TODO: perform intersection test for object selection
        
        return Number.MAX_VALUE;
    }
}


////////////////////////////////////////////////////////////////////////////////
// class Sphere centered at origin
////////////////////////////////////////////////////////////////////////////////
 
class Sphere extends Primitive {
    constructor(id, radius) {
        super(id);
        this.radius = radius;
        this.radiusStr = 'sphereRadius' + id;
    }

    getGlobalCode() {
        return super.getGlobalCode() + `
uniform float ` + this.radiusStr + `;
        `;
    }

    getIntersectCode() {
        return `
float ` + this.intersectStr + ` = intersectSphere(rayOrigin, rayDir, ` 
+ this.radiusStr + `, ` + this.modelStr  + `, ` + this.invModelStr + `);
        `;
    }
    
    setUniforms(renderer) {
        super.setUniforms(renderer);
        renderer.uniforms[this.radiusStr] = this.radius;
    }

    intersect(ray) {
        // TODO: perform intersection test for object selection
        
        return Number.MAX_VALUE;
    }
}


////////////////////////////////////////////////////////////////////////////////
// class Cube represented by min corner and max corner points
////////////////////////////////////////////////////////////////////////////////

class Cube extends Primitive { 
    constructor(id, minCorner, maxCorner) {
        super(id);
        this.minCorner = minCorner;
        this.maxCorner = maxCorner;
        this.minStr = 'cubeMin' + id;
        this.maxStr = 'cubeMax' + id;
    }
    
    getGlobalCode() {
        return super.getGlobalCode() + `
uniform vec3 ` + this.minStr + `;
uniform vec3 ` + this.maxStr + `;
        `;
    }

    getIntersectCode() {
        return `
float ` + this.intersectStr + ` = intersectCube(rayOrigin, rayDir, `
+ this.minStr + `, ` + this.maxStr + `, ` 
+ this.modelStr  + `, ` + this.invModelStr + `);
        `;
    }
    
    setUniforms(renderer) {
        super.setUniforms(renderer);
        renderer.uniforms[this.minStr] = this.minCorner;
        renderer.uniforms[this.maxStr] = this.maxCorner;
    }

    intersect(ray) {
        // TODO: perform intersection test for object selection
        
        return Number.MAX_VALUE;
    }
}


////////////////////////////////////////////////////////////////////////////////
// class Torus
////////////////////////////////////////////////////////////////////////////////

class Torus extends Primitive { 
    constructor(id, a, b) {
        super(id);
        this.a = a;
        this.b = b;
        this.aStr = 'torusA' + id;
        this.bStr = 'torusB' + id;
    }
    
    getGlobalCode() {
        return super.getGlobalCode() + `
uniform float ` + this.aStr + `;
uniform float ` + this.bStr + `;
        `;
    }

    getIntersectCode() {
        return `
float ` + this.intersectStr + ` = intersectTorus(rayOrigin, rayDir, ` 
+ this.aStr + `, ` + this.bStr + `, ` 
+ this.modelStr  + `, ` + this.invModelStr + `);
        `;
    }
    
    setUniforms(renderer) {
        super.setUniforms(renderer);
        renderer.uniforms[this.aStr] = this.a;
        renderer.uniforms[this.bStr] = this.b;
    }

    intersect(ray) {
        // TODO: calculate bounding box and perform intersection test for object selection
        
        return Number.MAX_VALUE;
    }
}


////////////////////////////////////////////////////////////////////////////////
// class Cylinder with y-axis as centre, 'bottom' as bottom cap's y coordinate,
// 'top' as top cap's y coordinate, and radius
////////////////////////////////////////////////////////////////////////////////

class Cylinder extends Primitive { 
    constructor(id, bottom, top, radius) {
        super(id);
        this.bottom = bottom;
        this.top = top;
        this.radius = radius;
        this.bottomStr = 'cylBottom' + id;
        this.topStr = 'cylTop' + id;
        this.radiusStr = 'cylRadius' + id;
    }
    
    getGlobalCode() {
        return super.getGlobalCode() + `
uniform float ` + this.bottomStr + `;
uniform float ` + this.topStr + `;
uniform float ` + this.radiusStr + `;
        `;
    }

    getIntersectCode() {
        return `
float ` + this.intersectStr + ` = intersectCylinder(rayOrigin, rayDir, ` 
+ this.bottomStr + `, ` + this.topStr + `, ` + this.radiusStr + `, ` 
+ this.modelStr  + `, ` + this.invModelStr + `);
        `;
    }
    
    setUniforms(renderer) {
        super.setUniforms(renderer);
        renderer.uniforms[this.bottomStr] = this.bottom;
        renderer.uniforms[this.topStr] = this.top;
        renderer.uniforms[this.radiusStr] = this.radius;
    }

    intersect(ray) {
        // TODO: calculate bounding box and perform intersection test for object selection
        
        return Number.MAX_VALUE;
    }
}



////////////////////////////////////////////////////////////////////////////////
// class RayTracer
////////////////////////////////////////////////////////////////////////////////

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
        
        // initialize objects and ray tracing shader
        this.objects = [];
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
        //console.log(tracerFragmentSource);
        this.tracerProgram = compileShader(tracerVertexSource, tracerFragmentSource);
        this.tracerVertexAttribute = gl.getAttribLocation(this.tracerProgram, 'vertex');
        gl.enableVertexAttribArray(this.tracerVertexAttribute);
    }
    
    update(time) {
        // calculate uniforms
        for (var i = 0; i < this.objects.length; i++) {
            this.objects[i].setUniforms(this);
        }
        this.uniforms.cameraPos = camera.position.clone();
        this.uniforms.ray00 = getPrimaryRay(-1, -1);
        this.uniforms.ray01 = getPrimaryRay(-1, +1);
        this.uniforms.ray10 = getPrimaryRay(+1, -1);
        this.uniforms.ray11 = getPrimaryRay(+1, +1);
        this.uniforms.time = time;
        
        gl.useProgram(this.tracerProgram);
        
        // set uniforms
        setUniforms(this.tracerProgram, this.uniforms);
    }

    render() {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.vertexAttribPointer(this.tracerVertexAttribute, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
}



////////////////////////////////////////////////////////////////////////////////
// class Renderer
////////////////////////////////////////////////////////////////////////////////
 
class Renderer {
    constructor() {
        this.objects = [];
        this.rayTracer = new RayTracer();
    }
    
    setObjects(objects) {
        this.objects = objects;
        this.rayTracer.setObjects(objects);
    }
    
    update(time) {
        this.rayTracer.update(time);
    }
    
    render() {
        this.rayTracer.render();
    }
}



////////////////////////////////////////////////////////////////////////////////
// class UI
////////////////////////////////////////////////////////////////////////////////
 
class UI {
    constructor() {
        camera = new THREE.PerspectiveCamera( VIEW_ANGLE, WIDTH / HEIGHT, NEAR, FAR);
        
        this.renderer = new Renderer();
        
        // Create controls
    	this.controls = new THREE.TrackballControls( camera, canvas );
    	this.controls.rotateSpeed = ROTATE;
    	this.controls.zoomSpeed = ZOOM;
    	this.controls.panSpeed = PAN;
    	this.controls.noZoom = false;
    	this.controls.noPan = false;
    	this.controls.staticMoving = false;
    	this.controls.dynamicDampingFactor = 0.3;
    	this.resetControls();
    }
    
    setObjects(objects) {
      this.objects = objects;
      this.renderer.setObjects(this.objects);
    }
    
    update(time) {
        this.controls.update();
        camera.updateMatrixWorld(true);
        this.renderer.update(time);
    }
    
    render() {
        this.renderer.render();
    }
    
    resetControls() {
    	this.controls.reset();
    	this.controls.target.set(0, 0, -270);
    	camera.updateMatrixWorld(true);
    }
}


/**
 * Initializes WebGL and creates UI
 */

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
		result.ui = ui;
        var start = new Date();
        setInterval(function(){ tick((new Date() - start) * 0.001); }, 1000 / 60);
    } else {
        alert('Your browser does not support WebGL.');
    }
}

/**
 * Update function
 */

function tick(time) {
    ui.update(time);
    ui.render();
}

/**
 * Generates the scene to be rendered
 */

function generateScene() {
    var objects = [];
    
    var sphere1 = new Sphere(nextId++, 50);
    sphere1.translate(new THREE.Vector3(2, 5, -240));
    sphere1.setColor(new THREE.Color(1, 0, 0));
    
    var sphere2 = new Sphere(nextId++, 40);
    sphere2.translate(new THREE.Vector3(-65, -30, -300));
    sphere2.setColor(new THREE.Color(0, 0, 1));
    
    var floor = new Rectangle(nextId++, 400, -1600);
    floor.translate(new THREE.Vector3(-285, -80, 500));
    floor.setColor(new THREE.Color(0, 1, 0));
    
    var cube = new Cube(nextId++, new THREE.Vector3(-30, -30, 30), new THREE.Vector3(30, 30, -30));
    cube.scale(new THREE.Vector3(1.5, 1, 1));
    cube.rotate(new THREE.Euler(-Math.PI / 4, Math.PI / 4, 0, 'XYZ'));
    cube.translate(new THREE.Vector3(-70, 25, -370));
    cube.setColor(new THREE.Color(0.5, 0, 0.5));
    
    var cylinder = new Cylinder(nextId++, -30, 30, 30);
    cylinder.rotate(new THREE.Euler(-Math.PI / 3, 0, -Math.PI / 6, 'XYZ'));
    cylinder.translate(new THREE.Vector3(50, 70, -300));
    cylinder.setColor(new THREE.Color(0, 0.5, 0.5));
    
    var torus = new Torus(nextId++, 20, 10);
    torus.setColor(new THREE.Color(0.5, 0.5, 0));
    
    objects.push(sphere1);
    objects.push(sphere2);
    objects.push(floor);
    objects.push(cube);
    objects.push(cylinder);
    //objects.push(torus);
    
    return objects;
}

/**
 * Public methods and variables to be returned
 */

var result = {
	initWebGL: initWebGL,
	ui: ui
};

return result;

}();

raytracer2.initWebGL();