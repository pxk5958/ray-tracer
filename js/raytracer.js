/* global THREE */


// Set the controls speeds
const ROTATE = 15;
const ZOOM = 10;
const PAN = 6;
const VIEW_ANGLE = 45;
const NEAR = 0.1;
const FAR = 10000;

var gl = null;
var canvas = document.getElementById('canvas');
var camera, ui, WIDTH = 800, HEIGHT = 600, nextId = 0;



var tracerVertexSource = `
attribute vec3 vertex;
uniform vec3 ray00, ray01, ray10, ray11;
varying vec3 primaryRayDir;

void main() {
    vec2 fraction = vertex.xy * 0.5 + 0.5;
    primaryRayDir = mix(mix(ray00, ray01, fraction.y), mix(ray10, ray11, fraction.y), fraction.x);
    normalize(primaryRayDir);
    gl_Position = vec4(vertex, 1.0);
}
`;

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

float intersectRect(vec3 rayOrigin, vec3 rayDir, vec3 rectP0, vec3 rectA, vec3 rectB, vec3 rectNormal) {
    float t = (dot(rectP0 - rayOrigin, rectNormal)) / dot(rayDir, rectNormal);
	
	if(t > 0.0){
	    vec3 p = rayOrigin + rayDir * t;
	    vec3 d = p - rectP0;
	    
	    float ddota = dot(d, rectA);
	    float ddotb = dot(d, rectB);
	    if (ddota > 0.0 && ddota < dot(rectA, rectA) && ddotb > 0.0 && ddotb < dot(rectB, rectB)) {
	        return t;
	    }
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

bool isZero(double x) {
    return (x > -1E-9 && x < 1E-9);
}

double cbrt(double x) {
    return (x > 0.0 ? pow(x, 1.0/3.0) : (x < 0.0 ? -pow(-x, 1.0/3.0) : 0.0));
}

int solveQuadric(double c[3], inout double s[2]) {
    double p, q, D;

    // normal form: x^2 + px + q = 0

    p = c[1] / (2.0 * c[2]);
    q = c[0] / c[2];

    D = p * p - q;

    if (isZero(D)) {
    	s[0] = -p;
    	return 1;
    } else if (D > 0.0) {
    	double sqrt_D = sqrt(D);
    
    	s[0] =  sqrt_D - p;
    	s[1] = -sqrt_D - p;
    	return 2;
    }
    
    // else if (D < 0.0)
    return 0;
}

int solveCubic(double c[4], inout double s[3]) {
    int    num;
    double  sub;
    double  A, B, C;
    double  sq_A, p, q;
    double  cb_p, D;

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
    	    double u = cbrt(-q);
    	    s[ 0 ] = 2.0 * u;
    	    s[ 1 ] = - u;
    	    num = 2;
	    }
    } else if (D < 0.0) { 
        // Casus irreducibilis: three real solutions
		double phi = 1.0/3.0 * acos(-q / sqrt(-cb_p));
		double t = 2.0 * sqrt(-p);

		s[ 0 ] =   t * cos(phi);
		s[ 1 ] = - t * cos(phi + PI / 3.0);
		s[ 2 ] = - t * cos(phi - PI / 3.0);
		num = 3;
    } else { 
        // one real solution
		double sqrt_D = sqrt(D);
		double u = cbrt(sqrt_D - q);
		double v = - cbrt(sqrt_D + q);

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

int solveQuartic(double c[5], inout double s[4]) {
    double  coeffs[4];
    double  z, u, v, sub;
    double  A, B, C, D;
    double  sq_A, p, q, r;
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
		double cubeS[3];
		
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
		double cubeS[3];
		
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

        double quadCoeffs[3];
        double quadS[2];
        
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

float intersectTorus(vec3 rayOrigin, vec3 rayDir, float a, float b) {
    float x1 = rayOrigin.x, y1 = rayOrigin.y, z1 = rayOrigin.z,
    d1 = rayDir.x, d2 = rayDir.y, d3 = rayDir.z;
    
    double coeffs[5];    // coefficient array for the quartic equation
    double roots[4];    // solution array for the quartic equation
    
    // define the coefficients of the quartic equation
	double sum_d_sqrd 	= d1 * d1 + d2 * d2 + d3 * d3;
	double e			    = x1 * x1 + y1 * y1 + z1 * z1 - a * a - b * b;
	double f			    = x1 * d1 + y1 * d2 + z1 * d3;
	double four_a_sqrd	= 4.0 * a * a;
	
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

function getPrimaryRay(x, y) {
    var vector = new THREE.Vector3( x, y, 0.5 ).unproject( camera );
    var dir = vector.sub(camera.position).normalize();
    return dir;
}

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
        
        this.modelMatrix = new THREE.Matrix4().identity();
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
    
    translate(translation) {
        this.modelMatrix = new THREE.Matrix4().makeTranslation(translation.x, translation.y, translation.z).multiply(this.modelMatrix);
    }
    
    rotate(rotationEuler) {
        this.modelMatrix = new THREE.Matrix4().makeRotationFromEuler(rotationEuler).multiply(this.modelMatrix);
    }
    
    scale(scale) {
        this.modelMatrix = new THREE.Matrix4().makeScale(scale.x, scale.y, scale.z).multiply(this.modelMatrix);
    }
}


/**
 * class Plane
 */

class Plane extends Primitive {
    constructor(id, point, normal) {
        super();
        this.point = point;
        this.normal = normal.normalize();
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
    
    translate(translation) {
        super.translate(translation);
        this.point.applyMatrix4(this.modelMatrix);
    }
    
    rotate(rotationEuler) {
        throw new Error("Plane: cannot call abstract method 'rotate'");
    }
    
    scale(scale) {
        throw new Error("Plane: cannot call abstract method 'scale'");
    }
}



/**
 * class Rectangle
 */

class Rectangle extends Primitive {
    constructor(id, p0, a, b, normal) {
        super();
        this.p0 = p0;
        this.a = a;
        this.b = b;
        if (typeof normal === 'undefined') {
            this.normal = new THREE.Vector3().crossVectors(a, b).normalize();
        } else {
            this.normal = normal.normalize();
        }
        this.p0Str = 'rectP0' + id;
        this.aStr = 'rectA' + id;
        this.bStr = 'rectB' + id;
        this.normalStr = 'rectNormal' + id;
        this.intersectStr = 'tRect' + id;
        this.colorStr = 'rectColor' + id;
    }

    getGlobalCode() {
        return `
uniform vec3 ` + this.p0Str + `;
uniform vec3 ` + this.aStr + `;
uniform vec3 ` + this.bStr + `;
uniform vec3 ` + this.normalStr + `;
uniform vec3 ` + this.colorStr + `;
        `;
    }

    getIntersectCode() {
        return `
float ` + this.intersectStr + ` = intersectRect(rayOrigin, rayDir, ` 
+ this.p0Str + `, ` + this.aStr + `, ` + this.bStr + `, ` + this.normalStr + `);
        `;
    }
    
    setUniforms(renderer) {
        super.setUniforms(renderer);
        renderer.uniforms[this.p0Str] = this.p0;
        renderer.uniforms[this.aStr] = this.a;
        renderer.uniforms[this.bStr] = this.b;
        renderer.uniforms[this.normalStr] = this.normal;
    }

    intersect(ray) {
        var t = (this.p0.clone().sub(ray.origin)).dot(this.normal) 
                / (ray.dir.clone().dot(this.normal));
        
        if (t > 0) {
            var p = ray.origin.clone().add(ray.dir.clone().multiplyScalar(t));
            var d = p.clone().sub(this.p0);
            
            var ddota = d.dot(this.a);
            var ddotb = d.dot(this.b);
            if (ddota > 0.0 && ddota < this.a.lengthSq() 
                && ddotb > 0.0 && ddotb < this.b.lengthSq()) {
                return t;
            }
        }
        
        return Number.MAX_VALUE;
    }
    
    translate(translation) {
        super.translate(translation);
        this.p0.applyMatrix4(this.modelMatrix);
    }
    
    rotate(rotationEuler) {
        throw new Error("Rectangle: cannot call abstract method 'rotate'");
    }
    
    scale(scale) {
        throw new Error("Rectangle: cannot call abstract method 'scale'");
    }
}



/**
 * class Sphere
 */
 
class Sphere extends Primitive {
    constructor(id, center, radius) {
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
    
    translate(translation) {
        super.translate(translation);
        this.center.applyMatrix4(this.modelMatrix);
    }
    
    rotate(rotationEuler) {
        super.rotate(rotationEuler);
        this.center.applyMatrix4(this.modelMatrix);
    }
    
    scale(scale) {
        super.scale(new THREE.Vector3(scale, scale, scale));
        this.center.applyMatrix4(this.modelMatrix);
        this.radius = this.radius * scale;
    }
}


/**
 * class Cube
 */

class Cube extends Primitive { 
    constructor(id, minCorner, maxCorner) {
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
    
    translate(translation) {
        super.translate(translation);
        this.minCorner.applyMatrix4(this.modelMatrix);
        this.maxCorner.applyMatrix4(this.modelMatrix);
    }
    
    rotate(rotationEuler) {
        throw new Error("Cube: cannot call abstract method 'rotate'");
    }
    
    scale(scale) {
        throw new Error("Cube: cannot call abstract method 'scale'");
    }
}



/**
 * class Torus
 */

class Torus extends Primitive { 
    constructor(id, a, b) {
        super();
        this.a = a;
        this.b = b;
        this.aStr = 'torusA' + id;
        this.bStr = 'torusB' + id;
        this.intersectStr = 'tTorus' + id;
        this.colorStr = 'torusColor' + id;
    }
    
    getGlobalCode() {
        return `
uniform float ` + this.aStr + `;
uniform float ` + this.bStr + `;
uniform vec3 ` + this.colorStr + `;
        `;
    }

    getIntersectCode() {
        return `
float ` + this.intersectStr + ` = intersectTorus(rayOrigin, rayDir, ` + this.aStr + `, ` + this.bStr + `);
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
    
    translate(translation) {
        throw new Error("Torus: cannot call abstract method 'translate'");
    }
    
    rotate(rotationEuler) {
        throw new Error("Torus: cannot call abstract method 'rotate'");
    }
    
    scale(scale) {
        throw new Error("Torus: cannot call abstract method 'scale'");
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
        console.log(tracerFragmentSource);
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
    
    update(time) {
        this.rayTracer.update(time);
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
    
    var sphere1 = new Sphere(nextId++, new THREE.Vector3(), 50);
    sphere1.translate(new THREE.Vector3(2, 5, -240));
    sphere1.setColor(new THREE.Color(1, 0, 0));
    
    var sphere2 = new Sphere(nextId++, new THREE.Vector3(), 40);
    sphere2.translate(new THREE.Vector3(-65, -30, -300));
    sphere2.setColor(new THREE.Color(0, 0, 1));
    
    var floorP0 = new THREE.Vector3();
    var floorA = new THREE.Vector3(400, 0, 0);
    var floorB = new THREE.Vector3(0, 0, -1600);
    var floor = new Rectangle(nextId++, floorP0, floorA, floorB);
    floor.translate(new THREE.Vector3(-285, -80, 500));
    floor.setColor(new THREE.Color(0, 1, 0));
    
    var cube = new Cube(nextId++, new THREE.Vector3(-30, -30, 30), new THREE.Vector3(30, 30, -30));
    cube.translate(new THREE.Vector3(-50, 75, -320));
    cube.setColor(new THREE.Color(0.5, 0, 0.5));
    
    var torus = new Torus(nextId++, 20, 10);
    torus.setColor(new THREE.Color(0.5, 0.5, 0));
    
    objects.push(sphere1);
    objects.push(sphere2);
    objects.push(floor);
    objects.push(cube);
    objects.push(torus);
    
    return objects;
}