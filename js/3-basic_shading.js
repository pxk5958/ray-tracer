/* global THREE */

var raytracer2 = function() {
	
////////////////////////////////////////////////////////////////////////////////
// Namespace constants
////////////////////////////////////////////////////////////////////////////////

// camera constants
const ROTATE = 15.0;
const ZOOM = 10.0;
const PAN = 6.0;
const VIEW_ANGLE = 45.0;
const NEAR = 0.1;
const FAR = 10000.0;

const INFINITY = 100000.1;
const EPSILON = 0.0001;
const SPHERE_EPSILON = 0.095;
const PLANE_RECT_EPSILON = 0.095;
const PI = 3.1415926535897932384;
const MAX_RECURSION_DEPTH = 1;
const SUPERSAMPLING = 5;   // takes average of NxN pixels


////////////////////////////////////////////////////////////////////////////////
// Namespace variables
////////////////////////////////////////////////////////////////////////////////

var gl = null;
var canvas = document.getElementById('canvas2');
var camera, ui, WIDTH = 800, HEIGHT = 600;
var nextObjectId = 0;
var nextPhongId = 0, nextPhongBlinnId = 0;
var nextPointLightId = 0;


////////////////////////////////////////////////////////////////////////////////
// Shaders
////////////////////////////////////////////////////////////////////////////////

/**
 * Vertex shader for rendering to canvas 
 */

var renderVertexSource = `
attribute vec3 vertex;
varying vec2 texCoord;

void main() {
    texCoord = vertex.xy * 0.5 + 0.5;
    gl_Position = vec4(vertex, 1.0);
}
`;

/**
 * Fragment shader for rendering to canvas 
 */

var renderFragmentSource = `
precision highp float;

const int SUPERSAMPLING = ` + SUPERSAMPLING + `;

varying vec2 texCoord;
uniform sampler2D texture;

void main() {
    // weighted samples from the larger render-to-texture to the canvas
    vec4 sampledColor = vec4(0.0, 0.0, 0.0, 1.0);
    if (mod(float(SUPERSAMPLING), 2.0) > 0.0) {
        const int oddN = SUPERSAMPLING / 2;
        for (int i = -oddN; i <= oddN; i++) {
            for (int j = -oddN; j <= oddN; j++) {
                vec2 offset = vec2(float(i) / float(` + WIDTH * SUPERSAMPLING + `), 
                                float(j) / float(` + HEIGHT * SUPERSAMPLING + `));
                sampledColor += texture2D(texture, clamp(texCoord + offset, vec2(0.0), vec2(1.0)));
            }
        }
    } else {
        const float evenN = float(SUPERSAMPLING / 2) - 0.5;
        for (float i = -evenN; i <= evenN; i += 1.0) {
            for (float j = -evenN; j <= evenN; j += 1.0) {
                vec2 offset = vec2(i / float(` + WIDTH * SUPERSAMPLING + `), 
                                j / float(` + HEIGHT * SUPERSAMPLING + `));
                sampledColor += texture2D(texture, clamp(texCoord + offset, vec2(0.0), vec2(1.0)));
            }
        }
    }
    gl_FragColor = sampledColor / float(SUPERSAMPLING * SUPERSAMPLING);
}
`;

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

const int MAX_RECURSION_DEPTH = ` + MAX_RECURSION_DEPTH + `;
const float PI = ` + PI + `;
const float INFINITY = ` + INFINITY + `;
const float EPSILON = ` + EPSILON + `;
const float SPHERE_EPSILON = ` + SPHERE_EPSILON + `;
const float PLANE_RECT_EPSILON = ` + PLANE_RECT_EPSILON + `;

uniform vec3 bgColor;
uniform vec3 ambientLight;
uniform vec3 cameraPos;
varying vec3 primaryRayDir;
uniform float time;

// Intersection information
struct HitInfo {
    bool hit;
    vec3 hitPoint;
    vec3 localHitPoint;
    vec3 normal;
    float t;
    int materialType;
    int materialId;
};

// Materials
int PHONG_MATERIAL = 0;
int PHONG_BLINN_MATERIAL = 1;
struct Phong {
    float ka;
    float kd;
    float ks;
    float ke;
    vec3 Co;
    vec3 Cs;
};
uniform Phong phongMaterials[` + (nextPhongId > 0 ? nextPhongId : 1) + `];
uniform Phong phongBlinnMaterials[` + (nextPhongBlinnId > 0 ? nextPhongBlinnId : 1) + `];

// Lights
struct PointLight {
    vec3 position;
    vec3 color;
};
uniform PointLight pointLights[` + (nextPointLightId > 0 ? nextPointLightId : 1) + `];
`
+ concat(objects, function(o){ return o.getGlobalCode(); }) +
`
bool intersectSphere(vec3 rayOrigin, vec3 rayDir, inout HitInfo hitInfo,
    mat4 sphereModel, mat4 sphereInvModel) {
    vec3 localRayOrigin = vec3(sphereInvModel * vec4(rayOrigin, 1.0));
    vec3 localRayDir = vec3(sphereInvModel * vec4(rayDir, 0.0));

    float a = dot(localRayDir, localRayDir);
    float b = 2.0 * dot(localRayOrigin, localRayDir);
    float c = dot(localRayOrigin, localRayOrigin) - 1.0;
    float disc = b*b - 4.0*a*c;
    
    if (disc > 0.0) {
        float e = sqrt(disc);
        float denom = 2.0*a;
        
        float t0 = (-b - e) / denom;  // smaller root
        if (t0 > SPHERE_EPSILON) {
            hitInfo.t = t0;
            hitInfo.localHitPoint = localRayOrigin + localRayDir * hitInfo.t;
            hitInfo.normal = hitInfo.localHitPoint;
            return true;
        }
        
        t0 = (-b + e) / denom;  // larger root
        if (t0 > SPHERE_EPSILON) {
            hitInfo.t = t0;
            hitInfo.localHitPoint = localRayOrigin + localRayDir * hitInfo.t;
            hitInfo.normal = hitInfo.localHitPoint;  // model space
            hitInfo.normal = vec3(sphereModel * vec4(hitInfo.normal, 0.0));  // world space
            return true;
        }
    }
    
    return false;
}

bool intersectPlane(vec3 rayOrigin, vec3 rayDir, inout HitInfo hitInfo,
    mat4 planeModel, mat4 planeInvModel) {
    vec3 localRayOrigin = vec3(planeInvModel * vec4(rayOrigin, 1.0));
    vec3 localRayDir = vec3(planeInvModel * vec4(rayDir, 0.0));
    
    vec3 planeNormal = vec3(0, 1, 0);
    float t0 = (dot(-localRayOrigin, planeNormal)) / dot(localRayDir, planeNormal);
	
	if(t0 > PLANE_RECT_EPSILON){
		hitInfo.t = t0;
		hitInfo.localHitPoint = localRayOrigin + localRayDir * hitInfo.t;
        hitInfo.normal = planeNormal;  // model space
        hitInfo.normal = vec3(planeModel * vec4(hitInfo.normal, 0.0));  // world space
		return true;
	}
	
	return false;
}

bool intersectRect(vec3 rayOrigin, vec3 rayDir, inout HitInfo hitInfo,
    mat4 rectModel, mat4 rectInvModel) {
    
    vec3 localRayOrigin = vec3(rectInvModel * vec4(rayOrigin, 1.0));
    vec3 localRayDir = vec3(rectInvModel * vec4(rayDir, 0.0));
    
    vec3 rectAVec = vec3(1.0, 0, 0), rectBVec = vec3(0, 0, -1.0);
    vec3 rectNormal = cross(rectAVec, rectBVec);
    
    float t0 = (dot(-localRayOrigin, rectNormal)) / dot(localRayDir, rectNormal);
	
	if(t0 > PLANE_RECT_EPSILON){
	    vec3 d = localRayOrigin + localRayDir * t0;
	    
	    float ddota = dot(d, rectAVec);
	    float ddotb = dot(d, rectBVec);
	    if (ddota > 0.0 && ddota < 1.0 && ddotb > 0.0 && ddotb < 1.0) {
	        hitInfo.t = t0;
	        hitInfo.localHitPoint = localRayOrigin + localRayDir * hitInfo.t;
            hitInfo.normal = rectNormal;  // model space
            hitInfo.normal = vec3(rectModel * vec4(hitInfo.normal, 0.0));  // world space
	        return true;
	    }
	}
	
	return false;
}

bool intersectCube(vec3 rayOrigin, vec3 rayDir, inout HitInfo hitInfo,
    mat4 cubeModel, mat4 cubeInvModel) {
    
    vec3 localRayOrigin = vec3(cubeInvModel * vec4(rayOrigin, 1.0));
    vec3 localRayDir = vec3(cubeInvModel * vec4(rayDir, 0.0));
    
    vec3 t0 = (vec3(-0.5, -0.5, -0.5) / abs(localRayDir)) - (localRayOrigin / localRayDir);
    vec3 t1 = (vec3(0.5, 0.5, 0.5) / abs(localRayDir)) - (localRayOrigin / localRayDir);
    
    float tMin = max( max(t0.x, t0.y), t0.z );
    float tMax = min( min(t1.x, t1.y), t1.z );
    
    if (tMin > tMax || tMax < 0.0) return false;
    
    hitInfo.t = tMin;
    hitInfo.localHitPoint = localRayOrigin + localRayDir * hitInfo.t;
    // normal calculation code taken from https://www.shadertoy.com/view/ld23DV
    hitInfo.normal = -sign(localRayDir)*step(t0.yzx,t0.xyz)*step(t0.zxy,t0.xyz);
    hitInfo.normal = vec3(cubeModel * vec4(hitInfo.normal, 0.0));  // world space
    return true;
}

bool intersectCylinder(vec3 rayOrigin, vec3 rayDir, inout HitInfo hitInfo, 
    mat4 cylModel, mat4 cylInvModel) {
    
    vec3 localRayOrigin = vec3(cylInvModel * vec4(rayOrigin, 1.0));
    vec3 localRayDir = vec3(cylInvModel * vec4(rayDir, 0.0));
    
    float ox = localRayOrigin.x, oy = localRayOrigin.y, oz = localRayOrigin.z,
        dx = localRayDir.x, dy = localRayDir.y, dz = localRayDir.z;
        
    float a = dx * dx + dz * dz;
    float b = 2.0 * (ox * dx + oz * dz);
    float c = ox * ox + oz * oz - 1.0;
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
        
        if (y0 < -0.5) {
        	if (y1 >= -0.5) {
        		// hit the bottom cap
        		float th = t0 + (t1-t0) * (y0 + 0.5) / (y0-y1);
        		if (th > 0.0) {
        		    // normal: vec3(0, -1, 0)
        		    hitInfo.t = th;
        		    hitInfo.localHitPoint = localRayOrigin + localRayDir * hitInfo.t;
                    hitInfo.normal = vec3(0, -1, 0);  // local space
                    hitInfo.normal = vec3(cylModel * vec4(hitInfo.normal, 0.0));  // world space
                    return true;
        		}
        	}
        } else if (y0 >= -0.5 && y0 <= 0.5) {
        	// hit the cylinder part
        	if (t0 > 0.0) {
        	    hitInfo.t = t0;
        	    hitInfo.localHitPoint = localRayOrigin + localRayDir * hitInfo.t;
                hitInfo.normal = vec3(hitInfo.localHitPoint.x, 0, hitInfo.localHitPoint.z);  // local space
                hitInfo.normal = vec3(cylModel * vec4(hitInfo.normal, 0.0));  // world space
                return true;
        	}
        } else if (y0 > 0.5) {
        	if (y1 <= 0.5) {
        		// hit the top cap
        		float th = t0 + (t1-t0) * (y0 - 0.5) / (y0-y1);
        		if (th > 0.0) {
        		    // normal: vec3(0, 1, 0)
        		    hitInfo.t = th;
        		    hitInfo.localHitPoint = localRayOrigin + localRayDir * hitInfo.t;
                    hitInfo.normal = vec3(0, 1, 0);  // local space
                    hitInfo.normal = vec3(cylModel * vec4(hitInfo.normal, 0.0));  // world space
                    return true;
        		}
        	}
        }
    }
    
    return false;
}

bool castShadow(vec3 rayOrigin, vec3 rayDir) {
    rayDir = normalize(rayDir);
    HitInfo hitInfo;
    hitInfo.hit = false;
    
    // find if shadow ray intersects any object
`
+ concat(objects, function(o){ return o.getShadowCode(); }) +
`
    return false;
}

vec3 illuminate(HitInfo hitInfo, vec3 rayDir) {
    vec3 accumulatedColor = vec3(0.0);
    
    Phong material;
    // TODO: due to limitation of GLSL to have constant array indices, have to
    // iterate over the materials to find the correct one. Any better workaround?
    if (hitInfo.materialType == PHONG_MATERIAL) {
        for (int i = 0; i < ` + nextPhongId + `; i++) {
            if (i == hitInfo.materialId) {
                material = phongMaterials[i];
                break;
            }
        }
    } else if (hitInfo.materialType == PHONG_BLINN_MATERIAL) {
        for (int i = 0; i < ` + nextPhongBlinnId + `; i++) {
            if (i == hitInfo.materialId) {
                material = phongBlinnMaterials[i];
                break;
            }
        }
    }
        
    vec3 N = normalize(hitInfo.normal);
    vec3 V = normalize(-rayDir);
    
    vec3 ambient = material.ka * ambientLight * material.Co;
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);
    
    // Iterate over the point lights
    for (int i = 0; i < ` + nextPointLightId + `; i++) {
        vec3 S = normalize(pointLights[i].position - hitInfo.hitPoint);
        bool isInShadow = castShadow(hitInfo.hitPoint + hitInfo.normal * EPSILON, S);
        if (isInShadow == false) {
            float SdotN = max(0.0, dot(S, N));
            float diffuseCoeff = material.kd * SdotN;
            diffuseCoeff = clamp(diffuseCoeff, 0.0, 1.0);
            diffuse += diffuseCoeff * pointLights[i].color * material.Co;
            
            if (SdotN > 0.0) {
                float specularCoeff;
                if (hitInfo.materialType == PHONG_MATERIAL) {
                    vec3 R = normalize(reflect(-S, N));
                    float RdotV = max(0.0, dot(R, V));
                    specularCoeff = material.ks * pow(RdotV, material.ke);
                } else if (hitInfo.materialType == PHONG_BLINN_MATERIAL) {
                    vec3 H = normalize( S + V );
                    float NdotH = max(0.0, dot(N, H));
                    specularCoeff = material.ks * pow(NdotH, material.ke);
                }
                specularCoeff = clamp(specularCoeff, 0.0, 1.0);
                specular += specularCoeff * pointLights[i].color * material.Cs;
            }
        }
    }
    
    accumulatedColor += ambient + specular + diffuse;
    
    return accumulatedColor;
}

vec3 castRay(vec3 rayOrigin, vec3 rayDir) {
    rayDir = normalize(rayDir);
    vec3 colorMask = vec3(1.0);
    vec3 accumulatedColor = vec3(0.0);
    
    for (int depth = 0; depth < MAX_RECURSION_DEPTH; depth++) {
            float tMin = INFINITY;
            vec3 normal;
            vec3 localHitPoint;
            vec3 hitPoint;
            int materialType;
            int materialId;
            HitInfo hitInfo;
            hitInfo.hit = false;
            
            // find closest intersection
`
+ concat(objects, function(o){ return o.getClosestIntersectCode(); }) +
`
            if (hitInfo.hit) {
                hitInfo.t = tMin;
                hitInfo.normal = normal;
                hitInfo.localHitPoint = localHitPoint;
                hitInfo.hitPoint = hitPoint;
                hitInfo.materialType = materialType;
                hitInfo.materialId = materialId;
            } else {
                // ray did not hit any object
                accumulatedColor += colorMask * bgColor;
                break;
            }
        
            accumulatedColor += colorMask * illuminate(hitInfo, rayDir);
        
        // next reflected ray
        rayOrigin = hitInfo.hitPoint;
        rayDir = reflect(rayDir, hitInfo.normal);
    }
    
    return accumulatedColor;
}

void main() {
    gl_FragColor = vec4(castRay(cameraPos, primaryRayDir), 1.0);
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
        this.material = null;
        
        this.modelStr = 'model' + id;
        this.invModelStr = 'invModel' + id;
    }
    
    
    // Sets material of the primitive
    
    setMaterial(material) {
        this.material = material;
    }
    
    getGlobalCode() {
        return `
uniform mat4 ` + this.modelStr + `;
uniform mat4 ` + this.invModelStr + `;
        `;
    }
    
    getClosestIntersectCode() {
        return `
if(` + this.getIntersectCode() + ` && (hitInfo.t < tMin)) {
    hitInfo.hit = true;
    hitPoint = rayOrigin + rayDir * hitInfo.t;
    tMin = hitInfo.t;
    normal = hitInfo.normal;
    localHitPoint = hitInfo.localHitPoint;
    materialType = ` + this.material.getType() + `;
    materialId = ` + this.material.getId() + `;
}
        `;
    }
    
    getShadowCode() {
        return `
if(` + this.getIntersectCode() + `) {
    return true;
}
        `;
    }
    
    getIntersectCode() {
        throw new Error("Primitive: cannot call abstract method 'getIntersectCode'");
    }
    
    setUniforms(renderer) {
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
intersectPlane(rayOrigin, rayDir, hitInfo, ` + this.modelStr  + `, ` 
+ this.invModelStr + `)
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
    constructor(id) {
        super(id);
    }

    getIntersectCode() {
        return `
intersectRect(rayOrigin, rayDir, hitInfo, `
+ this.modelStr  + `, ` + this.invModelStr + `)
        `;
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
    constructor(id) {
        super(id);
    }

    getIntersectCode() {
        return `
intersectSphere(rayOrigin, rayDir, hitInfo, ` 
+ this.modelStr + `, ` + this.invModelStr + `)
        `;
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
    constructor(id) {
        super(id);
    }

    getIntersectCode() {
        return `
intersectCube(rayOrigin, rayDir, hitInfo, ` 
+ this.modelStr  + `, ` + this.invModelStr + `)
        `;
    }

    intersect(ray) {
        // TODO: perform intersection test for object selection
        
        return Number.MAX_VALUE;
    }
}


////////////////////////////////////////////////////////////////////////////////
// class Cylinder with y-axis as centre, 'bottom' as bottom cap's y coordinate,
// 'top' as top cap's y coordinate, and radius
////////////////////////////////////////////////////////////////////////////////

class Cylinder extends Primitive { 
    constructor(id) {
        super(id);
    }
    
    getIntersectCode() {
        return `
intersectCylinder(rayOrigin, rayDir, hitInfo, `
+ this.modelStr  + `, ` + this.invModelStr + `)
        `;
    }

    intersect(ray) {
        // TODO: calculate bounding box and perform intersection test for object selection
        
        return Number.MAX_VALUE;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Abstract class for materials
////////////////////////////////////////////////////////////////////////////////

class Material {
    static get PHONG_MATERIAL() {
        return 0;
    }
    
    static get PHONG_BLINN_MATERIAL() {
        return 1;
    }
    
    constructor(type, id) {
        if (this.constructor === Material) {
            throw new Error("Material: cannot instantiate abstract class.");
        }
        
        this.type = type;
        this.id = id;
    }
    
    getType() {
        return this.type;
    }
    
    getId() {
        return this.id;
    }
    
    setUniforms(renderer) {
        throw new Error("Material: cannot call abstract method 'setUniforms'");
    }
}


////////////////////////////////////////////////////////////////////////////////
// class Phong
////////////////////////////////////////////////////////////////////////////////

class Phong extends Material {
    constructor(id, ka, kd, ks, ke, Co, Cs) {
        super(Material.PHONG_MATERIAL, id);
        this.ka = ka;
        this.kd = kd;
        this.ks = ks;
        this.ke = ke;
        this.Co = Co;
        this.Cs = Cs;
    }
    
    setUniforms(renderer) {
        renderer.uniforms["phongMaterials[" + this.id + "].ka"] = this.ka;
        renderer.uniforms["phongMaterials[" + this.id + "].kd"] = this.kd;
        renderer.uniforms["phongMaterials[" + this.id + "].ks"] = this.ks;
        renderer.uniforms["phongMaterials[" + this.id + "].ke"] = this.ke;
        renderer.uniforms["phongMaterials[" + this.id + "].Co"] = this.Co;
        renderer.uniforms["phongMaterials[" + this.id + "].Cs"] = this.Cs;
    }
}


////////////////////////////////////////////////////////////////////////////////
// class PhongBlinn
////////////////////////////////////////////////////////////////////////////////

class PhongBlinn extends Material {
    constructor(id, ka, kd, ks, ke, Co, Cs) {
        super(Material.PHONG_BLINN_MATERIAL, id);
        this.ka = ka;
        this.kd = kd;
        this.ks = ks;
        this.ke = ke;
        this.Co = Co;
        this.Cs = Cs;
    }
    
    setUniforms(renderer) {
        renderer.uniforms["phongBlinnMaterials[" + this.id + "].ka"] = this.ka;
        renderer.uniforms["phongBlinnMaterials[" + this.id + "].kd"] = this.kd;
        renderer.uniforms["phongBlinnMaterials[" + this.id + "].ks"] = this.ks;
        renderer.uniforms["phongBlinnMaterials[" + this.id + "].ke"] = this.ke;
        renderer.uniforms["phongBlinnMaterials[" + this.id + "].Co"] = this.Co;
        renderer.uniforms["phongBlinnMaterials[" + this.id + "].Cs"] = this.Cs;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Abstract class for lights
////////////////////////////////////////////////////////////////////////////////

class Light {
    constructor(id) {
        if (this.constructor === Light) {
            throw new Error("Light: cannot instantiate abstract class.");
        }
        
        this.id = id;
    }
    
    setUniforms(renderer) {
        throw new Error("Light: cannot call abstract method 'setUniforms'");
    }
}


////////////////////////////////////////////////////////////////////////////////
// class PointLight
////////////////////////////////////////////////////////////////////////////////

class PointLight extends Light {
    constructor(id, position, color) {
        super(id);
        this.position = position;
        this.color = color;
    }
    
    setUniforms(renderer) {
        renderer.uniforms["pointLights[" + this.id + "].position"] = this.position;
        renderer.uniforms["pointLights[" + this.id + "].color"] = this.color;
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
        
        // create frame buffer
        this.framebuffer = gl.createFramebuffer();
        
        // create textures to render to
        var type = gl.getExtension('OES_texture_float') ? gl.FLOAT : gl.UNSIGNED_BYTE;
        this.textures = [];
        for (var i = 0; i < 1; i++) {
            this.textures.push(gl.createTexture());
            gl.bindTexture(gl.TEXTURE_2D, this.textures[i]);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, WIDTH * SUPERSAMPLING, HEIGHT * SUPERSAMPLING, 0, gl.RGB, type, null);
        }
        gl.bindTexture(gl.TEXTURE_2D, null);
        
        // create render shader
        this.renderProgram = compileShader(renderVertexSource, renderFragmentSource);
        this.renderVertexAttribute = gl.getAttribLocation(this.renderProgram, 'vertex');
        gl.enableVertexAttribArray(this.renderVertexAttribute);
        
        // initialize objects and ray tracing shader
        this.scene = {
            objects: [],
            materials: [],
            lights: []
        };
        this.tracerProgram = null;
    }

    setScene(scene) {
        this.uniforms = {};
        this.sampleCount = 0;
        this.scene = scene;
        
        // create ray tracing shader
        if (this.tracerProgram != null) {
            gl.deleteProgram(this.tracerProgram);
        }
        var tracerFragmentSource = generateTracerFragmentSource(this.scene.objects);
        //console.log(tracerFragmentSource);
        this.tracerProgram = compileShader(tracerVertexSource, tracerFragmentSource);
        this.tracerVertexAttribute = gl.getAttribLocation(this.tracerProgram, 'vertex');
        gl.enableVertexAttribArray(this.tracerVertexAttribute);
    }
    
    update(time) {
        // calculate uniforms for objects
        for (var i = 0; i < this.scene.objects.length; i++) {
            this.scene.objects[i].setUniforms(this);
        }
        // calculate uniforms for materials
        for (var i = 0; i < this.scene.materials.length; i++) {
            this.scene.materials[i].setUniforms(this);
        }
        // calculate uniforms for lights
        for (var i = 0; i < this.scene.lights.length; i++) {
            this.scene.lights[i].setUniforms(this);
        }
        this.uniforms.ambientLight = this.scene.ambientLight;
        this.uniforms.bgColor = this.scene.bgColor;
        this.uniforms.cameraPos = camera.position.clone();
        this.uniforms.ray00 = getPrimaryRay(-1, -1);
        this.uniforms.ray01 = getPrimaryRay(-1, +1);
        this.uniforms.ray10 = getPrimaryRay(+1, -1);
        this.uniforms.ray11 = getPrimaryRay(+1, +1);
        this.uniforms.time = time;
        
        // render to texture
        gl.useProgram(this.tracerProgram);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
        gl.viewport(0, 0, WIDTH * SUPERSAMPLING, HEIGHT * SUPERSAMPLING);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.textures[0], 0);
        gl.vertexAttribPointer(this.tracerVertexAttribute, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, WIDTH, HEIGHT);
        
        // set uniforms
        setUniforms(this.tracerProgram, this.uniforms);
    }

    render() {
        // render from texture to canvas
        gl.useProgram(this.renderProgram);
        gl.bindTexture(gl.TEXTURE_2D, this.textures[0]);
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
    
    setScene(scene) {
        this.scene = scene;
        this.rayTracer.setScene(this.scene);
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
    
    setScene(scene) {
      this.scene = scene;
      this.renderer.setScene(this.scene);
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
        var scene = generateScene();
        ui.setScene(scene);
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
    var materials = [];
    var lights = [];
    
    var pointLight1 = new PointLight(nextPointLightId++,
        new THREE.Vector3(10, 300, -10), 
        new THREE.Vector3(1, 1, 1));
        
    var pointLight2 = new PointLight(nextPointLightId++,
        new THREE.Vector3(-300, 50, 200), 
        new THREE.Vector3(0.1, 1, 0.1));
        
    lights.push(pointLight1);
    lights.push(pointLight2);
    
    var sphere1Material = new Phong(nextPhongId++, 0.7, 0.6, 0.7, 16, 
        new THREE.Vector3(0.75, 0.1, 0.1), 
        new THREE.Vector3(0.7, 0.7, 0.7));
        
    var sphere2Material = new Phong(nextPhongId++, 0.55, 0.7, 0.75, 8, 
        new THREE.Vector3(0.1, 0.1, 0.75), 
        new THREE.Vector3(0.7, 0.7, 0.7));
        
    var cubeMaterial = new Phong(nextPhongId++, 0.5, 0.9, 0.2, 24, 
        new THREE.Vector3(0.8, 0.1, 0.7), 
        new THREE.Vector3(0.7, 0.7, 0.7));
    
    var cylMaterial = new Phong(nextPhongId++, 0.5, 0.3, 0.9, 20, 
        new THREE.Vector3(0.85, 0.7, 0.1), 
        new THREE.Vector3(0.7, 0.7, 0.7));
    
    var floorMaterial = new Phong(nextPhongId++, 0.1, 0.9, 0.2, 12, 
        new THREE.Vector3(0.1, 0.75, 0.1), 
        new THREE.Vector3(0.7, 0.7, 0.7));
    
    materials.push(sphere1Material);
    materials.push(sphere2Material);
    materials.push(cubeMaterial);
    materials.push(cylMaterial);
    materials.push(floorMaterial);
    
    var sphere1 = new Sphere(nextObjectId++);
    sphere1.scale(new THREE.Vector3(50, 50, 50));
    sphere1.translate(new THREE.Vector3(2, 15, -240));
    sphere1.setMaterial(sphere1Material);
    
    var sphere2 = new Sphere(nextObjectId++);
    sphere2.scale(new THREE.Vector3(40, 40, 40));
    sphere2.translate(new THREE.Vector3(-75, -20, -325));
    sphere2.setMaterial(sphere2Material);
    
    var floor = new Rectangle(nextObjectId++);
    floor.scale(new THREE.Vector3(450, 1, 1600));
    floor.translate(new THREE.Vector3(-330, -80, 500));
    floor.setMaterial(floorMaterial);
    
    var cube = new Cube(nextObjectId++);
    cube.scale(new THREE.Vector3(90, 60, 60));
    cube.rotate(new THREE.Euler(-Math.PI / 4, Math.PI / 4, 0, 'XYZ'));
    cube.translate(new THREE.Vector3(-70, 40, -400));
    cube.setMaterial(cubeMaterial);
    
    var cylinder = new Cylinder(nextObjectId++);
    cylinder.scale(new THREE.Vector3(30, 60, 30));
    cylinder.rotate(new THREE.Euler(-Math.PI / 3, 0, -Math.PI / 6, 'XYZ'));
    cylinder.translate(new THREE.Vector3(50, 75, -300));
    cylinder.setMaterial(cylMaterial);
    
    objects.push(sphere1);
    objects.push(sphere2);
    objects.push(floor);
    objects.push(cube);
    objects.push(cylinder);
    
    return {
        objects: objects, 
        materials: materials,
        lights: lights,
        ambientLight: new THREE.Vector3(0.75, 0.75, 0.75),
        bgColor: new THREE.Vector3(0.1, 0.5, 0.9)
    };
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