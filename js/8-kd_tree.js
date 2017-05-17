/**
 * WebGL Ray Tracer
 *
 * @author Pratith Kanagaraj <pxk5958@rit.edu>, 2017
 */

/* global THREE */
/* global Stats */

////////////////////////////////////////////////////////////////////////////////
// Global constants
////////////////////////////////////////////////////////////////////////////////

const STATS = 0;   // 0: fps, 1: ms, 2: mb, 3+: custom

// Camera constants
const ROTATE = 15.0;
const ZOOM = 10.0;
const PAN = 6.0;
const VIEW_ANGLE = 45.0;
const NEAR = 0.1;
const FAR = 10000.0;

// Tone Reproduction constants
const TONE_REPRODUCTION_METHOD = 0;    // 0 = Ward's TR operator; 1 = Reinhard model
const L_MAX = 1000;
const L_DMAX = 500;
const A = 0.18;

// Ray tracing quality constants
const MAX_RECURSION_DEPTH = 1;
const SUPERSAMPLING = 1;   // takes average of NxN pixels

// Triangle meshes constants
const RENDER_TRIANGLES = true;
const MODEL_FILE_NAME = 'models/bun_zipper_res3.ply';
const USE_KD_TREE = true;
const MAX_KD_TREE_DEPTH = 24;
const STATIC_GEOMETRY = false;

// Math constants
const INFINITY = 100000.0;
const EPSILON = 0.0001;
const SPHERE_EPSILON = 0.095;
const PLANE_RECT_EPSILON = 0.095;
const TRIANGLE_EPSILON = 0.0001;
const PI = 3.1415926535897932384;

////////////////////////////////////////////////////////////////////////////////
// Namespace variables
////////////////////////////////////////////////////////////////////////////////

var gl = null;
var canvas = document.getElementById('canvas');
var camera, ui, WIDTH = 800, HEIGHT = 600;
var nextObjectId = 0;
var nextPhongId = 0, nextPhongBlinnId = 0, nextPhongCheckeredId = 0;
var nextPointLightId = 0;
var g_verts = [];
var g_faces = [];

////////////////////////////////////////////////////////////////////////////////
// Shaders
////////////////////////////////////////////////////////////////////////////////

/**
 * Vertex shader for rendering to canvas 
 */

var renderVertexSource = `#version 300 es

in vec3 vertex;
out vec2 texCoord;

void main() {
    texCoord = vertex.xy * 0.5 + 0.5;
    gl_Position = vec4(vertex, 1.0);
}
`;

/**
 * Fragment shader for rendering to canvas 
 */

var renderFragmentSource = `#version 300 es

precision highp float;

const int SUPERSAMPLING = ` + SUPERSAMPLING + `;

const int TR_METHOD = ` + TONE_REPRODUCTION_METHOD + `;
const int L_MAX = ` + L_MAX + `;
const int L_DMAX = ` + L_DMAX + `;
const float A = ` + A + `;

out vec4 fragmentColor;

in vec2 texCoord;
uniform sampler2D tex;
uniform float sf;
uniform float Lbar;

void main() {
    // weighted samples from the larger render-to-texture to the canvas
    vec3 sampledColor = vec3(0.0, 0.0, 0.0);
    if (mod(float(SUPERSAMPLING), 2.0) > 0.0) {
        const int oddN = SUPERSAMPLING / 2;
        for (int i = -oddN; i <= oddN; i++) {
            for (int j = -oddN; j <= oddN; j++) {
                vec2 offset = vec2(float(i) / float(` + WIDTH * SUPERSAMPLING + `), 
                                float(j) / float(` + HEIGHT * SUPERSAMPLING + `));
                sampledColor += vec3(texture(tex, clamp(texCoord + offset, vec2(0.0), vec2(1.0))));
            }
        }
    } else {
        const float evenN = float(SUPERSAMPLING / 2) - 0.5;
        for (float i = -evenN; i <= evenN; i += 1.0) {
            for (float j = -evenN; j <= evenN; j += 1.0) {
                vec2 offset = vec2(i / float(` + WIDTH * SUPERSAMPLING + `), 
                                j / float(` + HEIGHT * SUPERSAMPLING + `));
                sampledColor += vec3(texture(tex, clamp(texCoord + offset, vec2(0.0), vec2(1.0))));
            }
        }
    }
    sampledColor = sampledColor / float(SUPERSAMPLING * SUPERSAMPLING);
    
    // Tone reproduction
    sampledColor *= float(L_MAX);
    if (TR_METHOD == 0) {
        // Ward Tone Reproduction
        sampledColor *= sf;
    } else {
        // Reinhard model
        sampledColor = sampledColor * float(A) / Lbar;
        for (int i = 0; i < 3; i++) {
            sampledColor[i] = sampledColor[i] * float(L_DMAX) / (1.0 + sampledColor[i]);
        }
    }
    sampledColor /= float(L_DMAX);
    
    fragmentColor = vec4(sampledColor, 1.0);
}
`;

/**
 * Vertex shader for drawing line
 */

var lineVertexSource = `#version 300 es

in vec3 vertex;
uniform vec3 cubeMin;
uniform vec3 cubeMax;
uniform mat4 modelViewProjection;
void main() {
    gl_Position = modelViewProjection * vec4(mix(cubeMin, cubeMax, vertex), 1.0);
}
`;

/**
 * Fragment shader for drawing line
 */

var lineFragmentSource = `#version 300 es

precision highp float;
out vec4 fragmentColor;
void main() {
    fragmentColor = vec4(1.0);
}
`;

/**
 * Vertex shader for Ray tracing 
 */
 
var tracerVertexSource = `#version 300 es

in vec3 vertex;
uniform vec3 ray00, ray01, ray10, ray11;
out vec3 primaryRayDir;
out vec2 texCoord;

void main() {
    vec2 fraction = vertex.xy * 0.5 + 0.5;
    texCoord = fraction;
    primaryRayDir = mix(mix(ray00, ray01, fraction.y), mix(ray10, ray11, fraction.y), fraction.x);
    gl_Position = vec4(vertex, 1.0);
}
`;

/**
 * Generates Fragment shader for Ray tracing based on the objects in the scene
 */

function generateTracerFragmentSource(objects) {
    return `#version 300 es

// kd-pushdown traversal algorithm from "Review: Kd-tree Traversal Algorithms for Ray Tracing"  by M. Hapala and V. Havran

precision highp float;
precision highp int;
precision highp usampler2D;
precision highp isampler2D;

const float PI = ` + PI + `;
const float INFINITY = float(` + INFINITY + `);
const float EPSILON = ` + EPSILON + `;
const float SPHERE_EPSILON = ` + SPHERE_EPSILON + `;
const float PLANE_RECT_EPSILON = ` + PLANE_RECT_EPSILON + `;
const float TRIANGLE_EPSILON = ` + TRIANGLE_EPSILON + `;

in vec2 texCoord;
out vec4 fragmentColor;

uniform vec3 bgColor;
uniform vec3 ambientLight;
uniform vec3 cameraPos;
in vec3 primaryRayDir;
uniform float time;


const int MAX_RECURSION_DEPTH = ` + MAX_RECURSION_DEPTH + `;
const int RAY_TREE_SIZE = int(` + (Math.pow(2, (MAX_RECURSION_DEPTH)) - 1) + `);

// Ray tracing tree node information
struct Node {
    bool valid;
    vec3 rayOrigin;
    vec3 rayDir;
    vec3 color;
    float kReflect;
    float kRefract;
};

Node rayTree[RAY_TREE_SIZE];


// Intersection information
struct HitInfo {
    bool hit;
    vec3 hitPoint;
    vec3 localHitPoint;  // is this needed?
    vec3 normal;
    float t;
    int materialType;
    int materialId;
};

// Materials
const int PHONG_MATERIAL = ` + Material.PHONG_MATERIAL + `;
const int PHONG_BLINN_MATERIAL = ` + Material.PHONG_BLINN_MATERIAL + `;
const int PHONG_CHECKERED_MATERIAL = ` + Material.PHONG_CHECKERED_MATERIAL + `;

struct Phong {
    float ka;
    float kd;
    float ks;
    float ke;
    float kReflect;
    float kRefract;
    float ior;
    vec3 Co;
    vec3 Cs;
};
uniform Phong phongMaterials[` + (nextPhongId > 0 ? nextPhongId : 1) + `];
uniform Phong phongBlinnMaterials[` + (nextPhongBlinnId > 0 ? nextPhongBlinnId : 1) + `];

struct PhongCheckered {
    float ka;
    float kd;
    float ks;
    float ke;
    float kReflect;
    float kRefract;
    float ior;
    vec3 Co1;
    vec3 Co2;
    vec3 Cs;
};
uniform PhongCheckered phongCheckeredMaterials[` + (nextPhongCheckeredId > 0 ? nextPhongCheckeredId : 1) + `];

// Lights
struct PointLight {
    vec3 position;
    vec3 color;
};
uniform PointLight pointLights[` + (nextPointLightId > 0 ? nextPointLightId : 1) + `];

uniform sampler2D vertsTexture;
uniform int vertsTextureSize;

uniform usampler2D facesTexture;
uniform usampler2D facesMatTexture;
uniform int facesTextureSize;
const int numFaces = ` + g_faces.length + `;

// kd-tree node types
const int INVALID = ` + KdNode.INVALID + `;
const int INTERNAL = ` + KdNode.INTERNAL + `;
const int EMPTY_LEAF = ` + KdNode.EMPTY_LEAF + `;
const int LEAF = ` + KdNode.LEAF + `;

const bool RENDER_TRIANGLES = ` + RENDER_TRIANGLES + `;
const bool USE_KD_TREE = ` + USE_KD_TREE + `;

uniform usampler2D kdTreeTexture;
uniform isampler2D kdLeavesTexture;
uniform int kdTreeTextureSize;
uniform int kdLeavesTextureSize;
uniform vec3 kdBoxMin;
uniform vec3 kdBoxMax;

int getKdNodeType(uint metadata) {
    return int(metadata >> 30u);
}

int getKdLeafIndex(uint metadata) {
    return int(metadata & 0x3fffffffu);
}

bool error = false;

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
    vec3 rectP0 = vec3(-0.5, 0, 0.5);
    
    float t0 = (dot((rectP0 - localRayOrigin), rectNormal)) / dot(localRayDir, rectNormal);
	
	if(t0 > PLANE_RECT_EPSILON){
	    vec3 p = localRayOrigin + localRayDir * t0;
	    vec3 d = p - rectP0;
	    
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

bool intersectTriangle(vec3 rayOrigin, vec3 rayDir, inout HitInfo hitInfo, vec3 triVerts[3]) {
    vec3 e1 = triVerts[1] - triVerts[0];
    vec3 e2 = triVerts[2] - triVerts[0];

    vec3 T = rayOrigin - triVerts[0];
    vec3 P = cross(rayDir, e2);
    vec3 Q = cross(T, e1);
    
    // check if ray is parallel to triangle
    float PdotE1 = dot(P, e1);
    if (abs(PdotE1) < TRIANGLE_EPSILON) {
        return false;
    }
    
    vec3 rhs = vec3(dot(Q, e2), dot(P, T), dot(Q, rayDir));
    vec3 lhs = (1.0/PdotE1) * rhs;
    
    hitInfo.t = lhs.x;
    // check if intersection point is behind ray origin
    if (hitInfo.t < 0.0) {
        return false;
    }
    
    float u = lhs.y;
    float v = lhs.z;
    // check if intersection point is outside of triangle
    if (u < 0.0 || v < 0.0 || u+v > 1.0) {
        return false;
    }
    
    hitInfo.normal = normalize(cross(e1, e2));
    return true;
}

vec2 getTexCoord(int i, int textureSize) {
    return vec2(
        (mod(float(i), float(textureSize)) + 0.5) / float(textureSize), 
        (float(i / textureSize) + 0.5) / float(textureSize)
    );
}

bool intersectRayAABB(vec3 rayOrigin, vec3 rayDir, vec3 boxMin, vec3 boxMax, 
    out float tMin, out float tMax) {
    tMin = -INFINITY;
    tMax = INFINITY;
    for (int i = 0; i < 3; i++) {
        float dimLo = (boxMin[i] - rayOrigin[i]) / rayDir[i];
        float dimHi = (boxMax[i] - rayOrigin[i]) / rayDir[i];
        if (dimLo > dimHi) {
            float tmp = dimLo;
            dimLo = dimHi;
            dimHi = tmp;
        }
        // if (dimHi < tMin || dimLo > tMax) {
        //     return false;
        // }
        if (dimLo > tMin) {
            tMin = dimLo;
        }
        if (dimHi < tMax) {
            tMax = dimHi;
        }
    }
    
    if (tMin > tMax || tMax < 0.0) {
        return false;
    }
    
    return true;
}

bool castShadow(vec3 rayOrigin, vec3 rayDir) {
    rayDir = normalize(rayDir);
    HitInfo hitInfo;
    hitInfo.hit = false;
    
    // find if shadow ray intersects any object
`
+ concat(objects, function(o){ return o.getShadowCode(); }) +
`
    // triangles
    
    if (RENDER_TRIANGLES) {
        if (USE_KD_TREE) {
            float kdTMin, kdTMax;
            intersectRayAABB(rayOrigin, rayDir, kdBoxMin, kdBoxMax, kdTMin, kdTMax);
            float maxDist = kdTMax;
            kdTMax = kdTMin;
            uint pushedNode = uint(texture(kdTreeTexture, getTexCoord(0, kdTreeTextureSize)));
            int pushedNodeIndex = 0;
            vec3 pushedBoxMin = kdBoxMin;
            vec3 pushedBoxMax = kdBoxMax;
            int pushedDepth = 0;
            if (getKdNodeType(pushedNode) != INVALID) {
                bool pushdown = true;
                while (kdTMax < maxDist) {
                    uint currNode = pushedNode;
                    int currNodeIndex = pushedNodeIndex;
                    vec3 currBoxMin = pushedBoxMin;
                    vec3 currBoxMax = pushedBoxMax;
                    int currDepth = pushedDepth;
                    kdTMin = kdTMax;
                    kdTMax = maxDist;
                    while (getKdNodeType(currNode) != EMPTY_LEAF
                            && getKdNodeType(currNode) != LEAF) {
                        int axis = currDepth % 3;
                        float currSplitPos = currBoxMin[axis] + 0.5 * (currBoxMax[axis] - currBoxMin[axis]);
                        float t = (currSplitPos - rayOrigin[axis]) / rayDir[axis];
                        
                        // classify children as near and far
                        int nearIndex = 2*currNodeIndex+1;
                        int farIndex = 2*currNodeIndex+2;
                        uint near = uint(texture(kdTreeTexture, getTexCoord(nearIndex, kdTreeTextureSize)));
                        uint far = uint(texture(kdTreeTexture, getTexCoord(farIndex, kdTreeTextureSize)));
                        vec3 nearBoxMin = currBoxMin;
                        vec3 nearBoxMax = currBoxMax;
                        vec3 farBoxMin = currBoxMin;
                        vec3 farBoxMax = currBoxMax;
                        nearBoxMax[axis] = farBoxMin[axis] = currSplitPos;
                        if (rayDir[axis] <= 0.0) {
                            // swap near and far
                            uint temp = near;
                            near = far;
                            far = temp;
                            
                            int temp2 = nearIndex;
                            nearIndex = farIndex;
                            farIndex = temp2;
                            
                            vec3 temp3 = nearBoxMin;
                            nearBoxMin = farBoxMin;
                            farBoxMin = temp3;
                            
                            temp3 = nearBoxMax;
                            nearBoxMax = farBoxMax;
                            farBoxMax = temp3;
                        }
                        
                        if (t >= kdTMax/* || t < 0.0*/) {
                            currNode = near;
                            currNodeIndex = nearIndex;
                            currBoxMin = nearBoxMin;
                            currBoxMax = nearBoxMax;
                        } else if (t <= kdTMin) {
                            currNode = far;
                            currNodeIndex = farIndex;
                            currBoxMin = farBoxMin;
                            currBoxMax = farBoxMax;
                        } else {
                            currNode = near;
                            currNodeIndex = nearIndex;
                            currBoxMin = nearBoxMin;
                            currBoxMax = nearBoxMax;
                            kdTMax = t;
                            pushdown = false;
                        }
                        ++currDepth;
                        if (pushdown) {
                            pushedNode = currNode;
                            pushedNodeIndex = currNodeIndex;
                            pushedBoxMin = currBoxMin;
                            pushedBoxMax = currBoxMax;
                            pushedDepth = currDepth;
                        }
                    }
                    if (getKdNodeType(currNode) != EMPTY_LEAF) {
                        int leafIndex = getKdLeafIndex(currNode);
                        ivec4 faceIndices = ivec4(texture(kdLeavesTexture, getTexCoord(leafIndex, kdLeavesTextureSize)));
                        for (int i = 0; i < 4; i++) {
                            if (faceIndices[i] > -1) { // if it has a valid face index
                                vec2 fCoord = getTexCoord(faceIndices[i], facesTextureSize);
                                ivec3 vIndices = ivec3(texture(facesTexture, fCoord));
                                vec2 vCoords[3];
                                for (int j = 0; j < 3; j++) {
                                    vCoords[j] = getTexCoord(vIndices[j], vertsTextureSize);
                                }
                                vec3 vertices[3];
                                for (int j = 0; j < 3; j++) {
                                    vertices[j] = vec3(texture(vertsTexture, vCoords[j]));
                                }
                                
                                if(intersectTriangle(rayOrigin, rayDir, hitInfo, vertices)) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            for (int i = 0; i < numFaces; i++) {
                vec2 fCoord = getTexCoord(i, facesTextureSize);
                ivec3 indices = ivec3(texture(facesTexture, fCoord));
                vec2 vCoords[3];
                for (int j = 0; j < 3; j++) {
                    vCoords[j] = getTexCoord(indices[j], vertsTextureSize);
                }
                vec3 vertices[3];
                for (int j = 0; j < 3; j++) {
                    vertices[j] = vec3(texture(vertsTexture, vCoords[j]));
                }
                
                if(intersectTriangle(rayOrigin, rayDir, hitInfo, vertices)) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

vec3 illuminate(HitInfo hitInfo, vec3 rayDir, out float reflectionMix, 
                out float refractionMix, out float refractiveIndex) {
    vec3 accumulatedColor = vec3(0.0);
    
    float ka, kd, ks, ke, kReflect, kRefract, ior;
    vec3 Co, Cs;
    
    if (hitInfo.materialType == PHONG_MATERIAL) {
        Phong material = phongMaterials[hitInfo.materialId];
        ka = material.ka;
        kd = material.kd;
        ks = material.ks;
        ke = material.ke;
        kReflect = material.kReflect;
        kRefract = material.kRefract;
        ior = material.ior;
        Co = material.Co;
        Cs = material.Cs;
    } else if (hitInfo.materialType == PHONG_BLINN_MATERIAL) {
        Phong material = phongBlinnMaterials[hitInfo.materialId];
        ka = material.ka;
        kd = material.kd;
        ks = material.ks;
        ke = material.ke;
        kReflect = material.kReflect;
        kRefract = material.kRefract;
        ior = material.ior;
        Co = material.Co;
        Cs = material.Cs;
    } else if (hitInfo.materialType == PHONG_CHECKERED_MATERIAL) {
        PhongCheckered material = phongCheckeredMaterials[hitInfo.materialId];
        ka = material.ka;
        kd = material.kd;
        ks = material.ks;
        ke = material.ke;
        kReflect = material.kReflect;
        kRefract = material.kRefract;
        ior = material.ior;
        Cs = material.Cs;
        // Compute Co from Co1 and Co2 based on (u,v) coordinates
        float x = hitInfo.localHitPoint.x;
        float z = hitInfo.localHitPoint.z;
        float u = x + 0.5;
        float v = z + 0.5;
        float sizeU = 1.0 / 12.0;
        float sizeV = 450.0 / (1600.0 * 12.0);
        float row = float(int(u / sizeU));
        float col = float(int(v / sizeV));
        if (int(mod(row, 2.0)) == 0) {
            if (int(mod(col, 2.0)) == 0) {
                Co = material.Co1;
            } else {
                Co = material.Co2;
            }
        } else {
            if (int(mod(col, 2.0)) == 0) {
                Co = material.Co2;
            } else {
                Co = material.Co1;
            }
        }
    }
        
    vec3 N = normalize(hitInfo.normal);
    vec3 V = normalize(-rayDir);
    
    vec3 ambient = ka * ambientLight * Co;
    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);
    
    // Iterate over the point lights
    for (int i = 0; i < ` + nextPointLightId + `; i++) {
        vec3 S = normalize(pointLights[i].position - hitInfo.hitPoint);
        bool isInShadow = castShadow(hitInfo.hitPoint + hitInfo.normal * EPSILON, S);
        if (isInShadow == false) {
            float SdotN = max(0.0, dot(S, N));
            float diffuseCoeff = kd * SdotN;
            diffuseCoeff = clamp(diffuseCoeff, 0.0, 1.0);
            diffuse += diffuseCoeff * pointLights[i].color * Co;
            
            if (SdotN > 0.0) {
                float specularCoeff;
                if (hitInfo.materialType == PHONG_MATERIAL) {
                    vec3 R = normalize(reflect(-S, N));
                    float RdotV = max(0.0, dot(R, V));
                    specularCoeff = ks * pow(RdotV, ke);
                } else if (hitInfo.materialType == PHONG_BLINN_MATERIAL) {
                    vec3 H = normalize( S + V );
                    float NdotH = max(0.0, dot(N, H));
                    specularCoeff = ks * pow(NdotH, ke);
                }
                specularCoeff = clamp(specularCoeff, 0.0, 1.0);
                specular += specularCoeff * pointLights[i].color * Cs;
            }
        }
    }
    
    accumulatedColor += ambient + specular + diffuse;
    
    reflectionMix = kReflect;
    refractionMix = kRefract;
    refractiveIndex = ior;
    
    return accumulatedColor;
}

vec3 castRayRecursive(vec3 primaryRayOrigin, vec3 primaryRayDir) {
    primaryRayDir = normalize(primaryRayDir);
    rayTree[0].valid = true;
    rayTree[0].rayOrigin = primaryRayOrigin;
    rayTree[0].rayDir = primaryRayDir;
    
    // Find color for each node and update children 
    // (left child: reflected ray, right child: refracted ray)
    for (int node = 0; node < RAY_TREE_SIZE; ++node) {
        if (rayTree[node].valid == true) {
            vec3 rayOrigin = rayTree[node].rayOrigin;
            vec3 rayDir = rayTree[node].rayDir;
            
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
            // triangles
            if (RENDER_TRIANGLES) {
                if (USE_KD_TREE) {
                    float kdTMin, kdTMax;
                    intersectRayAABB(rayOrigin, rayDir, kdBoxMin, kdBoxMax, kdTMin, kdTMax);
                    float maxDist = kdTMax;
                    kdTMax = kdTMin;
                    uint pushedNode = uint(texture(kdTreeTexture, getTexCoord(0, kdTreeTextureSize)));
                    int pushedNodeIndex = 0;
                    vec3 pushedBoxMin = kdBoxMin;
                    vec3 pushedBoxMax = kdBoxMax;
                    int pushedDepth = 0;
                    if (getKdNodeType(pushedNode) != INVALID) {
                        bool pushdown = true;
                        while (kdTMax < maxDist) {
                            uint currNode = pushedNode;
                            int currNodeIndex = pushedNodeIndex;
                            vec3 currBoxMin = pushedBoxMin;
                            vec3 currBoxMax = pushedBoxMax;
                            int currDepth = pushedDepth;
                            kdTMin = kdTMax;
                            kdTMax = maxDist;
                            while (getKdNodeType(currNode) != EMPTY_LEAF
                                    && getKdNodeType(currNode) != LEAF) {
                                int axis = currDepth % 3;
                                float currSplitPos = currBoxMin[axis] + 0.5 * (currBoxMax[axis] - currBoxMin[axis]);
                                float t = (currSplitPos - rayOrigin[axis]) / rayDir[axis];
                                
                                // classify children as near and far
                                int nearIndex = 2*currNodeIndex+1;
                                int farIndex = 2*currNodeIndex+2;
                                uint near = uint(texture(kdTreeTexture, getTexCoord(nearIndex, kdTreeTextureSize)));
                                uint far = uint(texture(kdTreeTexture, getTexCoord(farIndex, kdTreeTextureSize)));
                                vec3 nearBoxMin = currBoxMin;
                                vec3 nearBoxMax = currBoxMax;
                                vec3 farBoxMin = currBoxMin;
                                vec3 farBoxMax = currBoxMax;
                                nearBoxMax[axis] = farBoxMin[axis] = currSplitPos;
                                if (rayDir[axis] <= 0.0) {
                                    // swap near and far
                                    uint temp = near;
                                    near = far;
                                    far = temp;
                                    
                                    int temp2 = nearIndex;
                                    nearIndex = farIndex;
                                    farIndex = temp2;
                                    
                                    vec3 temp3 = nearBoxMin;
                                    nearBoxMin = farBoxMin;
                                    farBoxMin = temp3;
                                    
                                    temp3 = nearBoxMax;
                                    nearBoxMax = farBoxMax;
                                    farBoxMax = temp3;
                                }
                                
                                if (t >= kdTMax/* || t < 0.0*/) {
                                    currNode = near;
                                    currNodeIndex = nearIndex;
                                    currBoxMin = nearBoxMin;
                                    currBoxMax = nearBoxMax;
                                } else if (t <= kdTMin) {
                                    currNode = far;
                                    currNodeIndex = farIndex;
                                    currBoxMin = farBoxMin;
                                    currBoxMax = farBoxMax;
                                } else {
                                    currNode = near;
                                    currNodeIndex = nearIndex;
                                    currBoxMin = nearBoxMin;
                                    currBoxMax = nearBoxMax;
                                    kdTMax = t;
                                    pushdown = false;
                                }
                                ++currDepth;
                                if (pushdown) {
                                    pushedNode = currNode;
                                    pushedNodeIndex = currNodeIndex;
                                    pushedBoxMin = currBoxMin;
                                    pushedBoxMax = currBoxMax;
                                    pushedDepth = currDepth;
                                }
                            }
                            if (getKdNodeType(currNode) != EMPTY_LEAF) {
                                int leafIndex = getKdLeafIndex(currNode);
                                ivec4 faceIndices = ivec4(texture(kdLeavesTexture, getTexCoord(leafIndex, kdLeavesTextureSize)));
                                for (int i = 0; i < 4; i++) {
                                    if (faceIndices[i] > -1) { // if it has a valid face index
                                        vec2 fCoord = getTexCoord(faceIndices[i], facesTextureSize);
                                        ivec3 vIndices = ivec3(texture(facesTexture, fCoord));
                                        vec2 vCoords[3];
                                        for (int j = 0; j < 3; j++) {
                                            vCoords[j] = getTexCoord(vIndices[j], vertsTextureSize);
                                        }
                                        vec3 vertices[3];
                                        for (int j = 0; j < 3; j++) {
                                            vertices[j] = vec3(texture(vertsTexture, vCoords[j]));
                                        }
                                        
                                        if (intersectTriangle(rayOrigin, rayDir, hitInfo, vertices) 
                                            && (hitInfo.t < tMin)) {
                                            hitInfo.hit = true;
                                            hitPoint = rayOrigin + rayDir * hitInfo.t;
                                            tMin = hitInfo.t;
                                            normal = hitInfo.normal;
                                            localHitPoint = hitInfo.localHitPoint;
                                            vec2 mat = vec2(texture(facesMatTexture, fCoord));
                                            materialType = int(mat.r);  // Material type is stored as R value
                                            materialId = int(mat.g);  // Material id is stored as G value
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < numFaces; i++) {
                        vec2 fCoord = getTexCoord(i, facesTextureSize);
                        ivec3 indices = ivec3(texture(facesTexture, fCoord));
                        vec2 vCoords[3];
                        for (int j = 0; j < 3; j++) {
                            vCoords[j] = getTexCoord(indices[j], vertsTextureSize);
                        }
                        vec3 vertices[3];
                        for (int j = 0; j < 3; j++) {
                            vertices[j] = vec3(texture(vertsTexture, vCoords[j]));
                        }
                        
                        if (intersectTriangle(rayOrigin, rayDir, hitInfo, vertices) 
                            && (hitInfo.t < tMin)) {
                            hitInfo.hit = true;
                            hitPoint = rayOrigin + rayDir * hitInfo.t;
                            tMin = hitInfo.t;
                            normal = hitInfo.normal;
                            localHitPoint = hitInfo.localHitPoint;
                            vec2 mat = vec2(texture(facesMatTexture, fCoord));
                            materialType = int(mat.r);  // Material type is stored as R value
                            materialId = int(mat.g);  // Material id is stored as G value
                        }
                    }
                }
            }
            
            if (hitInfo.hit) {
                hitInfo.t = tMin;
                hitInfo.normal = normal;
                hitInfo.localHitPoint = localHitPoint;
                hitInfo.hitPoint = hitPoint;
                hitInfo.materialType = materialType;
                hitInfo.materialId = materialId;
            } else {
                // ray did not hit any object
                rayTree[node].color = bgColor;
                continue;
            }
        
            float kReflect = 0.0;
            float kRefract = 0.0;
            float ior = 0.0;
            rayTree[node].color = illuminate(hitInfo, rayDir, kReflect, kRefract, ior);
            
            bool outside = dot(rayDir, hitInfo.normal) < 0.0;
            vec3 bias = hitInfo.normal * EPSILON;
            if (kRefract > EPSILON && ior > EPSILON && (node*2 + 2) < RAY_TREE_SIZE) {
                // set up for child refracted ray
                rayTree[node].kRefract = kRefract;
                rayTree[node*2 + 2].valid = true;
                rayTree[node*2 + 2].rayOrigin = outside ? (hitInfo.hitPoint - bias) : (hitInfo.hitPoint + bias);
                vec3 N = outside ? hitInfo.normal : -(hitInfo.normal);
                float eta = outside ? (1.0 / ior) : ior;
                float cosi = dot(-N, rayDir);
                float k = 1.0 + eta * eta * (cosi * cosi - 1.0);
                vec3 refrDir = rayDir * eta + N * (eta * cosi - sqrt(k));
                normalize(refrDir);
                rayTree[node*2 + 2].rayDir = refrDir;
            } else {
                rayTree[node*2 + 2].valid = false;
            }
            
            if (kReflect > EPSILON && (node*2 + 1) < RAY_TREE_SIZE) {
                // set up for child reflected ray
                rayTree[node].kReflect = kReflect;
                rayTree[node*2 + 1].valid = true;
                rayTree[node*2 + 1].rayOrigin = outside ? (hitInfo.hitPoint + bias) : (hitInfo.hitPoint - bias);
                rayTree[node*2 + 1].rayDir = reflect(rayDir, hitInfo.normal);
            } else {
                rayTree[node*2 + 1].valid = false;
            }
        }
    }
    
    // Accumulate child colors in parent node
    for (int node = RAY_TREE_SIZE-1; node > 0; --node) {
        if (rayTree[node].valid == true) {
            // odd nodes are reflected rays and even nodes are refracted rays
            float mix = 0.0;
            if (mod(float(node), 2.0) != 0.0) {
                // current node is reflected ray
                mix = rayTree[(node-1) / 2].kReflect;
            } else {
                // current node is refracted ray
                mix = rayTree[(node-1) / 2].kRefract;
            }
            rayTree[(node-1) / 2].color += mix * rayTree[node].color;
        }
    }
    
    return rayTree[0].color;
}


void main() {
    for (int i = 0; i < RAY_TREE_SIZE; i++) {
        rayTree[i].valid = false;
    }
    
    fragmentColor = vec4(castRayRecursive(cameraPos, primaryRayDir), 1.0);
    if (error == true) {
        fragmentColor = vec4(1.0, 0.0, 0.0, 1.0);
    }
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
    var vector = new THREE.Vector3( x, y, -1 ).unproject( camera );
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
    
    getMinCorner() {
        throw new Error("Primitive: cannot call abstract method 'getMinCorner'");
    }
    
    getMaxCorner() {
        throw new Error("Primitive: cannot call abstract method 'getMaxCorner'");
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
    
    getMinCorner() {
        return new THREE.Vector3(-10, -10, -10);
    }
    
    getMaxCorner() {
        return new THREE.Vector3(10, 10, 10);
    }

    intersect(ray) {
        var localRayOrigin = ray.origin.clone().applyMatrix4(this.invModelTransform);
        var localRayDir = ray.dir.clone().transformDirection(this.invModelTransform);
        
        var planeNormal = new THREE.Vector3(0, 1, 0);
        var t0 = localRayOrigin.clone().negate().dot(planeNormal) / localRayDir.dot(planeNormal);
    	
    	if(t0 > PLANE_RECT_EPSILON){
    		return t0;
    	}
        
        return Number.MAX_VALUE;
    }

    scale(scale) {
        throw new Error("Plane: cannot call abstract method 'scale'");
    }
}


////////////////////////////////////////////////////////////////////////////////
// class Rectangle with left bottom corner at origin, left side of length
// 1, bottom side of length 1 and normal pointing upwards
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
    
    getMinCorner() {
        return new THREE.Vector3(-0.5, -10, -0.5);
    }
    
    getMaxCorner() {
        return new THREE.Vector3(0.5, 10, 0.5);
    }

    intersect(ray) {
        var localRayOrigin = ray.origin.clone().applyMatrix4(this.invModelTransform);
        var localRayDir = ray.dir.clone().transformDirection(this.invModelTransform);
        
        var rectAVec = new THREE.Vector3(1.0, 0, 0), rectBVec = new THREE.Vector3(0, 0, -1.0);
        var rectNormal = rectAVec.clone().cross(rectBVec);
        var rectP0 = new THREE.Vector3(-0.5, 0, 0.5);
        
        var t0 = rectP0.clone().sub(localRayOrigin).dot(rectNormal) / localRayDir.dot(rectNormal);
    	
    	if(t0 > PLANE_RECT_EPSILON){
    	    var p = localRayOrigin.clone().add(localRayDir.clone().multiplyScalar(t0));
    	    var d = p.clone().sub(rectP0);
    	    
    	    var ddota = d.dot(rectAVec);
    	    var ddotb = d.dot(rectBVec);
    	    if (ddota > 0.0 && ddota < 1.0 && ddotb > 0.0 && ddotb < 1.0) {
    	        return t0;
    	    }
    	}
        
        return Number.MAX_VALUE;
    }
}


////////////////////////////////////////////////////////////////////////////////
// class Sphere of radius 1 centered at origin
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
    
    getMinCorner() {
        return new THREE.Vector3(-1, -1, -1);
    }
    
    getMaxCorner() {
        return new THREE.Vector3(1, 1, 1);
    }

    intersect(ray) {
        var localRayOrigin = ray.origin.clone().applyMatrix4(this.invModelTransform);
        var localRayDir = ray.dir.clone().transformDirection(this.invModelTransform);
    
        var a = localRayDir.dot(localRayDir);
        var b = 2.0 * localRayOrigin.dot(localRayDir);
        var c = localRayOrigin.dot(localRayOrigin) - 1.0;
        var disc = b*b - 4.0*a*c;
        
        if (disc > 0.0) {
            var e = Math.sqrt(disc);
            var denom = 2.0*a;
            
            var t0 = (-b - e) / denom;  // smaller root
            if (t0 > SPHERE_EPSILON) {
                return t0;
            }
            
            t0 = (-b + e) / denom;  // larger root
            if (t0 > SPHERE_EPSILON) {
                return t0;
            }
        }
        
        return Number.MAX_VALUE;
    }
}


////////////////////////////////////////////////////////////////////////////////
// class Cube of side length 1 represented by min corner and max corner points
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
    
    getMinCorner() {
        return new THREE.Vector3(-0.5, -0.5, -0.5);
    }
    
    getMaxCorner() {
        return new THREE.Vector3(0.5, 0.5, 0.5);
    }

    intersect(ray) {
        return Cube.intersect(ray, this.getMinCorner(), this.getMaxCorner(), this.invModelTransform);
    }

    static intersect(ray, minCorner, maxCorner, invModelTransform) {
        var localRayOrigin = ray.origin.clone().applyMatrix4(invModelTransform);
        var localRayDir = ray.dir.clone().transformDirection(invModelTransform);
        
        var absLocalRayDir = new THREE.Vector3(Math.abs(localRayDir.x), Math.abs(localRayDir.y), Math.abs(localRayDir.z));
        var t0 = (minCorner.clone().divide(absLocalRayDir)).sub(localRayOrigin.clone().divide(localRayDir));
        var t1 = (maxCorner.clone().divide(absLocalRayDir)).sub(localRayOrigin.clone().divide(localRayDir));
        
        var tMin = Math.max( t0.x, t0.y, t0.z );
        var tMax = Math.min( t1.x, t1.y, t1.z );
        
        if (tMin > tMax || tMax < 0.0) return Number.MAX_VALUE;
        
        return tMin;
    }
}


////////////////////////////////////////////////////////////////////////////////
// class Cylinder of height 1 with origin as centre, 'bottom' as bottom cap's y 
// coordinate, 'top' as top cap's y coordinate, and radius 1
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
    
    getMinCorner() {
        return new THREE.Vector3(-1, -0.5, -1);
    }
    
    getMaxCorner() {
        return new THREE.Vector3(1, 0.5, 1);
    }

    intersect(ray) {
        var localRayOrigin = ray.origin.clone().applyMatrix4(this.invModelTransform);
        var localRayDir = ray.dir.clone().transformDirection(this.invModelTransform);
        
        var ox = localRayOrigin.x, oy = localRayOrigin.y, oz = localRayOrigin.z,
        dx = localRayDir.x, dy = localRayDir.y, dz = localRayDir.z;
        
        var a = dx * dx + dz * dz;
        var b = 2.0 * (ox * dx + oz * dz);
        var c = ox * ox + oz * oz - 1.0;
        var disc = b * b - 4.0 * a * c;
        
        if (disc > 0.0) {
            var e = Math.sqrt(disc);
            var denom = 2.0 * a;
            
            var t0 = (-b - e) / denom;  // smaller root
            var t1 = (-b + e) / denom;  // larger root
            if (t0 > t1) {
                // swap t0 and t1
                var tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            
            var y0 = oy + t0 * dy;
            var y1 = oy + t1 * dy;
            
            var th;
            if (y0 < -0.5) {
            	if (y1 >= -0.5) {
            		// hit the bottom cap
            		th = t0 + (t1-t0) * (y0 + 0.5) / (y0-y1);
            		if (th > 0.0) {
            		    return th;
            		}
            	}
            } else if (y0 >= -0.5 && y0 <= 0.5) {
            	// hit the cylinder part
            	if (t0 > 0.0) {
            	    return t0;
            	}
            } else if (y0 > 0.5) {
            	if (y1 <= 0.5) {
            		// hit the top cap
            		th = t0 + (t1-t0) * (y0 - 0.5) / (y0-y1);
            		if (th > 0.0) {
            		    return th;
            		}
            	}
            }
        }
    
        return Number.MAX_VALUE;
    }
}

////////////////////////////////////////////////////////////////////////////////
// class KdNode is a node in the k-d Tree. It also contains a 32 bit integer
// metadata which has information as follows:
//
// 2 bits: Type of Node
// 0 = Invalid node
// 1 = Internal node
// 2 = Leaf node empty
// 3 = Leaf node with primitives
// 
// 30 bits: Leaf node index
////////////////////////////////////////////////////////////////////////////////

class KdNode {
    static get INVALID() {
        return 0;
    }
    
    static get INTERNAL() {
        return 1;
    }
    
    static get EMPTY_LEAF() {
        return 2;
    }
    
    static get LEAF() {
        return 3;
    }
    
    constructor() {
        this.metadata = 0;
        this.childNodes = [];
        this.childNodes.push(null);
        this.childNodes.push(null);
        this.min = null;
        this.max = null;
        this.depth = 0;
        this.faceIndices = [];
    }
    
    getType() {
        return (this.metadata >> 30);
    }
    
    getLeafIndex() {
        return (this.metadata & 0x3fffffff);
    }
    
    setInvalid() {
        this.metadata = 0;
    }
    
    setInternal() {
        this.metadata = 1 << 30;
    }
    
    setEmptyLeaf() {
        this.metadata = 2 << 30;
    }
    
    setLeaf(leafIndex) {
        this.metadata = 3 << 30;
        this.metadata = this.metadata | leafIndex;
    }
}

////////////////////////////////////////////////////////////////////////////////
// class KdLeaf is a leaf node in the k-d Tree. It has indices to 4 faces
////////////////////////////////////////////////////////////////////////////////

class KdLeaf {
    constructor(faceIndices) {
        this.faceIndices = [-1, -1, -1, -1];
        for (var i = 0; i < Math.min(faceIndices.length, 4); i++) {
            this.faceIndices[i] = faceIndices[i];
        }
    }
}

class KdTree {
    constructor() {
        this.root = null;
        this.depth = 0;
        this.leafNodes = [];
    }
    
    fillDataArray(dataArray) {
        if (this.root) {
            this.traverseTree(this.root, 0, dataArray);
        }
    }
    
    traverseTree(node, index, dataArray) {
        dataArray[index] = node.metadata;
        if (node.getType() == KdNode.INTERNAL) {
            for (var i = 0; i < 2; i++) {
                this.traverseTree(node.childNodes[i], 2*index+i+1, dataArray);
            }
        }
    }
    
    buildTree(faceIndices, maxDepth = MAX_KD_TREE_DEPTH) {
        maxDepth = Math.min(maxDepth, MAX_KD_TREE_DEPTH);
        
        this.root = new KdNode();
        this.root.depth = 1;
        this.root.faceIndices = faceIndices;
        this.root.min = new THREE.Vector3(INFINITY, INFINITY, INFINITY);
        this.root.max = new THREE.Vector3(-INFINITY, -INFINITY, -INFINITY);
        
        // Calculate the outermost bounding box extents
        for (var i = 0; i < faceIndices.length; i++) {
            var faceIndex = faceIndices[i];
            var face = g_faces[faceIndex];
            var v_index = face.indices[0];
            this.root.min.x = Math.min(this.root.min.x, g_verts[v_index].x);
            this.root.max.x = Math.max(this.root.max.x, g_verts[v_index].x);
            v_index = face.indices[1];
            this.root.min.y = Math.min(this.root.min.y, g_verts[v_index].y);
            this.root.max.y = Math.max(this.root.max.y, g_verts[v_index].y);
            v_index = face.indices[2];
            this.root.min.z = Math.min(this.root.min.z, g_verts[v_index].z);
            this.root.max.z = Math.max(this.root.max.z, g_verts[v_index].z);
        }
        
        // iteratively build the kd-tree
        
        var nodeStack = [];
        nodeStack.push(this.root);
        
        while (nodeStack.length != 0) {
            var node = nodeStack.pop();
            
            if (node.depth > this.depth) {
                this.depth = node.depth;
            }
            
            if (node.faceIndices.length > 4 && node.depth < maxDepth) {
                node.setInternal();
                for (var i = 0; i < 2; i++) {
                    var childMin = node.min.clone();
                    var childMax = node.max.clone();
                    node.childNodes[i] = new KdNode();
                    var axis = node.depth % 3;
                    if (axis == 1) {
                        // splitting plane in X-axis
                        childMin.x = node.min.x + 0.5 * i * (node.max.x - node.min.x);
                        childMax.x = node.min.x + 0.5 * (i+1) * (node.max.x - node.min.x);
                    } else if (axis == 2) {
                        // splitting plane in Y-axis
                        childMin.y = node.min.y + 0.5 * i * (node.max.y - node.min.y);
                        childMax.y = node.min.y + 0.5 * (i+1) * (node.max.y - node.min.y);
                    } else {
                        // splitting plane in Z-axis
                        childMin.z = node.min.z + 0.5 * i * (node.max.z - node.min.z);
                        childMax.z = node.min.z + 0.5 * (i+1) * (node.max.z - node.min.z);
                    }
                    
                    // add faceIndices which are in child bounding box
                    var childFaceIndices = [];
                    for (var j = 0; j < node.faceIndices.length; j++) {
                        faceIndex = node.faceIndices[j];
                        if (this.faceBoxOverlap(childMin, childMax, faceIndex)) {
                            childFaceIndices.push(faceIndex);
                        }
                    }
                    
                    node.childNodes[i].min = childMin;
                    node.childNodes[i].max = childMax;
                    node.childNodes[i].depth = node.depth+1;
                    node.childNodes[i].faceIndices = childFaceIndices;
                    
                    nodeStack.push(node.childNodes[i]);
                }
                continue;
            }
            
            if (node.faceIndices.length == 0) {
                node.setEmptyLeaf();
                continue;
            }
            
            var leaf = new KdLeaf(node.faceIndices);
            this.leafNodes.push(leaf);
            node.setLeaf(this.leafNodes.length - 1);
        }
    }
    
    /* Triangle-AABB intersection from “Fast 3D Triangle-Box Overlap Testing” by Tomas Akenine-Moller 
     * [http://www.cs.lth.se/home/Tomas_Akenine_Moller/code/]
     */
    faceBoxOverlap(boxMin, boxMax, faceIndex) {
        var boxCenter = new THREE.Vector3(
            (boxMin.x + boxMax.x) * 0.5,
            (boxMin.y + boxMax.y) * 0.5,
            (boxMin.z + boxMax.z) * 0.5
        );
        var boxHalfSize = new THREE.Vector3(
            Math.abs(boxMax.x - boxMin.x) * 0.5,
            Math.abs(boxMax.y - boxMin.y) * 0.5,
            Math.abs(boxMax.z - boxMin.z) * 0.5
        );
        
        var face = g_faces[faceIndex];
        var faceVerts = [
            g_verts[face.indices[0]].clone().sub(boxCenter), 
            g_verts[face.indices[1]].clone().sub(boxCenter), 
            g_verts[face.indices[2]].clone().sub(boxCenter)
        ];
        var faceEdges = [
            faceVerts[1].clone().sub(faceVerts[0]),
            faceVerts[2].clone().sub(faceVerts[1]),
            faceVerts[0].clone().sub(faceVerts[2])
        ];
        var faceNormal = faceEdges[0].cross(faceEdges[1]);
        
        var fex = Math.abs(faceEdges[0].x);
        var fey = Math.abs(faceEdges[0].y);
        var fez = Math.abs(faceEdges[0].z);
        if (!this.axisTest_X01(faceEdges[0].z, faceEdges[0].y, fez, fey, faceVerts, boxHalfSize)) {
            return false;
        }
        if (!this.axisTest_Y02(faceEdges[0].z, faceEdges[0].x, fez, fex, faceVerts, boxHalfSize)) {
            return false;
        }
        if (!this.axisTest_Z12(faceEdges[0].y, faceEdges[0].x, fey, fex, faceVerts, boxHalfSize)) {
            return false;
        }
        
        fex = Math.abs(faceEdges[1].x);
        fey = Math.abs(faceEdges[1].y);
        fez = Math.abs(faceEdges[1].z);
        if (!this.axisTest_X01(faceEdges[1].z, faceEdges[1].y, fez, fey, faceVerts, boxHalfSize)) {
            return false;
        }
        if (!this.axisTest_Y02(faceEdges[1].z, faceEdges[1].x, fez, fex, faceVerts, boxHalfSize)) {
            return false;
        }
        if (!this.axisTest_Z0(faceEdges[1].y, faceEdges[1].x, fey, fex, faceVerts, boxHalfSize)) {
            return false;
        }
        
        fex = Math.abs(faceEdges[2].x);
        fey = Math.abs(faceEdges[2].y);
        fez = Math.abs(faceEdges[2].z);
        if (!this.axisTest_X2(faceEdges[2].z, faceEdges[2].y, fez, fey, faceVerts, boxHalfSize)) {
            return false;
        }
        if (!this.axisTest_Y1(faceEdges[2].z, faceEdges[2].x, fez, fex, faceVerts, boxHalfSize)) {
            return false;
        }
        if (!this.axisTest_Z12(faceEdges[2].y, faceEdges[2].x, fey, fex, faceVerts, boxHalfSize)) {
            return false;
        }
        
        // test in X-direction
        var min = Math.min(faceVerts[0].x, Math.min(faceVerts[1].x, faceVerts[2].x));
        var max = Math.max(faceVerts[0].x, Math.max(faceVerts[1].x, faceVerts[2].x));
        if (min > boxHalfSize.x || max < -(boxHalfSize.x)) {
            return false;
        }
        
        // test in Y-direction
        min = Math.min(faceVerts[0].y, Math.min(faceVerts[1].y, faceVerts[2].y));
        max = Math.max(faceVerts[0].y, Math.max(faceVerts[1].y, faceVerts[2].y));
        if (min > boxHalfSize.y || max < -(boxHalfSize.y)) {
            return false;
        }
        
        // test in Z-direction
        min = Math.min(faceVerts[0].z, Math.min(faceVerts[1].z, faceVerts[2].z));
        max = Math.max(faceVerts[0].z, Math.max(faceVerts[1].z, faceVerts[2].z));
        if (min > boxHalfSize.z || max < -(boxHalfSize.z)) {
            return false;
        }
        
        if (!this.planeBoxOverlap(faceNormal, faceVerts[0], boxHalfSize)) {
            return false;
        }
        
        return true;
    }
    
    axisTest_X01(a, b, fa, fb, verts, boxHalfSize) {
        var min, max;
        var p0 = a*verts[0].y - b*verts[0].z;
        var p2 = a*verts[2].y - b*verts[2].z;
        if (p0 < p2) {
            min = p0; max = p2;
        } else {
            min = p2; max = p0;
        }
        var rad = fa*boxHalfSize.y + fb*boxHalfSize.z;
        if (min > rad || max < -rad) {
            return false;
        } else {
            return true;
        }
    }
    
    axisTest_X2(a, b, fa, fb, verts, boxHalfSize) {
        var min, max;
        var p0 = a*verts[0].y - b*verts[0].z;
        var p1 = a*verts[1].y - b*verts[1].z;
        if (p0 < p1) {
            min = p0; max = p1;
        } else {
            min = p1; max = p0;
        }
        var rad = fa*boxHalfSize.y + fb*boxHalfSize.z;
        if (min > rad || max < -rad) {
            return false;
        } else {
            return true;
        }
    }
    
    axisTest_Y02(a, b, fa, fb, verts, boxHalfSize) {
        var min, max;
        var p0 = -a*verts[0].x + b*verts[0].z;
        var p2 = -a*verts[2].x + b*verts[2].z;
        if (p0 < p2) {
            min = p0; max = p2;
        } else {
            min = p2; max = p0;
        }
        var rad = fa*boxHalfSize.x + fb*boxHalfSize.z;
        if (min > rad || max < -rad) {
            return false;
        } else {
            return true;
        }
    }
    
    axisTest_Y1(a, b, fa, fb, verts, boxHalfSize) {
        var min, max;
        var p0 = -a*verts[0].x + b*verts[0].z;
        var p1 = -a*verts[1].x + b*verts[1].z;
        if (p0 < p1) {
            min = p0; max = p1;
        } else {
            min = p1; max = p0;
        }
        var rad = fa*boxHalfSize.x + fb*boxHalfSize.z;
        if (min > rad || max < -rad) {
            return false;
        } else {
            return true;
        }
    }
    
    axisTest_Z12(a, b, fa, fb, verts, boxHalfSize) {
        var min, max;
        var p1 = a*verts[1].x - b*verts[1].y;
        var p2 = a*verts[2].x - b*verts[2].y;
        if (p2 < p1) {
            min = p2; max = p1;
        } else {
            min = p1; max = p2;
        }
        var rad = fa*boxHalfSize.x + fb*boxHalfSize.y;
        if (min > rad || max < -rad) {
            return false;
        } else {
            return true;
        }
    }
    
    axisTest_Z0(a, b, fa, fb, verts, boxHalfSize) {
        var min, max;
        var p0 = a*verts[0].x - b*verts[0].y;
        var p1 = a*verts[1].x - b*verts[1].y;
        if (p0 < p1) {
            min = p0; max = p1;
        } else {
            min = p1; max = p0;
        }
        var rad = fa*boxHalfSize.x + fb*boxHalfSize.y;
        if (min > rad || max < -rad) {
            return false;
        } else {
            return true;
        }
    }
    
    planeBoxOverlap(normal, vert, maxbox) {
        var vmin, vmax, v;
        vmin = new THREE.Vector3();
        vmax = new THREE.Vector3();
        v = vert.x;
        if (normal.x > 0) {
            vmin.x = -maxbox.x - v;
            vmax.x = maxbox.x - v;
        } else {
            vmin.x = maxbox.x - v;
            vmax.x = -maxbox.x - v;
        }
        
        v = vert.y;
        if (normal.y > 0) {
            vmin.y = -maxbox.y - v;
            vmax.y = maxbox.y - v;
        } else {
            vmin.y = maxbox.y - v;
            vmax.y = -maxbox.y - v;
        }
        
        v = vert.z;
        if (normal.z > 0) {
            vmin.z = -maxbox.z - v;
            vmax.z = maxbox.z - v;
        } else {
            vmin.z = maxbox.z - v;
            vmax.z = -maxbox.z - v;
        }
        
        if (normal.dot(vmin) > 0) {
            return false;
        }
        if (normal.dot(vmax) >= 0) {
            return true;
        }
        
        return false;
    }
}

////////////////////////////////////////////////////////////////////////////////
// class Triangle
////////////////////////////////////////////////////////////////////////////////

class Triangle {
    constructor(v0, v1, v2) {
        this.indices = [v0, v1, v2];
        this.material = null;
    }
    
    getMaterial() {
        return this.material;
    }
    
    setMaterial(material) {
        this.material = material;
    }
}

////////////////////////////////////////////////////////////////////////////////
// class Mesh, which is made up of Triangles
////////////////////////////////////////////////////////////////////////////////

class Mesh extends Primitive {
    constructor(id, verts = [], faces = []) {
        super(id);
        this.verts = verts;
        this.faces = faces;
        
        this.min = new THREE.Vector3(INFINITY, INFINITY, INFINITY);
        this.max = new THREE.Vector3(-INFINITY, -INFINITY, -INFINITY);
        
        // Calculate bounding box
        for (var i = 0; i < faces.length; i++) {
            var face = faces[i];
            var v_index = face.indices[0];
            this.min.x = Math.min(this.min.x, verts[v_index].x);
            this.max.x = Math.max(this.max.x, verts[v_index].x);
            v_index = face.indices[1];
            this.min.y = Math.min(this.min.y, verts[v_index].y);
            this.max.y = Math.max(this.max.y, verts[v_index].y);
            v_index = face.indices[2];
            this.min.z = Math.min(this.min.z, verts[v_index].z);
            this.max.z = Math.max(this.max.z, verts[v_index].z);
        }
        
        this.pushToGlobal();
    }
    
    setMaterial(material) {
        for (var t = 0; t < this.faces.length; t++) {
            this.faces[t].setMaterial(material);
        }
    }
    
    getGlobalCode() {
        return ``;
    }

    getClosestIntersectCode() {
        return ``;
    }
    
    getShadowCode() {
        return ``;
    }
    
    getMinCorner() {
        return this.min;
    }
    
    getMaxCorner() {
        return this.max;
    }

    intersect(ray) {
        if (!RENDER_TRIANGLES || STATIC_GEOMETRY) {
            return Number.MAX_VALUE;
        }
        return Cube.intersect(ray, this.min, this.max, this.invModelTransform);
    }
    
    translate(translation) {
        this.transformToModel();
        
        this.modelTransform = new THREE.Matrix4().makeTranslation(translation.x, translation.y, translation.z).multiply(this.modelTransform);
        this.invModelTransform = new THREE.Matrix4().getInverse(this.modelTransform, true);
        
        this.transformToWorld();
        
        this.replaceInGlobal();
    }
    
    rotate(rotationEuler) {
        this.transformToModel();
        
        this.modelTransform = new THREE.Matrix4().makeRotationFromEuler(rotationEuler).multiply(this.modelTransform);
        this.invModelTransform = new THREE.Matrix4().getInverse(this.modelTransform, true);
        
        this.transformToWorld();
        
        this.replaceInGlobal();
    }
    
    scale(scale) {
        this.transformToModel();
        
        this.modelTransform = new THREE.Matrix4().makeScale(scale.x, scale.y, scale.z).multiply(this.modelTransform);
        this.invModelTransform = new THREE.Matrix4().getInverse(this.modelTransform, true);
        
        this.transformToWorld();
        
        this.replaceInGlobal();
    }
    
    transformToWorld() {
        for (var t = 0; t < this.verts.length; t++) {
            this.verts[t].applyMatrix4(this.modelTransform);
        }
    }
    
    transformToModel() {
        for (var t = 0; t < this.verts.length; t++) {
            this.verts[t].applyMatrix4(this.invModelTransform);
        }
    }
    
    replaceInGlobal() {
        Array.prototype.splice.apply(g_faces, [this.startFaceIndex, this.facesLength].concat(this.faces));
        Array.prototype.splice.apply(g_verts, [this.startVertIndex, this.vertsLength].concat(this.verts));
        
        this.facesLength = this.faces.length;
        this.vertsLength = this.verts.length;
    }
    
    pushToGlobal() {
        this.startFaceIndex = g_faces.length;
        this.startVertIndex = g_verts.length;
        this.facesLength = this.faces.length;
        this.vertsLength = this.verts.length;
        
        Array.prototype.push.apply(g_faces, this.faces);
        Array.prototype.push.apply(g_verts, this.verts);
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
    
    static get PHONG_CHECKERED_MATERIAL() {
        return 2;
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
    constructor(id, ka, kd, ks, ke, kReflect, kRefract, ior, Co, Cs) {
        super(Material.PHONG_MATERIAL, id);
        this.ka = ka;
        this.kd = kd;
        this.ks = ks;
        this.ke = ke;
        this.kReflect = kReflect;
        this.kRefract = kRefract;
        this.ior = ior;
        this.Co = Co;
        this.Cs = Cs;
    }
    
    setUniforms(renderer) {
        renderer.uniforms["phongMaterials[" + this.id + "].ka"] = this.ka;
        renderer.uniforms["phongMaterials[" + this.id + "].kd"] = this.kd;
        renderer.uniforms["phongMaterials[" + this.id + "].ks"] = this.ks;
        renderer.uniforms["phongMaterials[" + this.id + "].ke"] = this.ke;
        renderer.uniforms["phongMaterials[" + this.id + "].kReflect"] = this.kReflect;
        renderer.uniforms["phongMaterials[" + this.id + "].kRefract"] = this.kRefract;
        renderer.uniforms["phongMaterials[" + this.id + "].ior"] = this.ior;
        renderer.uniforms["phongMaterials[" + this.id + "].Co"] = this.Co;
        renderer.uniforms["phongMaterials[" + this.id + "].Cs"] = this.Cs;
    }
}


////////////////////////////////////////////////////////////////////////////////
// class PhongBlinn
////////////////////////////////////////////////////////////////////////////////

class PhongBlinn extends Material {
    constructor(id, ka, kd, ks, ke, kReflect, kRefract, ior, Co, Cs) {
        super(Material.PHONG_BLINN_MATERIAL, id);
        this.ka = ka;
        this.kd = kd;
        this.ks = ks;
        this.ke = ke;
        this.kReflect = kReflect;
        this.kRefract = kRefract;
        this.ior = ior;
        this.Co = Co;
        this.Cs = Cs;
    }
    
    setUniforms(renderer) {
        renderer.uniforms["phongBlinnMaterials[" + this.id + "].ka"] = this.ka;
        renderer.uniforms["phongBlinnMaterials[" + this.id + "].kd"] = this.kd;
        renderer.uniforms["phongBlinnMaterials[" + this.id + "].ks"] = this.ks;
        renderer.uniforms["phongBlinnMaterials[" + this.id + "].ke"] = this.ke;
        renderer.uniforms["phongBlinnMaterials[" + this.id + "].kReflect"] = this.kReflect;
        renderer.uniforms["phongBlinnMaterials[" + this.id + "].kRefract"] = this.kRefract;
        renderer.uniforms["phongBlinnMaterials[" + this.id + "].ior"] = this.ior;
        renderer.uniforms["phongBlinnMaterials[" + this.id + "].Co"] = this.Co;
        renderer.uniforms["phongBlinnMaterials[" + this.id + "].Cs"] = this.Cs;
    }
}


////////////////////////////////////////////////////////////////////////////////
// class Phong Checkered
////////////////////////////////////////////////////////////////////////////////

class PhongCheckered extends Material {
    constructor(id, ka, kd, ks, ke, kReflect, kRefract, ior, Co1, Co2, Cs) {
        super(Material.PHONG_CHECKERED_MATERIAL, id);
        this.ka = ka;
        this.kd = kd;
        this.ks = ks;
        this.ke = ke;
        this.kReflect = kReflect;
        this.kRefract = kRefract;
        this.ior = ior;
        this.Co1 = Co1;
        this.Co2 = Co2;
        this.Cs = Cs;
    }
    
    setUniforms(renderer) {
        renderer.uniforms["phongCheckeredMaterials[" + this.id + "].ka"] = this.ka;
        renderer.uniforms["phongCheckeredMaterials[" + this.id + "].kd"] = this.kd;
        renderer.uniforms["phongCheckeredMaterials[" + this.id + "].ks"] = this.ks;
        renderer.uniforms["phongCheckeredMaterials[" + this.id + "].ke"] = this.ke;
        renderer.uniforms["phongCheckeredMaterials[" + this.id + "].kReflect"] = this.kReflect;
        renderer.uniforms["phongCheckeredMaterials[" + this.id + "].kRefract"] = this.kRefract;
        renderer.uniforms["phongCheckeredMaterials[" + this.id + "].ior"] = this.ior;
        renderer.uniforms["phongCheckeredMaterials[" + this.id + "].Co1"] = this.Co1;
        renderer.uniforms["phongCheckeredMaterials[" + this.id + "].Co2"] = this.Co2;
        renderer.uniforms["phongCheckeredMaterials[" + this.id + "].Cs"] = this.Cs;
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
        this.pixels = new Uint8Array(WIDTH * SUPERSAMPLING * HEIGHT * SUPERSAMPLING * 4);
        
        // create textures to render to
        var type = gl.UNSIGNED_BYTE;
        this.textures = [];
        for (var i = 0; i < 1; i++) {
            this.textures.push(gl.createTexture());
            gl.bindTexture(gl.TEXTURE_2D, this.textures[i]);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, WIDTH * SUPERSAMPLING, HEIGHT * SUPERSAMPLING, 0, gl.RGBA, type, null);
        }
        gl.bindTexture(gl.TEXTURE_2D, null);
        
        // create render shader
        console.log(renderFragmentSource);
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
        
        this.startTime = 0;
    }

    setScene(scene) {
        this.uniforms = {};
        this.sampleCount = 0;
        this.scene = scene;
    
    
        // create texture to store vertices data
        this.vertsTextureSize = Math.ceil(Math.sqrt(g_verts.length));
        this.vertsData = new Float32Array(this.vertsTextureSize * this.vertsTextureSize * 3);
        gl.getExtension('OES_texture_float');
        this.vertsTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.vertsTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB32F, this.vertsTextureSize, this.vertsTextureSize, 0, gl.RGB, gl.FLOAT, this.vertsData);
        gl.bindTexture(gl.TEXTURE_2D, null);
        
        // create textures to store triangles data
        this.facesData = [];
        this.facesTextureSize = Math.ceil(Math.sqrt(g_faces.length));
        for (var i = 0; i < 2; i++) {
            this.facesData[i] = new Uint32Array(this.facesTextureSize * this.facesTextureSize * 3);
        }
        this.facesTextures = [];
        for (var i = 0; i < 2; i++) {
            this.facesTextures.push(gl.createTexture());
            gl.bindTexture(gl.TEXTURE_2D, this.facesTextures[i]);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB32UI, this.facesTextureSize, this.facesTextureSize, 0, gl.RGB_INTEGER, gl.UNSIGNED_INT, this.facesData[i]);
        }
        gl.bindTexture(gl.TEXTURE_2D, null);
        
        
        // create texture to store kd-tree structure
        var maxNumberOfKdNodes = (1 << MAX_KD_TREE_DEPTH) - 1;
        this.kdTreeTextureSize = Math.ceil(Math.sqrt(maxNumberOfKdNodes));
        this.kdTreeData = new Uint32Array(this.kdTreeTextureSize * this.kdTreeTextureSize);
        this.kdTreeTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.kdTreeTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32UI, this.kdTreeTextureSize, this.kdTreeTextureSize, 0, gl.RED_INTEGER, gl.UNSIGNED_INT, this.kdTreeData);
        gl.bindTexture(gl.TEXTURE_2D, null);
        
        // create texture to store kd-tree leaves
        var maxNumberOfKdLeaves = 1 << (MAX_KD_TREE_DEPTH-1);
        this.kdLeavesTextureSize = Math.ceil(Math.sqrt(maxNumberOfKdLeaves));
        this.kdLeavesData = new Int32Array(this.kdLeavesTextureSize * this.kdLeavesTextureSize * 4);
        this.kdLeavesTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, this.kdLeavesTexture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32I, this.kdLeavesTextureSize, this.kdLeavesTextureSize, 0, gl.RGBA_INTEGER, gl.INT, this.kdLeavesData);
        gl.bindTexture(gl.TEXTURE_2D, null);
        
        
        // create ray tracing shader
        if (this.tracerProgram != null) {
            gl.deleteProgram(this.tracerProgram);
        }
        var tracerFragmentSource = generateTracerFragmentSource(this.scene.objects);
        console.log(tracerFragmentSource);
        this.tracerProgram = compileShader(tracerVertexSource, tracerFragmentSource);
        this.tracerVertexAttribute = gl.getAttribLocation(this.tracerProgram, 'vertex');
        gl.enableVertexAttribArray(this.tracerVertexAttribute);
        
        gl.useProgram(this.tracerProgram);
        gl.uniform1i(gl.getUniformLocation(this.tracerProgram, "vertsTexture"), 0);
        gl.uniform1i(gl.getUniformLocation(this.tracerProgram, "facesTexture"), 1);
        gl.uniform1i(gl.getUniformLocation(this.tracerProgram, "facesMatTexture"), 2);
        gl.uniform1i(gl.getUniformLocation(this.tracerProgram, "kdTreeTexture"), 3);
        gl.uniform1i(gl.getUniformLocation(this.tracerProgram, "kdLeavesTexture"), 4);
        
        // used to update geometry at least once, when STATIC_GEOMETRY is true
        this.geometryUpdatedOnce = false;
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
        
        if (RENDER_TRIANGLES) {
            if (!STATIC_GEOMETRY || !this.geometryUpdatedOnce) {
                // update vertex data array
                for (var i = 0; i < g_verts.length; i++) {
                    this.vertsData[i*3+0] = g_verts[i].x;
                    this.vertsData[i*3+1] = g_verts[i].y;
                    this.vertsData[i*3+2] = g_verts[i].z;
                }
                
                // update face data array
                for (var i = 0; i < g_faces.length; i++) {
                    for (var j = 0; j < 3; j++) {
                        this.facesData[0][i*3+j] = g_faces[i].indices[j];
                    }
                    this.facesData[1][i*3+0] = g_faces[i].getMaterial().getType();
                    this.facesData[1][i*3+1] = g_faces[i].getMaterial().getId();
                }
                
                // update vertex and face data textures
                gl.activeTexture(gl.TEXTURE0);
                gl.bindTexture(gl.TEXTURE_2D, this.vertsTexture);
                gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.vertsTextureSize, this.vertsTextureSize, gl.RGB, gl.FLOAT, this.vertsData);
                for (var i = 0; i < 2; i++) {
                    gl.activeTexture(gl.TEXTURE1 + i);
                    gl.bindTexture(gl.TEXTURE_2D, this.facesTextures[i]);
                    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.facesTextureSize, this.facesTextureSize, gl.RGB_INTEGER, gl.UNSIGNED_INT, this.facesData[i]);
                }
            } else {
                gl.activeTexture(gl.TEXTURE0);
                gl.bindTexture(gl.TEXTURE_2D, this.vertsTexture);
                for (var i = 0; i < 2; i++) {
                    gl.activeTexture(gl.TEXTURE1 + i);
                    gl.bindTexture(gl.TEXTURE_2D, this.facesTextures[i]);
                }
            }
        
            if ((!STATIC_GEOMETRY || !this.geometryUpdatedOnce) 
                && USE_KD_TREE && (this.startTime == 0 || (time - this.startTime >= 0.30))) {
                // Build the kd-tree
                this.kdTree = new KdTree();
                var faceIndices = [];
                for (var i = 0; i < g_faces.length; i++) {
                    faceIndices.push(i);
                }
                this.kdTree.buildTree(faceIndices);
                this.kdTree.fillDataArray(this.kdTreeData);
                for (var i = 0; i < this.kdTree.leafNodes.length; i++) {
                    for (var j = 0; j < 4; j++) {
                        this.kdLeavesData[i*4+j] = this.kdTree.leafNodes[i].faceIndices[j];
                    }
                }
                
                gl.activeTexture(gl.TEXTURE3);
                gl.bindTexture(gl.TEXTURE_2D, this.kdTreeTexture);
                gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.kdTreeTextureSize, this.kdTreeTextureSize, gl.RED_INTEGER, gl.UNSIGNED_INT, this.kdTreeData);
                
                gl.activeTexture(gl.TEXTURE4);
                gl.bindTexture(gl.TEXTURE_2D, this.kdLeavesTexture);
                gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.kdLeavesTextureSize, this.kdLeavesTextureSize, gl.RGBA_INTEGER, gl.INT, this.kdLeavesData);
                
                // set kd-tree texture sizes
                this.uniforms.kdBoxMin = this.kdTree.root.min;
                this.uniforms.kdBoxMax = this.kdTree.root.max;
                
                this.startTime = time;
            } else {
                gl.activeTexture(gl.TEXTURE3);
                gl.bindTexture(gl.TEXTURE_2D, this.kdTreeTexture);
                
                gl.activeTexture(gl.TEXTURE4);
                gl.bindTexture(gl.TEXTURE_2D, this.kdLeavesTexture);
               
                if (USE_KD_TREE) {
                    this.uniforms.kdBoxMin = this.kdTree.root.min;
                    this.uniforms.kdBoxMax = this.kdTree.root.max;
                }
            }
            
            if (!this.geometryUpdatedOnce) {
                this.geometryUpdatedOnce = true;
            }
        }
        
        // set uniforms
        gl.uniform1i(gl.getUniformLocation(this.tracerProgram, "vertsTextureSize"), this.vertsTextureSize);
        gl.uniform1i(gl.getUniformLocation(this.tracerProgram, "facesTextureSize"), this.facesTextureSize);
        gl.uniform1i(gl.getUniformLocation(this.tracerProgram, "kdTreeTextureSize"), this.kdTreeTextureSize);
        gl.uniform1i(gl.getUniformLocation(this.tracerProgram, "kdLeavesTextureSize"), this.kdLeavesTextureSize);
        
        setUniforms(this.tracerProgram, this.uniforms);
        
        // render to frame buffer
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
        gl.viewport(0, 0, WIDTH * SUPERSAMPLING, HEIGHT * SUPERSAMPLING);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.textures[0], 0);
        gl.vertexAttribPointer(this.tracerVertexAttribute, 2, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        
        //var canRead = (gl.checkFramebufferStatus(gl.READ_FRAMEBUFFER) == gl.FRAMEBUFFER_COMPLETE);
        gl.readPixels(0, 0, WIDTH * SUPERSAMPLING, HEIGHT * SUPERSAMPLING, gl.RGBA, gl.UNSIGNED_BYTE, this.pixels, 0);
        this.Lbar = 0;
        var numPixels = this.pixels.length / 4;
        for (var i = 0; i < numPixels; i++) {
            var r = this.pixels[i*4+0] * L_MAX / 256.0;
            var g = this.pixels[i*4+1] * L_MAX / 256.0;
            var b = this.pixels[i*4+2] * L_MAX / 256.0;
            var L = 0.27*r + 0.67*g + 0.07*b;
            this.Lbar += Math.log(EPSILON + L) / numPixels;
        }
        this.Lbar = Math.exp(this.Lbar);
        this.sf = Math.pow((1.219 + Math.pow(L_DMAX*0.5, 0.4)) / (1.219 + Math.pow(this.Lbar, 0.4)), 2.5);
        
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, WIDTH, HEIGHT);
    }

    render() {
        // render from texture to canvas
        gl.useProgram(this.renderProgram);
        var dict = {};
        dict.Lbar = this.Lbar;
        dict.sf = this.sf;
        setUniforms(this.renderProgram, dict);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures[0]);
        gl.uniform1i(gl.getUniformLocation(this.renderProgram, "texture"), 0);
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
        var vertices = [
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
            1, 1, 0,
            0, 0, 1,
            1, 0, 1,
            0, 1, 1,
            1, 1, 1
        ];
        var indices = [
            0, 1, 1, 3, 3, 2, 2, 0,
            4, 5, 5, 7, 7, 6, 6, 4,
            0, 4, 1, 5, 2, 6, 3, 7
        ];
        
        // create vertex buffer
        this.vertexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
    
        // create index buffer
        this.indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
    
        // create line shader
        this.lineProgram = compileShader(lineVertexSource, lineFragmentSource);
        this.vertexAttribute = gl.getAttribLocation(this.lineProgram, 'vertex');
        gl.enableVertexAttribArray(this.vertexAttribute);

        this.objects = [];
        this.selectedObject = null;
        this.rayTracer = new RayTracer();
    }
    
    setScene(scene) {
        this.scene = scene;
        this.selectedObject = null;
        this.rayTracer.setScene(this.scene);
    }
    
    update(viewProjection, time) {
        this.viewProjection = viewProjection;
        this.rayTracer.update(time);
    }
    
    render() {
        this.rayTracer.render();
        
        if (this.selectedObject != null) {
            gl.useProgram(this.lineProgram);
            gl.bindTexture(gl.TEXTURE_2D, null);
            gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
            gl.vertexAttribPointer(this.vertexAttribute, 3, gl.FLOAT, false, 0, 0);
            setUniforms(this.lineProgram, {
                cubeMin: this.selectedObject.getMinCorner(),
                cubeMax: this.selectedObject.getMaxCorner(),
                modelViewProjection: 
                    new THREE.Matrix4().multiplyMatrices(this.viewProjection, 
                                            this.selectedObject.modelTransform)
            });
            gl.drawElements(gl.LINES, 24, gl.UNSIGNED_SHORT, 0);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
// class UI
////////////////////////////////////////////////////////////////////////////////
 
class UI {
    constructor() {
        camera = new THREE.PerspectiveCamera( VIEW_ANGLE, WIDTH / HEIGHT, NEAR, FAR);
        
        this.renderer = new Renderer();
        this.moving = false;
        
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
    	
    	this.stats = new Stats();
    	document.body.appendChild( this.stats.dom );
    }
    
    setScene(scene) {
      this.scene = scene;
      this.renderer.setScene(this.scene);
    }
    
    update(time) {
        this.animate(time);
        
        this.controls.update();
        camera.updateMatrixWorld(true);
        camera.updateProjectionMatrix();
        this.renderer.update(
            new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, 
                            new THREE.Matrix4().getInverse(camera.matrixWorld)), 
            time);
    }
    
    animate(time) {
        var light = this.scene.lights[0];
        
        light.position.x = 500*Math.cos((time%60)/60 * Math.PI * 2);
        light.position.z = 500*Math.cos((time%60)/60 * Math.PI * 2) - 280;
    }
    
    mouseDown(x, y) {
        var t;
        var ray = new Ray(camera.position.clone(), 
                    getPrimaryRay((x / WIDTH) * 2 - 1, 1 - (y / HEIGHT) * 2));
                    
        // test the selection box first
        if (this.renderer.selectedObject != null) {
            var minBounds = this.renderer.selectedObject.getMinCorner();
            var maxBounds = this.renderer.selectedObject.getMaxCorner();
            var invModelTransform = this.renderer.selectedObject.invModelTransform;
            t = Cube.intersect(ray, minBounds, maxBounds, invModelTransform);
        
            if(t < Number.MAX_VALUE) {
                var position = new THREE.Vector3().setFromMatrixPosition(this.renderer.selectedObject.modelTransform);
                this.oldPosition = position.clone();
                position.applyMatrix4(new THREE.Matrix4().getInverse(camera.matrixWorld));
                this.zPos = position.z;
                
                this.moving = true;
                this.controls.enabled = false;
        
                return true;
            }
        }
                    
        t = Number.MAX_VALUE;
        this.renderer.selectedObject = null;

        for (var i = 0; i < this.scene.objects.length; i++) {
            var objectT = this.scene.objects[i].intersect(ray);
            if(objectT < t) {
                t = objectT;
                this.renderer.selectedObject = this.scene.objects[i];
            }
        }
        
        if (t < Number.MAX_VALUE) {
            this.controls.enabled = false;
        } else {
            this.controls.enabled = true;
        }
        
        return (t < Number.MAX_VALUE);
    }
    
    mouseMove(x, y) {
        if (this.moving) {
            var pos = new THREE.Vector3( (x / WIDTH) * 2 - 1, 1 - (y / HEIGHT) * 2, 0.5 ).unproject( camera );
            var dir = pos.sub( camera.position ).normalize().transformDirection(new THREE.Matrix4().getInverse(camera.matrixWorld));
            var distance = this.zPos / dir.z;
            var newPosition = dir.multiplyScalar( distance ).applyMatrix4(camera.matrixWorld);

            // Move the object
            this.renderer.selectedObject.translate(newPosition.clone().sub(this.oldPosition));
            
            this.oldPosition = newPosition;
        }
    }
    
    mouseUp(x, y) {
        if (this.moving) {
            var pos = new THREE.Vector3( (x / WIDTH) * 2 - 1, 1 - (y / HEIGHT) * 2, 0.5 ).unproject( camera );
            var dir = pos.sub( camera.position ).normalize().transformDirection(new THREE.Matrix4().getInverse(camera.matrixWorld));
            var distance = this.zPos / dir.z;
            var newPosition = dir.multiplyScalar( distance ).applyMatrix4(camera.matrixWorld);
            
            // Move the object
            this.renderer.selectedObject.translate(newPosition.clone().sub(this.oldPosition));
            
            this.oldPosition = newPosition;
            
            this.moving = false;
        }
        this.controls.enabled = true;
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
        gl = canvas.getContext('webgl2'); 
    } catch(e) {
    }
    
    if (gl) {
        WIDTH = canvas.clientWidth.toFixed(1);
        HEIGHT = canvas.clientHeight.toFixed(1);
        
        // TODO: find better way to load 3D models
        loadPLY(MODEL_FILE_NAME).then(function(data) {
            var meshes = {
                [MODEL_FILE_NAME]: data
            };
            
            ui = new UI();
            var scene = generateScene(meshes);
            ui.setScene(scene);
            var start = new Date();
            setInterval(function(){ tick((new Date() - start) * 0.001); }, 1000 / 60);
        });
        
    } else {
        alert('Your browser does not support WebGL.');
    }
}

/**
 * Update function
 */

function tick(time) {
    ui.stats.begin();
    
    ui.update(time);
    ui.render();
    
    ui.stats.end();
}

function loadOBJ(filename) {
    function getTris() {
        return new Promise( function( resolve, reject ) {
            var loader = new THREE.OBJLoader();    
            loader.load(filename, function (object) {
        		object.traverse( function ( child ) {
        			if ( child instanceof THREE.Mesh ) {
        			    var verts = [];
        			    var faces = [];
        			    
        			    var geometry = new THREE.Geometry().fromBufferGeometry( child.geometry );
        			    //geometry.computeFaceNormals();
                        //geometry.mergeVertices();
                        //geometry.computeVertexNormals();
                        for (var i = 0; i < geometry.vertices.length; i++) {
                            verts.push(geometry.vertices[i].clone());
                        }
                        for (var i = 0; i < geometry.faces.length; i++) {
                            faces.push(new Triangle(
                                geometry.faces[i].a,
                                geometry.faces[i].b,
                                geometry.faces[i].c
                            ));
                        }
                        
                        resolve({
                            verts: verts,
                            faces: faces
                        });
        			}
        		});
            });
        });
    }
    
    return getTris();
}

function loadPLY(filename) {
    function getTris() {
        return new Promise( function( resolve, reject ) {
            var loader = new THREE.PLYLoader();    
            loader.load(filename, function (object) {
			    var verts = [];
			    var faces = [];
			    
			    var geometry = new THREE.Geometry().fromBufferGeometry( object );
			    //geometry.computeFaceNormals();
                //geometry.mergeVertices();
                //geometry.computeVertexNormals();
                for (var i = 0; i < geometry.vertices.length; i++) {
                    verts.push(geometry.vertices[i].clone());
                }
                for (var i = 0; i < geometry.faces.length; i++) {
                    faces.push(new Triangle(
                        geometry.faces[i].a,
                        geometry.faces[i].b,
                        geometry.faces[i].c
                    ));
                }
                
                resolve({
                    verts: verts,
                    faces: faces
                });
            });
        });
    }
    
    return getTris();
}

/**
 * Generates the scene to be rendered
 */

function generateScene(meshes) {
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
    
    var sphere1Material = new Phong(nextPhongId++, 0.075, 0.075, 0.2, 20, 0.01, 0.8, 0.95, 
        new THREE.Vector3(1.0, 1.0, 1.0), 
        new THREE.Vector3(1.0, 1.0, 1.0));
        
    var sphere2Material = new Phong(nextPhongId++, 0.15, 0.25, 1.0, 20, 0.75, 0, 0, 
        new THREE.Vector3(0.7, 0.7, 0.7), 
        new THREE.Vector3(1.0, 1.0, 1.0));
        
    var cubeMaterial = new Phong(nextPhongId++, 0.5, 0.9, 0.2, 24, 0, 0, 0, 
        new THREE.Vector3(0.8, 0.1, 0.7), 
        new THREE.Vector3(0.7, 0.7, 0.7));
    
    var cylMaterial = new Phong(nextPhongId++, 0.5, 0.3, 0.9, 20, 0, 0, 0, 
        new THREE.Vector3(0.85, 0.7, 0.1), 
        new THREE.Vector3(0.7, 0.7, 0.7));
    
    var floorMaterial = new PhongCheckered(nextPhongCheckeredId++, 0.5, 0.9, 0.2, 12, 0.4, 0, 0, 
        new THREE.Vector3(1, 0, 0), new THREE.Vector3(1, 1, 0), 
        new THREE.Vector3(0.7, 0.7, 0.7));
    
    var meshMaterial = new Phong(nextPhongId++, 0.7, 0.6, 0.7, 16, 0, 0, 0, 
        new THREE.Vector3(0.1, 0.75, 0.1), 
        new THREE.Vector3(0.7, 0.7, 0.7));
    
    materials.push(sphere1Material);
    materials.push(sphere2Material);
    materials.push(cubeMaterial);
    materials.push(cylMaterial);
    materials.push(floorMaterial);
    materials.push(meshMaterial);
    
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
    floor.translate(new THREE.Vector3(-105, -80, -300));
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

    g_verts = [];
    g_faces = [];
    
    var mesh = new Mesh(nextObjectId++, meshes[MODEL_FILE_NAME].verts, meshes[MODEL_FILE_NAME].faces);
    mesh.scale(new THREE.Vector3(1000, 1000, 1000));
    //mesh.rotate(new THREE.Euler(-Math.PI / 3, 0, -Math.PI / 6, 'XYZ'));
    mesh.translate(new THREE.Vector3(15, 27, -245));
    mesh.setMaterial(meshMaterial);
    
    objects.push(sphere1);
    objects.push(sphere2);
    objects.push(floor);
    objects.push(cube);
    objects.push(cylinder);
    objects.push(mesh);
    
    return {
        objects: objects, 
        materials: materials,
        lights: lights,
        ambientLight: new THREE.Vector3(0.75, 0.75, 0.75),
        bgColor: new THREE.Vector3(0.1, 0.5, 0.9)
    };
}

initWebGL();

function elementPos(element) {
  var x = 0, y = 0;
  while(element.offsetParent) {
    x += element.offsetLeft;
    y += element.offsetTop;
    element = element.offsetParent;
  }
  return { x: x, y: y };
}

function eventPos(event) {
  return {
    x: event.clientX + document.body.scrollLeft + document.documentElement.scrollLeft,
    y: event.clientY + document.body.scrollTop + document.documentElement.scrollTop
  };
}

function canvasMousePos(event) {
  var mousePos = eventPos(event);
  var canvasPos = elementPos(canvas);
  return {
    x: mousePos.x - canvasPos.x,
    y: mousePos.y - canvasPos.y
  };
}

var mouseDown = false, oldX, oldY;

canvas.onmousedown = function(event) {
    var mouse = canvasMousePos(event);
    oldX = mouse.x;
    oldY = mouse.y;

    if(mouse.x >= 0 && mouse.x < WIDTH && mouse.y >= 0 && mouse.y < HEIGHT) {
        mouseDown = !ui.mouseDown(mouse.x, mouse.y);

        // disable selection because dragging is used for moving objects
        return false;
    }

    return true;
};

canvas.onmousemove = function(event) {
    var mouse = canvasMousePos(event);

    if(mouseDown) {
        // remember this coordinate
        oldX = mouse.x;
        oldY = mouse.y;
    } else {
        ui.mouseMove(mouse.x, mouse.y);
    }
};

canvas.onmouseup = function(event) {
    mouseDown = false;

    var mouse = canvasMousePos(event);
    ui.mouseUp(mouse.x, mouse.y);
};