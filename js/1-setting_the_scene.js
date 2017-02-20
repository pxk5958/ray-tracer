/* global THREE */

var raytracer1 = function() {

// Set the scene size.
const WIDTH = 640;
const HEIGHT = 480;

// Set some camera attributes.
const VIEW_ANGLE = 45;
const NEAR = 0.1;
const FAR = 10000;

// Set the controls speeds
const ROTATE = 15;
const ZOOM = 10;
const PAN = 6;

var scene, camera, renderer, controls;
var sphere1, sphere2, ground, pointLight;


/**
 * Initializes WebGL using three.js and sets up the scene
 */
function init() {
	// Create a WebGL scene
	scene = new THREE.Scene();
	
	// Start the renderer
	renderer = new THREE.WebGLRenderer();
	renderer.setSize( WIDTH, HEIGHT );

	// Add the render window to the document
    var container = document.getElementById( 'canvas1' );
	container.appendChild( renderer.domElement );
	
	// Create a camera
	camera =
	    new THREE.PerspectiveCamera(
	        VIEW_ANGLE,
	        WIDTH/HEIGHT,
	        NEAR,
	        FAR
	    );
	
	// Add the camera to the scene.
	scene.add(camera);
	
	// Create controls
	controls = new THREE.TrackballControls( camera, renderer.domElement );
	controls.rotateSpeed = ROTATE;
	controls.zoomSpeed = ZOOM;
	controls.panSpeed = PAN;
	controls.noZoom = false;
	controls.noPan = false;
	controls.staticMoving = true;
	controls.dynamicDampingFactor = 0.3;
	controls.addEventListener( 'change', render );
	resetControls();
	
	
    // world
    
    var groundMaterial = new THREE.MeshPhongMaterial( { color: 0x00ff00, wireframe: false } );
    ground = new THREE.Mesh(
    	new THREE.PlaneGeometry(400, 1600, 32, 32),
    	groundMaterial
    	);
    ground.position.set(-85, -80, -300);
    ground.rotation.x = -Math.PI / 2;
    scene.add(ground);
	
	var sphere1Material = new THREE.MeshPhongMaterial( { color: 0xff0000, wireframe: false } );
    sphere1 = new THREE.Mesh(
		new THREE.SphereGeometry(50, 16, 16),
		sphere1Material
		);
	sphere1.position.set(2, 5, -240);
	scene.add(sphere1);
	
	var sphere2Material = new THREE.MeshPhongMaterial( { color: 0x0000ff, wireframe: false } );
    sphere2 = new THREE.Mesh(
		new THREE.SphereGeometry(40, 16, 16),
		sphere2Material
		);
	sphere2.position.set(-65, -30, -300);
	scene.add(sphere2);

	
	// lights
	
	pointLight = new THREE.PointLight(0xFFFFFF);
	pointLight.position.set(50, 200, 150);
	scene.add(pointLight);
	
	var light = new THREE.AmbientLight( 0x222222 );
	scene.add( light );
}

/**
 * Animates the scene
 */
function animate() {
	// Schedule the next frame.
	requestAnimationFrame(animate);
	
	controls.update();

	render();
}

/**
 * Renders the scene
 */
function render() {
	renderer.render( scene, camera );
}

/**
 * Resets the controls (and camera) to default state
 */
function resetControls() {
	controls.reset();
	controls.target.set(0, 0, -270);
}

return {
	init: init,
	animate: animate,
	resetControls: resetControls
}

}();

raytracer1.init();
raytracer1.animate();