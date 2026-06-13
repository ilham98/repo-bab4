// Kerr (rotating) black hole — real-time geodesic ray tracer on Three.js.
//
// The black hole is ray-traced entirely in a fragment shader drawn onto a
// full-screen triangle. A Three.js PerspectiveCamera driven by OrbitControls
// supplies the view: each frame its position + basis feed the shader, so
// orbiting / zooming / panning literally moves the observer through curved
// spacetime. UnrealBloomPass adds the glare.
//
// Build:  esbuild src/black-hole.js --bundle --minify --format=iife \
//                 --outfile=black-hole.bundle.js

import * as THREE from 'three';
import { OrbitControls }   from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer }  from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass }      from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass }      from 'three/examples/jsm/postprocessing/OutputPass.js';

const errEl = document.getElementById('err');
function fail(m){ if(errEl){ errEl.style.display='grid'; errEl.textContent=m; } throw new Error(m); }
window.addEventListener('error', e => { if(errEl && errEl.style.display==='none') fail(String(e.message||e)); });

const vert = /* glsl */`
in vec3 position;
out vec2 vNdc;
void main(){ vNdc = position.xy; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const frag = /* glsl */`
precision highp float;
in vec2 vNdc;
out vec4 fragColor;

uniform float uTime;
uniform vec2  uResolution;
uniform vec3  uCamPos;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform vec3  uCamFwd;
uniform float uTanHalfFov;

/*
 * Rotating (Kerr) black hole. Geometrized units: M = 0.5 so r_s = 2M = 1,
 * spin a = 0.42 (a/M = 0.84). Spin axis = world +Y, disk in the y = 0 plane.
 * Photons are null geodesics of the Kerr metric, integrated with a Hamiltonian
 * formulation in Boyer-Lindquist (r, theta, phi): we evolve (r,theta,phi,p_r,
 * p_theta) by Hamilton's equations with conserved E = -p_t and L = p_phi,
 * taking dH/dr, dH/dtheta by finite differences of the inverse metric.
 */
#define MASS   0.5
#define A_SPIN 0.42
#define DISK_IN  1.9
#define DISK_OUT 9.0
#define ESCAPE   30.0
#define MAX_STEPS 240

float hash31(vec3 p){
  p = fract(p * 0.3183099 + 0.1); p *= 17.0;
  return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float vnoise(vec3 p){
  vec3 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
  return mix(mix(mix(hash31(i+vec3(0,0,0)),hash31(i+vec3(1,0,0)),f.x),
                 mix(hash31(i+vec3(0,1,0)),hash31(i+vec3(1,1,0)),f.x),f.y),
             mix(mix(hash31(i+vec3(0,0,1)),hash31(i+vec3(1,0,1)),f.x),
                 mix(hash31(i+vec3(0,1,1)),hash31(i+vec3(1,1,1)),f.x),f.y),f.z);
}
float fbm(vec3 p){ float a=0.5,s=0.0; for(int i=0;i<5;i++){ s+=a*vnoise(p); p*=2.02; a*=0.5; } return s; }

vec3 starField(vec3 dir){
  vec3 col = vec3(0.0);
  for(int k=0;k<3;k++){
    float sc = 70.0 + float(k)*110.0;
    vec3 p = dir*sc, id = floor(p), fr = fract(p)-0.5;
    float h = hash31(id + float(k)*11.7);
    if(h > 0.94){
      vec3 jit = vec3(hash31(id+1.3),hash31(id+2.7),hash31(id+4.1)) - 0.5;
      float d = length(fr - jit*0.7);
      float br = smoothstep(0.16,0.0,d) * (h-0.94)/0.06;
      vec3 sc2 = mix(vec3(0.65,0.78,1.0), vec3(1.0,0.86,0.62), hash31(id+9.1));
      col += sc2 * br * 1.4;
    }
  }
  float n = fbm(dir*3.0 + 11.0);
  vec3 neb = mix(vec3(0.015,0.01,0.035), vec3(0.05,0.02,0.06), n) * n * 0.5;
  float band = exp(-pow(dir.y*2.6, 2.0));
  neb += vec3(0.03,0.045,0.08) * band * 0.35 * (0.4 + 0.6*fbm(dir*5.0));
  return col + neb;
}

void metric(float r, float th, out float gtt, out float gtp, out float gpp,
            out float grr, out float gthth){
  float s2 = max(sin(th)*sin(th), 1e-4);
  float c2 = 1.0 - s2;
  float a  = A_SPIN;
  float Sig = r*r + a*a*c2;
  float Del = r*r - 2.0*MASS*r + a*a;
  float r2a2 = r*r + a*a;
  gtt   = -((r2a2*r2a2 - a*a*Del*s2) / (Sig*Del));
  gtp   = -(2.0*MASS*r*a) / (Sig*Del);
  gpp   = (Del - a*a*s2) / (Sig*Del*s2);
  grr   = Del / Sig;
  gthth = 1.0 / Sig;
}
float hamiltonian(float r, float th, float pr, float pth, float pt, float L){
  float gtt,gtp,gpp,grr,gthth;
  metric(r, th, gtt,gtp,gpp,grr,gthth);
  return 0.5*(gtt*pt*pt + 2.0*gtp*pt*L + gpp*L*L + grr*pr*pr + gthth*pth*pth);
}
void geoDeriv(float r,float th,float ph,float pr,float pth,float pt,float L,
              out float dr,out float dth,out float dph,out float dpr,out float dpth){
  float gtt,gtp,gpp,grr,gthth;
  metric(r, th, gtt,gtp,gpp,grr,gthth);
  dr  = grr*pr;
  dth = gthth*pth;
  dph = gtp*pt + gpp*L;
  float e = 1e-3;
  dpr  = -(hamiltonian(r+e,th,pr,pth,pt,L) - hamiltonian(r-e,th,pr,pth,pt,L)) / (2.0*e);
  dpth = -(hamiltonian(r,th+e,pr,pth,pt,L) - hamiltonian(r,th-e,pr,pth,pt,L)) / (2.0*e);
}
void deriv(float y[5], float pt, float L, out float d[5]){
  geoDeriv(y[0],y[1],y[2],y[3],y[4], pt, L, d[0],d[1],d[2],d[3],d[4]);
}

vec3 diskEmission(float cr, float ph, vec3 pdir){
  float rad = clamp((cr - DISK_IN) / (DISK_OUT - DISK_IN), 0.0, 1.0);
  float orbit = uTime * (0.9 / pow(cr, 1.5));
  vec3  q = vec3(cos(ph+orbit), sin(ph+orbit), 0.0) * cr;
  float turb = fbm(vec3(ph*2.0 - orbit*3.0, cr*0.9, 0.0))*0.6 + fbm(q*1.3 + 5.0)*0.6;
  float density = smoothstep(0.35, 1.0, turb) * (0.6 + 0.7*(1.0-rad));
  float temp = pow(1.0 - rad, 0.75);
  vec3  base = mix(vec3(1.0,0.55,0.18), vec3(0.75,0.86,1.0), temp);
  float sp = sin(ph), cp = cos(ph);
  vec3  vorb  = vec3(-sp, 0.0, cp);                 // prograde orbital direction
  float beta  = clamp(sqrt(0.5 / max(cr,1.2)), 0.0, 0.88);
  float gamma = 1.0 / sqrt(1.0 - beta*beta);
  vec3  toObs = -pdir;
  float doppler = 1.0 / (gamma * (1.0 - beta*dot(vorb,toObs)));
  float grav    = sqrt(max(1e-4, 1.0 - 1.0/cr));    // gravitational redshift
  float g = doppler * grav;
  float boost = clamp(pow(g, 4.0), 0.0, 7.0);
  vec3  shift = mix(vec3(1.0,0.42,0.20), vec3(0.7,0.82,1.25), clamp((g-0.55)/0.9, 0.0, 1.0));
  float brightness = density * (0.5 + 1.1*(1.0-rad)) * boost;
  return base * shift * brightness * 0.42;
}

void blBasis(float th, float ph, out vec3 er, out vec3 et, out vec3 ep){
  float st=sin(th), ct=cos(th), sp=sin(ph), cp=cos(ph);
  er = vec3(st*cp, ct, st*sp);
  et = vec3(ct*cp, -st, ct*sp);
  ep = vec3(-sp, 0.0, cp);
}

void main(){
  vec2 uv = vec2(vNdc.x * uResolution.x / uResolution.y, vNdc.y);
  vec3 rd = normalize(uCamFwd + uCamRight*uv.x*uTanHalfFov + uCamUp*uv.y*uTanHalfFov);

  vec3  P = uCamPos;
  float a = A_SPIN;
  float rho2 = dot(P,P);
  float r2 = 0.5*((rho2 - a*a) + sqrt((rho2 - a*a)*(rho2 - a*a) + 4.0*a*a*P.y*P.y));
  float r0  = sqrt(r2);
  float th0 = acos(clamp(P.y / r0, -1.0, 1.0));
  float ph0 = atan(P.z, P.x);

  vec3 er, et, ep; blBasis(th0, ph0, er, et, ep);
  float pr0  = dot(rd, er);
  float pth0 = r0 * dot(rd, et);
  float pphi = r0 * sin(th0) * dot(rd, ep);
  float pt   = -1.0;
  float L    = pphi;

  float y[5];
  y[0]=r0; y[1]=th0; y[2]=ph0; y[3]=pr0; y[4]=pth0;

  float rplus = MASS + sqrt(max(0.0, MASS*MASS - a*a));
  vec3  diskCol = vec3(0.0); float diskA = 0.0;
  bool  captured = false;
  float prevCos = cos(y[1]);

  for(int i=0;i<MAX_STEPS;i++){
    float r   = y[0];
    float Del = r*r - 2.0*MASS*r + a*a;
    if(r < rplus*1.04 || Del < 0.03){ captured = true; break; }
    if(r > ESCAPE) break;

    float h = 0.14 * (1.0 + 0.12*r);
    float k1[5],k2[5],k3[5],k4[5],tmp[5],yn[5];
    deriv(y, pt, L, k1);
    for(int j=0;j<5;j++) tmp[j] = y[j] + 0.5*h*k1[j];
    deriv(tmp, pt, L, k2);
    for(int j=0;j<5;j++) tmp[j] = y[j] + 0.5*h*k2[j];
    deriv(tmp, pt, L, k3);
    for(int j=0;j<5;j++) tmp[j] = y[j] + h*k3[j];
    deriv(tmp, pt, L, k4);
    for(int j=0;j<5;j++) yn[j] = y[j] + h/6.0*(k1[j]+2.0*k2[j]+2.0*k3[j]+k4[j]);
    yn[1] = clamp(yn[1], 0.02, 3.12139);

    float newCos = cos(yn[1]);
    if(prevCos*newCos < 0.0 && diskA < 0.99){
      float t  = prevCos / (prevCos - newCos);
      float rc = mix(y[0], yn[0], t);
      if(rc > DISK_IN && rc < DISK_OUT){
        float phc = mix(y[2], yn[2], t);
        float ic0=mix(y[0],yn[0],t), ic1=mix(y[1],yn[1],t),
              ic3=mix(y[3],yn[3],t), ic4=mix(y[4],yn[4],t);
        float d0,d1,d2,d3,d4;
        geoDeriv(ic0,ic1,phc,ic3,ic4,pt,L,d0,d1,d2,d3,d4);
        vec3 er2,et2,ep2; blBasis(ic1, phc, er2, et2, ep2);
        vec3 pdir = normalize(d0*er2 + ic0*d1*et2 + ic0*sin(ic1)*d2*ep2);
        vec3 em = diskEmission(rc, phc, pdir);
        float al = clamp(length(em)*0.6, 0.0, 1.0) * 0.85;
        diskCol += (1.0 - diskA) * em;
        diskA    = diskA + (1.0 - diskA) * al;
      }
    }
    prevCos = newCos;
    for(int j=0;j<5;j++) y[j] = yn[j];
  }

  vec3 bg = vec3(0.0);
  if(!captured){
    float d0,d1,d2,d3,d4;
    geoDeriv(y[0],y[1],y[2],y[3],y[4],pt,L,d0,d1,d2,d3,d4);
    vec3 er3,et3,ep3; blBasis(y[1], y[2], er3, et3, ep3);
    vec3 dir = normalize(d0*er3 + y[0]*d1*et3 + y[0]*sin(y[1])*d2*ep3);
    bg = starField(dir);
  }
  fragColor = vec4(diskCol + (1.0 - diskA) * bg, 1.0);   // linear HDR for bloom
}
`;

// ---- renderer ----
const renderer = new THREE.WebGLRenderer({ antialias:false, powerPreference:'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
document.body.appendChild(renderer.domElement);

// ---- full-screen quad scene that the shader draws into ----
const quadScene = new THREE.Scene();
const quadCam   = new THREE.Camera();
const uniforms = {
  uTime:       { value: 0 },
  uResolution: { value: new THREE.Vector2() },
  uCamPos:     { value: new THREE.Vector3() },
  uCamRight:   { value: new THREE.Vector3() },
  uCamUp:      { value: new THREE.Vector3() },
  uCamFwd:     { value: new THREE.Vector3() },
  uTanHalfFov: { value: 0 },
};
const material = new THREE.RawShaderMaterial({
  vertexShader: vert, fragmentShader: frag, uniforms,
  glslVersion: THREE.GLSL3, depthTest:false, depthWrite:false,
});
quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));

// ---- observer camera driven by OrbitControls ----
const bhCamera = new THREE.PerspectiveCamera(
  50, window.innerWidth/window.innerHeight, 0.1, 1000);
bhCamera.position.set(0.0, 2.6, 15.0);

const controls = new OrbitControls(bhCamera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 7.0;
controls.maxDistance = 40.0;
controls.autoRotate = true;          // keep it gently orbiting even when idle
controls.autoRotateSpeed = 0.45;
controls.update();

// ---- post-processing: bloom glare ----
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(quadScene, quadCam));
const bloom = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  0.75,   // strength
  0.45,   // radius
  0.85);  // threshold
composer.addPass(bloom);
composer.addPass(new OutputPass());

// ---- camera -> shader uniforms ----
const _fwd = new THREE.Vector3(), _right = new THREE.Vector3(),
      _up = new THREE.Vector3(), _wUp = new THREE.Vector3(0,1,0);
function syncCamera(){
  _fwd.copy(controls.target).sub(bhCamera.position).normalize();
  _right.crossVectors(_fwd, _wUp).normalize();
  _up.crossVectors(_right, _fwd);
  uniforms.uCamPos.value.copy(bhCamera.position);
  uniforms.uCamFwd.value.copy(_fwd);
  uniforms.uCamRight.value.copy(_right);
  uniforms.uCamUp.value.copy(_up);
  uniforms.uTanHalfFov.value = Math.tan(0.5 * bhCamera.fov * Math.PI/180.0);
}

function onResize(){
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h);
  composer.setSize(w, h);
  bhCamera.aspect = w/h; bhCamera.updateProjectionMatrix();
  const dpr = renderer.getPixelRatio();
  uniforms.uResolution.value.set(w*dpr, h*dpr);
}
window.addEventListener('resize', onResize);
onResize();

const clock = new THREE.Clock();
function animate(){
  requestAnimationFrame(animate);
  uniforms.uTime.value = clock.getElapsedTime();
  controls.update();          // damping + auto-rotate
  syncCamera();
  composer.render();
}
animate();
