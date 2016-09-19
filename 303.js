// Events
// init() once the page has finished loading.
window.onload = init;
var ctx, stag;
var t = 0;

// 64 samples per envelope update
var ENVINC = 64;
var f_smp = 44100; // samplerate
var bpm = 141.1927;

var vco_inc = 0.0,
    vco_k = 0;

var vcf_cutoff = 0, vcf_envmod = 0,
    vcf_reso = 0, vcf_rescoeff = 0,
    vcf_decay = 0, vcf_envdecay = 0,
    vcf_envpos = ENVINC;

// define two biquad filter sections; each has a different conjugate pole pair
// and one zero.

// filter update:
// y[i] = x[i] * filtergain + x[i-1] * coef[0] +
//        y[i-1] * coef[1] + y[i-2] * coef[2]
// state[0] === x[i-1]
// state[1] === y[i-1]
// state[2] === y[i-2]
var filtergain1 = 0, filtergain2 = 0;
var f1coef = new Float32Array(3);
var f2coef = new Float32Array(3);
var f1state = new Float32Array(3);
var f2state = new Float32Array(3);

var vca_mode = 2, vca_a = 0,
    vca_attack = 1.0 - 0.94406088,
    vca_decay = 0.99897516,
    vca_a0 = 0.5;

var distortion = 1;
var cliplevel = 1;

var pat_idx = 0;
function getNextRow() {
  // note, accent, slide, cutoff, resonance, envmod, decay
  var pattern = [[39], [], [27], [], [39], [42], [27], [], [], [39], [39],
  [30], [], [30], [30], [], [39], [], [27], [], [39], [42], [27], [], [39], [],
  [39], [], [30], [30], [30], [39]];
  var row = pattern[pat_idx % pattern.length];
/*
  // sweep cutoff
  row[3] = 0.5+Math.sin(pat_idx*0.1)/2;
  // sweep reso
  row[4] = 0.5+Math.sin(pat_idx*0.01)/2;
  // sweep envmod
  row[5] = 0.5+Math.cos(pat_idx*0.03)/2;
  // sweep decay
  row[6] = 0.5+Math.cos(pat_idx*0.005)/2;
  */
  pat_idx++;
  return row;
}

function recalcParams()
{
  // this was sort of reverse engineered badly by looking at ReBirth output
  // waveforms umm, 18 years ago by a much more naive yours truly. this will
  // need revisiting.

  // vcf_rescoeff = Math.exp(-1.20 + 3.455*vcf_reso);
  vcf_rescoeff = Math.exp(1.4 + 2*vcf_reso);
  var d = (0.2 + (vcf_decay)) * f_smp;
  vcf_envdecay = Math.pow(0.1, 1.0/d * ENVINC);

  // vcf_e0 and vcf_e1 define the exponential curve envelope of the VCF cutoff
  // frequency, which is impacted by the various knobs in various ways
  // vcf_e1 = Math.exp(6.109 + 1.5876*vcf_envmod + 2.1553*vcf_cutoff - 1.2*(1.0-vcf_reso));
  // vcf_e0 = Math.exp(5.613 - 0.8*vcf_envmod + 2.1553*vcf_cutoff - 0.7696*(1.0-vcf_reso));
  // vcf_e0 *= Math.PI/f_smp;
  // vcf_e1 *= Math.PI/f_smp;
  // vcf_e1 -= vcf_e0;

  vcf_e0 = 0.4 * vcf_cutoff;
  vcf_e1 = 0.1 * vcf_envmod;
  vcf_envpos = ENVINC;
  console.log('vcf e: ', vcf_e0, vcf_e1, ' Q:', vcf_rescoeff);
}

function setCutoff(x) { vcf_cutoff = x; recalcParams(); }
function setReso(x) { vcf_reso = x; recalcParams(); }
function setEnvMod(x) { vcf_envmod = x; recalcParams(); }
function setDecay(x) { vcf_decay = x; recalcParams(); }
function setDistortion(x) { distortion = 1+4*x; cliplevel = Math.max(0.5, 1-x*3); }

function playNode(x) {
  vco_inc = (440.0/f_smp)*Math.pow(2, (x-57)/12.0);
  vca_mode = 0;
  vcf_c0 = vcf_e1;
  vcf_envpos = ENVINC;
}

function sustainNode() {
  vca_mode = 1;
}


function readRow(patdata) {
  //var smsg = "row " + pat_idx;
  // patdata[1] - accent (unsupported)
  // patdata[2] - slide  (same)
  if(patdata[3] !== undefined) { // cutoff
    vcf_cutoff = patdata[3];
    //smsg += " cutoff "+Math.floor(100*vcf_cutoff);
  }
  if(patdata[4] !== undefined) { // resonance
    vcf_reso = patdata[4];
    //smsg += " reso " + Math.floor(100*vcf_reso);
  }
  if(patdata[5] !== undefined) { // envmod
    vcf_envmod = patdata[5];
    // smsg += " envmod " + Math.floor(100*vcf_envmod);
  }
  if(patdata[6] !== undefined) { // decay
    vcf_decay = patdata[6];
    // smsg += " decay " + Math.floor(100*d);
  }

  // patdata[7] // accent amount

  recalcParams();
  // A-4 is concert A (440Hz)
  if(patdata[0] !== undefined) { // note
    playNode(patdata[0]);
    // smsg += " note " + patdata[0];
  } else {
    // sustainNode();
  }

  // stag.innerHTML = smsg;
}

function synth(outbufL, outbufR, offset, size) {
  var w,k;
  size += offset;
  for(var i=offset;i<size;i++) {
    // update vcf
    // FIXME: make a 64-sample inner loop and move this outside of it
    if(vcf_envpos >= ENVINC) {
      w = vcf_e0 + vcf_c0;
      vcf_c0 *= vcf_envdecay;

      // filter section 1
      // p1 = np.exp(w * (-1.0 / Q + 1j))
      // p2 = np.exp(w * (-1.15 / Q + 0.029j))
      // z1 = np.exp(w * 0.042)
      p1mag = Math.exp(-w / vcf_rescoeff);
      p1re = p1mag * Math.cos(w);
      z1 = Math.exp(w * 0.042);

      p2mag = Math.exp(-w * 1.15 / vcf_rescoeff);
      p2re = p2mag * Math.cos(w * 0.029);

      f1coef[0] = -z1;
      f1coef[1] = 2 * p1re;
      f1coef[2] = -p1mag * p1mag;

      // these two poles are right next to each other, causing the classic
      // confusion between 18dB, 3-pole filter vs 4 pole filter on the 303.
      f2coef[0] = -z1;
      f2coef[1] = 2 * p2re;
      f2coef[2] = -p2mag * p2mag;

      // evaluate gain at DC? problematic because of the high-pass zero
      // so we'll sort of fudge the gain due do the high pass effect

      // 0.95 is sort of a dumb hack here; we'll have to make sure this isn't
      // too crazy
      filtergain1 = (1 - f1coef[1] - f1coef[2]) / (0.98 + f1coef[0]);
      filtergain2 = (1 - f2coef[1] - f2coef[2]) / (0.98 + f1coef[0]);
      // / ((0.98 + f1coef[0]) * (0.98 + f2coef[0]));

      //console.log("filtergain=", filtergain, " z1=", z1);

      vcf_envpos = 0;
    }

    // first two-pole stage
    var x = vco_k * vca_a * filtergain1;
    var y = x + f1state[0] * f1coef[0] +
      f1state[1] * f1coef[1] + f1state[2] * f1coef[2];
    f1state[2] = f1state[1];
    f1state[1] = y;
    f1state[0] = x;

    // second two-pole stage (four poles total, 24dB/octave rolloff)
    x = y * filtergain2;
    y = x + f2state[0] * f2coef[0] +
      f2state[1] * f2coef[1] + f2state[2] * f2coef[2];
    f2state[2] = f2state[1];
    f2state[1] = y;
    f2state[0] = x;
    outbufL[i] = y;
    vcf_envpos++;

    outbufL[i] *= distortion;
    if(outbufL[i] > cliplevel) outbufL[i] = cliplevel;
    if(outbufL[i] < -cliplevel) outbufL[i] = -cliplevel;
    outbufR[i] = outbufL[i];

    // update vco
    vco_k += vco_inc;
    if(vco_k > 0.5) vco_k -= 1.0;

    // update vca
    if(!vca_mode) vca_a+=(vca_a0-vca_a)*vca_attack;
    else if(vca_mode == 1) {
      vca_a *= vca_decay;
    }
  }
}

var row_sample_idx = 0;
var samples_per_row = Math.floor(f_smp * 15.0 / bpm);
console.log("samples_per_row=", samples_per_row);
function audio_cb(e) {
  var buflen = e.outputBuffer.length;
  var dataL = e.outputBuffer.getChannelData(0);
  var dataR = e.outputBuffer.getChannelData(1);
  var offset = 0;

  while(buflen > 0) {
    var gen_length = Math.min(buflen, samples_per_row - row_sample_idx);
    synth(dataL, dataR, offset, gen_length);
    offset += gen_length;
    row_sample_idx += gen_length;
    if(row_sample_idx == samples_per_row) {
      readRow(getNextRow());
      row_sample_idx = 0;
    }
    buflen -= gen_length;
  }
  t += offset;
}

var active_knob, knob_mousepos;
function makeKnob(parent, desc, initialpos, move_cb) {
  var text = document.createElement('span');
  text.innerHTML = desc;
  text.style.fontSize = '8px';
  text.style.width = '32px';
  text.style.textAlign = 'center';
  text.style.display = 'inline-block';
  var div = document.createElement('div');
  var outerdiv = document.createElement('div');
  var pos = 28*4 - 96*initialpos;
  outerdiv.className = 'knobcontainer';
  div.className = 'knob';
  var update = function() {
    if(pos > 28*4) pos = 28*4;
    if(pos < 4*4) pos = 4*4;
    var qpos = ((pos>>2)-8)&31;
    div.style.backgroundPosition = (qpos*32)+"px 0px";
    move_cb((28*4-pos)/96.0);
  }
  div.addEventListener("mousedown", function(e) {
      active_knob = function(delta) {
        pos += delta;
        update();
      }
      knob_mousepos = e.clientX;
      e.preventDefault();
    }, true);
  update();
  outerdiv.appendChild(text);
  outerdiv.appendChild(div);
  parent.appendChild(outerdiv);
}

var jsNode, gainNode;
function init() {
  ctx = new AudioContext();
  gainNode = ctx.createGain();
  gainNode.gain.value = 0.15;

  jsNode = ctx.createScriptProcessor(2048, 0, 2);
  jsNode.onaudioprocess = audio_cb;
  jsNode.connect(gainNode);

  stag = document.getElementById('status');

  var body = document.getElementById('body');
  body.onmousemove = function(e) {
    if(!active_knob) return;
    var delta = knob_mousepos - e.clientX;
    knob_mousepos = e.clientX;
    active_knob(delta);
  }
  body.onmouseup = function(e) { active_knob = undefined; }
  var controltag = document.getElementById('controls');
  makeKnob(controltag, "cutoff", 0.5, function(pos) { setCutoff(pos); } );
  makeKnob(controltag, "reso", 0.7, function(pos) { setReso(pos); } );
  makeKnob(controltag, "envmod", 0.4, function(pos) { setEnvMod(pos); } );
  makeKnob(controltag, "decay", 0, function(pos) { setDecay(pos); } );
  makeKnob(controltag, "dist", 0.125, function(pos) { setDistortion(pos); } );

  readRow(getNextRow());
  stag.innerHTML = 'Initialized.  Press play to make terrible noise...';
}

var playing = false;
function playpause()
{
  if(playing) {
    gainNode.disconnect(0);
    playing = false;
    document.getElementById('playbutton').innerHTML = '&gt;';
  } else {
    gainNode.connect(ctx.destination);
    playing = true;
    document.getElementById('playbutton').innerHTML = 'pause';
  }
}


