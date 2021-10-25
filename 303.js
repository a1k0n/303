// Events
// init() once the page has finished loading.
window.onload = init;
var ctx, stag;
var t = 0;

// tb-303 / x0xb0x emulator. Comments in this file refer to x0xb0x components; see
// http://wiki.openmusiclabs.com/wiki/x0xb0x?action=AttachFile&do=get&target=mainboard2.png

// 640 samples per envelope update
var ENVINC = 64;
var f_smp = 44100; // samplerate
var bpm = 110;  // 141.1927;

var vco_period = 0.0, vco_scale,
    vco_k = 0;

var vcf_cutoff = 0, vcf_envmod = 0,
    vcf_reso = 0, vcf_rescoeff = 0,
    vcf_decay = 0, vcf_envdecay = 0,
    vcf_envpos = ENVINC;

// Five sets of filter coefficients...
// There are five poles and one zero in this filter. There is an entry in this
// table for each of 64 resonance settings and a linear fit for the pole
// location in the s plane given a cutoff frequency
//  filterpoles[0][reso] -> pole 0 constant
//  filterpoles[1][reso] -> pole 0 slope
//  filterpoles[2][reso] -> pole 1/2 pair constant real
//  filterpoles[3][reso] -> pole 1/2 pair constant imag
//  filterpoles[4][reso] -> pole 1/2 pair slope real
//  filterpoles[5][reso] -> pole 1/2 pair slope imag
//  filterpoles[6][reso] -> pole 3/4 pair constant real
//  filterpoles[7][reso] -> pole 3/4 pair constant imag
//  filterpoles[8][reso] -> pole 3/4 pair slope real
//  filterpoles[9][reso] -> pole 3/4 pair slope imag

// The first pole/zero pair forms a 100Hz high-pass filter (but resonance
// feedback slightly affects the pole location / cutoff) formed by C27/R113 +
// C25/R114.  This 100Hz high pass filter in the feedback path is part of what
// makes it sound like a 303; without it, it's too resonant at lower
// frequencies and sounds like it's whistling fifths and octaves of the
// fundamental.
//
// The other four poles are of course the diode ladder filter; the
// linear fit is good down to about 200Hz cutoff, which is about the minimum
// it gets down to anyway.

var filterpoles = [
  [ -1.42475857e-02,  -1.10558351e-02,  -9.58097367e-03,
  -8.63568249e-03,  -7.94942757e-03,  -7.41570560e-03,
  -6.98187179e-03,  -6.61819537e-03,  -6.30631927e-03,
  -6.03415378e-03,  -5.79333654e-03,  -5.57785533e-03,
  -5.38325013e-03,  -5.20612558e-03,  -5.04383985e-03,
  -4.89429884e-03,  -4.75581571e-03,  -4.62701254e-03,
  -4.50674977e-03,  -4.39407460e-03,  -4.28818259e-03,
  -4.18838855e-03,  -4.09410427e-03,  -4.00482112e-03,
  -3.92009643e-03,  -3.83954259e-03,  -3.76281836e-03,
  -3.68962181e-03,  -3.61968451e-03,  -3.55276681e-03,
  -3.48865386e-03,  -3.42715236e-03,  -3.36808777e-03,
  -3.31130196e-03,  -3.25665127e-03,  -3.20400476e-03,
  -3.15324279e-03,  -3.10425577e-03,  -3.05694308e-03,
  -3.01121207e-03,  -2.96697733e-03,  -2.92415989e-03,
  -2.88268665e-03,  -2.84248977e-03,  -2.80350622e-03,
  -2.76567732e-03,  -2.72894836e-03,  -2.69326825e-03,
  -2.65858922e-03,  -2.62486654e-03,  -2.59205824e-03,
  -2.56012496e-03,  -2.52902967e-03,  -2.49873752e-03,
  -2.46921570e-03,  -2.44043324e-03,  -2.41236091e-03,
  -2.38497108e-03,  -2.35823762e-03,  -2.33213577e-03,
  -2.30664208e-03,  -2.28173430e-03,  -2.25739130e-03,
  -2.23359302e-03],
  [  1.63323670e-16,  -1.61447133e-02,  -1.99932070e-02,
  -2.09872000e-02,  -2.09377795e-02,  -2.04470150e-02,
  -1.97637613e-02,  -1.90036975e-02,  -1.82242987e-02,
  -1.74550383e-02,  -1.67110053e-02,  -1.59995606e-02,
  -1.53237941e-02,  -1.46844019e-02,  -1.40807436e-02,
  -1.35114504e-02,  -1.29747831e-02,  -1.24688429e-02,
  -1.19916965e-02,  -1.15414484e-02,  -1.11162818e-02,
  -1.07144801e-02,  -1.03344362e-02,  -9.97465446e-03,
  -9.63374867e-03,  -9.31043725e-03,  -9.00353710e-03,
  -8.71195702e-03,  -8.43469084e-03,  -8.17081077e-03,
  -7.91946102e-03,  -7.67985179e-03,  -7.45125367e-03,
  -7.23299254e-03,  -7.02444481e-03,  -6.82503313e-03,
  -6.63422244e-03,  -6.45151640e-03,  -6.27645413e-03,
  -6.10860728e-03,  -5.94757730e-03,  -5.79299303e-03,
  -5.64450848e-03,  -5.50180082e-03,  -5.36456851e-03,
  -5.23252970e-03,  -5.10542063e-03,  -4.98299431e-03,
  -4.86501921e-03,  -4.75127814e-03,  -4.64156716e-03,
  -4.53569463e-03,  -4.43348032e-03,  -4.33475462e-03,
  -4.23935774e-03,  -4.14713908e-03,  -4.05795659e-03,
  -3.97167614e-03,  -3.88817107e-03,  -3.80732162e-03,
  -3.72901453e-03,  -3.65314257e-03,  -3.57960420e-03,
  -3.50830319e-03],
  [ -1.83545593e-06,  -1.35008051e-03,  -1.51527847e-03,
  -1.61437715e-03,  -1.68536679e-03,  -1.74064961e-03,
  -1.78587681e-03,  -1.82410854e-03,  -1.85719118e-03,
  -1.88632533e-03,  -1.91233586e-03,  -1.93581405e-03,
  -1.95719818e-03,  -1.97682215e-03,  -1.99494618e-03,
  -2.01177700e-03,  -2.02748155e-03,  -2.04219657e-03,
  -2.05603546e-03,  -2.06909331e-03,  -2.08145062e-03,
  -2.09317612e-03,  -2.10432901e-03,  -2.11496056e-03,
  -2.12511553e-03,  -2.13483321e-03,  -2.14414822e-03,
  -2.15309131e-03,  -2.16168985e-03,  -2.16996830e-03,
  -2.17794867e-03,  -2.18565078e-03,  -2.19309254e-03,
  -2.20029023e-03,  -2.20725864e-03,  -2.21401130e-03,
  -2.22056055e-03,  -2.22691775e-03,  -2.23309332e-03,
  -2.23909688e-03,  -2.24493730e-03,  -2.25062280e-03,
  -2.25616099e-03,  -2.26155896e-03,  -2.26682328e-03,
  -2.27196010e-03,  -2.27697514e-03,  -2.28187376e-03,
  -2.28666097e-03,  -2.29134148e-03,  -2.29591970e-03,
  -2.30039977e-03,  -2.30478562e-03,  -2.30908091e-03,
  -2.31328911e-03,  -2.31741351e-03,  -2.32145721e-03,
  -2.32542313e-03,  -2.32931406e-03,  -2.33313263e-03,
  -2.33688133e-03,  -2.34056255e-03,  -2.34417854e-03,
  -2.34773145e-03],
  [ -2.96292613e-06,   6.75138822e-04,   6.96581050e-04,
  7.04457808e-04,   7.07837502e-04,   7.09169651e-04,
  7.09415480e-04,   7.09031433e-04,   7.08261454e-04,
  7.07246872e-04,   7.06074484e-04,   7.04799978e-04,
  7.03460301e-04,   7.02080606e-04,   7.00678368e-04,
  6.99265907e-04,   6.97852005e-04,   6.96442963e-04,
  6.95043317e-04,   6.93656323e-04,   6.92284301e-04,
  6.90928882e-04,   6.89591181e-04,   6.88271928e-04,
  6.86971561e-04,   6.85690300e-04,   6.84428197e-04,
  6.83185182e-04,   6.81961088e-04,   6.80755680e-04,
  6.79568668e-04,   6.78399727e-04,   6.77248505e-04,
  6.76114631e-04,   6.74997722e-04,   6.73897392e-04,
  6.72813249e-04,   6.71744904e-04,   6.70691972e-04,
  6.69654071e-04,   6.68630828e-04,   6.67621875e-04,
  6.66626854e-04,   6.65645417e-04,   6.64677222e-04,
  6.63721940e-04,   6.62779248e-04,   6.61848835e-04,
  6.60930398e-04,   6.60023644e-04,   6.59128290e-04,
  6.58244058e-04,   6.57370684e-04,   6.56507909e-04,
  6.55655483e-04,   6.54813164e-04,   6.53980718e-04,
  6.53157918e-04,   6.52344545e-04,   6.51540387e-04,
  6.50745236e-04,   6.49958895e-04,   6.49181169e-04,
  6.48411873e-04],
  [ -1.00014774e+00,  -1.35336624e+00,  -1.42048887e+00,
  -1.46551548e+00,  -1.50035433e+00,  -1.52916086e+00,
  -1.55392254e+00,  -1.57575858e+00,  -1.59536715e+00,
  -1.61321568e+00,  -1.62963377e+00,  -1.64486333e+00,
  -1.65908760e+00,  -1.67244897e+00,  -1.68506052e+00,
  -1.69701363e+00,  -1.70838333e+00,  -1.71923202e+00,
  -1.72961221e+00,  -1.73956855e+00,  -1.74913935e+00,
  -1.75835773e+00,  -1.76725258e+00,  -1.77584919e+00,
  -1.78416990e+00,  -1.79223453e+00,  -1.80006075e+00,
  -1.80766437e+00,  -1.81505964e+00,  -1.82225940e+00,
  -1.82927530e+00,  -1.83611794e+00,  -1.84279698e+00,
  -1.84932127e+00,  -1.85569892e+00,  -1.86193740e+00,
  -1.86804360e+00,  -1.87402388e+00,  -1.87988413e+00,
  -1.88562983e+00,  -1.89126607e+00,  -1.89679760e+00,
  -1.90222885e+00,  -1.90756395e+00,  -1.91280679e+00,
  -1.91796101e+00,  -1.92303002e+00,  -1.92801704e+00,
  -1.93292509e+00,  -1.93775705e+00,  -1.94251559e+00,
  -1.94720328e+00,  -1.95182252e+00,  -1.95637561e+00,
  -1.96086471e+00,  -1.96529188e+00,  -1.96965908e+00,
  -1.97396817e+00,  -1.97822093e+00,  -1.98241904e+00,
  -1.98656411e+00,  -1.99065768e+00,  -1.99470122e+00,
  -1.99869613e+00],
  [  1.30592376e-04,   3.54780202e-01,   4.22050344e-01,
  4.67149412e-01,   5.02032084e-01,   5.30867858e-01,
  5.55650170e-01,   5.77501296e-01,   5.97121154e-01,
  6.14978238e-01,   6.31402872e-01,   6.46637440e-01,
  6.60865515e-01,   6.74229755e-01,   6.86843408e-01,
  6.98798009e-01,   7.10168688e-01,   7.21017938e-01,
  7.31398341e-01,   7.41354603e-01,   7.50925074e-01,
  7.60142923e-01,   7.69037045e-01,   7.77632782e-01,
  7.85952492e-01,   7.94016007e-01,   8.01841009e-01,
  8.09443333e-01,   8.16837226e-01,   8.24035549e-01,
  8.31049962e-01,   8.37891065e-01,   8.44568531e-01,
  8.51091211e-01,   8.57467223e-01,   8.63704040e-01,
  8.69808551e-01,   8.75787123e-01,   8.81645657e-01,
  8.87389629e-01,   8.93024133e-01,   8.98553916e-01,
  9.03983409e-01,   9.09316756e-01,   9.14557836e-01,
  9.19710291e-01,   9.24777540e-01,   9.29762800e-01,
  9.34669099e-01,   9.39499296e-01,   9.44256090e-01,
  9.48942030e-01,   9.53559531e-01,   9.58110882e-01,
  9.62598250e-01,   9.67023698e-01,   9.71389181e-01,
  9.75696562e-01,   9.79947614e-01,   9.84144025e-01,
  9.88287408e-01,   9.92379299e-01,   9.96421168e-01,
  1.00041442e+00],
  [ -2.96209812e-06,  -2.45794824e-04,  -8.18027564e-04,
  -1.19157447e-03,  -1.46371229e-03,  -1.67529045e-03,
  -1.84698016e-03,  -1.99058664e-03,  -2.11344205e-03,
  -2.22039065e-03,  -2.31478873e-03,  -2.39905115e-03,
  -2.47496962e-03,  -2.54390793e-03,  -2.60692676e-03,
  -2.66486645e-03,  -2.71840346e-03,  -2.76809003e-03,
  -2.81438252e-03,  -2.85766225e-03,  -2.89825096e-03,
  -2.93642247e-03,  -2.97241172e-03,  -3.00642174e-03,
  -3.03862912e-03,  -3.06918837e-03,  -3.09823546e-03,
  -3.12589065e-03,  -3.15226077e-03,  -3.17744116e-03,
  -3.20151726e-03,  -3.22456591e-03,  -3.24665644e-03,
  -3.26785166e-03,  -3.28820859e-03,  -3.30777919e-03,
  -3.32661092e-03,  -3.34474723e-03,  -3.36222800e-03,
  -3.37908995e-03,  -3.39536690e-03,  -3.41109012e-03,
  -3.42628855e-03,  -3.44098902e-03,  -3.45521647e-03,
  -3.46899410e-03,  -3.48234354e-03,  -3.49528498e-03,
  -3.50783728e-03,  -3.52001812e-03,  -3.53184405e-03,
  -3.54333061e-03,  -3.55449241e-03,  -3.56534320e-03,
  -3.57589590e-03,  -3.58616273e-03,  -3.59615520e-03,
  -3.60588419e-03,  -3.61536000e-03,  -3.62459235e-03,
  -3.63359049e-03,  -3.64236316e-03,  -3.65091867e-03,
  -3.65926491e-03],
  [ -7.75894750e-06,   3.11294169e-03,   3.41779455e-03,
  3.52160375e-03,   3.55957019e-03,   3.56903631e-03,
  3.56431495e-03,   3.55194570e-03,   3.53526954e-03,
  3.51613008e-03,   3.49560287e-03,   3.47434152e-03,
  3.45275527e-03,   3.43110577e-03,   3.40956242e-03,
  3.38823540e-03,   3.36719598e-03,   3.34648945e-03,
  3.32614343e-03,   3.30617351e-03,   3.28658692e-03,
  3.26738515e-03,   3.24856568e-03,   3.23012330e-03,
  3.21205091e-03,   3.19434023e-03,   3.17698219e-03,
  3.15996727e-03,   3.14328577e-03,   3.12692791e-03,
  3.11088400e-03,   3.09514449e-03,   3.07970007e-03,
  3.06454165e-03,   3.04966043e-03,   3.03504790e-03,
  3.02069585e-03,   3.00659636e-03,   2.99274180e-03,
  2.97912486e-03,   2.96573849e-03,   2.95257590e-03,
  2.93963061e-03,   2.92689635e-03,   2.91436713e-03,
  2.90203718e-03,   2.88990095e-03,   2.87795312e-03,
  2.86618855e-03,   2.85460234e-03,   2.84318974e-03,
  2.83194618e-03,   2.82086729e-03,   2.80994883e-03,
  2.79918673e-03,   2.78857707e-03,   2.77811607e-03,
  2.76780009e-03,   2.75762559e-03,   2.74758919e-03,
  2.73768761e-03,   2.72791768e-03,   2.71827634e-03,
  2.70876064e-03],
  [ -9.99869423e-01,  -6.38561407e-01,  -5.69514530e-01,
  -5.23990915e-01,  -4.89176780e-01,  -4.60615628e-01,
  -4.36195579e-01,  -4.14739573e-01,  -3.95520699e-01,
  -3.78056805e-01,  -3.62010728e-01,  -3.47136887e-01,
  -3.33250504e-01,  -3.20208824e-01,  -3.07899106e-01,
  -2.96230641e-01,  -2.85129278e-01,  -2.74533563e-01,
  -2.64391946e-01,  -2.54660728e-01,  -2.45302512e-01,
  -2.36285026e-01,  -2.27580207e-01,  -2.19163487e-01,
  -2.11013226e-01,  -2.03110249e-01,  -1.95437482e-01,
  -1.87979648e-01,  -1.80723016e-01,  -1.73655197e-01,
  -1.66764971e-01,  -1.60042136e-01,  -1.53477393e-01,
  -1.47062234e-01,  -1.40788856e-01,  -1.34650080e-01,
  -1.28639289e-01,  -1.22750366e-01,  -1.16977645e-01,
  -1.11315866e-01,  -1.05760138e-01,  -1.00305900e-01,
  -9.49488960e-02,  -8.96851464e-02,  -8.45109223e-02,
  -7.94227260e-02,  -7.44172709e-02,  -6.94914651e-02,
  -6.46423954e-02,  -5.98673139e-02,  -5.51636250e-02,
  -5.05288741e-02,  -4.59607376e-02,  -4.14570134e-02,
  -3.70156122e-02,  -3.26345497e-02,  -2.83119399e-02,
  -2.40459880e-02,  -1.98349851e-02,  -1.56773019e-02,
  -1.15713843e-02,  -7.51574873e-03,  -3.50897732e-03,
  4.50285508e-04],
  [  1.13389002e-04,   3.50509549e-01,   4.19971782e-01,
  4.66835760e-01,   5.03053790e-01,   5.32907131e-01,
  5.58475931e-01,   5.80942937e-01,   6.01050219e-01,
  6.19296203e-01,   6.36032925e-01,   6.51518847e-01,
  6.65949666e-01,   6.79477330e-01,   6.92222311e-01,
  7.04281836e-01,   7.15735567e-01,   7.26649641e-01,
  7.37079603e-01,   7.47072578e-01,   7.56668915e-01,
  7.65903438e-01,   7.74806427e-01,   7.83404383e-01,
  7.91720644e-01,   7.99775871e-01,   8.07588450e-01,
  8.15174821e-01,   8.22549745e-01,   8.29726527e-01,
  8.36717208e-01,   8.43532720e-01,   8.50183021e-01,
  8.56677208e-01,   8.63023619e-01,   8.69229911e-01,
  8.75303138e-01,   8.81249811e-01,   8.87075954e-01,
  8.92787154e-01,   8.98388600e-01,   9.03885123e-01,
  9.09281227e-01,   9.14581119e-01,   9.19788738e-01,
  9.24907772e-01,   9.29941684e-01,   9.34893728e-01,
  9.39766966e-01,   9.44564285e-01,   9.49288407e-01,
  9.53941905e-01,   9.58527211e-01,   9.63046630e-01,
  9.67502344e-01,   9.71896424e-01,   9.76230838e-01,
  9.80507456e-01,   9.84728057e-01,   9.88894335e-01,
  9.93007906e-01,   9.97070310e-01,   1.00108302e+00,
  1.00504744e+00]];

var f0b = new Float32Array(2);  // filter 0 numerator coefficients
var f0a = new Float32Array(2);  // filter 0 denominator coefficients
var f1b = 1;  // filter 1 numerator, really just a gain compensation
var f1a = new Float32Array(3);  // filter 1 denominator coefficients
var f2b = 1;  // filter 2 numerator, same
var f2a = new Float32Array(3);  // filter 2 denominator coefficients
var f0state = new Float32Array(1);
var f1state = new Float32Array(2);
var f2state = new Float32Array(2);

var oscopectx;
var oscopedatax, oscopedatay;
var oscopewidth, oscopeheight;

var vca_mode = 2, vca_a = 0,
    vca_attack = 1.0 - 0.94406088,
    vca_decay = 0.99897516,
    vca_a0 = 0.5;

var distortion = 1;
var cliplevel = 1;

function redrawScope() {
  oscopectx.fillcolor = '#000';
  oscopectx.fillRect(0, 0, oscopewidth, oscopeheight);
  oscopectx.strokeStyle = '#55acff';
  oscopectx.beginPath();
  for (var i = 0; i < oscopewidth; i++) {
    oscopectx.lineTo(i, oscopeheight * 0.5 * (1 + oscopedatax[i]));
  }
  oscopectx.stroke();
  oscopectx.beginPath();
  oscopectx.strokeStyle = '#55ffac';
  for (var i = 0; i < oscopewidth; i++) {
    oscopectx.lineTo(i, oscopeheight * 0.5 * (1 + oscopedatay[i]));
  }
  oscopectx.stroke();
}

var pat_idx = 0;
var _pattern = [
  [46], [46], [46], [46], [53], [34], [41], [49], [34], [41], [46], [49], [29], [46], [37], [41]];
for (var i = 0; i < _pattern.length; i++) _pattern[i][0] -= 7;
function getNextRow() {
  // note, accent, slide, cutoff, resonance, envmod, decay
  //var _pattern = [[39], [], [27], [], [39], [42], [27], [], [], [39], [39],
  //[30], [], [30], [30], [], [39], [], [27], [], [39], [42], [27], [], [39], [],
  //[39], [], [30], [30], [30], [39]];
  if (pat_idx & 1) {
    pat_idx++;
    return [];
  } else {
    pat_idx++;
    return _pattern[(pat_idx >> 1) % _pattern.length];
  }
}

function recalcParams() {
  var d = (0.1 + (vcf_decay)) * f_smp;
  vcf_envdecay = Math.pow(0.1, 1.0/d * ENVINC);

  // vcf_e0 and vcf_e1 define the exponential curve envelope of the VCF cutoff
  // frequency, which is impacted by the various knobs in various ways
  /* these are what i had, originally, from reverse-engineering rebirth or something, 20 years ago
  vcf_e1 = Math.exp(6.109 + 1.5876*vcf_envmod + 2.1553*vcf_cutoff);
  vcf_e0 = Math.exp(5.613 - 0.8*vcf_envmod + 2.1553*vcf_cutoff);
  */
  // these are from measuring the voltage across R69 on my x0xb0x while holding
  // the gate high or low, and linearly fitting the cutoff frequency to current
  // (appears to be about 27.6Hz/mV_R69 + 103 Hz
  vcf_e1 = Math.exp(5.55921003 + 2.17788267*vcf_cutoff + 1.99224351*vcf_envmod) + 103;
  vcf_e0 = Math.exp(5.22617147 + 1.70418937*vcf_cutoff - 0.68382928*vcf_envmod) + 103;
  console.log(vcf_e0, vcf_e1);
  vcf_e0 *= 2 * Math.PI / f_smp;
  vcf_e1 *= 2 * Math.PI / f_smp;
  vcf_e1 -= vcf_e0;

  vcf_envpos = ENVINC;
}

function setCutoff(x) { vcf_cutoff = x; recalcParams(); }
function setReso(x) { vcf_reso = x; recalcParams(); }
function setEnvMod(x) { vcf_envmod = x; recalcParams(); }
function setDecay(x) { vcf_decay = x; recalcParams(); }
function setDistortion(x) { distortion = 1+4*x; cliplevel = Math.max(0.5, 1-x*3); }

function playNote(x) {
  vco_period = (f_smp/440.0)*Math.pow(2, -(x-57-12)/12.0);
  vco_scale = 1.0 / (vco_period|0);
  vca_mode = 0;
  vcf_c0 = vcf_e1;
  vcf_envpos = ENVINC;
}

function releaseNote() {
  vca_mode = 1;
}


function readRow(patdata) {
  //var smsg = "row " + pat_idx;
  // patdata[1] - accent (unsupported)
  // patdata[2] - slide  (same)
  /*
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
  */

  // recalcParams();
  // A-4 is concert A (440Hz)
  if(patdata[0] !== undefined) { // note
    playNote(patdata[0]);
    // smsg += " note " + patdata[0];
  } else {
    releaseNote();
  }

  // stag.innerHTML = smsg;
}

function synth(outbufL, outbufR, offset, size) {
  var w,k;
  size += offset;
  var trig_offset;
  for(var i=offset;i<size;i++) {
    // update vcf
    // FIXME: make a 640-sample inner loop and move this outside of it
    if(vcf_envpos >= ENVINC) {
      w = vcf_e0 + vcf_c0;
      vcf_c0 *= vcf_envdecay;

      // filter section 1, one zero and one pole highpass
      // pole location is affected by feedback
      //
      //
      // theoretically we could interpolate but nah
      var resoIdx = 0|(vcf_reso * 60);
      var reso_k = vcf_reso * 4.0;  // feedback strength
      var p0 = filterpoles[0][resoIdx] + w * filterpoles[1][resoIdx];
      var p1r = filterpoles[2][resoIdx] + w * filterpoles[4][resoIdx];
      var p1i = filterpoles[3][resoIdx] + w * filterpoles[5][resoIdx];
      var p2r = filterpoles[6][resoIdx] + w * filterpoles[8][resoIdx];
      var p2i = filterpoles[7][resoIdx] + w * filterpoles[9][resoIdx];
      
      // filter section 1
      var z0 = 1;  // zero @ DC
      p0 = Math.exp(p0);
      // gain @inf -> 1/(1+k); boost volume by 2, and also compensate for
      // resonance (R72)
      var targetgain = 2/(1+reso_k) + 0.5*vcf_reso;
      f0b[0] = 1;     // (z - z0) * z^-1
      f0b[1] = -z0;
      f0a[0] = 1;     // (z - p0) * z^-1
      f0a[1] = -p0;

      // adjust gain
      f0b[0] *= targetgain * (-1 - p0) / (-1 - z0);
      f0b[1] *= targetgain * (-1 - p0) / (-1 - z0);

      // (z - exp(p)) (z - exp(p*)) ->
      // z^2 - 2 z exp(Re[p]) cos(Im[p]) + exp(Re[p])^2

      var exp_p1r = Math.exp(p1r);
      f1a[0] = 1;
      f1a[1] = -2 * exp_p1r * Math.cos(p1i);
      f1a[2] = exp_p1r*exp_p1r;
      f1b = f1a[0] + f1a[1] + f1a[2];

      var exp_p2r = Math.exp(p2r);
      f2a[0] = 1;
      f2a[1] = -2 * exp_p2r * Math.cos(p2i);
      f2a[2] = exp_p2r*exp_p2r;
      f2b = f2a[0] + f2a[1] + f2a[2];

      vcf_envpos = 0;
    }

    // first stage: preamp, one-zero, one-pole highpass
    // var square = (vco_k < (vco_period>>1) ? vco_k * vco_scale : 1);
    var saw = vco_k * vco_scale - 0.5;
    var x = saw;
    outbufL[i] = x;  // hack: show the original wave on the scope
    var y = f0b[0] * x + f0state[0];
    f0state[0] = f0b[1] * x - f0a[1] * y;

    // second stage two-pole

    // first two-pole stage
    x = y;  // input to this stage is output from last stage
    y = f1b * x + f1state[0];
    f1state[0] = f1state[1] - f1a[1] * y;
    f1state[1] = -f1a[2] * y;

    // second two-pole stage (four poles total, 24dB/octave rolloff)
    x = y;
    y = f2b * x + f2state[0];
    f2state[0] = f2state[1] - f2a[1] * y;
    f2state[1] = -f2a[2] * y;

    outbufR[i] = vca_a * y;
    vcf_envpos++;

    // outbufL[i] *= distortion;
    // if(outbufL[i] > cliplevel) outbufL[i] = cliplevel;
    // if(outbufL[i] < -cliplevel) outbufL[i] = -cliplevel;
    // outbufR[i] = outbufL[i];

    // update vco
    vco_k++;
    if(vco_k >= vco_period) {
      vco_k = 0;
      if (trig_offset == undefined) {
        trig_offset = i;
      }
    }

    // update vca
    if(!vca_mode) {
      vca_a += (vca_a0 - vca_a) * vca_attack;
    } else if(vca_mode == 1) {
      vca_a *= vca_decay;
    }
  }
  return trig_offset;
}

var row_sample_idx = 0;
var samples_per_row = Math.floor(f_smp * 7.5 / bpm);
console.log("samples_per_row=", samples_per_row);
function audio_cb(e) {
  var buflen = e.outputBuffer.length;
  var dataL = e.outputBuffer.getChannelData(0);
  var dataR = e.outputBuffer.getChannelData(1);
  var offset = 0;
  var oscope_trig = undefined;

  while(buflen > 0) {
    var gen_length = Math.min(buflen, samples_per_row - row_sample_idx);
    var trig_offset = synth(dataL, dataR, offset, gen_length);
    if (oscope_trig == undefined) {
      oscope_trig = trig_offset;
    }
    offset += gen_length;
    row_sample_idx += gen_length;
    if(row_sample_idx == samples_per_row) {
      readRow(getNextRow());
      row_sample_idx = 0;
    }
    buflen -= gen_length;
  }
  t += offset;

  if (oscope_trig != undefined) {
      // && oscope_trig + oscopewidth*4 < e.outputBuffer.length) {
    for (var i = 0; i < oscopewidth; i++) {
      oscopedatax[i] = dataL[i*4 + oscope_trig];
      oscopedatay[i] = dataR[i*4 + oscope_trig];
    }
    window.requestAnimationFrame(redrawScope);
    oscope_trig = true;
  }
  for (var i = 0; i < e.outputBuffer.length; i++) {
    // undo the earlier hack for the oscilloscope
    // dataL[i] *= 0.5;
    dataL[i] = dataR[i];
  }
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
  var canvasElem = document.getElementById('c');
  oscopectx = canvasElem.getContext('2d');
  oscopewidth = canvasElem.width;
  oscopeheight = canvasElem.height;
  oscopedatax = new Float32Array(oscopewidth);
  oscopedatay = new Float32Array(oscopewidth);

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
  // makeKnob(controltag, "dist", 0.125, function(pos) { setDistortion(pos); } );

  readRow(getNextRow());
  stag.innerHTML = 'Initialized.  Press play to make terrible noise...';
}

var playing = false;
function playpause()
{
  if (!ctx) {
    ctx = new AudioContext();
    gainNode = ctx.createGain();
    gainNode.gain.value = 0.8;

    jsNode = ctx.createScriptProcessor(2048, 0, 2);
    jsNode.onaudioprocess = audio_cb;
    jsNode.connect(gainNode);
  }
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


