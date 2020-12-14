let CLASSES = {
  0: 'Wearing Mask',
  1: 'Not Wearing Mask',
};


const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captures = []

function passImageToModel(){
        var context = canvas.getContext("2d").drawImage(video, 0, 0, 640, 480);
        let input = canvas.toDataURL("image/png")
        console.log(input)
}

if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({video: true}).then(stream => {
    video.srcObject = stream
    video.play();
  });

}



const MODEL_PATH =
    'model.json';

const IMAGE_SIZE = 192;
const TOPK_PREDICTIONS = 5;

let my_model;
const demo = async (img) => {
  status('Loading model...');

  my_model = await tf.loadLayersModel(MODEL_PATH);

  status("loaded model")
  console.log(my_model)

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  // my_model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  my_model.execute(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  status('');

  // Make a prediction through the locally hosted cat.jpg.
  const catElement = img
  if (catElement.complete && catElement.naturalHeight !== 0) {
    predict(catElement);
    catElement.style.display = '';
  } else {
    catElement.onload = () => {
      predict(catElement);
      catElement.style.display = '';
    }
  }

  document.getElementById('file-container').style.display = '';
};

/**
 * Given an image element, makes a prediction through my_model returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {
  status('Predicting...');
  console.log(imgElement)
  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElement).toFloat();

    // const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    // const normalized = img.sub(offset).div(offset);
    const normalized = img.div(255.0);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    // Make a prediction through my_model.
    return my_model.execute(batched);
  });

  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(`Done in ${Math.floor(totalTime1)} ms ` +
      `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);

  // Show the classes in the DOM.
  showResults(imgElement, classes);
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from my_model.
 * @param topK The number of top predictions to show.
 */
async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}

//
// UI
//

function showResults(imgElement, classes) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'cell';
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);

    const probsElement = document.createElement('div');
    probsElement.className = 'cell';
    probsElement.innerText = classes[i].probability.toFixed(3);
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);

  predictionsElement.insertBefore(
      predictionContainer, predictionsElement.firstChild);
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');
//
// demo();

// window.setInterval(passImageToModel(), 1000)

const loadModel = async () => {
const model = await tf.loadLayersModel('./model.json');
const webcamElement = document.getElementById('video');
const webcam = await tf.data.webcam(webcamElement);
// model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

let noMask = true
const sleep = m => new Promise(r => setTimeout(r, m))


while (noMask) {
    var context = canvas.getContext("2d").drawImage(webcamElement, 0, 0, 640, 480);
    let input = canvas.toDataURL("image/jpg")
    let img = document.getElementById('imageInput');
    img.width = IMAGE_SIZE;
    img.height = IMAGE_SIZE;
    img.src = input;

    const imgPixels = tf.browser.fromPixels(webcamElement).toFloat();
    // const imgPixels = tf.fromPixels(webcamElement).toFloat();
    const normalized = imgPixels.div(255.0);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    console.log(batched.dataSync())
    const result = await model.predict(batched);
    const probs = result.dataSync()

    if (probs[0] > 0.95) {
      noMask = false
    }
    // console.log(result)
    // console.log(result.dataSync())

    document.getElementById('console').innerText = `
      ${probs[0]}% of wearing mask
    `

    // document.getElementById('console').innerText = `
    //   prediction: ${result[0].className}\n
    //   probability: ${result[0].probability}
    // `;
    // status(`${result[0].className}`)
    // Dispose the tensor to release the memory.
    // img.dispose();

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await sleep(1000);
    // await tf.nextFrame();
  }
}

loadModel()




// todo note change to setInterval
// window.setTimeout(function(){
//   console.log('hi')
//   var context = canvas.getContext("2d").drawImage(video, 0, 0, 640, 480);
//   let input = canvas.toDataURL("image/jpg")
//   let img = document.getElementById('cat');
//   img.src = input;
//   demo(img)
// }, 3000)