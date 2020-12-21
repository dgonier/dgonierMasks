let CLASSES = {
  0: 'Wearing Mask',
  1: 'Not Wearing Mask',
};


const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captures = []

if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({video: true}).then(stream => {
    video.srcObject = stream
    video.play();
  });

}

const IMAGE_SIZE = 192;

const overlay = document.getElementById('overlay-container')
const mainApp = document.getElementById('main-app')

const loadModel = async () => {
  const model = await tf.loadLayersModel('./model.json');
  const webcamElement = document.getElementById('video');
  const webcam = await tf.data.webcam(webcamElement);
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
    const normalized = imgPixels.div(255.0);
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
    const result = await model.predict(batched);
    const probs = result.dataSync()

    if (probs[0] > 0.90) {
      noMask = false
      overlay.style.display = "none"
      mainApp.style.display = "block"
    }
    let maskWearProb = Math.round(probs[0]*100)
    let maskNotWearProb = Math.round(probs[1]* 100)
    // let maskImproperWear = Math.round(probs[2]* 100)
    
    document.getElementById('console').innerText = `
      ${maskWearProb}% prediction of wearing mask.
      ${maskNotWearProb}% prediction of not wearing mask.
    `
    await sleep(1000);
  }
}

loadModel()
